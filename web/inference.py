# ─────────────────────────────────────────────────────────────────────────────
# inference.py — Tile inference worker and full-map batch streaming
# ─────────────────────────────────────────────────────────────────────────────
import gc
import queue
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import zarr

import config
import geo_utils
import model_manager
import renderer

# ── Inference queue & tracking set ────────────────────────────────────────────
inference_queue = queue.Queue()
processing_set: set = set()



# ── Full-map state ─────────────────────────────────────────────────────────────
_fullmap_progress: dict = {}
_fullmap_lock            = threading.Lock()
_fullmap_running: set    = set()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fill_nan_columns(grid_2d: np.ndarray, max_gap: int = 15) -> np.ndarray:
    """
    Fill periodic horizontal NaN rows caused by Zarr/embedding chunking artefacts
    by 1-D vertical linear interpolation, column by column.
    """
    H, W = grid_2d.shape
    for c in range(W):
        col = grid_2d[:, c]
        for r in range(H):
            if np.isnan(col[r]):
                r_above = r - 1
                while r_above >= 0 and np.isnan(col[r_above]):
                    r_above -= 1
                r_below = r + 1
                while r_below < H and np.isnan(col[r_below]):
                    r_below += 1
                if r_above >= 0 and r_below < H:
                    gap = r_below - r_above
                    if gap <= max_gap:
                        w = (r - r_above) / gap
                        col[r] = (1 - w) * col[r_above] + w * col[r_below]
        grid_2d[:, c] = col
    return grid_2d


# ── Tile Inference ─────────────────────────────────────────────────────────────

def run_inference(species: str, tx: int, ty: int):
    if model_manager.is_stats_species(species):
        return run_inference_stats_only(species, tx, ty)
    else:
        return run_inference_cnn(species, tx, ty)


def run_inference_stats_only(species: str, tx: int, ty: int):
    torch.set_num_threads(6)

    slug = species.lower().replace(" ", "_")
    cp   = config.CACHE_DIR / slug / str(tx) / f"{ty}.npy"
    cp.parent.mkdir(parents=True, exist_ok=True)

    if cp.exists():
        return np.load(cp), True

    minlon, minlat, maxlon, maxlat = geo_utils.tile_to_bbox(config.BACKEND_Z, tx, ty)
    lons = np.arange(minlon + config.SPACING_DEG / 2, maxlon, config.SPACING_DEG)
    lats = np.arange(maxlat - config.SPACING_DEG / 2, minlat, -config.SPACING_DEG)

    final_probs = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)

    try:
        with config.species_lock:
            try:
                sp_idx = model_manager.stats_species_list.index(species)
                target_emb = model_manager.stats_embeds_tensor[sp_idx].unsqueeze(0).to(config.DEVICE)
            except ValueError:
                # Fallback for dynamic/new species added via /species/add
                try:
                    global_idx = model_manager.species_list.index(species)
                    target_emb = model_manager.species_embeds_tensor[global_idx].unsqueeze(0).to(config.DEVICE)
                except ValueError:
                    target_emb = torch.zeros((1, 768), device=config.DEVICE)

        grid_lons, grid_lats = np.meshgrid(lons, lats)
        flat_lons = grid_lons.ravel()
        flat_lats = grid_lats.ravel()

        flat_indices, valid_in_tile = geo_utils.get_stats_indices(flat_lons, flat_lats)

        if np.any(valid_in_tile):
            active_indices = flat_indices[valid_in_tile]
            
            stats_batch = model_manager.stats_mmap[active_indices].astype(np.float32)
            stats_tensor = torch.from_numpy(stats_batch).to(config.DEVICE)
            
            with torch.no_grad():
                stats_model = model_manager.stats_model
                text_proj = F.normalize(stats_model.text_proj(target_emb), dim=-1)
                
                K = stats_tensor.shape[0]
                batch_text = target_emb.unsqueeze(0).expand(K, -1, -1)
                
                visual_features = stats_model(stats=stats_tensor, text_embeds=batch_text)
                v_norm = F.normalize(visual_features, dim=2)
                
                # Dot product similarity
                sim = (v_norm.squeeze(1) * text_proj).sum(dim=-1)
                
                t_val = model_manager.stats_t_val
                b_val = model_manager.stats_b_val
                probs = torch.sigmoid(sim * t_val + b_val).cpu().numpy()

            flat_probs = np.full(len(flat_lons), np.nan, dtype=np.float32)
            flat_probs[valid_in_tile] = probs
            final_probs = flat_probs.reshape(len(lats), len(lons))
            
            final_probs = _fill_nan_columns(final_probs, max_gap=15)

        with config.progress_lock:
            config.progress_store.setdefault(
                species, {"total": 0, "done": 0, "status": "running", "min_prob": 1.0, "max_prob": 0.0}
            )
            config.progress_store[species]['done'] += len(flat_lons)
            valid_vals = final_probs[~np.isnan(final_probs)]
            if valid_vals.size > 0:
                lmax = float(valid_vals.max())
                pos  = valid_vals[valid_vals > 0]
                lmin = float(pos.min()) if pos.size > 0 else 0.0
                if lmax > config.progress_store[species].get('max_prob', 0.0):
                    config.progress_store[species]['max_prob'] = lmax
                if lmin > 0 and lmin < config.progress_store[species].get('min_prob', 1.0):
                    config.progress_store[species]['min_prob'] = lmin

        if np.isnan(final_probs).all():
            cp.unlink(missing_ok=True)
            return None, False

        np.save(cp, final_probs)

        with config._scale_cache_lock:
            config._scale_cache.pop(species, None)
        config._tile_png_cache.invalidate_species(species)

        queued = len([t for t in processing_set if t[0] == species])
        if queued <= 1:
            threading.Thread(
                target=renderer.save_global_maps_background,
                args=(species,), daemon=True
            ).start()

        return final_probs, False

    except Exception as e:
        print(f"Stats-only inference error: {e}")
        import traceback
        traceback.print_exc()
        return final_probs, False


def run_inference_cnn(species: str, tx: int, ty: int):
    """
    Run model inference for a single backend tile (tx, ty) at BACKEND_Z zoom.
    Results are saved to the tile cache as a .npy file.
    Returns (probs_array, from_cache: bool).
    """
    import zarr
    torch.set_num_threads(2)

    slug = species.lower().replace(" ", "_")
    cp   = config.CACHE_DIR / slug / str(tx) / f"{ty}.npy"
    cp.parent.mkdir(parents=True, exist_ok=True)

    if cp.exists():
        return np.load(cp), True

    minlon, minlat, maxlon, maxlat = geo_utils.tile_to_bbox(config.BACKEND_Z, tx, ty)
    lons = np.arange(minlon + config.SPACING_DEG / 2, maxlon, config.SPACING_DEG)
    lats = np.arange(maxlat - config.SPACING_DEG / 2, minlat, -config.SPACING_DEG)

    final_probs = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)

    _store = None
    try:
        # ── Load precomputed embeddings ────────────────────────────────────────────
        zip_path = config.EMB_DIR / "context_map.zip"
        zarr_path = config.EMB_DIR / "context_map.zarr"
        mask_path = config.EMB_DIR / "valid_mask.npy"
        try:
            if zarr_path.exists():
                store = zarr.open(str(zarr_path), mode='r')
            elif zip_path.exists():
                print("📦 Auto-extracting context_map.zip into context_map.zarr...", flush=True)
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(config.EMB_DIR.resolve())
                store = zarr.open(str(zarr_path), mode='r')
            else:
                raise FileNotFoundError(f"Neither {zarr_path} nor {zip_path} exists")
            valid_mask = np.load(mask_path)
        except Exception as e:
            print(f"❌ Error loading pre-computed data in run_inference: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return final_probs, False

        # ── Project species text embedding (768D → 128D) ───────────────────────────
        cohort_indices = model_manager.get_species_cohort(species, cohort_size=15)
        with config.species_lock:
            cohort_embeds = model_manager.species_embeds_tensor[cohort_indices]  # (16, 768)
            cohort_embeds = cohort_embeds.unsqueeze(0)  # (1, 16, 768)

        with torch.no_grad():
            projected_text = model_manager.model.text_proj(cohort_embeds)  # (1, 16, 128)
            
            # Precompute the self-attention step of the first layer (layer0)
            layer0 = model_manager.model.decoders[0]
            q_sa = layer0.self_attn(projected_text, projected_text, projected_text)[0]
            queries_precomputed = layer0.norm1(projected_text + q_sa)  # (1, 16, 128)
            
            target_proj = projected_text[:, 0:1, :]  # (1, 1, 128)
            t_norm_global  = F.normalize(target_proj.float(), dim=2)

        t_val = model_manager.model.loss_fn.t.exp().item() if hasattr(model_manager.model, 'loss_fn') else 10.0
        b_val = model_manager.model.loss_fn.b.item()       if hasattr(model_manager.model, 'loss_fn') else -5.0

        # ── Build grid coords and map to embedding indices ─────────────────────────
        grid_lons, grid_lats = np.meshgrid(lons, lats)
        flat_lons = grid_lons.ravel()
        flat_lats = grid_lats.ravel()

        flat_indices, in_extent = geo_utils.get_embedding_indices(flat_lons, flat_lats)

        grid_to_flat = geo_utils.get_grid_to_flat_map()

        # Map flat indices to Zarr indices
        zarr_indices = np.full_like(flat_indices, -1, dtype=np.int32)
        if np.any(in_extent):
            zarr_indices[in_extent] = grid_to_flat[flat_indices[in_extent]]

        # Valid cells are inside extent, map to a valid Zarr index, and are marked True in valid_mask
        valid_in_tile = (zarr_indices != -1) & valid_mask[np.where(zarr_indices != -1, zarr_indices, 0)]

        if np.any(valid_in_tile):
            active_zarr_indices = zarr_indices[valid_in_tile]
            valid_embeddings = store[active_zarr_indices.tolist()].astype(np.float32)
            batch_ctx = torch.from_numpy(valid_embeddings).float()

            # ── Sub-batch decoder forward ──────────────────────────────────────────
            SUB_BATCH = 1024
            preds = []
            for j in range(0, len(batch_ctx), SUB_BATCH):
                sub = batch_ctx[j : j + SUB_BATCH]
                with torch.no_grad():
                    # Expand precomputed queries to sub-batch size
                    queries = queries_precomputed.expand(sub.shape[0], -1, -1)
                    
                    # Remaining of layer 0
                    layer0 = model_manager.model.decoders[0]
                    queries_cross = layer0.multihead_attn(queries, sub, sub)[0]
                    queries = layer0.norm2(queries + queries_cross)
                    
                    queries_ffn = layer0.linear2(layer0.activation(layer0.linear1(queries)))
                    queries = layer0.norm3(queries + queries_ffn)
                    
                    # Subsequent layers
                    for layer in model_manager.model.decoders[1:]:
                        queries = layer(queries, sub)

                    # Extract target species prediction (index 0)
                    queries_target = queries[:, 0:1, :]

                    v_norm = F.normalize(queries_target.float(), dim=-1)
                    t_norm = t_norm_global.expand(len(queries_target), -1, -1)
                    sim    = (v_norm * t_norm).sum(dim=-1)
                    scaled = sim * t_val + b_val
                    probs  = torch.sigmoid(scaled).squeeze(-1).cpu().numpy()

                preds.extend(np.atleast_1d(probs))

            flat_probs                = np.full(len(flat_lons), np.nan, dtype=np.float32)
            flat_probs[valid_in_tile] = preds
            final_probs               = flat_probs.reshape(len(lats), len(lons))

            # Heal chunking-induced horizontal NaN rows
            final_probs = _fill_nan_columns(final_probs, max_gap=15)

        # ── Update progress ────────────────────────────────────────────────────────
        with config.progress_lock:
            config.progress_store.setdefault(
                species, {"total": 0, "done": 0, "status": "running", "min_prob": 1.0, "max_prob": 0.0}
            )
            config.progress_store[species]['done'] += len(flat_lons)
            valid_vals = final_probs[~np.isnan(final_probs)]
            if valid_vals.size > 0:
                lmax = float(valid_vals.max())
                pos  = valid_vals[valid_vals > 0]
                lmin = float(pos.min()) if pos.size > 0 else 0.0
                if lmax > config.progress_store[species].get('max_prob', 0.0):
                    config.progress_store[species]['max_prob'] = lmax
                if lmin > 0 and lmin < config.progress_store[species].get('min_prob', 1.0):
                    config.progress_store[species]['min_prob'] = lmin

        if np.isnan(final_probs).all():
            cp.unlink(missing_ok=True)
            return None, False

        np.save(cp, final_probs)

        # Bust the scale and PNG caches so next requests see fresh data
        with config._scale_cache_lock:
            config._scale_cache.pop(species, None)
        config._tile_png_cache.invalidate_species(species)

        # Trigger background global PNG export if the queue is nearly empty
        queued = len([t for t in processing_set if t[0] == species])
        if queued <= 1:
            threading.Thread(
                target=renderer.save_global_maps_background,
                args=(species,), daemon=True
            ).start()

        return final_probs, False

    finally:
        if _store is not None:
            _store.close()


# ── Inference queue worker ─────────────────────────────────────────────────────

def inference_worker():
    while True:
        task = inference_queue.get()
        if task is None:
            break
        species, tx, ty = task[:3]
        try:
            run_inference(species, tx, ty)
        except Exception as e:
            print(f"Inference error for {species} ({tx},{ty}): {e}")
        finally:
            processing_set.discard(task)
            inference_queue.task_done()


# The startup function that is called from server.py startup event handler.
# This ensures that only the main web worker process starts the background threads
# and persistent pool, preventing child processes (multiprocessing workers)
# from recursively spawning threads or pools when they import inference.py.
def startup_inference_workers():
    NUM_WORKERS = 1
    for i in range(NUM_WORKERS):
        threading.Thread(target=inference_worker, name=f"inference-worker-{i}", daemon=True).start()
    
    # Preload/initialize the persistent process pool in a background thread
    threading.Thread(target=init_fullmap_pool, name="init-fullmap-pool", daemon=True).start()
    print("🚀 Inference background workers and pool initialization started.", flush=True)


def shutdown_inference_workers():
    global _fullmap_pool
    print("Stopping inference workers and cleaning up ProcessPool...", flush=True)
    
    with _fullmap_pool_lock:
        if _fullmap_pool:
            try:
                import sys
                if sys.version_info >= (3, 9):
                    _fullmap_pool.shutdown(wait=False, cancel_futures=True)
                else:
                    _fullmap_pool.shutdown(wait=False)
            except Exception as e:
                print(f"Error shutting down ProcessPool: {e}", flush=True)
            _fullmap_pool = None

    # Forcefully terminate any remaining active children (multiprocessing pool processes)
    try:
        import multiprocessing
        active = multiprocessing.active_children()
        if active:
            print(f"Cleaning up {len(active)} active child processes...", flush=True)
            for child in active:
                child.terminate()
                child.join(timeout=0.5)
    except Exception as e:
        print(f"Error terminating children: {e}", flush=True)


# ── Full-map parallel persistent Process Pool ───────────────────────────────
_fullmap_pool = None
_fullmap_pool_lock = threading.Lock()

# Worker global variables inside child processes
_worker_embeddings_slice = None
_worker_model = None
_worker_slice_idx = None

def _init_worker_persistent(slice_idx: int, num_workers: int):
    global _worker_embeddings_slice, _worker_model, _worker_slice_idx
    _worker_slice_idx = slice_idx
    # Configure PyTorch to use a single thread per worker to prevent CPU over-subscription
    torch.set_num_threads(1)
    
    # Instantiate copy of model in child process
    import copy
    _worker_model = copy.deepcopy(model_manager.model)
    _worker_model.eval()
    
    # Load this worker's specific slice of the flat embeddings grid
    flat_cache_path = config.EMB_DIR / "valid_land_embeddings_250.npy"
    full_embeddings = np.load(flat_cache_path)
    chunk_size = int(np.ceil(full_embeddings.shape[0] / num_workers))
    start_idx = slice_idx * chunk_size
    end_idx = min(start_idx + chunk_size, full_embeddings.shape[0])
    
    _worker_embeddings_slice = full_embeddings[start_idx:end_idx].astype(np.float32)

def _set_worker_slice_task(args):
    slice_idx, num_workers = args
    _init_worker_persistent(slice_idx, num_workers)
    return True

def _worker_predict_task(args):
    # args: (queries_precomputed, t_norm_val, t_val, b_val)
    queries_precomputed, t_norm_val, t_val, b_val = args
    batch_ctx = torch.from_numpy(_worker_embeddings_slice)
    
    with torch.no_grad():
        B = batch_ctx.shape[0]
        queries = queries_precomputed.expand(B, -1, -1)
        
        # Remaining of layer 0
        layer0 = _worker_model.decoders[0]
        queries_cross = layer0.multihead_attn(queries, batch_ctx, batch_ctx)[0]
        queries = layer0.norm2(queries + queries_cross)
        
        queries_ffn = layer0.linear2(layer0.activation(layer0.linear1(queries)))
        queries = layer0.norm3(queries + queries_ffn)
        
        # Subsequent layers (layer 1)
        for layer in _worker_model.decoders[1:]:
            queries = layer(queries, batch_ctx)
            
        # Extract target species prediction (index 0)
        queries_target = queries[:, 0:1, :]
        
        v_norm = F.normalize(queries_target.float(), dim=-1)
        sim = (v_norm * t_norm_val).sum(dim=-1)
        scaled = sim * t_val + b_val
        probs = torch.sigmoid(scaled).squeeze(-1).numpy()
    return _worker_slice_idx, probs

def init_fullmap_pool():
    global _fullmap_pool
    with _fullmap_pool_lock:
        if _fullmap_pool is not None:
            return
        try:
            num_workers = 4
            from concurrent.futures import ProcessPoolExecutor
            _fullmap_pool = ProcessPoolExecutor(max_workers=num_workers)
            # Initialize each worker with its slice of the flat embeddings cache
            futures = [_fullmap_pool.submit(_set_worker_slice_task, (i, num_workers)) for i in range(num_workers)]
            for f in futures:
                f.result()
            print("✅ Persistent Process Pool initialized with 4 workers.", flush=True)
        except Exception as e:
            print(f"❌ Failed to initialize parallel Process Pool: {e}. Falling back to sequential.", flush=True)
            _fullmap_pool = False


# ── Full-map batch streaming ───────────────────────────────────────────────────

def _run_fullmap(species: str):
    if model_manager.is_stats_species(species):
        return _run_fullmap_stats_only(species)
    else:
        return _run_fullmap_cnn(species)


def _run_fullmap_stats_only(species: str):
    """
    Run a 250×250 uniform grid inference over the stats extent for the full Quebec.
    """
    slug    = species.lower().replace(" ", "_")
    out_dir = config.CACHE_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    fm_path = out_dir / config.FULLMAP_NPY

    grid_n = config.FULLMAP_GRID_N
    ext = config.STATS_EXTENT
    
    lons = np.linspace(ext['minlon'], ext['maxlon'], grid_n)
    lats = np.linspace(ext['maxlat'], ext['minlat'], grid_n)
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    
    flat_lons = grid_lons.ravel()
    flat_lats = grid_lats.ravel()

    flat_indices, valid_mask_250 = geo_utils.get_stats_indices(flat_lons, flat_lats)
    total_valid = int(np.sum(valid_mask_250))

    full_grid_flat = np.full(grid_n * grid_n, np.nan, dtype=np.float32)

    with _fullmap_lock:
        _fullmap_progress[species] = {
            "status": "running", "done": 0, "total": total_valid,
            "pct": 0.0, "min_prob": 1.0, "max_prob": 0.0,
        }

    try:
        # Get target text embedding
        with config.species_lock:
            try:
                sp_idx = model_manager.stats_species_list.index(species)
                target_emb = model_manager.stats_embeds_tensor[sp_idx].unsqueeze(0).to(config.DEVICE)
            except ValueError:
                try:
                    global_idx = model_manager.species_list.index(species)
                    target_emb = model_manager.species_embeds_tensor[global_idx].unsqueeze(0).to(config.DEVICE)
                except ValueError:
                    target_emb = torch.zeros((1, 768), device=config.DEVICE)

        if total_valid > 0:
            active_indices = flat_indices[valid_mask_250]
            stats_batch_all = model_manager.stats_mmap[active_indices].astype(np.float32)

            with torch.no_grad():
                stats_model = model_manager.stats_model
                text_proj = F.normalize(stats_model.text_proj(target_emb), dim=-1)
                t_val = model_manager.stats_t_val
                b_val = model_manager.stats_b_val

                SUB_BATCH = 16384
                all_probs = []
                total_pts = len(stats_batch_all)

                for j in range(0, total_pts, SUB_BATCH):
                    chunk_stats = stats_batch_all[j : j + SUB_BATCH]
                    stats_tensor = torch.from_numpy(chunk_stats).to(config.DEVICE)
                    K = stats_tensor.shape[0]
                    batch_text = target_emb.unsqueeze(0).expand(K, -1, -1)
                    
                    visual_features = stats_model(stats=stats_tensor, text_embeds=batch_text)
                    v_norm = F.normalize(visual_features, dim=2)
                    sim = (v_norm.squeeze(1) * text_proj).sum(dim=-1)
                    
                    chunk_probs = torch.sigmoid(sim * t_val + b_val).cpu().numpy()
                    all_probs.append(chunk_probs)

                    # Update progress for real-time frontend streaming
                    done_count = min(j + K, total_pts)
                    pct = round((done_count / total_pts) * 100.0, 1)
                    with _fullmap_lock:
                        _fullmap_progress[species] = {
                            "status": "running", "done": done_count, "total": total_pts,
                            "pct": pct, "min_prob": float(chunk_probs.min()), "max_prob": float(chunk_probs.max()),
                        }

                probs = np.concatenate(all_probs)

            valid_flat_pos = np.where(valid_mask_250)[0]
            full_grid_flat[valid_flat_pos] = probs

        # Reshape to (grid_n, grid_n)
        final_probs = full_grid_flat.reshape(grid_n, grid_n)
        
        # Save to disk
        np.save(fm_path, final_probs)
        print(f"✅ Stats full map saved for: {species}", flush=True)

        # Update progress to 100% and done status
        with _fullmap_lock:
            p = _fullmap_progress.get(species)
            if p:
                p.update({"status": "done", "pct": 100.0, "done": total_valid})
                valid_probs = full_grid_flat[~np.isnan(full_grid_flat)]
                if len(valid_probs) > 0:
                    p['max_prob'] = float(valid_probs.max())
                    p['min_prob'] = float(valid_probs.min())

        with config._scale_cache_lock:
            config._scale_cache.pop(species, None)
        config._tile_png_cache.invalidate_species(species)

        _fullmap_running.discard(species)

        # Launch background map rendering for global res levels
        threading.Thread(
            target=renderer.save_global_maps_background,
            args=(species,), daemon=True
        ).start()

    except Exception as e:
        print(f"Stats full-map generation error: {e}")
        with _fullmap_lock:
            _fullmap_progress[species] = {"status": "error", "pct": 0.0}
        _fullmap_running.discard(species)


def _run_fullmap_cnn(species: str):
    """
    Run a 250×250 uniform grid inference over the full embedding extent.
    Uses a persistent Process Pool to parallelize predictions across 4 CPU cores.
    Saves result to fullmap.npy in the species tile cache directory.
    """
    slug    = species.lower().replace(" ", "_")
    out_dir = config.CACHE_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    fm_path = out_dir / config.FULLMAP_NPY

    flat_cache_path = config.EMB_DIR / "valid_land_embeddings_250.npy"
    mask_path       = config.EMB_DIR / "valid_mask.npy"

    try:
        valid_mask = np.load(mask_path)
    except Exception as e:
        print(f"❌ Failed to load valid mask: {e}")
        with _fullmap_lock:
            _fullmap_progress[species] = {"status": "error", "pct": 0.0}
        _fullmap_running.discard(species)
        return

    # Build 250×250 index grid over the 836×1134 embedding raster
    grid_n  = 250
    r_origs = np.round(np.linspace(835, 0, grid_n)).astype(np.int32)
    c_origs = np.round(np.linspace(0, 1133, grid_n)).astype(np.int32)

    grid_to_flat = geo_utils.get_grid_to_flat_map()

    zarr_indices   = []
    valid_mask_250 = []
    for r in range(grid_n):
        r_orig = r_origs[r]
        for c in range(grid_n):
            c_orig = c_origs[c]
            flat_idx = int(r_orig) * 1134 + int(c_orig)
            zarr_idx = grid_to_flat[flat_idx]
            if zarr_idx != -1 and valid_mask[zarr_idx]:
                zarr_indices.append(int(zarr_idx))
                valid_mask_250.append(True)
            else:
                zarr_indices.append(-1)
                valid_mask_250.append(False)

    valid_mask_250      = np.array(valid_mask_250, dtype=bool)
    total_valid         = int(np.sum(valid_mask_250))

    full_grid_flat = np.full(grid_n * grid_n, np.nan, dtype=np.float32)

    with _fullmap_lock:
        _fullmap_progress[species] = {
            "status": "running", "done": 0, "total": total_valid,
            "pct": 0.0, "min_prob": 1.0, "max_prob": 0.0,
        }

    # ── Project text embedding once ───────────────────────────────────────────
    cohort_indices = model_manager.get_species_cohort(species, cohort_size=15)
    with config.species_lock:
        cohort_embeds = model_manager.species_embeds_tensor[cohort_indices]  # (16, 768)

    with torch.no_grad():
        projected_text = model_manager.model.text_proj(cohort_embeds).unsqueeze(0)  # (1, 16, 128)
        
        # Precompute the self-attention step of the first layer (layer0)
        layer0 = model_manager.model.decoders[0]
        q_sa = layer0.self_attn(projected_text, projected_text, projected_text)[0]
        queries_precomputed = layer0.norm1(projected_text + q_sa)  # (1, 16, 128)
        
        target_proj = projected_text[:, 0:1, :]  # (1, 1, 128)
        t_norm = F.normalize(target_proj.float(), dim=-1)  # (1, 1, 128)

    t_val = model_manager.model.loss_fn.t.exp().item() if hasattr(model_manager.model, 'loss_fn') else 10.0
    b_val = model_manager.model.loss_fn.b.item()       if hasattr(model_manager.model, 'loss_fn') else -5.0

    global _fullmap_pool
    if _fullmap_pool is None:
        init_fullmap_pool()

    try:
        if _fullmap_pool:
            # Parallel inference using the Process Pool
            print(f"Full-map inference: {total_valid} valid cells, running in parallel on 4 worker processes...", flush=True)
            start_time = time.time()
            
            # Submit prediction slices to all workers
            futures = [_fullmap_pool.submit(_worker_predict_task, (queries_precomputed, t_norm, t_val, b_val)) for i in range(4)]
            results = [f.result() for f in futures]
            # Sort results by slice index (first element in returned tuple) to ensure correct row stitching order
            results.sort(key=lambda x: x[0])
            probs = np.concatenate([r[1] for r in results])
            
            valid_flat_pos = np.where(valid_mask_250)[0]
            full_grid_flat[valid_flat_pos] = probs
            done_valid = len(probs)
            
            # Update progress to 100%
            with _fullmap_lock:
                p = _fullmap_progress.get(species)
                if p:
                    p['done'] = total_valid
                    p['pct']  = 100.0
                    if len(probs) > 0:
                        p['max_prob'] = float(probs.max())
                        p['min_prob'] = float(probs.min())
            print(f"Parallel inference completed in {time.time() - start_time:.3f}s", flush=True)
        else:
            # Fallback to sequential execution
            print(f"Full-map inference: {total_valid} valid cells, running sequentially (fallback)...", flush=True)
            flat_embeddings = np.load(flat_cache_path, mmap_mode='r')
            BATCH_SIZE      = 2048
            done_valid      = 0
            valid_flat_pos  = np.where(valid_mask_250)[0]
            for j in range(0, total_valid, BATCH_SIZE):
                batch_np  = flat_embeddings[j : j + BATCH_SIZE].astype(np.float32)
                batch_ctx = torch.from_numpy(batch_np)
                batch_pos = valid_flat_pos[j : j + BATCH_SIZE]

                with torch.no_grad():
                    # Expand precomputed queries to batch size
                    queries = queries_precomputed.expand(batch_ctx.shape[0], -1, -1)
                    
                    # Remaining of layer 0
                    layer0 = model_manager.model.decoders[0]
                    queries_cross = layer0.multihead_attn(queries, batch_ctx, batch_ctx)[0]
                    queries = layer0.norm2(queries + queries_cross)
                    
                    queries_ffn = layer0.linear2(layer0.activation(layer0.linear1(queries)))
                    queries = layer0.norm3(queries + queries_ffn)
                    
                    # Subsequent layers (layer 1)
                    for layer in model_manager.model.decoders[1:]:
                        queries = layer(queries, batch_ctx)
                        
                    # Extract target species prediction (index 0)
                    queries_target = queries[:, 0:1, :]
                    
                    v_norm = F.normalize(queries_target.float(), dim=-1)
                    sim    = (v_norm * t_norm).sum(dim=-1)
                    scaled = sim * t_val + b_val
                    probs  = torch.sigmoid(scaled).squeeze(-1).cpu().numpy()

                probs = np.atleast_1d(probs)
                full_grid_flat[batch_pos] = probs

                done_valid += len(probs)
                pct = round(done_valid / total_valid * 100, 1)
                with _fullmap_lock:
                    p = _fullmap_progress.get(species)
                    if p:
                        p['done'] = done_valid
                        p['pct']  = pct
                        if len(probs) > 0:
                            p['max_prob'] = max(p['max_prob'], float(probs.max()))
                            p['min_prob'] = min(p['min_prob'], float(probs.min()))
                gc.collect()

        if done_valid == 0:
            raise Exception("No valid predictions returned.")
            
        # ── Reshape, heal, save ───────────────────────────────────────────────────
        full_grid_2d = full_grid_flat.reshape(grid_n, grid_n)
        full_grid_2d = _fill_nan_columns(full_grid_2d, max_gap=10)
        
        # Apply invalid mask (flipped vertically because full_grid_2d is South-up,
        # whereas get_global_invalid_mask returns a North-up mask)
        mask = geo_utils.get_global_invalid_mask(grid_n, config.EMB_EXTENT)
        full_grid_2d[np.flipud(mask)] = np.nan

        np.save(fm_path, full_grid_2d)
        print(f"✅ Full map saved for: {species}", flush=True)

        with _fullmap_lock:
            p = _fullmap_progress.get(species)
            if p:
                p.update({"status": "done", "pct": 100.0})

        with config._scale_cache_lock:
            config._scale_cache.pop(species, None)
        config._tile_png_cache.invalidate_species(species)

        _fullmap_running.discard(species)
        threading.Thread(
            target=renderer.save_global_maps_background,
            args=(species,), daemon=True
        ).start()

    except Exception as e:
        print(f"❌ Failed to run fullmap inference: {e}")
        with _fullmap_lock:
            _fullmap_progress[species] = {"status": "error", "pct": 0.0}
        _fullmap_running.discard(species)



