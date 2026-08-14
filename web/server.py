# ─────────────────────────────────────────────────────────────────────────────
# server.py — FastAPI route handler
# All heavy logic lives in: config, geo_utils, model_manager, renderer, inference
# ─────────────────────────────────────────────────────────────────────────────
import threading
import asyncio
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

# ── Domain modules ─────────────────────────────────────────────────────────────
import config
import geo_utils          # noqa: F401 — triggers mask loading at import time
import model_manager
import renderer
import inference as inf_mod

from text_inference import generate_species_description, generate_species_embedding

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    inf_mod.startup_inference_workers()

@app.on_event("shutdown")
def shutdown_event():
    inf_mod.shutdown_inference_workers()

# ── Render thread pool ─────────────────────────────────────────────────────────
_render_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="render")

# Alias for convenience inside route handlers
progress_store = config.progress_store   # shared dict from config


# ══════════════════════════════════════════════════════════════════════════════
# Helper: compute scale stats from cached .npy tiles
# ══════════════════════════════════════════════════════════════════════════════

def get_cached_species_list() -> list:
    """Return all species that have at least one .npy tile on disk."""
    cached = []
    if not config.CACHE_DIR.exists():
        return cached
    for slug_dir in config.CACHE_DIR.iterdir():
        if not slug_dir.is_dir():
            continue
        tiles = list(slug_dir.rglob("*.npy"))
        if not tiles:
            continue
        slug = slug_dir.name
        with config.species_lock:
            match = next(
                (s for s in model_manager.species_list
                 if s.lower().replace(" ", "_") == slug),
                None
            )
        display = match if match else slug.replace("_", " ").title()
        cached.append({
            "name":       display,
            "slug":       slug,
            "tile_count": len(tiles),
            "registered": match is not None,
        })
    return sorted(cached, key=lambda x: x["name"])


def _persist_dynamic_species():
    """Saves all non-initial species embeddings to disk for next restart."""
    with config.species_lock:
        n      = len(model_manager._INITIAL_SPECIES)
        names  = model_manager.species_list[n:]
        if not names:
            return
        embeds = model_manager.species_embeds_tensor[n:].clone()
    torch.save({"names": names, "embeddings": embeds}, config.DYNAMIC_EMBEDDINGS_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def index():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/map.html")


@app.get("/health")
def health():
    with config.species_lock:
        names = list(model_manager.species_list)
    return {"status": "ok", "species_count": len(names), "species": names}


# ── Species registry ───────────────────────────────────────────────────────────

@app.get("/species/list")
def list_species():
    with config.species_lock:
        names = list(model_manager.species_list)
    cached_slugs = {c["slug"] for c in get_cached_species_list()}
    return {"species": [
        {"name": n, "slug": n.lower().replace(" ", "_"),
         "category": model_manager.get_species_category(n),
         "cached": n.lower().replace(" ", "_") in cached_slugs}
        for n in names
    ]}


@app.get("/species/cached")
def list_cached():
    return {"species": get_cached_species_list()}


class GenerateDescriptionRequest(BaseModel):
    name: str


@app.post("/species/generate_description")
def generate_description_endpoint(req: GenerateDescriptionRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "name cannot be empty")
    return {"name": name, "description": generate_species_description(name)}


class AddSpeciesRequest(BaseModel):
    name:        str
    description: Optional[str] = None


class AddSpeciesResponse(BaseModel):
    name:            str
    slug:            str
    description:     str
    already_existed: bool


@app.post("/species/add", response_model=AddSpeciesResponse)
def add_species(req: AddSpeciesRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "species name cannot be empty")

    with config.species_lock:
        if name in model_manager.species_list:
            return AddSpeciesResponse(
                name=name, slug=name.lower().replace(" ", "_"),
                description="(already registered)", already_existed=True
            )

    description = req.description or generate_species_description(name)
    try:
        embedding = generate_species_embedding(description)
    except Exception as e:
        raise HTTPException(500, f"embedding generation failed: {e}")
    if embedding.shape != (768,):
        raise HTTPException(500, f"expected shape (768,), got {embedding.shape}")

    with config.species_lock:
        if name in model_manager.species_list:
            return AddSpeciesResponse(
                name=name, slug=name.lower().replace(" ", "_"),
                description=description, already_existed=True
            )
        model_manager.species_list.append(name)
        model_manager.species_embeds_tensor = torch.cat(
            [model_manager.species_embeds_tensor, embedding.unsqueeze(0)], dim=0
        )

    _persist_dynamic_species()
    print(f"New species: {name} (total: {len(model_manager.species_list)})")
    return AddSpeciesResponse(
        name=name, slug=name.lower().replace(" ", "_"),
        description=description, already_existed=False
    )


# ── Progress ───────────────────────────────────────────────────────────────────

@app.get("/progress/{species:path}")
def get_progress(species: str):
    with config.progress_lock:
        p = progress_store.get(species, {"total": 0, "done": 0, "status": "idle"}).copy()
    queued = len([t for t in inf_mod.processing_set if t[0] == species])
    pct    = (p['done'] / p['total'] * 100) if p['total'] > 0 else 0
    status = 'running' if queued > 0 else p['status']
    if queued == 0 and p['total'] > 0 and p['done'] >= p['total']:
        status = 'done'
    return {
        "total": p['total'], "done": p['done'], "queued": queued,
        "pct": round(pct, 1), "status": status,
        "max_prob": 1.0, "min_prob": 0.0,
    }


# ── Viewport generation ────────────────────────────────────────────────────────

class ViewportRequest(BaseModel):
    species: str
    minlon:  float
    minlat:  float
    maxlon:  float
    maxlat:  float


@app.post("/generate_viewport")
def generate_viewport(req: ViewportRequest):
    with config.species_lock:
        known = req.species in model_manager.species_list
    if not known:
        raise HTTPException(404, f"Unknown species '{req.species}'. Call POST /species/add first.")

    sdm_tiles = geo_utils.overlapping_sdm_tiles(req.minlon, req.minlat, req.maxlon, req.maxlat)
    slug      = req.species.lower().replace(" ", "_")

    cached_patches   = 0
    uncached_patches = 0
    to_queue         = []

    for tx, ty in sdm_tiles:
        cp = config.CACHE_DIR / slug / str(tx) / f"{ty}.npy"
        minlon_t, minlat_t, maxlon_t, maxlat_t = geo_utils.tile_to_bbox(config.BACKEND_Z, tx, ty)
        n_lons    = len(np.arange(minlon_t + config.SPACING_DEG / 2, maxlon_t, config.SPACING_DEG))
        n_lats    = len(np.arange(minlat_t + config.SPACING_DEG / 2, maxlat_t, config.SPACING_DEG))
        n_patches = n_lons * n_lats

        if cp.exists():
            cached_patches += n_patches
        else:
            uncached_patches += n_patches
            task = (req.species, tx, ty)
            if task not in inf_mod.processing_set:
                to_queue.append((task, n_patches))

    total_patches = cached_patches + uncached_patches

    with config.progress_lock:
        old_min = progress_store.get(req.species, {}).get('min_prob', 1.0)
        old_max = progress_store.get(req.species, {}).get('max_prob', 0.0)
        progress_store[req.species] = {
            "total":    total_patches,
            "done":     cached_patches,
            "status":   "running" if to_queue else "done",
            "min_prob": old_min,
            "max_prob": old_max,
        }
        for task, _ in to_queue:
            inf_mod.processing_set.add(task)
            inf_mod.inference_queue.put(task)

    return {"status": "success", "queued": len(to_queue), "sdm_z": config.BACKEND_Z}


# ── Tile serving ───────────────────────────────────────────────────────────────

@app.get("/tile/{species:path}/{z}/{x}/{y}.png")
async def get_tile(request: Request, species: str, z: int, x: int, y: int,
                   vmin: float = 0.0, vmax: float = 1.0,
                   percentile: Optional[str] = None,
                   render_mode: Optional[str] = 'smooth'):
    with config.species_lock:
        known = species in model_manager.species_list
    if not known:
        return Response(status_code=404)

    is_raw = (render_mode == 'raw')
    cache_key  = (species, z, x, y, vmin, vmax, percentile, is_raw)
    cached_png = config._tile_png_cache.get(cache_key)
    if cached_png is not None:
        return Response(content=cached_png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    loop = asyncio.get_event_loop()

    def _render():
        return renderer.render_merged_tile(species, z, x, y, vmin=vmin, vmax=vmax,
                                          percentile=percentile, raw_pixels=is_raw)

    png    = await loop.run_in_executor(_render_pool, _render)
    status = progress_store.get(species, {}).get('status')

    if not png and status == 'running':
        for _ in range(60):
            await asyncio.sleep(0.5)
            if await request.is_disconnected():
                return Response(status_code=204)
            png = await loop.run_in_executor(_render_pool, _render)
            if png:
                break

    if not png:
        return Response(status_code=204)

    config._tile_png_cache.set(cache_key, png)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# ── Scale / Percentiles endpoints ──────────────────────────────────────────────

@app.get("/species/scale/{species:path}")
def get_species_scale(species: str):
    with config._scale_cache_lock:
        cached = config._scale_cache.get(species)
        if cached is None or cached.get('_algo_version', 0) < 5:
            config._scale_cache[species] = renderer.compute_scale_from_disk(species)
        scale = config._scale_cache[species]
    return {
        "min_prob":   scale["min_prob"],
        "max_prob":   scale["max_prob"],
        "tile_count": scale["tile_count"],
    }


@app.get("/species/percentiles/{species:path}")
def get_species_percentiles(species: str):
    """Return precomputed percentile threshold values in raw suitability space."""
    with config._scale_cache_lock:
        cached = config._scale_cache.get(species)
        if cached is None or cached.get('_algo_version', 0) < 5:
            config._scale_cache[species] = renderer.compute_scale_from_disk(species)
        scale = config._scale_cache[species]
    return {
        "tile_count": scale["tile_count"],
        "adaptive_raw": scale.get("adaptive_raw"),
        "adaptive_percentile": scale.get("adaptive_percentile"),
        "otsu_raw": scale.get("otsu_raw"),
        "otsu_percentile": scale.get("otsu_percentile"),
        "mean_raw": scale.get("mean_raw"),
        "mean_percentile": scale.get("mean_percentile"),
    }


@app.get("/species/cohort/{species:path}")
def get_species_cohort_endpoint(species: str):
    """Return the JSDM cohort associates (similar niche species) for the given species."""
    target_name = species
    with config.species_lock:
        match = next(
            (s for s in model_manager.species_list
             if s.lower().replace(" ", "_") == species.lower().replace(" ", "_") or s.lower() == species.lower()),
            None
        )
        if match:
            target_name = match

    # Get cohort indices (target is at index 0, associates at 1..15)
    cohort_indices = model_manager.get_species_cohort(target_name, cohort_size=15)
    with config.species_lock:
        cohort_names = [model_manager.species_list[idx] for idx in cohort_indices]

    return {
        "target": cohort_names[0],
        "associates": cohort_names[1:]
    }


# ── Full-map generation ────────────────────────────────────────────────────────

@app.post("/generate_fullmap")
def generate_fullmap(req: GenerateDescriptionRequest):
    """
    Start a 250×250 uniform grid inference over the full embedding extent.
    Poll /fullmap_progress/{species} for status.
    """
    species = req.name.strip()
    with config.species_lock:
        if species not in model_manager.species_list:
            raise HTTPException(404, f"Unknown species '{species}'. Register first.")

    if species in inf_mod._fullmap_running:
        return {"status": "already_running"}

    fm_path = config.CACHE_DIR / species.lower().replace(" ", "_") / config.FULLMAP_NPY
    if fm_path.exists():
        with inf_mod._fullmap_lock:
            prog = inf_mod._fullmap_progress.get(species, {"status": "done", "pct": 100})
        return {"status": "cached", "progress": prog}

    inf_mod._fullmap_running.add(species)
    threading.Thread(target=inf_mod._run_fullmap, args=(species,), daemon=True).start()
    return {"status": "started", "grid_n": config.FULLMAP_GRID_N}


@app.get("/fullmap_progress/{species:path}")
def fullmap_progress(species: str):
    fm_path = config.CACHE_DIR / species.lower().replace(" ", "_") / config.FULLMAP_NPY
    with inf_mod._fullmap_lock:
        p = inf_mod._fullmap_progress.get(species)
    if p is None:
        if fm_path.exists():
            return {"status": "done", "pct": 100,
                    "done": config.FULLMAP_GRID_N, "total": config.FULLMAP_GRID_N}
        return {"status": "idle", "pct": 0, "done": 0, "total": config.FULLMAP_GRID_N}
    return p


# ── Debug endpoint ─────────────────────────────────────────────────────────────

@app.get("/debug/{species:path}")
def debug_cache(species: str):
    slug    = species.lower().replace(" ", "_")
    files   = list((config.CACHE_DIR / slug).rglob("*.npy")) if (config.CACHE_DIR / slug).exists() else []
    fm_path = config.CACHE_DIR / slug / config.FULLMAP_NPY
    fm_info = {}
    if fm_path.exists():
        fm    = np.load(fm_path)
        valid = fm[~np.isnan(fm)]
        fm_info = {
            "shape":        list(fm.shape),
            "valid_pixels": int(valid.size),
            "min":          round(float(valid.min()), 4) if valid.size else None,
            "max":          round(float(valid.max()), 4) if valid.size else None,
        }
    return {
        "cached_files": len(files),
        "fullmap":      fm_info,
        "zarr_extent":  config._ZARR_EXTENT,
        "examples":     [str(f) for f in files[:5]],
        "progress":     progress_store.get(species, {}),
    }

from fastapi.responses import FileResponse

@app.get("/geo/quebec_border.geojson")
def get_quebec_border():
    geojson_path = Path("data/quebec_border.geojson")
    if geojson_path.exists():
        return FileResponse(geojson_path, media_type="application/json",
                            headers={"Cache-Control": "public, max-age=86400"})
    return Response(status_code=404)

# Serve static files from the frontend directory
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend"), name="frontend")