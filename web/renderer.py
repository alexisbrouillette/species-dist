import io
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
import config
import geo_utils
from model_manager import is_stats_species

def compute_scale_from_disk(species: str) -> dict:
    slug  = species.lower().replace(" ", "_")
    target_dir = config.CACHE_DIR / slug
    files = list(target_dir.rglob("*.npy")) if target_dir.exists() else []

    if not files:
        return {
            "min_prob": 0.0, "max_prob": 1.0, "tile_count": 0,
            "p90": None, "p95": None, "p98": None, "p99": None,
        }

    valid_pixels = []
    tile_count = 0

    for f in files:
        try:
            arr = np.load(f)
            # Filter out NaNs and absolute zeroes
            valid = arr[~np.isnan(arr) & (arr > 0)]
            if valid.size > 0:
                valid_pixels.append(valid)
            
            if f.name != config.FULLMAP_NPY:
                tile_count += 1
            else:
                tile_count += 1   # count fullmap as a tile
        except Exception:
            continue

    if not valid_pixels:
        return {
            "min_prob": 0.0, "max_prob": 1.0, "tile_count": tile_count,
            "p90": None, "p95": None, "p98": None, "p99": None,
        }

    all_valid = np.concatenate(valid_pixels)
    
    # Scale from absolute min to absolute max prediction
    vmin_raw = float(all_valid.min())
    vmax_raw = float(all_valid.max())

    if vmax_raw <= vmin_raw:
        vmax_raw = vmin_raw + 0.01

    # Compute optimal threshold using an adaptive method based on distribution skewness:
    n_bins = 256
    hist, bin_edges = np.histogram(all_valid, bins=n_bins, range=(vmin_raw, vmax_raw))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Compute the percentile of the peak value to detect distribution skewness.
    # Widespread/common species will have a peak at the middle or high end (high percentile),
    # while localized/rare species will have a peak at the low end (low percentile).
    peak_idx = int(np.argmax(hist))
    peak_val = bin_centers[peak_idx]
    peak_pct = float(np.sum(all_valid <= peak_val) / all_valid.size)

    # ── 1. Adaptive (Triangle/p10) ─────────────────────────────────────────
    if peak_pct < 0.30:
        # Localized/rare species: use the Triangle method to find the elbow
        # on the right tail of the histogram (from peak to max value).
        nonzero = np.nonzero(hist)[0]
        tail_idx = int(nonzero[-1]) if len(nonzero) else n_bins - 1

        if tail_idx <= peak_idx:
            triangle_raw = float(np.median(all_valid))
        else:
            x1, y1 = float(peak_idx), float(hist[peak_idx])
            x2, y2 = float(tail_idx), float(hist[tail_idx])
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            denom = np.sqrt(a * a + b * b)
            if denom < 1e-12:
                triangle_raw = float(np.median(all_valid))
            else:
                indices = np.arange(peak_idx, tail_idx + 1, dtype=np.float64)
                heights = hist[peak_idx:tail_idx + 1].astype(np.float64)
                distances = np.abs(a * indices + b * heights + c) / denom
                best_offset = int(np.argmax(distances))
                triangle_raw = float(bin_centers[peak_idx + best_offset])

        # Apply adaptive cap so at least 15% of the habitat signal survives
        MAX_MASK_FRAC = 0.85
        triangle_pct = float(np.sum(all_valid <= triangle_raw) / all_valid.size)
        if triangle_pct > MAX_MASK_FRAC:
            adaptive_raw = float(np.percentile(all_valid, MAX_MASK_FRAC * 100))
        else:
            adaptive_raw = triangle_raw
    else:
        # Widespread/common species: use 10th percentile
        adaptive_raw = float(np.percentile(all_valid, 10.0))
    adaptive_percentile = float(np.sum(all_valid <= adaptive_raw) / all_valid.size * 100)

    # ── 2. Otsu's Threshold ────────────────────────────────────────────────
    weight1 = np.cumsum(hist).astype(np.float64)
    weight2 = np.cumsum(hist[::-1])[::-1].astype(np.float64)
    mean1 = np.cumsum(hist * bin_centers).astype(np.float64) / (weight1 + 1e-12)
    mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1].astype(np.float64) / (weight2 + 1e-12))
    variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
    otsu_idx = int(np.argmax(variance_between))
    otsu_raw = float(bin_centers[otsu_idx])
    otsu_percentile = float(np.sum(all_valid <= otsu_raw) / all_valid.size * 100)

    # ── 3. Mean Threshold ──────────────────────────────────────────────────
    mean_raw = float(np.mean(all_valid))
    mean_percentile = float(np.sum(all_valid <= mean_raw) / all_valid.size * 100)

    print(f"[scale] {species}: range=[{vmin_raw:.4f}, {vmax_raw:.4f}] | "
          f"adaptive={adaptive_raw:.4f} (~p{adaptive_percentile:.0f}) | "
          f"otsu={otsu_raw:.4f} (~p{otsu_percentile:.0f}) | "
          f"mean={mean_raw:.4f} (~p{mean_percentile:.0f})")

    return {
        "min_prob":  max(0.0, round(vmin_raw, 5)),
        "max_prob":  min(1.0, round(vmax_raw, 5)),
        "tile_count": tile_count,
        "adaptive_raw": round(adaptive_raw, 5),
        "adaptive_percentile": round(adaptive_percentile, 2),
        "otsu_raw": round(otsu_raw, 5),
        "otsu_percentile": round(otsu_percentile, 2),
        "mean_raw": round(mean_raw, 5),
        "mean_percentile": round(mean_percentile, 2),
        "_algo_version": 12,      # bump to invalidate stale cache entries
    }


def _resolve_threshold(scale: dict, percentile) -> float:
    """Convert a percentile param (adaptive/otsu/mean/None) to a rescaled [0,1] threshold.

    The returned value is in the *rescaled* space ([0,1] after vmin/vmax stretch)
    so it can be compared directly against `rescaled` inside the render functions.
    """
    if percentile is None:
        return 0.0  # No masking

    # Handle string cases
    if isinstance(percentile, str):
        p_lower = percentile.lower()
        if p_lower == 'adaptive' or p_lower == 'optimal':
            raw_threshold = scale.get('adaptive_raw')
        elif p_lower == 'otsu':
            raw_threshold = scale.get('otsu_raw')
        elif p_lower == 'mean':
            raw_threshold = scale.get('mean_raw')
        else:
            # Fallback check for numbers in string format
            try:
                percentile = int(percentile)
            except ValueError:
                return 0.0
    
    if not isinstance(percentile, str):
        percentile = int(percentile)
        # Fallback mapping: any historic numeric request maps to adaptive
        key_map = {90: 'adaptive_raw', 95: 'adaptive_raw', 98: 'adaptive_raw', 99: 'adaptive_raw'}
        raw_key = key_map.get(percentile)
        raw_threshold = scale.get(raw_key) if raw_key else None

    if raw_threshold is None:
        return 0.0

    s_min = scale['min_prob']
    s_max = scale['max_prob']
    span = (s_max - s_min) if (s_max - s_min) > 1e-5 else 0.0001
    return float(np.clip((raw_threshold - s_min) / span, 0.0, 1.0))


def render_merged_tile(species, map_z, map_x, map_y, vmin=0.0, vmax=1.0, percentile=None, raw_pixels=False):
    minlon, minlat, maxlon, maxlat = geo_utils.tile_to_bbox(map_z, map_x, map_y)
    
    pad = 36  # feather_width
    GRID_RES_PADDED = config.GRID_RES + 2 * pad
    
    lon_step = (maxlon - minlon) / config.GRID_RES
    lat_step = (maxlat - minlat) / config.GRID_RES
    
    # Construct coordinates for the padded grid
    lons_pad = np.linspace(minlon - pad * lon_step + lon_step/2, 
                           maxlon + pad * lon_step - lon_step/2, 
                           GRID_RES_PADDED)
    lats_pad = np.linspace(maxlat + pad * lat_step - lat_step/2, 
                           minlat - pad * lat_step + lat_step/2, 
                           GRID_RES_PADDED)
    LON_pad, LAT_pad = np.meshgrid(lons_pad, lats_pad)

    high_res_grid_pad = np.full((GRID_RES_PADDED, GRID_RES_PADDED), np.nan, dtype=np.float32)
    high_res_mask_pad = np.zeros((GRID_RES_PADDED, GRID_RES_PADDED), dtype=bool)
    slug = species.lower().replace(" ", "_")
    EPS  = 1e-7

    # Find overlapping backend tiles covering the padded area
    minlon_pad_val = minlon - pad * lon_step
    maxlon_pad_val = maxlon + pad * lon_step
    minlat_pad_val = minlat - pad * lat_step
    maxlat_pad_val = maxlat + pad * lat_step
    overlapping_tiles = geo_utils.overlapping_sdm_tiles(minlon_pad_val, minlat_pad_val, maxlon_pad_val, maxlat_pad_val)

    # ── 1. Fill from viewport tiles first (higher resolution, takes priority) ──────
    for tx, ty in overlapping_tiles:
        p = config.CACHE_DIR / slug / str(tx) / f"{ty}.npy"
        if not p.exists(): continue
        npy = np.load(p)
        npy_valid = ~np.isnan(npy)
        if npy_valid.any() and not npy_valid.all():
            npy_dists, npy_idx = distance_transform_edt(~npy_valid, return_indices=True)
            npy = np.where(npy_dists <= 3.0, npy[tuple(npy_idx)], np.nan)

        t_minlon, t_minlat, t_maxlon, t_maxlat = geo_utils.tile_to_bbox(config.BACKEND_Z, tx, ty)
        nr, nc = npy.shape

        inside = ((LON_pad >= t_minlon - EPS) & (LON_pad <= t_maxlon + EPS) &
                  (LAT_pad >= t_minlat - EPS) & (LAT_pad <= t_maxlat + EPS))
        if not np.any(inside): continue

        fc = np.clip((LON_pad[inside] - t_minlon) / (t_maxlon - t_minlon) * (nc - 1), 0, nc - 1)
        fr = np.clip((t_maxlat - LAT_pad[inside]) / (t_maxlat - t_minlat) * (nr - 1), 0, nr - 1)

        if raw_pixels:
            rc = np.clip(np.rint(fc), 0, nc - 1).astype(int)
            rr = np.clip(np.rint(fr), 0, nr - 1).astype(int)
            sampled = npy[rr, rc]
            high_res_grid_pad[inside] = sampled
            high_res_mask_pad[inside] = ~np.isnan(sampled)
        else:
            c0 = np.floor(fc).astype(int);  c1 = np.minimum(c0 + 1, nc - 1)
            r0 = np.floor(fr).astype(int);  r1 = np.minimum(r0 + 1, nr - 1)
            wc = (fc - c0).astype(np.float32)
            wr = (fr - r0).astype(np.float32)

            s00 = npy[r0, c0]; s01 = npy[r0, c1]
            s10 = npy[r1, c0]; s11 = npy[r1, c1]

            bilinear = (s00 * (1-wr) * (1-wc) +
                        s01 * (1-wr) *    wc  +
                        s10 *    wr  * (1-wc) +
                        s11 *    wr  *    wc)

            high_res_grid_pad[inside] = bilinear
            high_res_mask_pad[inside] = ~np.isnan(bilinear)

    # ── 2. Sample from fullmap.npy for remaining empty regions (low-res fallback) ──
    low_res_grid_pad = np.full((GRID_RES_PADDED, GRID_RES_PADDED), np.nan, dtype=np.float32)
    fm_path = config.CACHE_DIR / slug / config.FULLMAP_NPY
    if fm_path.exists():
        fm     = np.load(fm_path)
        fm_valid = ~np.isnan(fm)
        if fm_valid.any() and not fm_valid.all():
            fm_dists, fm_idx = distance_transform_edt(~fm_valid, return_indices=True)
            fm = np.where(fm_dists <= 3.0, fm[tuple(fm_idx)], np.nan)
        nr, nc = fm.shape
        if is_stats_species(species):
            ext = config.STATS_EXTENT
        else:
            ext = config.EMB_EXTENT if fm.shape in [(250, 250), (836, 1134)] else config._ZARR_EXTENT
        inside = ((LON_pad >= ext['minlon'] - EPS) & (LON_pad <= ext['maxlon'] + EPS) &
                  (LAT_pad >= ext['minlat'] - EPS) & (LAT_pad <= ext['maxlat'] + EPS))
        if np.any(inside):
            fc = np.clip((LON_pad[inside] - ext['minlon']) /
                         (ext['maxlon'] - ext['minlon']) * (nc - 1), 0, nc - 1)
            fr = np.clip((ext['maxlat'] - LAT_pad[inside]) /
                         (ext['maxlat'] - ext['minlat']) * (nr - 1), 0, nr - 1)
            if raw_pixels:
                rc = np.clip(np.rint(fc), 0, nc - 1).astype(int)
                rr = np.clip(np.rint(fr), 0, nr - 1).astype(int)
                low_res_grid_pad[inside] = fm[rr, rc]
            else:
                c0 = np.floor(fc).astype(int);  c1 = np.minimum(c0 + 1, nc - 1)
                r0 = np.floor(fr).astype(int);  r1 = np.minimum(r0 + 1, nr - 1)
                wc = (fc - c0).astype(np.float32)
                wr = (fr - r0).astype(np.float32)
                s00 = fm[r0, c0]; s01 = fm[r0, c1]
                s10 = fm[r1, c0]; s11 = fm[r1, c1]
                low_res_grid_pad[inside] = (s00*(1-wr)*(1-wc) + s01*(1-wr)*wc +
                                            s10*wr*(1-wc)      + s11*wr*wc)

    # ── 3. Blend high-res and low-res with feathering ──
    grid_pad = np.full((GRID_RES_PADDED, GRID_RES_PADDED), np.nan, dtype=np.float32)
    if high_res_mask_pad.any():
        dist_inside = distance_transform_edt(high_res_mask_pad)
        feather_width = 36.0
        w_high = np.clip(dist_inside / feather_width, 0.0, 1.0)

        blend_mask = high_res_mask_pad & ~np.isnan(low_res_grid_pad)
        grid_pad[blend_mask] = w_high[blend_mask] * high_res_grid_pad[blend_mask] + (1.0 - w_high[blend_mask]) * low_res_grid_pad[blend_mask]

        hr_only = high_res_mask_pad & np.isnan(low_res_grid_pad)
        grid_pad[hr_only] = high_res_grid_pad[hr_only]

        lr_only = ~high_res_mask_pad & ~np.isnan(low_res_grid_pad)
        grid_pad[lr_only] = low_res_grid_pad[lr_only]
    else:
        grid_pad = low_res_grid_pad

    if np.isnan(grid_pad).all(): return None

    # Apply invalid mask to padded grid
    grid_pad[geo_utils.get_invalid_mask(map_z, map_x, map_y, LON_pad, LAT_pad, species=species)] = np.nan
    valid_pad = ~np.isnan(grid_pad)
    if not valid_pad.any(): return None

    lat_span = maxlat - minlat
    n_cells = lat_span / config.SPACING_DEG
    cell_px = config.GRID_RES / max(n_cells, 1.0)

    # Keep sigma small (max 0.5) to preserve crisp, distinct isoline contours
    sigma = 0.0 if raw_pixels else 0.5

    # Fill NaN borders locally on padded grid so blur doesn't shrink valid regions,
    # but restrict max filling distance to prevent bridging disconnected patches.
    filled_pad = grid_pad.copy()
    if not valid_pad.all() and sigma > 0:
        dists, idx = distance_transform_edt(~valid_pad, return_distances=True, return_indices=True)
        max_fill_dist = max(3.0, sigma * 2.0)
        near_mask = dists <= max_fill_dist
        filled_pad[near_mask] = grid_pad[tuple(idx)][near_mask]

    # Apply Gaussian filter or keep raw pixel values
    if raw_pixels or sigma == 0:
        smoothed_pad = filled_pad
    else:
        smoothed_pad = gaussian_filter(filled_pad, sigma=sigma)
        smoothed_pad[~valid_pad] = np.nan

    # Crop the padded outputs back to the original tile dimensions (256x256)
    grid = smoothed_pad[pad : pad + config.GRID_RES, pad : pad + config.GRID_RES]
    valid = valid_pad[pad : pad + config.GRID_RES, pad : pad + config.GRID_RES]
    if not valid.any(): return None

    # ── 4. Rescale to [0, 1] range, apply percentile-based threshold, and map to custom LUT ──
    with config._scale_cache_lock:
        cached = config._scale_cache.get(species)
        if cached is None or cached.get('_algo_version', 0) < 12:
            config._scale_cache[species] = compute_scale_from_disk(species)
        scale = config._scale_cache[species]
        s_min = scale['min_prob']
        s_max = scale['max_prob']

    span = (s_max - s_min) if (s_max - s_min) > 1e-5 else 0.0001
    grid_clean = np.where(valid, grid, 0.0)
    rescaled = np.clip((grid_clean - s_min) / span, 0.0, 1.0)

    # No opacity fading (always 100% opaque for active pixels)
    opacity_factor = 1.0

    threshold = _resolve_threshold(scale, percentile)
    under_threshold = valid & (rescaled < threshold)

    # Re-scale color gradient from threshold to 1.0 for the chosen threshold method
    t_span = max(1.0 - threshold, 1e-4)
    norm = np.nan_to_num(np.clip((rescaled - threshold) / t_span, 0.0, 1.0), nan=0.0)
    palette = config.RAW_CONTINUOUS_COLORS if raw_pixels else config.DISCRETE_COLORS

    num_steps = len(palette) - 1
    color_idx = np.clip(np.rint(norm * num_steps), 0, num_steps).astype(np.int32)

    rgba8 = palette[color_idx].copy()
    rgba8[~valid, 3] = 0
    rgba8[under_threshold, 3] = 0
    
    alpha_val = int(255 * opacity_factor)
    rgba8[valid & ~under_threshold, 3] = alpha_val

    buf = io.BytesIO()
    Image.fromarray(rgba8, mode='RGBA').save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


def save_global_map_at_res(species: str, grid_n: int, suffix: str):
    slug = species.lower().replace(" ", "_")
    species_cache_dir = config.CACHE_DIR / slug
    
    if is_stats_species(species):
        ext = config.STATS_EXTENT
    else:
        ext = config._ZARR_EXTENT
        fm_path = species_cache_dir / config.FULLMAP_NPY
        if fm_path.exists():
            try:
                fm_shape = np.load(fm_path).shape
                if fm_shape in [(250, 250), (836, 1134)]:
                    ext = config.EMB_EXTENT
            except Exception:
                pass

    lons = np.linspace(ext['minlon'], ext['maxlon'], grid_n)
    lats = np.linspace(ext['maxlat'], ext['minlat'], grid_n)  # North-up
    LON, LAT = np.meshgrid(lons, lats)
    
    high_res_grid = np.full((grid_n, grid_n), np.nan, dtype=np.float32)
    high_res_mask = np.zeros((grid_n, grid_n), dtype=bool)
    EPS = 1e-7
    
    # 2. Blend overlapping viewport tiles
    if species_cache_dir.exists():
        for tx_dir in species_cache_dir.iterdir():
            if not tx_dir.is_dir() or not tx_dir.name.isdigit():
                continue
            tx = int(tx_dir.name)
            for ty_file in tx_dir.glob("*.npy"):
                try:
                    ty = int(ty_file.stem)
                except ValueError:
                    continue
                
                npy = np.load(ty_file)
                npy_valid = ~np.isnan(npy)
                if npy_valid.any() and not npy_valid.all():
                    npy_dists, npy_idx = distance_transform_edt(~npy_valid, return_indices=True)
                    npy = np.where(npy_dists <= 3.0, npy[tuple(npy_idx)], np.nan)
                t_minlon, t_minlat, t_maxlon, t_maxlat = geo_utils.tile_to_bbox(config.BACKEND_Z, tx, ty)
                nr, nc = npy.shape
                
                inside = ((LON >= t_minlon - EPS) & (LON <= t_maxlon + EPS) &
                          (LAT >= t_minlat - EPS) & (LAT <= t_maxlat + EPS))
                if not np.any(inside):
                    continue
                
                fc = np.clip((LON[inside] - t_minlon) / (t_maxlon - t_minlon) * (nc - 1), 0, nc - 1)
                fr = np.clip((LAT[inside] - t_minlat) / (t_maxlat - t_minlat) * (nr - 1), 0, nr - 1)
                
                c0 = np.floor(fc).astype(int); c1 = np.minimum(c0 + 1, nc - 1)
                r0 = np.floor(fr).astype(int); r1 = np.minimum(r0 + 1, nr - 1)
                wc = (fc - c0).astype(np.float32)
                wr = (fr - r0).astype(np.float32)
                
                s00 = npy[r0, c0]; s01 = npy[r0, c1]
                s10 = npy[r1, c0]; s11 = npy[r1, c1]
                
                bilinear = (s00 * (1-wr) * (1-wc) +
                            s01 * (1-wr) *    wc  +
                            s10 *    wr  * (1-wc) +
                            s11 *    wr  *    wc)
                
                high_res_grid[inside] = bilinear
                high_res_mask[inside] = ~np.isnan(bilinear)
                
    # 3. Interpolate the low-res fullmap background
    low_res_grid = np.full((grid_n, grid_n), np.nan, dtype=np.float32)
    if fm_path.exists():
        fm = np.load(fm_path)
        fm_valid = ~np.isnan(fm)
        if fm_valid.any() and not fm_valid.all():
            fm_dists, fm_idx = distance_transform_edt(~fm_valid, return_indices=True)
            fm = np.where(fm_dists <= 3.0, fm[tuple(fm_idx)], np.nan)
        nr, nc = fm.shape
        inside = ((LON >= ext['minlon'] - EPS) & (LON <= ext['maxlon'] + EPS) &
                  (LAT >= ext['minlat'] - EPS) & (LAT <= ext['maxlat'] + EPS))
        if np.any(inside):
            fc = np.clip((LON[inside] - ext['minlon']) /
                         (ext['maxlon'] - ext['minlon']) * (nc - 1), 0, nc - 1)
            fr = np.clip((ext['maxlat'] - LAT[inside]) /
                         (ext['maxlat'] - ext['minlat']) * (nr - 1), 0, nr - 1)
            c0 = np.floor(fc).astype(int); c1 = np.minimum(c0 + 1, nc - 1)
            r0 = np.floor(fr).astype(int); r1 = np.minimum(r0 + 1, nr - 1)
            wc = (fc - c0).astype(np.float32)
            wr = (fr - r0).astype(np.float32)
            s00 = fm[r0, c0]; s01 = fm[r0, c1]
            s10 = fm[r1, c0]; s11 = fm[r1, c1]
            low_res_grid[inside] = (s00*(1-wr)*(1-wc) + s01*(1-wr)*wc +
                                    s10*wr*(1-wc)      + s11*wr*wc)
            
    # 4. Blend high-res and low-res
    grid = np.full((grid_n, grid_n), np.nan, dtype=np.float32)
    if high_res_mask.any():
        dist_inside = distance_transform_edt(high_res_mask)
        feather_width = max(1.0, grid_n * 0.05)
        w_high = np.clip(dist_inside / feather_width, 0.0, 1.0)
        
        blend_mask = high_res_mask & ~np.isnan(low_res_grid)
        grid[blend_mask] = w_high[blend_mask] * high_res_grid[blend_mask] + (1.0 - w_high[blend_mask]) * low_res_grid[blend_mask]
        
        hr_only = high_res_mask & np.isnan(low_res_grid)
        grid[hr_only] = high_res_grid[hr_only]
        
        lr_only = ~high_res_mask & ~np.isnan(low_res_grid)
        grid[lr_only] = low_res_grid[lr_only]
    else:
        grid = low_res_grid
        
    # 5. Mask out-of-extent and masked pixels (apply invalid mask)
    mask = geo_utils.get_global_invalid_mask(grid_n, ext, species=species)
    grid[mask] = np.nan
    
    valid = ~np.isnan(grid)
    if not valid.any():
        return
        
    # Light gaussian smooth
    filled = grid.copy()
    if not valid.all():
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = filled[tuple(idx)]
    smoothed = gaussian_filter(filled, sigma=0.8)
    smoothed[~valid] = np.nan
    grid = smoothed

    # 6. Colormap scaling
    scale = compute_scale_from_disk(species)
    s_min = scale['min_prob']
    s_max = scale['max_prob']
    
    span = (s_max - s_min) if (s_max - s_min) > 1e-5 else 0.0001
    grid_clean = np.where(valid, grid, 0.0)
    rescaled = np.clip((grid_clean - s_min) / span, 0.0, 1.0)
    
    # No opacity fading (always 100% opaque for active pixels)
    opacity_factor = 1.0

    # Static maps default to Otsu masking for a clean look
    threshold = _resolve_threshold(scale, 'otsu')
    under_threshold = valid & (rescaled < threshold)

    t_span = max(1.0 - threshold, 1e-4)
    norm = np.clip((rescaled - threshold) / t_span, 0.0, 1.0)

    color_idx = np.minimum(np.floor(norm * 7.0).astype(np.int32), 6)

    rgba8 = config.DISCRETE_COLORS[color_idx].copy()
    rgba8[~valid, 3] = 0
    rgba8[under_threshold, 3] = 0
    
    alpha_val = int(255 * opacity_factor)
    rgba8[valid & ~under_threshold, 3] = alpha_val
    
    # Save the PNG
    img = Image.fromarray(rgba8, mode='RGBA')
    img.save(species_cache_dir / f"map_{suffix}.png")
    img.save(f"sdm_map_{slug}_{suffix}.png")


def save_global_maps_background(species: str):
    try:
        save_global_map_at_res(species, 250, "lowres")
        save_global_map_at_res(species, 1000, "highres")
    except Exception as e:
        print(f"Error saving global maps for {species}: {e}", flush=True)
