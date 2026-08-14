import numpy as np
import rasterio
import threading
from pyproj import Transformer
import config
import shapely
import shapely.geometry as sg

# ── Mask Loading (runs once on module load) ───────────────────────────────────
print("Loading masks...", flush=True)
with rasterio.open(config.WATER_MASK_PATH) as src:
    water_mask      = src.read(1)
    water_transform = src.transform
    water_crs       = src.crs
    water_shape     = water_mask.shape
water_transformer = Transformer.from_crs("EPSG:4326", water_crs, always_xy=True)

with rasterio.open(config.COMBINED_MASK_PATH) as src:
    combined_mask      = src.read(1)
    combined_transform = src.transform
    combined_crs       = src.crs
    combined_shape     = combined_mask.shape
combined_transformer = Transformer.from_crs("EPSG:4326", combined_crs, always_xy=True)
print("✅ Masks loaded", flush=True)

# Load full extent water mask for stats-only model
print("Loading full extent water mask...", flush=True)
with rasterio.open(config.WATER_FULL_EXTENT_PATH) as src:
    water_full_mask      = src.read(1)
    water_full_transform = src.transform
    water_full_shape     = water_full_mask.shape
water_full_transformer = Transformer.from_crs("EPSG:4326", "EPSG:6624", always_xy=True)
print("✅ Full extent water mask loaded", flush=True)

# --- Coordinate matching and index retrieval for 1.49M points ---
_grid_to_stats_idx = None
_grid_to_stats_lock = threading.Lock()
_transformer_4326_to_32198 = None
_transformer_lock = threading.Lock()

def get_grid_to_stats_map():
    global _grid_to_stats_idx
    with _grid_to_stats_lock:
        if _grid_to_stats_idx is None:
            import os
            from pathlib import Path
            minx, miny = -830844.1785036868, 117973.2977605052
            x_len, y_len = 1615, 1974
            coords_path = Path(__file__).parent / "data" / "grid_coords_32198.npy"
            coords = np.load(coords_path)
            ix = np.round((coords[:, 0] - minx) / 1000).astype(np.int32)
            iy = np.round((coords[:, 1] - miny) / 1000).astype(np.int32)
            flat_indices = ix * y_len + iy
            _grid_to_stats_idx = np.full(x_len * y_len, -1, dtype=np.int32)
            _grid_to_stats_idx[flat_indices] = np.arange(len(coords), dtype=np.int32)
    return _grid_to_stats_idx

def get_transformer_4326_to_32198():
    global _transformer_4326_to_32198
    with _transformer_lock:
        if _transformer_4326_to_32198 is None:
            _transformer_4326_to_32198 = Transformer.from_crs("EPSG:4326", "EPSG:32198", always_xy=True)
    return _transformer_4326_to_32198

def get_stats_indices(lons, lats):
    lons = np.atleast_1d(lons)
    lats = np.atleast_1d(lats)
    tr = get_transformer_4326_to_32198()
    xs, ys = tr.transform(lons, lats)
    
    minx, miny = -830844.1785036868, 117973.2977605052
    x_len, y_len = 1615, 1974
    
    ix = np.round((xs - minx) / 1000).astype(np.int32)
    iy = np.round((ys - miny) / 1000).astype(np.int32)
    
    valid = (ix >= 0) & (ix < x_len) & (iy >= 0) & (iy < y_len)
    
    grid_to_stats = get_grid_to_stats_map()
    
    flat_indices = np.full(len(lons), -1, dtype=np.int32)
    if np.any(valid):
        idx_2d = ix[valid] * y_len + iy[valid]
        flat_indices[valid] = grid_to_stats[idx_2d]
        
    valid &= (flat_indices != -1)
    return flat_indices, valid


# ── Coordinate and Index Mapping Helpers ─────────────────────────────────────

def get_embedding_indices(lons, lats):
    lons = np.atleast_1d(lons)
    lats = np.atleast_1d(lats)
    cols = np.round((lons - config.EMB_EXTENT['minlon']) / 
                    (config.EMB_EXTENT['maxlon'] - config.EMB_EXTENT['minlon']) * 1133).astype(np.int32)
    rows = np.round((lats - config.EMB_EXTENT['minlat']) / 
                    (config.EMB_EXTENT['maxlat'] - config.EMB_EXTENT['minlat']) * 835).astype(np.int32)
    valid = (cols >= 0) & (cols < 1134) & (rows >= 0) & (rows < 836)
    flat_indices = np.where(valid, rows * 1134 + cols, -1)
    return flat_indices, valid


def tile_to_bbox(z, x, y):
    """Web Mercator tile to lat/lon bounding box (minlon, minlat, maxlon, maxlat)"""
    n = 2.0 ** z
    lon_deg = 360.0
    
    minlon = x / n * lon_deg - 180.0
    maxlon = (x + 1) / n * lon_deg - 180.0
    
    # Latitudes
    minlat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + 1) / n)))
    maxlat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / n)))
    
    return minlon, np.degrees(minlat_rad), maxlon, np.degrees(maxlat_rad)


def lonlat_to_tile(z, lon, lat):
    n     = 2 ** z
    tx    = int((lon + 180.0) / 360.0 * n)
    lat_r = np.radians(lat)
    ty    = int((1.0 - np.log(np.tan(lat_r) + 1.0/np.cos(lat_r)) / np.pi) / 2.0 * n)
    return np.clip(tx, 0, n-1), np.clip(ty, 0, n-1)


def overlapping_sdm_tiles(minlon, minlat, maxlon, maxlat):
    tx0, ty0 = lonlat_to_tile(config.BACKEND_Z, minlon, maxlat)
    tx1, ty1 = lonlat_to_tile(config.BACKEND_Z, maxlon, minlat)
    return [(tx, ty) for tx in range(tx0, tx1+1) for ty in range(ty0, ty1+1)]


# ── Fused invalid-mask cache ───────────────────────────────────────────────────

_study_area_poly = None
_study_area_poly_lock = threading.Lock()

def get_study_area_polygon():
    global _study_area_poly
    with _study_area_poly_lock:
        if _study_area_poly is None:
            import shapely.wkb as wkb
            poly_path = config.EMB_DIR / "study_area_polygon.wkb"
            if poly_path.exists():
                try:
                    with open(poly_path, "rb") as f:
                        _study_area_poly = wkb.loads(f.read())
                except Exception as ex:
                    print(f"Failed to load study area polygon WKB: {ex}. Recomputing...", flush=True)
            
            if _study_area_poly is None:
                print("study_area_polygon.wkb not found or failed to load. Computing on the fly...", flush=True)
                from shapely.ops import unary_union
                from shapely.geometry import Polygon, MultiPolygon
                import shapely.affinity as sa
                from rasterio.features import shapes
                
                lats_path = config.EMB_DIR / "latitudes.npy"
                lons_path = config.EMB_DIR / "longitudes.npy"
                new_lats = np.load(lats_path)
                new_lons = np.load(lons_path)
                
                min_lat, max_lat = -1.0008863, -0.14605425
                min_lon, max_lon = -0.9418238, 0.33796838
                
                cols = np.round((new_lons - min_lon) / (max_lon - min_lon) * 1133).astype(np.int32)
                rows = np.round((new_lats - min_lat) / (max_lat - min_lat) * 835).astype(np.int32)
                
                active_mask_2d = np.zeros((836, 1134), dtype=np.uint8)
                active_mask_2d[rows, cols] = 1
                
                shapes_gen = list(shapes(active_mask_2d, mask=active_mask_2d == 1))
                geoms = [sg.shape(g) for g, val in shapes_gen]
                merged = unary_union(geoms)
                
                def fill_holes(geom):
                    if isinstance(geom, Polygon):
                        return Polygon(geom.exterior)
                    elif isinstance(geom, MultiPolygon):
                        return MultiPolygon([Polygon(g.exterior) for g in geom.geoms])
                    return geom
                
                filled = fill_holes(merged)
                buffered = filled.buffer(3.0).buffer(-3.0)
                simplified = buffered.simplify(1.0, preserve_topology=True)
                
                # Affine transform to EPSG:4326
                ext = config.EMB_EXTENT
                a = (ext['maxlon'] - ext['minlon']) / 1133.0
                b = 0.0
                xoff = ext['minlon']
                d = 0.0
                e = (ext['maxlat'] - ext['minlat']) / 835.0
                yoff = ext['minlat']
                
                _study_area_poly = sa.affine_transform(simplified, [a, b, d, e, xoff, yoff])
                
                # Save to disk
                try:
                    with open(poly_path, "wb") as f:
                        f.write(wkb.dumps(_study_area_poly))
                    print(f"Successfully computed and saved study area polygon to {poly_path}", flush=True)
                except Exception as ex:
                    print(f"Failed to save computed study area polygon: {ex}", flush=True)
                    
    return _study_area_poly


_quebec_study_area = None
_quebec_study_area_lock = threading.Lock()

def get_quebec_study_area():
    global _quebec_study_area
    with _quebec_study_area_lock:
        if _quebec_study_area is None:
            import shapely.affinity as sa
            import rasterio.features
            
            poly_4326 = get_study_area_polygon()
            ext = config.EMB_EXTENT
            inv_a = 1133.0 / (ext['maxlon'] - ext['minlon'])
            inv_e = 835.0 / (ext['maxlat'] - ext['minlat'])
            poly_pixel = sa.affine_transform(poly_4326, [inv_a, 0.0, 0.0, inv_e, -ext['minlon'] * inv_a, -ext['minlat'] * inv_e])
            
            _quebec_study_area = rasterio.features.rasterize(
                [poly_pixel],
                out_shape=(836, 1134),
                fill=0,
                default_value=1,
                dtype=np.uint8
            ).astype(bool)
    return _quebec_study_area


def _build_invalid_mask(LON, LAT):
    flat_lon, flat_lat = LON.ravel(), LAT.ravel()

    # Check Zarr extent boundary
    ext = config._ZARR_EXTENT
    out_of_extent = (flat_lon < ext['minlon']) | (flat_lon > ext['maxlon']) | \
                    (flat_lat < ext['minlat']) | (flat_lat > ext['maxlat'])

    # Query water mask
    xx, yy = water_transformer.transform(flat_lon, flat_lat)
    rw = np.clip(((yy - water_transform.f) / water_transform.e).astype(np.int32), 0, water_shape[0]-1)
    cw = np.clip(((xx - water_transform.c) / water_transform.a).astype(np.int32), 0, water_shape[1]-1)
    invalid = (water_mask[rw, cw] == 1) | out_of_extent

    # Query combined land mask
    xx2, yy2 = combined_transformer.transform(flat_lon, flat_lat)
    rc = np.clip(((yy2 - combined_transform.f) / combined_transform.e).astype(np.int32), 0, combined_shape[0]-1)
    cc = np.clip(((xx2 - combined_transform.c) / combined_transform.a).astype(np.int32), 0, combined_shape[1]-1)
    invalid |= (combined_mask[rc, cc] == 0)

    # Restrict invalid mask using the rasterized vector study area polygon
    # This prevents any predictions from displaying outside Quebec (e.g. Ontario, Labrador)
    # while allowing internal lakes and rivers to be filled and then high-res masked.
    flat_indices, in_extent = get_embedding_indices(flat_lon, flat_lat)
    study_area = get_quebec_study_area()
    
    rows = np.clip(flat_indices // 1134, 0, 835)
    cols = np.clip(flat_indices % 1134, 0, 1133)
    
    outside_study_area = ~in_extent
    if np.any(in_extent):
        outside_study_area[in_extent] = ~study_area[rows[in_extent], cols[in_extent]]
        
    invalid |= outside_study_area

    return invalid.reshape(LON.shape)


def _build_invalid_mask_stats_only(LON, LAT):
    flat_lon, flat_lat = LON.ravel(), LAT.ravel()

    # Query full extent water mask for ocean / nodata
    xx, yy = water_full_transformer.transform(flat_lon, flat_lat)
    rw = np.clip(((yy - water_full_transform.f) / water_full_transform.e).astype(np.int32), 0, water_full_shape[0]-1)
    cw = np.clip(((xx - water_full_transform.c) / water_full_transform.a).astype(np.int32), 0, water_full_shape[1]-1)
    
    water_val = water_full_mask[rw, cw]
    # Invalid ONLY if nodata (-9999) outside Quebec or outside terrestrial grid
    invalid = (water_val == -9999.0)

    # Check if outside Quebec Lambert stats grid
    flat_indices, valid_in_grid = get_stats_indices(flat_lon, flat_lat)
    invalid |= ~valid_in_grid

    return invalid.reshape(LON.shape)


def get_invalid_mask(z, x, y, LON, LAT, species=None):
    from model_manager import is_stats_species
    
    key = (z, x, y, LON.shape)
    with config._mask_cache_lock:
        if key in config._mask_cache:
            return config._mask_cache[key]

    if species is not None and is_stats_species(species):
        mask = _build_invalid_mask_stats_only(LON, LAT)
    else:
        mask = _build_invalid_mask(LON, LAT)
    
    with config._mask_cache_lock:
        config._mask_cache[key] = mask
        
    return mask


_global_mask_cache = {}
_global_mask_cache_lock = threading.Lock()

def get_global_invalid_mask(grid_n: int, ext: dict, species: str = None) -> np.ndarray:
    from model_manager import is_stats_species
    
    key = (grid_n, ext['minlon'], ext['minlat'], ext['maxlon'], ext['maxlat'])
    with _global_mask_cache_lock:
        if key in _global_mask_cache:
            return _global_mask_cache[key]
            
    lons = np.linspace(ext['minlon'], ext['maxlon'], grid_n)
    lats = np.linspace(ext['maxlat'], ext['minlat'], grid_n)
    LON, LAT = np.meshgrid(lons, lats)
    
    if species is not None and is_stats_species(species):
        mask = _build_invalid_mask_stats_only(LON, LAT)
    else:
        mask = _build_invalid_mask(LON, LAT)
    
    with _global_mask_cache_lock:
        _global_mask_cache[key] = mask
        
    return mask


_grid_to_flat_v2 = None
_grid_to_flat_lock = threading.Lock()

def get_grid_to_flat_map():
    global _grid_to_flat_v2
    with _grid_to_flat_lock:
        if _grid_to_flat_v2 is None:
            lats_path = config.EMB_DIR / "latitudes.npy"
            lons_path = config.EMB_DIR / "longitudes.npy"
            if not lats_path.exists() or not lons_path.exists():
                raise FileNotFoundError(f"Coordinate files not found in {config.EMB_DIR}")
            
            new_lats = np.load(lats_path)
            new_lons = np.load(lons_path)
            
            min_lat, max_lat = -1.0008863, -0.14605425
            min_lon, max_lon = -0.9418238, 0.33796838
            
            cols = np.round((new_lons - min_lon) / (max_lon - min_lon) * 1133).astype(np.int32)
            rows = np.round((new_lats - min_lat) / (max_lat - min_lat) * 835).astype(np.int32)
            flat_indices = rows * 1134 + cols
            
            _grid_to_flat_v2 = np.full(948024, -1, dtype=np.int32)
            _grid_to_flat_v2[flat_indices] = np.arange(len(flat_indices), dtype=np.int32)
    return _grid_to_flat_v2
