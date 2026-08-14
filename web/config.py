import threading
import torch
import numcodecs
import numpy as np
from pathlib import Path

# Disable blosc threads to prevent issues in multi-threaded/FastAPI context
numcodecs.blosc.use_threads = False

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("tile_cache_v2")
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(exist_ok=True)

ZARR_PATH               = "../data/data_layers/climate_stack.zarr"
EMB_DIR                 = Path("../data/data_layers/embedding_map_v2")
EMBEDDING_CSV           = "../data/saved_df/species_embeddings_v3.csv"
SPECIES_NAMES_JSON      = Path("data/web_portal_assets/species_names.json")
SPECIES_EMBEDDINGS_NPY  = Path("data/web_portal_assets/species_embeddings.npy")
CHECKPOINT_PATH         = Path("model/model_simplified_v2_inference.pt")
DYNAMIC_EMBEDDINGS_PATH = Path("dynamic_embeddings.pt")

# Stats-only insect model configuration
STATS_MODEL_PATH         = Path("data/model_stats_only_test_v2_PU.pt")
STATS_INDEX_PATH         = Path("data/quebec_stats_index_1km.npy")
STATS_SPECIES_EMBEDDINGS = Path("../notebooks/species_embeddings_v4_insects.csv")
STATS_LONS_PATH          = Path("data/stats_longitudes.npy")
STATS_LATS_PATH          = Path("data/stats_latitudes.npy")
WATER_FULL_EXTENT_PATH   = Path("data/water_full_extent.tif")

STATS_EXTENT = {
    'minlon': -79.75659,
    'maxlon': -57.10598,
    'minlat': 44.99242,
    'maxlat': 62.58214
}

WATER_MASK_PATH    = "../data/data_layers/predictors_100_QC_normalized/water.tif"
COMBINED_MASK_PATH = "../data/data_layers/predictors_100_QC_normalized/combined_mask.tif"

FULLMAP_NPY    = "fullmap.npy"

# ── Bounding Box Extents ──────────────────────────────────────────────────────
EMB_EXTENT = {
    'minlon': -79.1397,
    'maxlon': -64.61405897436104,
    'minlat': 44.9922,
    'maxlat': 52.51472252251985
}

def _compute_zarr_extent():
    try:
        import zarr as _zarr
        from affine import Affine as _Affine
        from pyproj import Transformer as _Transformer
        store  = _zarr.open(ZARR_PATH, mode='r', zarr_format=2)
        t      = store.attrs['transform']
        crs    = store.attrs['crs']
        H, W   = store['predictors'].shape[1], store['predictors'].shape[2]
        aff    = _Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        corners_proj = [aff * (c, r) for c, r in [(0,0),(W,0),(0,H),(W,H)]]
        tr           = _Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        corners_ll   = [tr.transform(x, y) for x, y in corners_proj]
        return dict(
            minlon = min(c[0] for c in corners_ll),
            maxlon = max(c[0] for c in corners_ll),
            minlat = min(c[1] for c in corners_ll),
            maxlat = max(c[1] for c in corners_ll),
        )
    except Exception:
        # Fallback precomputed Zarr extent if climate_stack.zarr is not present (e.g. Hugging Face Space)
        return {
            'minlon': -79.506411,
            'maxlon': -64.4044751060257,
            'minlat': 44.620892771770805,
            'maxlat': 52.43903936426831
        }

_ZARR_EXTENT = _compute_zarr_extent()

# ── General Constants ─────────────────────────────────────────────────────────
RESOLUTION_KM = 1.0
BACKEND_Z     = 8
SPACING_DEG   = RESOLUTION_KM / 111.0
GRID_RES      = 256
FULLMAP_GRID_N = 100

DEVICE = torch.device("cpu")

# ── Global Thread Locks ───────────────────────────────────────────────────────
species_lock     = threading.Lock()
progress_lock    = threading.Lock()
_fullmap_lock    = threading.Lock()
_scale_cache_lock = threading.Lock()
_mask_cache_lock = threading.Lock()

# ── Caches and Progress Stores ────────────────────────────────────────────────
_scale_cache   = {}
progress_store = {}
_mask_cache    = {}

def _build_continuous_palette(stops, n_levels=256):
    stops = np.array(stops, dtype=np.float32)
    n_stops = len(stops)
    x_stops = np.linspace(0, 1, n_stops)
    x_interp = np.linspace(0, 1, n_levels)
    palette = np.zeros((n_levels, 4), dtype=np.uint8)
    for c in range(4):
        palette[:, c] = np.clip(np.interp(x_interp, x_stops, stops[:, c]), 0, 255).astype(np.uint8)
    return palette

COLOR_STOPS = [
    [185, 212, 218, 255], # #b9d4da (soft pale blue)
    [140, 192, 202, 255], # #8cc0ca (light teal)
    [106, 172, 184, 255], # #6aacb8 (medium blue)
    [30,  111, 126, 255], # #1e6f7e (dark blue/teal)
    [10,  61,  74,  255], # #0a3d4a (very dark teal)
    [255, 217, 125, 255], # #ffd97d (golden)
    [232, 160, 32,  255]  # #e8a020 (dark gold)
]

DISCRETE_COLORS = _build_continuous_palette(COLOR_STOPS, n_levels=10)
RAW_CONTINUOUS_COLORS = _build_continuous_palette(COLOR_STOPS, n_levels=256)

# ── Viewport PNG Tile Cache ───────────────────────────────────────────────────
class TilePngCache:
    def __init__(self, maxsize=1024):
        from collections import OrderedDict
        self.cache   = OrderedDict()
        self.maxsize = maxsize
        self.lock    = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

    def invalidate_species(self, species: str):
        with self.lock:
            to_del = [k for k in self.cache.keys() if isinstance(k, tuple) and len(k) > 0 and str(k[0]).lower() == species.lower()]
            for k in to_del:
                self.cache.pop(k, None)

_tile_png_cache = TilePngCache()

# ── Hugging Face Dataset Download Fallback ────────────────────────────────────
def _download_hf_assets():
    import os
    # Check if we are running in Hugging Face or if assets are missing
    checkpoint_exists = CHECKPOINT_PATH.exists()
    
    needed_emb_files = [
        "context_map.zip",
        "valid_land_embeddings_250.npy",
        "latitudes.npy",
        "longitudes.npy",
        "valid_mask.npy",
        "study_area_polygon.wkb",
        "context_map_meta.json"
    ]
    
    needed_mask_files = [
        "water.tif",
        "combined_mask.tif"
    ]
    
    mask_dir = Path("../data/data_layers/predictors_100_QC_normalized")
    
    missing_emb = [f for f in needed_emb_files if not (EMB_DIR / f).exists()]
    missing_mask = [f for f in needed_mask_files if not (mask_dir / f).exists()]

    if not checkpoint_exists or missing_emb or missing_mask:
        dataset_repo = os.environ.get("DATASET_REPO", "alexisBb/sdm-explorer-data")
        print(f"📦 Hugging Face Space startup: Missing large assets locally.", flush=True)
        print(f"📂 Checking/Downloading from HF Dataset repository: '{dataset_repo}'...", flush=True)
        try:
            from huggingface_hub import hf_hub_download
            
            # Ensure folders exist
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            EMB_DIR.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model weights if missing
            if not checkpoint_exists:
                print("  Downloading model weights model_simplified_v2_inference.pt...", flush=True)
                hf_hub_download(
                    repo_id=dataset_repo,
                    filename="model_simplified_v2_inference.pt",
                    repo_type="dataset",
                    local_dir=str(CHECKPOINT_PATH.parent),
                    token=os.environ.get("HF_TOKEN")
                )
            
            # Download Zarr and metadata files if missing
            for filename in missing_emb:
                print(f"  Downloading {filename}...", flush=True)
                hf_hub_download(
                    repo_id=dataset_repo,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(EMB_DIR),
                    token=os.environ.get("HF_TOKEN")
                )
                
            # Download mask TIFFs if missing
            for filename in missing_mask:
                print(f"  Downloading {filename}...", flush=True)
                hf_hub_download(
                    repo_id=dataset_repo,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(mask_dir),
                    token=os.environ.get("HF_TOKEN")
                )
            print("✅ All Hugging Face Hub assets are downloaded and ready!", flush=True)
        except Exception as e:
            print(f"⚠️ Error downloading assets from Hugging Face: {e}", flush=True)
            print("Please ensure your dataset repository exists and contains model_simplified_v2_inference.pt, all embedding_map_v2 files, and predictors_100_QC_normalized files.", flush=True)

_download_hf_assets()

