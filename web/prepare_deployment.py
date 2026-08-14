import os
import shutil
from pathlib import Path

# Paths
ROOT_DIR = Path("/home/alexis/Documents/projets/species-dist")
WEB_DIR = ROOT_DIR / "web"
DATA_DIR = ROOT_DIR / "data"
DEPLOY_DIR = ROOT_DIR / "deploy_hf"

# 1. Unused files in the web directory to delete
REMOVE_FILES = [
    "app.py",
    "benchmark_int8.py",
    "benchmark_kv_cache.py",
    "benchmark_new_model.py",
    "benchmark_results.csv",
    "context_10k.npy",
    "context_grid.npy",
    "gen_comparison_map_violins.py",
    "grid_lats.npy",
    "grid_lons.npy",
    "grid_shape.npy",
    "inference_pipeline.py",
    "lats_10k.npy",
    "lons_10k.npy",
    "main.ipynb",
    "model.py",
    "model_optimized.py",
    "new_model_proposed_by_user.py",
    "pred_v1.csv",
    "preds.csv",
    "sdm_map_danaus_plexippus_highres.png",
    "sdm_map_danaus_plexippus_lowres.png",
    "simplified_model.py",
    "sort_idx.npy",
    "species_predictions.csv",
    "test.ipynb",
    "test.py",
    "dynamic_embeddings.pt"
]

def clean_web_directory():
    print("🧹 Cleaning up unused/test files in web/ directory...")
    for filename in REMOVE_FILES:
        filepath = WEB_DIR / filename
        if filepath.exists():
            try:
                if filepath.is_dir():
                    shutil.rmtree(filepath)
                else:
                    filepath.unlink()
                print(f"  Deleted: {filename}")
            except Exception as e:
                print(f"  ❌ Error deleting {filename}: {e}")
        else:
            print(f"  (Not found: {filename})")
    
    # Remove any __pycache__ directory
    pycache = WEB_DIR / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
        print("  Deleted: __pycache__")

def build_deployment_package():
    print(f"📦 Packaging files for deployment into {DEPLOY_DIR}...")
    
    # Recreate deploy directory
    if DEPLOY_DIR.exists():
        print(f"  Removing existing {DEPLOY_DIR}...")
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy Dockerfile
    dockerfile = ROOT_DIR / "Dockerfile"
    if dockerfile.exists():
        shutil.copy(dockerfile, DEPLOY_DIR / "Dockerfile")
        print("  Copied Dockerfile")
    else:
        print("  ❌ Dockerfile not found at root!")
        
    # Copy web directory (necessary files only)
    web_dest = DEPLOY_DIR / "web"
    web_dest.mkdir(parents=True, exist_ok=True)
    
    keep_web_files = [
        "config.py",
        "geo_utils.py",
        "model_manager.py",
        "renderer.py",
        "inference.py",
        "text_inference.py",
        "utils.py",
        "server.py"
    ]
    
    for filename in keep_web_files:
        src = WEB_DIR / filename
        if src.exists():
            shutil.copy(src, web_dest / filename)
            print(f"  Copied web/{filename}")
            
    # Copy web/model directory (EXCLUDING model weights checkpoint)
    model_src = WEB_DIR / "model"
    model_dest = web_dest / "model"
    model_dest.mkdir(parents=True, exist_ok=True)
    if model_src.exists():
        for item in model_src.iterdir():
            if item.name == "model_simplified_v2_inference.pt" or item.name == "__pycache__":
                continue
            if item.is_dir():
                shutil.copytree(item, model_dest / item.name)
            else:
                shutil.copy(item, model_dest / item.name)
        print("  Copied web/model/ (excluding model weights checkpoint)")
        
    # Copy web/frontend directory
    frontend_src = WEB_DIR / "frontend"
    if frontend_src.exists():
        shutil.copytree(frontend_src, web_dest / "frontend")
        print("  Copied web/frontend/")
        
    # Copy data directory (necessary files only)
    data_dest = DEPLOY_DIR / "data"
    data_dest.mkdir(parents=True, exist_ok=True)
    
    # Exclude all data layers on Git to prevent binary reject hooks.
    # The server will download the entire embedding_map_v2 folder from the HF Dataset on startup.
    print("  Skipped copying data/data_layers/ embedding files (handled dynamically at startup)")
        
    # Copy web/data/web_portal_assets directory (species names and embeddings)
    assets_src = WEB_DIR / "data" / "web_portal_assets"
    assets_dest = web_dest / "data" / "web_portal_assets"
    if assets_src.exists():
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_src, assets_dest)
        print("  Copied web/data/web_portal_assets/ (species names and embeddings)")
    else:
        print("  ❌ web/data/web_portal_assets/ not found!")
        
    # Create an empty or code-only .gitignore
    gitignore_content = """# Local environments
.venv/
venv/
__pycache__/
*.pyc

# Cache
tile_cache_v2/
dynamic_embeddings.pt
"""
    with open(DEPLOY_DIR / ".gitignore", "w") as f:
        f.write(gitignore_content)
    print("  Created .gitignore")
    
    print("\n✅ Packaging complete! You can find the deployable folder at:")
    print(f"   {DEPLOY_DIR}")

if __name__ == "__main__":
    clean_web_directory()
    build_deployment_package()
