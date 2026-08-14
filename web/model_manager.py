import torch
import torch.quantization
import config
from utils import load_species_embeddings
import sys
from pathlib import Path
_MODEL_DIR = str(Path(__file__).parent / "model")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)
from simplified_model_v2 import FlamingoSDM_ResNet, StatsProjectionBackbone, FlamingoSDM_StatsOnly

import json
import numpy as np

print("Loading model and species list...", flush=True)

# Load species list from JSON
try:
    with open(config.SPECIES_NAMES_JSON, "r") as f:
        species_list = json.load(f)
    print(f"Loaded {len(species_list)} species from JSON: {config.SPECIES_NAMES_JSON}")
except Exception as e:
    print(f"⚠️ Failed to load species list from JSON ({e}), falling back to default list.")
    species_list = [
        'Acer pensylvanicum', 'Pinus strobus', 'Betula populifolia',
        'Fraxinus americana', 'Rhododendron groenlandicum',
        'Populus grandidentata', 'Rubus pubescens', 'Thuja occidentalis'
    ]

# Set _INITIAL_SPECIES dynamically to match loaded list (or fallback list)
_INITIAL_SPECIES = list(species_list)

# Load species embeddings from NPY
try:
    embeds_np = np.load(config.SPECIES_EMBEDDINGS_NPY)
    _init_embeds = torch.tensor(embeds_np, dtype=torch.float32).to(config.DEVICE)
    print(f"Loaded species embeddings from npy: {config.SPECIES_EMBEDDINGS_NPY}, shape: {_init_embeds.shape}")
except Exception as e:
    print(f"⚠️ Failed to load species embeddings from NPY ({e}), calling fallback load_species_embeddings.")
    _init_embeds = load_species_embeddings(species_list, config.EMBEDDING_CSV, config.DEVICE)

# Load checkpoint
checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE, weights_only=False)

# Initialize model structure (eager mode) with embed_dim=128 matching v2 checkpoint
model = FlamingoSDM_ResNet(
    sat_branch=None,
    clim_branch=None,
    geo_branch=None,
    coord_embedder=None,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    ffn_multiplier=2,
    sat_pool_size=4,
    clim_pool_size=4,
    geo_pool_size=3,
    num_total_species=len(species_list)
)

state_dict = checkpoint['model_state_dict'].copy()

# Prevent vocabulary shape mismatch between checkpoint and current subset
if 'species_queries' in state_dict and state_dict['species_queries'].shape != model.species_queries.shape:
    state_dict.pop('species_queries')
if 'log_priors' in state_dict and state_dict['log_priors'].shape != model.log_priors.shape:
    state_dict.pop('log_priors')
    
model.load_state_dict(state_dict, strict=False)
model.eval()

# Attach FocalSigLIPLoss parameter holders
try:
    from torch import nn as _nn
    class FocalSigLIPLoss(_nn.Module):
        def __init__(self):
            super().__init__()
            self.t = _nn.Parameter(torch.tensor(2.3))
            self.b = _nn.Parameter(torch.tensor(-5.0))
            
    model.loss_fn = FocalSigLIPLoss()
    if isinstance(checkpoint, dict) and 'loss_fn_state_dict' in checkpoint:
        model.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        print(f"✅ loss_fn loaded from checkpoint: t={model.loss_fn.t.exp().item():.4f} b={model.loss_fn.b.item():.4f}", flush=True)
    else:
        print(f"✅ loss_fn using init values: t={model.loss_fn.t.exp().item():.4f} b={model.loss_fn.b.item():.4f}", flush=True)
except Exception as e:
    print(f"⚠️  loss_fn not attached ({e}) — using default scaling", flush=True)

# ── INT8 dynamic quantization of decoder Linear layers ────────────────────────
# quantize_dynamic replaces nn.Linear weights with int8 at call time (CPU only).
# No retraining or calibration data needed. Benchmarked at ~1.66× speedup
# on the decoder forward pass, saving ~19s per full-map inference.
try:
    model.decoders = torch.quantization.quantize_dynamic(
        model.decoders,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    print("✅ Decoder quantized to INT8 (dynamic)", flush=True)
except Exception as e:
    print(f"⚠️  INT8 quantization failed ({e}) — using float32", flush=True)

# Restore custom dynamic species saved on disk
if config.DYNAMIC_EMBEDDINGS_PATH.exists():
    _saved = torch.load(config.DYNAMIC_EMBEDDINGS_PATH, map_location=config.DEVICE)
    for name in _saved["names"]:
        if name not in species_list:
            species_list.append(name)
    species_embeds_tensor = torch.cat([_init_embeds, _saved["embeddings"]], dim=0)
    print(f"✅ Restored {len(_saved['names'])} dynamic species from disk", flush=True)
else:
    species_embeds_tensor = _init_embeds.clone()


# ── Load Stats-Only Insect Model and Assets ────────────────────────────────────
stats_species_list = []
stats_embeds_tensor = None
stats_model = None
stats_mmap = None
stats_t_val = torch.tensor(2.3, device=config.DEVICE)
stats_b_val = torch.tensor(-5.0, device=config.DEVICE)
species_categories = {}

def get_species_category(name: str) -> str:
    return species_categories.get(name, "Trees & Plants")

stats_species_set = set()

if config.STATS_MODEL_PATH.exists() and config.STATS_INDEX_PATH.exists():
    try:
        import pandas as pd
        import ast
        print("Loading stats-only model and species list...", flush=True)
        
        # Load insects, birds, and trees v4 CSV files
        stats_files = [
            (config.STATS_SPECIES_EMBEDDINGS, "Insects"),
            (Path("../notebooks/species_embeddings_v4_birds.csv"), "Birds"),
            (Path("../notebooks/species_embeddings_v4.csv"), "Trees & Plants")
        ]
        
        all_stats_species = []
        all_stats_embeds = []
        
        for path, cat in stats_files:
            if path.exists():
                df = pd.read_csv(path)
                names = df['species_name'].tolist()
                all_stats_species.extend(names)
                for n in names:
                    species_categories[n] = cat
                
                for emb_str in df['embedding']:
                    if isinstance(emb_str, str):
                        all_stats_embeds.append(ast.literal_eval(emb_str))
                    else:
                        all_stats_embeds.append(emb_str)
                print(f"Loaded {len(names)} species ({cat}) from CSV: {path}", flush=True)
        
        stats_species_list = all_stats_species
        stats_species_set = set(sp.lower().strip() for sp in stats_species_list)
        stats_embeds_tensor = torch.tensor(all_stats_embeds, dtype=torch.float32).to(config.DEVICE)
        print(f"✅ Loaded total of {len(stats_species_list)} stats-only species embeddings", flush=True)

        # Instantiate model structure
        stats_branch = StatsProjectionBackbone(group_dim=64, out_channels=128)
        stats_model = FlamingoSDM_StatsOnly(
            stats_branch=stats_branch, text_dim=768, embed_dim=128,
            num_heads=4, num_layers=2, ffn_multiplier=2,
            num_groups=8, group_dim=64
        )
        
        # Load checkpoint
        stats_checkpoint = torch.load(config.STATS_MODEL_PATH, map_location=config.DEVICE, weights_only=False)
        stats_model.load_state_dict(stats_checkpoint['model_state_dict'])
        stats_model.eval()
        
        # Quantize stats model linear layers for CPU speedups
        try:
            stats_model.decoders = torch.quantization.quantize_dynamic(
                stats_model.decoders,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            print("✅ Stats decoder quantized to INT8 (dynamic)", flush=True)
        except Exception as qe:
            print(f"⚠️ Stats INT8 quantization failed ({qe}) — using float32", flush=True)

        # Load t and b Scaling parameters
        if 'loss_fn_state_dict' in stats_checkpoint:
            stats_t_val = stats_checkpoint['loss_fn_state_dict']['t'].exp().to(config.DEVICE)
            stats_b_val = stats_checkpoint['loss_fn_state_dict']['b'].to(config.DEVICE)
            print(f"✅ stats loss_fn loaded: t={stats_t_val.item():.4f} b={stats_b_val.item():.4f}", flush=True)
            
        # Load stats memmap
        stats_mmap = np.memmap(config.STATS_INDEX_PATH, dtype='float16', mode='r', shape=(1495814, 512))
        print(f"✅ Loaded stats index memmap: {config.STATS_INDEX_PATH}, shape: {stats_mmap.shape}")

        # Append stats species to global species_list
        for sp in stats_species_list:
            if sp not in species_list:
                species_list.append(sp)

    except Exception as e:
        print(f"⚠️ Failed to load stats-only model/assets: {e}")
        import traceback
        traceback.print_exc()

def is_stats_species(species_name: str) -> bool:
    # Use stats-only model by default for everything EXCEPT species that are ONLY in the old CNN model
    name_clean = species_name.lower().strip()
    old_cnn_species_set = set(sp.lower().strip() for sp in _INITIAL_SPECIES)
    if name_clean in old_cnn_species_set and name_clean not in stats_species_set:
        return False
    return True


# ── Dynamic Registry Operations ───────────────────────────────────────────────

def get_cached_species() -> list:
    """Returns a list of all dynamic (non-initial) species added by the user."""
    with config.species_lock:
        return [sp for sp in species_list if sp not in _INITIAL_SPECIES]


def _persist_dynamic_species():
    """Saves dynamic species names and embeddings tensor back to disk."""
    with config.species_lock:
        dynamic_names = get_cached_species()
        if not dynamic_names:
            if config.DYNAMIC_EMBEDDINGS_PATH.exists():
                config.DYNAMIC_EMBEDDINGS_PATH.unlink()
            return

        dynamic_indices = [species_list.index(sp) for sp in dynamic_names]
        dynamic_embeds = species_embeds_tensor[dynamic_indices]
        
        torch.save({
            "names": dynamic_names,
            "embeddings": dynamic_embeds
        }, config.DYNAMIC_EMBEDDINGS_PATH)
        print(f"💾 Persisted {len(dynamic_names)} dynamic species embeddings to disk", flush=True)


def get_species_cohort(target_species: str, cohort_size: int = 15) -> list[int]:
    """
    Returns the indices of the target species and its top `cohort_size` most similar
    species (by cosine similarity of their embeddings).
    The target species itself will always be at index 0 of the returned list.
    """
    with config.species_lock:
        if is_stats_species(target_species):
            try:
                target_idx = stats_species_list.index(target_species)
            except ValueError:
                return [0]
            
            target_emb = stats_embeds_tensor[target_idx]
            dot_product = torch.mv(stats_embeds_tensor, target_emb)
            target_norm = torch.norm(target_emb)
            all_norms = torch.norm(stats_embeds_tensor, dim=1)
            similarities = dot_product / (target_norm * all_norms + 1e-8)
            similarities = similarities.clone()
            similarities[target_idx] = -999.0
            
            num_others = min(cohort_size, len(stats_species_list) - 1)
            if num_others > 0:
                top_indices = torch.topk(similarities, k=num_others).indices.tolist()
            else:
                top_indices = []
            
            cohort_global_indices = []
            try:
                cohort_global_indices.append(species_list.index(target_species))
            except ValueError:
                cohort_global_indices.append(0)
                
            for idx in top_indices:
                sp_name = stats_species_list[idx]
                try:
                    cohort_global_indices.append(species_list.index(sp_name))
                except ValueError:
                    pass
            return cohort_global_indices
        else:
            try:
                target_idx = species_list.index(target_species)
            except ValueError:
                return [0] + list(range(1, min(cohort_size + 1, len(species_list))))
            
            if target_idx >= species_embeds_tensor.shape[0]:
                return [0]
                
            target_emb = species_embeds_tensor[target_idx]
            dot_product = torch.mv(species_embeds_tensor, target_emb)
            target_norm = torch.norm(target_emb)
            all_norms = torch.norm(species_embeds_tensor, dim=1)
            
            similarities = dot_product / (target_norm * all_norms + 1e-8)
            similarities = similarities.clone()
            similarities[target_idx] = -999.0
            
            num_others = min(cohort_size, species_embeds_tensor.shape[0] - 1)
            if num_others > 0:
                top_indices = torch.topk(similarities, k=num_others).indices.tolist()
            else:
                top_indices = []
            
            cohort_indices = [target_idx] + top_indices
            return cohort_indices

