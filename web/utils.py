import torch
import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
import glob
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
import rasterio
from affine import Affine
from pyproj import Transformer
from numcodecs import Blosc
from tqdm import tqdm

def setup_static_inputs(dataset, species_names, csv_path, device='cuda'):
    """
    Args:
        dataset: Your Train Dataset (to get the biology targets)
        species_names: List of strings (The column names you used to create the dataset)
        csv_path: Path to the embeddings CSV
    """
    print("⚙️ Setting up Static Inputs...")
    
    # --- 1. Load Embeddings from CSV ---
    print(f"   📂 Reading CSV: {csv_path}")
    df_emb = pd.read_csv(csv_path)
    
    # Create Lookup: Name -> Vector
    name_to_embed = {}
    for idx, row in df_emb.iterrows():
        name = row['species_name'].strip()
        
        # Handle string representation of lists if needed
        if isinstance(row['embedding'], str):
            vec = ast.literal_eval(row['embedding'])
        else:
            vec = row['embedding']
            
        name_to_embed[name] = vec

    # --- 2. Align with Dataset Order ---
    aligned_vectors = []
    missing_count = 0
    
    print(f"   🔄 Aligning {len(species_names)} species...")
    
    for species in species_names:
        if species in name_to_embed:
            aligned_vectors.append(name_to_embed[species])
        else:
            # Fallback: Random noise (better than zero for transformers)
            print(f"      ❌ Warning: Missing embedding for '{species}'. Using Random.")
            aligned_vectors.append(np.random.randn(768)) 
            missing_count += 1
            
    if missing_count > 0:
        print(f"      ⚠️ Total missing embeddings: {missing_count}")
    
    # Final Tensor: [Num_Species, 768]
    all_species_embeds = torch.tensor(np.array(aligned_vectors), dtype=torch.float32).to(device)
    print(f"   ✅ Final Embedding Tensor: {all_species_embeds.shape}")

    # --- 3. Compute Biology Target ---
    # We grab the labels from the dataset directly
    print("   🧬 Computing Biology Target (Co-occurrence Matrix)...")
    
    # Convert numpy labels to torch
    targets_clean = torch.tensor(dataset.labels, dtype=torch.float32)
    targets_clean[targets_clean == -1] = 0 # Ensure no -1s
    
    # Covariance Calculation
    mean = targets_clean.mean(dim=0, keepdim=True)
    centered = targets_clean - mean
    cov = centered.T @ centered / (targets_clean.shape[0] - 1)
    std = targets_clean.std(dim=0)
    
    # Correlation Matrix
    corr = cov / (torch.outer(std, std) + 1e-8)
    biology_target = torch.clamp(corr, min=0.0)
    biology_target.fill_diagonal_(1.0) # Self-correlation is always 1
    
    biology_target = biology_target.to(device)
    
    return all_species_embeds, biology_target

def instantiate_model_from_data(train_ds, all_species_embeddings, device='cuda'):
    print("🔍 Inspecting data...")
    num_total_species, text_embed_dim = all_species_embeddings.shape
    
    # Create a dummy loader to peek at one batch
    temp_loader = DataLoader(train_ds, batch_size=1)
    sample = next(iter(temp_loader))
    
    # 1. SETUP CLIMATE
    clim_branch = None
    c_in = 0 # Default placeholder
    
    if 'clim_data' in sample:
        # Detect channels automatically (e.g., 67)
        c_in = sample['clim_data'].shape[1] 
        print(f"   ✅ Climate Branch Active ({c_in} channels)")
        
        # Initialize Backbone with detected channels
        clim_branch = ClimateSimpleBackbone(input_channels=c_in, out_channels=128)
        
    # 2. SETUP SATELLITE
    sat_branch = None
    if 'sat_image' in sample and sample['sat_image'] is not None:
        s_in = sample['sat_image'].shape[1]
        print(f"   ✅ Satellite Branch Active ({s_in} channels)")
        sat_branch = ResNet50SatBackbone(pretrained=True) 

    # 3. SETUP MODEL
    print("🏗️ Building FlamingoSDM_ResNet...")
    model = FlamingoSDM_ResNet(
        sat_branch=sat_branch,
        clim_branch=clim_branch,
        sat_dim=1024,      
        clim_dim=128,      
        text_dim=text_embed_dim, 
        embed_dim=768,           
        num_total_species=num_total_species,
        # 🟢 FIX: Pass the detected channels here!
        clim_channels_raw=c_in 
    )
    
    model = model.to(device)
    print("✅ Model Ready.")
    return model


import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchgeo.models import ResNet50_Weights, resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchgeo.models import resnet50, ResNet50_Weights
# -------------------------------------------------------
# 0. Basic Residual Block (Helper)
# -------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

import torch
import torch.nn as nn
import copy
from torchgeo.models import resnet50, ResNet50_Weights

import torch
import torch.nn as nn
import copy
from torchgeo.models import resnet50, ResNet50_Weights

class ResNet50SatBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS if pretrained else None
        backbone = resnet50(weights=weights, output_stride=8)
        
        # 1. Fix First Layer (12 Channels)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=12, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=old_conv.bias
        )
        
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                # Map standard bands: B2, B3, B4...
                mapping = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 10:7, 11:8}
                for my_idx, satlas_idx in mapping.items():
                    if satlas_idx < old_conv.weight.shape[1]:
                        new_conv.weight[:, my_idx] = old_conv.weight[:, satlas_idx]

        backbone.conv1 = new_conv
        
        # 2. Sequential Backbone (Returns Feature Map, NOT Tokens)
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, nn.ReLU(inplace=True), backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        )

    def forward(self, x):
        # Returns: [Batch, 1024, 16, 16]
        return self.backbone(x)
# -------------------------------------------------------
# 2. AlphaEarth Branch - MULTI-SCALE
# -------------------------------------------------------
class AlphaEarthDualBranch(nn.Module):
    def __init__(self, input_channels=64, out_channels=256):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            BasicBlock(64, 128, stride=2)
        )
        self.block2 = BasicBlock(128, 256, stride=2)

        def make_high_level(): return BasicBlock(256, out_channels, stride=1)
        self.shared_high = make_high_level()
        self.residual_high = make_high_level()
        
    def set_stage(self, stage):
        if stage == "stage1":
            self.shared_high.train()
            for p in self.shared_high.parameters(): p.requires_grad = True
            self.residual_high.eval()
            for p in self.residual_high.parameters(): p.requires_grad = False
        elif stage == "stage2":
            self.shared_high.eval()
            for p in self.shared_high.parameters(): p.requires_grad = False
            self.residual_high.train()
            for p in self.residual_high.parameters(): p.requires_grad = True

    def transfer_shared_to_residual(self):
        self.residual_high.load_state_dict(self.shared_high.state_dict())

    def _spatial_flatten(self, x):
        b, c, h, w = x.shape
        return x.view(b, c, h * w).permute(0, 2, 1)

    def forward(self, x):
        if x.dtype == torch.float16: x = x.float()
        is_trainable = next(self.block1.parameters()).requires_grad
        with torch.set_grad_enabled(is_trainable):
            x_l1 = self.block1(x)
            x_mid = self.block2(x_l1)
            x_shared_map = self.shared_high(x_mid)
            x_shared_tokens = self._spatial_flatten(x_shared_map)

        x_res_map = self.residual_high(x_mid)
        x_res_tokens = self._spatial_flatten(x_res_map)

        return x_shared_tokens, x_res_tokens

class ClimateSimpleBackbone(nn.Module):
    def __init__(self, input_channels=67, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            # 20x20 -> 10x10
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            
            # 10x10 -> 5x5
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            
            # 5x5 refinement
            nn.Conv2d(128, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # Returns Single Tensor [Batch, 128, 5, 5]
        return self.net(x)
# -------------------------------------------------------
# 4. Coordinate Embedder
# -------------------------------------------------------
class MultiScaleFourierEmbedding(nn.Module):
    def __init__(self, input_dim=2, num_frequencies=64, max_scale=1000.0):
        super().__init__()
        self.input_dim = input_dim
        exponent = torch.linspace(0, np.log2(max_scale), num_frequencies)
        self.scales = 2 ** exponent
        self.out_dim = input_dim * num_frequencies * 2
        
        # We register it as a buffer so it saves with state_dict
        self.register_buffer('scales_tensor', self.scales)

    def forward(self, x):
        # x shape: [Batch, 2]
        x_expanded = x.unsqueeze(-1)
        
        # --- THE FIX ---
        # Explicitly ensure scales_tensor is on the same device as x
        device = x.device
        scales = self.scales_tensor.to(device)
        # ---------------
        
        scaled_x = x_expanded * scales * torch.pi
        embeddings = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
        
        return embeddings.view(x.shape[0], -1)


import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------
# 1. THE GENERIC GAUSSIAN EYE (Handles Sat OR Clim)
# -------------------------------------------------------
class TextGuidedGaussianAttention(nn.Module):
    def __init__(self, channels, embed_dim=768, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        
        # A. Create Grid [-1, 1]
        y = torch.linspace(-1, 1, grid_size)
        x = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('grid', torch.stack([grid_x, grid_y], dim=-1).view(1, 1, grid_size, grid_size, 2))
        
        # B. Hypernetwork (Text -> Scale)
        self.text_to_sigma = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # [Sigma_Y, Sigma_X]
        )
        # Default: Start Focused (Small Sigma ~0.13)
        nn.init.constant_(self.text_to_sigma[-1].bias, -2.0) 
        nn.init.constant_(self.text_to_sigma[-1].weight, 0.0)

        # C. Adapter (Project Extracted Features)
        self.adapter = nn.Sequential(
            nn.Linear(channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        # Residual Init (Start at 0 impact)
        nn.init.constant_(self.adapter[-1].weight, 0.0)
        nn.init.constant_(self.adapter[-1].bias, 0.0)

    def forward(self, feature_map, text_embeds):
        """
        feature_map: [B, Channels, H, W]
        text_embeds: [B, N_Species, Embed_Dim]
        """
        B, C, H, W = feature_map.shape
        
        # 🛡️ SAFETY: Interpolate if dimensions don't match grid
        if H != self.grid_size or W != self.grid_size:
            feature_map = F.interpolate(feature_map, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False)
        
        # 1. Predict Scale
        log_sigma = self.text_to_sigma(text_embeds)
        sigma = torch.exp(log_sigma).view(B, -1, 1, 1, 2) # [B, N, 1, 1, 2]
        self.last_sigma = sigma.detach().cpu() # For debugging

        # 2. Gaussian Masks (Centered at 0,0)
        # Broadcasting: Grid [1,1,H,W,2] vs Sigma [B,N,1,1,2]
        grid = self.grid.to(dtype=feature_map.dtype, device=feature_map.device)
        dist_sq = ((grid - 0) / sigma).pow(2).sum(dim=-1)
        weights = torch.exp(-0.5 * dist_sq) # [B, N, H, W]
        
        # Normalize weights over H,W so they sum to 1
        weights = weights / (weights.sum(dim=[-2, -1], keepdim=True) + 1e-6)
        
        # 3. Apply Attention (Extract Local Features)
        # We want [B, N, C]
        feat_flat = feature_map.view(B, C, -1)     # [B, C, HW]
        attn_flat = weights.view(B, -1, self.grid_size**2) # [B, N, HW]
        
        attended = torch.einsum('bcp, bnp -> bnc', feat_flat, attn_flat)
        
        # 4. Project to Embed Dim
        return self.adapter(attended)

# -------------------------------------------------------
# 2. THE MAIN MODEL
# -------------------------------------------------------
class FlamingoSDM_ResNet(nn.Module):
    def __init__(self, 
                 sat_branch=None, 
                 clim_branch=None, 
                 coord_embedder=None,
                 sat_dim=1024,   # Output channels of ResNet Layer3
                 clim_dim=128,   # Output channels of Climate Backbone
                 clim_channels_raw=19, # Raw input channels (e.g. 19 or 172)
                 text_dim=768,
                 embed_dim=768,
                 num_heads=8, 
                 num_total_species=536, 
                 species_prevalence=None):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.num_total_species = num_total_species
        
        # --- BRANCHES ---
        self.sat_branch = sat_branch
        self.clim_branch = clim_branch
        self.coord_embedder = coord_embedder
        
        # 🟢 1. GAUSSIAN EYES (The Local Experts)
        # Sat Eye: Sees 16x16 feature map from ResNet
        if sat_branch:
            self.sat_eye = TextGuidedGaussianAttention(channels=sat_dim, grid_size=16, embed_dim=embed_dim)
        else: self.sat_eye = None

        # Clim Eye: Sees raw 20x20 input map
        if clim_branch:
            self.clim_eye = TextGuidedGaussianAttention(channels=clim_channels_raw, grid_size=20, embed_dim=embed_dim)
        else: self.clim_eye = None

        # --- PROJECTORS (The Global Context) ---
        if sat_branch:
            self.sat_proj = nn.Sequential(nn.Conv2d(sat_dim, embed_dim, 1), nn.Flatten(2))
            self.sat_pos_embed = nn.Parameter(torch.randn(1, 16*16, embed_dim) * 0.02)

        if clim_branch:
            self.clim_proj = nn.Sequential(nn.Conv2d(clim_dim, embed_dim, 1), nn.Flatten(2))
            self.clim_pos_embed = nn.Parameter(torch.randn(1, 5*5, embed_dim) * 0.02) # Note: 5x5 is backbone output
            
        if coord_embedder:
            self.loc_proj = nn.Sequential(nn.Linear(coord_embedder.out_dim, embed_dim), nn.LayerNorm(embed_dim))

        self.text_proj = nn.Linear(text_dim, embed_dim) if text_dim != embed_dim else nn.Identity()
        self.species_queries = nn.Parameter(torch.randn(1, num_total_species, embed_dim))
        
        # --- DECODER ---
        self.decoders = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1)
            for _ in range(2)
        ])
        
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        

    def forward(self, sat=None, clim=None, coords=None, text_embeds=None):
        batch_size = 0
        context_tokens = []

        # --- A. FEATURE EXTRACTION ---
        sat_feats_map = None
        if self.sat_branch and sat is not None:
            batch_size = sat.shape[0]
            sat_feats_map = self.sat_branch(sat)

            toks = self.sat_proj(sat_feats_map).transpose(1, 2)
            toks = toks + self.sat_pos_embed[:, :toks.shape[1], :]
            context_tokens.append(toks)

        if self.clim_branch and clim is not None:
            if batch_size == 0: batch_size = clim.shape[0]
            clim_feats_backbone = self.clim_branch(clim)

            toks = self.clim_proj(clim_feats_backbone).transpose(1, 2)
            toks = toks + self.clim_pos_embed
            context_tokens.append(toks)

        if self.coord_embedder and coords is not None:
            coord_toks = self.loc_proj(self.coord_embedder(coords)).unsqueeze(1)
            context_tokens.append(coord_toks)

        # --- B. GLOBAL STREAM ---
        if not context_tokens:
            n_species = text_embeds.shape[1] if text_embeds is not None else self.num_total_species
            return torch.zeros(batch_size, n_species, device=next(self.parameters()).device), \
                torch.zeros(batch_size, n_species, self.embed_dim, device=next(self.parameters()).device)

        context = torch.cat(context_tokens, dim=1)
        queries = self.text_proj(text_embeds).expand(batch_size, -1, -1)

        for layer in self.decoders:
            queries = layer(queries, context)

        global_features = queries

        # --- C. LOCAL STREAM ---
        local_features = 0

        if self.sat_eye and sat_feats_map is not None and text_embeds is not None:
            local_features = local_features + self.sat_eye(sat_feats_map, text_embeds)

        if self.clim_eye and clim is not None and text_embeds is not None:
            local_features = local_features + self.clim_eye(clim, text_embeds)

        # --- D. FUSION ---
        final_features = global_features + local_features
        logits = self.classifier(final_features).squeeze(-1)

        return logits, final_features

    def get_scales(self):
        return {
            "sat": self.sat_eye.last_sigma if self.sat_eye else None,
            "clim": self.clim_eye.last_sigma if self.clim_eye else None
        }
        
# =============================================================================
# 2. BUILD ZARR STACK FROM INDIVIDUAL TIFS
# =============================================================================
# Run once to convert your 67 individual TIF files into a single Zarr store.
# RAM usage: one band at a time (~safe).
# Output size: depends on your raster, typically 2-10GB.

def build_zarr_stack(clim_dir: str, zarr_path: str):
    band_files = sorted(glob.glob(os.path.join(clim_dir, "*.tif")))
    print(f"Found {len(band_files)} band files")

    with rasterio.open(band_files[0]) as src:
        H, W   = src.height, src.width
        affine = src.transform
        crs    = str(src.crs)
        nodata = src.nodata

    store = zarr.open(zarr_path, mode='w', zarr_format=2)

    # chunks=(N_bands, 20, 20) — ALL bands for one spatial tile
    # This is critical: matches your model's 20x20 input after interpolation
    ds = store.create_array(
        'predictors',
        shape=(len(band_files), H, W),
        chunks=(len(band_files), 20, 20),
        dtype='float32',
        compressors=Blosc(cname='lz4', clevel=5)
    )

    # Store affine in natural Affine order: (a, b, c, d, e, f)
    store.attrs['transform']   = [affine.a, affine.b, affine.c,
                                   affine.d, affine.e, affine.f]
    store.attrs['crs']         = crs
    store.attrs['nodata']      = nodata or 0
    store.attrs['band_names']  = [os.path.basename(f) for f in band_files]

    # Write one band at a time — safe RAM usage
    for i, f in tqdm(enumerate(band_files), total=len(band_files), desc="Writing bands"):
        with rasterio.open(f) as src:
            ds[i] = src.read(1).astype('float32')

    print(f"✅ Zarr stack written to {zarr_path}")
    print(f"   Shape: {ds.shape}, Chunks: {ds.chunks}")


# =============================================================================
# 3. LAZY CLIMATE READER
# =============================================================================
# Opens Zarr store once, holds only metadata in RAM (~5MB).
# Reads only the requested window on each call (~4ms, ~4.4MB per patch).

class ClimateReader:
    def __init__(self, zarr_path: str):
        self.store = zarr.open(zarr_path, mode='r', zarr_format=2)
        self.ds    = self.store['predictors']

        # Reconstruct affine from stored attrs
        t = self.store.attrs['transform']
        self.affine = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        self.crs    = self.store.attrs['crs']

        # Transformer: WGS84 → data CRS (EPSG:6624)
        self.transformer = Transformer.from_crs(
            "EPSG:4326", self.crs, always_xy=True
        )

        print(f"ClimateReader ready — shape: {self.ds.shape}, dtype: {self.ds.dtype}")

    def read_patch(self, lon: float, lat: float, patch_size: int = 20) -> np.ndarray:
        """
        Read ALL bands for a 20x20 pixel window centered on (lon, lat).
        Returns numpy array (67, 20, 20). ~4ms on local disk.
        """
        x, y = self.transformer.transform(lon, lat)
        col, row = ~self.affine * (x, y)
        col, row = int(col), int(row)
        half = patch_size // 2

        H, W = self.ds.shape[1], self.ds.shape[2]
        if not (half <= row < H - half and half <= col < W - half):
            raise ValueError(f"({lon}, {lat}) is outside raster extent")

        patch = np.array(self.ds[:, row-half:row+half, col-half:col+half])
        if patch.shape[0] > 6 and patch[6, half, half] == 0.0:
            raise ValueError(f"({lon}, {lat}) has no valid climate data (masked)")
        return patch

    def read_patch_px(self, row: int, col: int, patch_size: int = 20) -> np.ndarray:
        """
        Read by pixel index — used for grid-based map generation.
        Returns numpy array (67, patch_size, patch_size).
        """
        half = patch_size // 2
        H, W = self.ds.shape[1], self.ds.shape[2]
        if not (half <= row < H - half and half <= col < W - half):
            raise ValueError(f"Pixel ({row}, {col}) out of bounds")
        patch = np.array(self.ds[:, row-half:row+half, col-half:col+half])
        if patch.shape[0] > 6 and patch[6, half, half] == 0.0:
            raise ValueError(f"Pixel ({row}, {col}) is masked")
        return patch
    
    def read_grid_batch(self, coords: list, patch_size: int = 20):
        """
        MACRO-READ: Fetches a single large bounding box containing all requested
        coordinates into RAM, then slices out the patches to minimize IO overhead.
        coords: list of (lon, lat) tuples
        """
        if not coords:
            return []

        half = patch_size // 2
        H, W = self.ds.shape[1], self.ds.shape[2]

        # 1. Convert all lat/lons to row/col pixel indices
        rows, cols = [], []
        for lon, lat in coords:
            x, y = self.transformer.transform(lon, lat)
            col, row = ~self.affine * (x, y)
            cols.append(int(col))
            rows.append(int(row))

        # 2. Find the minimum bounding box that covers ALL needed patches
        min_row = max(0, min(rows) - half)
        max_row = min(H, max(rows) + half)
        min_col = max(0, min(cols) - half)
        max_col = min(W, max(cols) + half)

        # 3. THE MAGIC: Fetch the entire block from disk into RAM in one shot
        try:
            big_block = np.array(self.ds[:, min_row:max_row, min_col:max_col])
        except Exception as e:
            print(f"Zarr read error: {e}")
            return [None] * len(coords)

        # 4. Slice the 40 individual patches instantly out of the RAM block
        patches = []
        for r, c in zip(rows, cols):
            # Check if this specific patch is out of the actual raster bounds
            if not (half <= r < H - half and half <= c < W - half):
                patches.append(None)
                continue

            # Calculate where this patch lives inside our new 'big_block'
            local_r1 = (r - half) - min_row
            local_r2 = (r + half) - min_row
            local_c1 = (c - half) - min_col
            local_c2 = (c + half) - min_col

            patch = big_block[:, local_r1:local_r2, local_c1:local_c2]
            if patch.shape[0] > 6 and patch[6, half, half] == 0.0:
                patches.append(None)
            else:
                patches.append(patch)

        return patches


def prepare_clim(patch_np: np.ndarray, target_size: int = 20) -> torch.Tensor:
    """
    Convert numpy patch to model-ready tensor, resized to 20x20.
    Model was trained on 20x20 climate input (2km patch at 100m resolution).
    """
    return torch.from_numpy(patch_np).float().unsqueeze(0)  # (1, 67, H, W)



# =============================================================================
# 4. SATELLITE READER
# =============================================================================
# Uses TorchGeo's CombinedBandsDataset — R-tree indexed, lazy reads.
# Init cost: ~1s (index build). Per-read cost: ~2ms.
# IMPORTANT: CRS is EPSG:4326, query in degrees not meters.

def build_sat_dataset(sat_tiles_dir: str, bands: list):
    """
    Build satellite dataset. Call once at startup.
    RAM: index only (~50MB), no pixel data loaded.
    """
    from torchgeo.datasets import RasterDataset
    import torch.nn.functional as F
    from typing import Union, Any, List

    class CombinedBandsDataset(RasterDataset):
        def __init__(self, paths, all_bands, transforms=None,
                     resolution=30, target_size=None):
            super().__init__(paths, transforms=transforms)
            self.all_bands   = all_bands
            self.bands       = all_bands
            self.target_size = target_size

        def __getitem__(self, query):
            hits      = self.index.intersection(tuple(query), objects=True)
            filepaths = [hit.object for hit in hits]
            if not filepaths:
                return self._empty(query)
            try:
                data = self._merge_files(filepaths, query, self.band_indexes)
            except Exception:
                return self._empty(query)
            data = data.float()
            if self.target_size and len(data.shape) == 3:
                data = F.interpolate(
                    data.unsqueeze(0),
                    size=(self.target_size, self.target_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            return {'image': data, 'bounds': query}

        def _empty(self, query):
            C    = len(self.all_bands)
            size = self.target_size or 128
            return {'image': torch.zeros((C, size, size)), 'bounds': query}

    t0 = time.perf_counter()
    ds = CombinedBandsDataset(sat_tiles_dir, bands, target_size=128)
    print(f"Satellite dataset ready — {len(ds.files)} files, "
          f"init: {(time.perf_counter()-t0)*1000:.0f}ms")
    print(f"Satellite CRS: {ds.crs}")   # EPSG:4326 — query in degrees
    return ds


def read_sat_patch(sat_ds, lon: float, lat: float, radius_deg: float = 0.01):
    """
    Read satellite patch centered on (lon, lat).
    radius_deg: ~0.01° ≈ 1km. CRS is EPSG:4326.
    Returns dict with 'image' tensor (12, 128, 128).
    
    IMPORTANT: Do NOT convert to meters — sat_ds.crs is EPSG:4326.
    """
    from torchgeo.datasets.utils import BoundingBox
    return sat_ds[BoundingBox(
        minx=lon - radius_deg, maxx=lon + radius_deg,
        miny=lat - radius_deg, maxy=lat + radius_deg,
        mint=0.0, maxt=9.223372036854776e+18   # must match index maxt
    )]


# =============================================================================
# 5. MODEL INSTANTIATION (inference, no dataset needed)
# =============================================================================

def load_species_embeddings(species_names: list, csv_path: str,
                             device='cpu') -> torch.Tensor:
    """
    Load species text embeddings from CSV.
    No dataset required — inference-only version.
    Returns tensor (N_species, 768).
    """
    df          = pd.read_csv(csv_path)
    name_to_vec = {}
    for _, row in df.iterrows():
        name = row['species_name'].strip()
        vec  = ast.literal_eval(row['embedding']) \
               if isinstance(row['embedding'], str) else row['embedding']
        name_to_vec[name] = vec

    vectors = []
    for sp in species_names:
        if sp in name_to_vec:
            vectors.append(name_to_vec[sp])
        else:
            print(f"⚠️  Missing embedding for '{sp}', using random.")
            vectors.append(np.random.randn(768))

    return torch.tensor(np.array(vectors), dtype=torch.float32).to(device)


def build_model(clim_channels: int, sat_channels: int,
                species_embeds: torch.Tensor, device='cpu'):
    """
    Instantiate FlamingoSDM_ResNet without needing a dataset.
    Decoupled from training code — safe for inference.
    """
    num_species, text_dim = species_embeds.shape

    clim_branch = ClimateSimpleBackbone(input_channels=clim_channels, out_channels=128)
    sat_branch  = ResNet50SatBackbone(pretrained=True)

    # layer3 lives inside sat_branch.backbone[-1]
    for p in sat_branch.parameters():
        p.requires_grad = False
    for p in sat_branch.backbone[-1].parameters():
        p.requires_grad = True

    model = FlamingoSDM_ResNet(
        sat_branch=sat_branch,
        clim_branch=clim_branch,
        sat_dim=1024,
        clim_dim=128,
        text_dim=text_dim,
        embed_dim=768,
        num_total_species=num_species,
        clim_channels_raw=clim_channels
    )
    return model.to(device)


def load_model(checkpoint_path: str, species_embeds: torch.Tensor,
               clim_channels=67, sat_channels=12, device='cpu'):
    """
    Build model and load weights from checkpoint.
    Handles zero-shot transfer (drops species_queries, log_priors).
    """
    model = build_model(clim_channels, sat_channels, species_embeds, device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Drop keys that change shape for zero-shot transfer
    keys_to_ignore   = ["species_queries", "log_priors"]
    filtered         = {k: v for k, v in state_dict.items()
                        if k not in keys_to_ignore}
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print(f"✅ Weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
    model.eval()
    return model