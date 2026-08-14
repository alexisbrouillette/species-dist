import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchgeo.models import resnet50, ResNet50_Weights
import numpy as np


class ResNet50SatBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS if pretrained else None
        backbone = resnet50(weights=weights, output_stride=8)
        
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(12, old_conv.out_channels, old_conv.kernel_size, 
                             stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias)
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                mapping = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 10:7, 11:8}
                for my_idx, satlas_idx in mapping.items():
                    if satlas_idx < old_conv.weight.shape[1]:
                        new_conv.weight[:, my_idx] = old_conv.weight[:, satlas_idx]
        backbone.conv1 = new_conv
        
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, nn.ReLU(inplace=True), backbone.maxpool)
        self.layer1 = backbone.layer1 
        self.layer2 = backbone.layer2 
        self.layer3 = backbone.layer3 

    def forward(self, x):
        x = self.stem(x)
        x_local = self.layer1(x)
        x = self.layer2(x_local)
        x_global = self.layer3(x)
        return x_global, x_local
    
# ── 1. Basic Residual Block (Helper) ──────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


# ── 2. Climate Simple Backbone ───────────────────────────────────────────────

class ClimateSimpleBackbone(nn.Module):
    def __init__(self, input_channels=67, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(128, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.net(x)   # (B, out_channels, 5, 5)


# ── 3. Geology Simple Backbone ───────────────────────────────────────────────

import torch
import torch.nn as nn

class GeologyCategoricalBackbone(nn.Module):
    def __init__(self, num_classes=11000, embed_dim=16, out_channels=128):
        super().__init__()
        self.geo_embed = nn.Embedding(num_classes, embed_dim)
        # Replace the two-layer conv net with a single linear projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.geo_embed(x)          # (B, 20, 20, 16)
        x = self.proj(x)               # (B, 20, 20, 128)
        return x.permute(0, 3, 1, 2)   # (B, 128, 20, 20)

# ── 4. Coordinate Embedder (Fourier Features) ────────────────────────────────

class MultiScaleFourierEmbedding(nn.Module):
    def __init__(self, input_dim=2, num_frequencies=64, max_scale=1000.0):
        super().__init__()
        self.input_dim = input_dim
        exponent = torch.linspace(0, np.log2(max_scale), num_frequencies)
        self.out_dim = input_dim * num_frequencies * 2
        self.register_buffer('scales_tensor', 2 ** exponent)

    def forward(self, x):
        scales    = self.scales_tensor.to(x.device)
        x_exp     = x.unsqueeze(-1)
        scaled_x  = x_exp * scales * torch.pi
        embeddings = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
        return embeddings.view(x.shape[0], -1)


# ── 5. CenterPixelMLP (Text-Independent Local Expert) ──────────────────────────

class CenterPixelMLP(nn.Module):
    """
    Extracts a K×K center crop from a raw spatial raster and projects it
    to embed_dim via a 3-layer MLP. This provides high-resolution "foveal"
    context at the exact location while remaining 100% precomputable.
    """
    def __init__(self, in_channels: int, embed_dim: int = 128, crop_size: int = 3):
        super().__init__()
        self.crop_size = crop_size
        in_dim = in_channels * crop_size * crop_size
        hidden_dim = max(128, embed_dim * 2)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Residual initialization — zero impact at the start of training
        nn.init.constant_(self.mlp[-2].weight, 0.0)
        nn.init.constant_(self.mlp[-2].bias,   0.0)

    def _center_crop(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.crop_size

        if H < k or W < k:
            return F.adaptive_avg_pool2d(x, 1).view(B, C)

        row = (H - k) // 2
        col = (W - k) // 2
        crop = x[:, :, row: row + k, col: col + k]
        return crop.reshape(B, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center = self._center_crop(x)        # (B, C*K*K)
        out    = self.mlp(center)             # (B, embed_dim)
        return out.unsqueeze(1)               # (B, 1, embed_dim)


# ── 6. Optimized FlamingoSDM_ResNet ───────────────────────────────────────────

class FlamingoSDM_ResNet(nn.Module):
    def __init__(
        self,
        sat_branch=None,
        clim_branch=None,
        geo_branch=None,
        coord_embedder=None,
        sat_dim=1024,
        clim_dim=128,
        geo_dim=128,
        clim_channels_raw=67,
        text_dim=768,
        embed_dim=128,              # Parameterized: default 128
        num_heads=4,                # Divides 128 cleanly
        num_layers=2,               # Number of cross-attention decoder layers
        num_total_species=536,
        center_crop_size=3,
        sat_pool_size=8,            # 16x16 -> 8x8 (64 tokens)
        clim_pool_size=5,           # 5x5   -> 5x5 (25 tokens)
        geo_pool_size=5,            # 20x20 -> 5x5 (25 tokens)
        ffn_multiplier=4,
    ):
        super().__init__()
        self.embed_dim          = embed_dim
        self.num_total_species  = num_total_species
        self.sat_pool_size      = sat_pool_size
        self.clim_pool_size     = clim_pool_size
        self.geo_pool_size      = geo_pool_size

        # ── Branches ──────────────────────────────────────────────────────────
        self.sat_branch    = sat_branch
        self.clim_branch   = clim_branch
        self.geo_branch    = geo_branch
        self.coord_embedder = coord_embedder
        self.register_buffer('log_priors', torch.zeros(num_total_species))

        # ── Global Projectors & Position Embeddings ───────────────────────────
        if sat_branch:
            self.sat_proj = nn.Sequential(nn.Conv2d(sat_dim, embed_dim, 1), nn.Flatten(2))
            self.sat_pool = nn.AdaptiveAvgPool2d((sat_pool_size, sat_pool_size))
            self.sat_pos_embed = nn.Parameter(torch.randn(1, sat_pool_size * sat_pool_size, embed_dim) * 0.02)

        if clim_branch:
            self.clim_proj = nn.Sequential(nn.Conv2d(clim_dim, embed_dim, 1), nn.Flatten(2))
            self.clim_pool = nn.AdaptiveAvgPool2d((clim_pool_size, clim_pool_size))
            self.clim_pos_embed = nn.Parameter(torch.randn(1, clim_pool_size * clim_pool_size, embed_dim) * 0.02)

        if geo_branch:
            self.geo_proj = nn.Sequential(nn.Conv2d(geo_dim, embed_dim, 1), nn.Flatten(2))
            self.geo_pool = nn.AdaptiveAvgPool2d((geo_pool_size, geo_pool_size))
            self.geo_pos_embed = nn.Parameter(torch.randn(1, geo_pool_size * geo_pool_size, embed_dim) * 0.02)

        if coord_embedder:
            self.loc_proj = nn.Sequential(
                nn.Linear(coord_embedder.out_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        # ── Center MLPs (Local Foveal Experts) ────────────────────────────────
        
        self.sat_center_mlp = None

        
        self.clim_center_mlp = None

        
        self.geo_center_mlp = None


        # ── Text Projection & Species Queries ───────────────────────────────
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        ) if text_dim != embed_dim else nn.Identity()
        self.species_queries = nn.Parameter(torch.randn(1, num_total_species, embed_dim))

        # ── Transformer Decoder with SDPA and Optimized FFN Multiplier ────────
        # dim_feedforward is set to 2x embed_dim instead of the default 4x
        self.decoders = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * ffn_multiplier,  # ← was hardcoded *2
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(num_layers)   # ← was hardcoded 2
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

    # ── Context Builder (purely visual/spatial - 100% precomputable) ──────────

    def _build_context(
        self,
        sat:   torch.Tensor | None = None,
        clim:  torch.Tensor | None = None,
        geo:   torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        batch_size     = 0
        context_tokens = []

        # Satellite tokens
        if self.sat_branch and sat is not None:
            batch_size = sat.shape[0]
            
            # 🟢 THE FIX: Unpack the tuple and keep only x_global
            sat_feats_tuple = self.sat_branch(sat)
            sat_feats = sat_feats_tuple[0] 

            sat_feats_pooled = self.sat_pool(sat_feats)
            toks = self.sat_proj(sat_feats_pooled).transpose(1, 2)
            toks = toks + self.sat_pos_embed
            context_tokens.append(toks)

            if self.sat_center_mlp is not None:
                context_tokens.append(self.sat_center_mlp(sat))

        # Climate tokens
        if self.clim_branch and clim is not None:
            if batch_size == 0:
                batch_size = clim.shape[0]
            clim_feats = self.clim_branch(clim)
            clim_feats_pooled = self.clim_pool(clim_feats)
            toks = self.clim_proj(clim_feats_pooled).transpose(1, 2)
            toks = toks + self.clim_pos_embed
            context_tokens.append(toks)

            if self.clim_center_mlp is not None:
                context_tokens.append(self.clim_center_mlp(clim))

        # Geology tokens
        if self.geo_branch and geo is not None:
            if batch_size == 0:
                batch_size = geo.shape[0]
            geo_feats = self.geo_branch(geo)
            geo_feats_pooled = self.geo_pool(geo_feats)
            toks = self.geo_proj(geo_feats_pooled).transpose(1, 2)
            toks = toks + self.geo_pos_embed
            context_tokens.append(toks)

            if self.geo_center_mlp is not None:
                context_tokens.append(self.geo_center_mlp(geo))

        # Coordinates tokens
        if self.coord_embedder and coords is not None:
            if batch_size == 0:
                batch_size = coords.shape[0]
            coord_toks = self.loc_proj(self.coord_embedder(coords)).unsqueeze(1)
            context_tokens.append(coord_toks)

        if not context_tokens:
            return None, batch_size

        return torch.cat(context_tokens, dim=1), batch_size

    # ── Main Forward Pass ─────────────────────────────────────────────────────

    def forward(
        self,
        sat=None,
        clim=None,
        geo=None,
        coords=None,
        text_embeds=None,
        species_indices=None,   # ← add this
    ):
        # 1. Build purely visual spatial context sequence
        context, batch_size = self._build_context(sat, clim, geo, coords)

        if context is None:
            n_sp = text_embeds.shape[1] if text_embeds is not None else self.num_total_species
            dev  = next(self.parameters()).device
            return (
                torch.zeros(batch_size, n_sp, device=dev),
                torch.zeros(batch_size, n_sp, self.embed_dim, device=dev),
            )

        # 2. Project and expand text queries
        queries = self.text_proj(text_embeds)

        # 3. Cross-attention layers
        for layer in self.decoders:
            queries = layer(queries, context)

        final_features = queries
        logits         = self.classifier(final_features).squeeze(-1)

        # 4. Add log priors during training only — zero cost at inference
        if self.training and species_indices is not None:
            logits = logits + self.log_priors[species_indices]

        return logits, final_features
    # ── Helper for offline precomputation ─────────────────────────────────────

    @torch.no_grad()
    def precompute_context(
        self,
        sat:    torch.Tensor | None = None,
        clim:   torch.Tensor | None = None,
        geo:    torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Runs the backbones and Center MLPs to output the complete precomputable
        context sequence. Save this sequence to disk to bypass backbones entirely!
        """
        self.eval()
        context, _ = self._build_context(sat=sat, clim=clim, geo=geo, coords=coords)
        return context


def instantiate_model_from_data1(
    train_ds,
    all_species_embeddings,
    device='cpu',
    # ── All tuning knobs in one place ──
    embed_dim=128,
    num_layers=2,
    num_heads=4,
    ffn_multiplier=2,
    # ── Backbone capacity ──
    clim_out_channels=128,
    geo_embed_dim=16,
    geo_out_channels=128,
    geo_num_classes=11000,
    # ── Token budget (affects precomputed map file size) ──
    sat_pool_size=4,        # 4x4 = 16 tokens  (was 64)
    clim_pool_size=4,       # 4x4 = 16 tokens  (was 25)
    geo_pool_size=3,        # 3x3 = 9 tokens   (was 25)
    center_crop_size=3,
):
    print("🔍 Inspecting data...")
    num_total_species, text_embed_dim = all_species_embeddings.shape

    temp_loader = DataLoader(train_ds, batch_size=1)
    sample = next(iter(temp_loader))

    # 1. Climate Branch
    clim_branch = None
    c_in = 0
    clim_key = 'image' if 'image' in sample else 'clim_data'
    if clim_key in sample:
        c_in = sample[clim_key].shape[1]
        print(f"   ✅ Climate Branch Active ({c_in} channels via '{clim_key}')")
        clim_branch = ClimateSimpleBackbone(
            input_channels=c_in,
            out_channels=clim_out_channels,
        )

    # 2. Satellite Branch
    sat_branch = None
    if 'sat_image' in sample and sample['sat_image'] is not None:
        s_in = sample['sat_image'].shape[1]
        print(f"   ✅ Satellite Branch Active ({s_in} channels)")
        sat_branch = ResNet50SatBackbone(pretrained=True)
    if sat_branch:
        for p in sat_branch.parameters():
            p.requires_grad = False
        for p in sat_branch.layer3[-1].parameters():
            p.requires_grad = True

    # 3. Geology Branch
    geo_branch = None
    if 'geology' in sample and sample['geology'] is not None:
        print(f"   ✅ Geology Branch Active (Spatial Categorical, "
              f"embed_dim={geo_embed_dim}, out={geo_out_channels})")
        geo_branch = GeologyCategoricalBackbone(
            num_classes=geo_num_classes,
            embed_dim=geo_embed_dim,
            out_channels=geo_out_channels,
        )
    else:
        print("   ⚠️ Geology data not found in sample; skipping branch.")

    # 4. Coordinate Embedder
    coord_embedder = MultiScaleFourierEmbedding()
    print("   ✅ Coordinate Embedder Active")

    assert embed_dim % num_heads == 0, \
        f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

    # ── Token count summary (useful for estimating precomputed map file size) ──
    n_sat_tokens   = sat_pool_size * sat_pool_size + 1 if sat_branch else 0   # +1 for center MLP
    n_clim_tokens  = clim_pool_size * clim_pool_size + 1 if clim_branch else 0
    n_geo_tokens   = geo_pool_size * geo_pool_size if geo_branch else 0
    n_coord_tokens = 1
    total_tokens   = n_sat_tokens + n_clim_tokens + n_geo_tokens + n_coord_tokens
    map_size_mb_per_point = total_tokens * embed_dim * 4 / 1024**2  # float32
    print(f"   📦 Context tokens per location: {total_tokens} "
          f"({map_size_mb_per_point*1000:.2f} KB/point at embed_dim={embed_dim})")

    print(f"🏗️ Building FlamingoSDM_ResNet "
          f"(embed_dim={embed_dim}, layers={num_layers}, "
          f"heads={num_heads}, ffn={ffn_multiplier}x)...")

    model = FlamingoSDM_ResNet(
        sat_branch=sat_branch,
        clim_branch=clim_branch,
        geo_branch=geo_branch,
        coord_embedder=coord_embedder,
        sat_dim=1024,
        clim_dim=clim_out_channels,
        geo_dim=geo_out_channels,
        text_dim=text_embed_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_multiplier=ffn_multiplier,
        num_total_species=num_total_species,
        clim_channels_raw=c_in,
        sat_pool_size=sat_pool_size,
        clim_pool_size=clim_pool_size,
        geo_pool_size=geo_pool_size,
        center_crop_size=center_crop_size,
    )

    model = model.to(device)
    print("✅ Model Ready.")
    return model


class StatsProjectionBackbone(nn.Module):
    def __init__(self, group_dim=64, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(group_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class FlamingoSDM_StatsOnly(nn.Module):
    def __init__(
        self, stats_branch, text_dim=768, embed_dim=128,
        num_heads=4, num_layers=2, ffn_multiplier=4,
        num_groups=8, group_dim=64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.stats_branch = stats_branch
        self.stats_pos_embed = nn.Parameter(torch.randn(1, num_groups, embed_dim) * 0.02)

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim), nn.LayerNorm(embed_dim)
        ) if text_dim != embed_dim else nn.Identity()

        self.decoders = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * ffn_multiplier,
                batch_first=True, dropout=0.1,
            )
            for _ in range(num_layers)
        ])

    def forward(self, stats, text_embeds):
        x = stats.view(stats.shape[0], self.num_groups, self.group_dim)
        context = self.stats_branch(x) + self.stats_pos_embed

        queries = self.text_proj(text_embeds)
        for layer in self.decoders:
            queries = layer(queries, context)

        if torch.isnan(queries).any() or torch.isinf(queries).any():
            print("❌ WARNING: NaNs/Infs detected in model features!")

        return queries