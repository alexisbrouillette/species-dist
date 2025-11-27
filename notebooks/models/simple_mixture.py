import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm 
from torchgeo.models import ResNet50_Weights, resnet50
from utils.model_pretraining import transfer_pretrained_weights, ClimatePretrainingCNN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import copy
# ------------------------------
# 1. Sentinel-2 branch (UNCOMPRESSED)
# ------------------------------

class ResNet50DualPath(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS if pretrained else None
        backbone = resnet50(weights=weights)

        # 1. SHARED TRUNK (Frozen)
        # Contains Stem, Layer 1, Layer 2
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, nn.ReLU(inplace=True), backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        
        # Freezing Trunk
        for p in self.stem.parameters(): p.requires_grad = False
        for p in self.layer1.parameters(): p.requires_grad = False
        for p in self.layer2.parameters(): p.requires_grad = False

        # 2. PATH A: SHARED BRANCH (Frozen)
        self.layer3_shared = backbone.layer3
        for p in self.layer3_shared.parameters(): p.requires_grad = False
        
        # 3. PATH B: RESIDUAL BRANCH (Unfrozen Clone)
        # We deepcopy layer3 so it starts with the same pre-trained weights
        # but can diverge during training.
        self.layer3_residual = copy.deepcopy(backbone.layer3)
        for p in self.layer3_residual.parameters(): p.requires_grad = True
        
        print(">>> Dual-Path ResNet Initialized.")
        print("    - Trunk (Stem+L1+L2): Frozen")
        print("    - Branch A (Shared L3): Frozen")
        print("    - Branch B (Residual L3): Unfrozen (Trainable)")

    def forward(self, x):
        # --- Common Path ---
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # --- Divergent Paths ---
        
        # Path A: The "Safe" Generalist Features
        # (Used by Shared Head)
        feat_shared = self.layer3_shared(x)
        feat_shared = F.adaptive_avg_pool2d(feat_shared, (1, 1)).flatten(1)
        
        # Path B: The "Adaptive" Specialist Features
        # (Used by Residual Heads)
        feat_residual = self.layer3_residual(x)
        feat_residual = F.adaptive_avg_pool2d(feat_residual, (1, 1)).flatten(1)
        
        return feat_shared, feat_residual
    
class ResNet50SpatialBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS if pretrained else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            nn.ReLU(inplace=True),
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        # --- CHANGE 1: REMOVED COMPRESSION LAYER ---
        # We no longer squeeze 1024 -> 256. 
        # We let the full feature volume flow through.
        
        # Freeze everything by default
        for param in backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze layer3 for fine-tuning
        print("Freezing ResNet50 up to layer2. Training layer3.")
        for param in self.layer3.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)                    # (B, 1024, H, W)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (B, 1024, 1, 1)
        return x.flatten(1)                   # (B, 1024)

# ------------------------------
# 2. Climate branch (PERFECTLY SIZED FOR 20x20)
# ------------------------------
class FlexibleCNNBranchSpatial(nn.Module):
    def __init__(self, input_channels, out_channels=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # 20x20 -> 10x10
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # 10x10 -> 5x5
            
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            # No final pool here, we let global avg pool handle the 5x5
        )

    def forward(self, x):
        x = self.conv_layers(x)        # (B, out_channels, 5, 5)
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        return x.flatten(1)            # (B, out_channels)

# ------------------------------
# 3. Fourier Embedder (Unchanged)
# ------------------------------
class MultiScaleFourierEmbedding(nn.Module):
    def __init__(self, input_dim=2, num_frequencies=64, max_scale=1000.0):
        super().__init__()
        self.input_dim = input_dim
        exponent = torch.linspace(0, np.log2(max_scale), num_frequencies)
        self.scales = 2 ** exponent
        self.out_dim = input_dim * num_frequencies * 2
        self.register_buffer('scales_tensor', self.scales)

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        scaled_x = x_expanded * self.scales_tensor * torch.pi
        sin_x = torch.sin(scaled_x)
        cos_x = torch.cos(scaled_x)
        embeddings = torch.cat([sin_x, cos_x], dim=-1)
        return embeddings.view(x.shape[0], -1)

# -------------------------------------------------------
# 4. The Updated Hybrid Model (NO ANCHORS, NO BOTTLENECK)
# -------------------------------------------------------
class HybridMultiSpeciesModel(nn.Module):
    def __init__(
        self,
        input_shapes,
        num_species,
        # --- CHANGE 2: WIDER SHARED HEAD ---
        # Input is now ~1200 dims, so we start wide (1024) to avoid immediate compression
        shared_head_dim=(1024, 512, 256), 
        species_head_dims=(256, 256, 128), 
        dropout=0.0,
        use_latent=False, # Simplified: removed latent logic for clarity, can add back if needed
        training_mode="stage1",
        use_satelitte=True,
        use_climate = True,
        use_fourier_coords=True,
        use_shared_heads=True,
        use_independent_heads=True
    ):
        super().__init__()
        self.num_species = num_species
        self.dropout = dropout
        self.training_mode = training_mode
        self.use_fourier_coords = use_fourier_coords
        self.use_shared_heads = use_shared_heads
        self.use_independent_heads = use_independent_heads
        
        # --- 1. SHARED TOWER ---
        self.shared_sat_branch = None
        self.shared_clim_branch = None
        self.coord_embedder = None
        self.shared_coords_mlp = None
        
        shared_total_features = 0

        for input_shape in input_shapes:
            source = input_shape['source']
            shape = input_shape['shape']
            
            if source == 'sentinel2' and use_satelitte:
                #self.shared_sat_branch = ResNet50SpatialBranch()
                self.shared_sat_branch = ResNet50DualPath()
                # --- CHANGE 3: UPDATED DIMENSION ---
                shared_total_features += 1024 
                
            elif source == 'pred_100' and use_climate:
                self.shared_clim_branch = FlexibleCNNBranchSpatial(shape[0])
                shared_total_features += 128
                
            elif source == 'coordinates':
                if self.use_fourier_coords:
                    self.coord_embedder = MultiScaleFourierEmbedding(
                        input_dim=2, num_frequencies=32, max_scale=1000.0
                    )
                # Simple MLP for raw coords to merge into shared stream
                self.shared_coords_mlp = nn.Sequential(
                    nn.Linear(2, 64), 
                    nn.LeakyReLU(0.1), 
                    nn.BatchNorm1d(64)
                )
                shared_total_features += 64
        
        print(f"Shared Tower Feature Dim: {shared_total_features}") # Should be ~1216

        # --- 2. SHARED HEAD (MLP) ---
        layers = []
        in_dim = shared_total_features
            
        for h in shared_head_dim:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(self.dropout))
            in_dim = h
        
        layers.append(nn.Linear(in_dim, self.num_species))
        self.shared_head_mlp = nn.Sequential(*layers)
            
        # --- 3. SPECIALIST HEADS ---
        species_heads_mlps = []
        
        # Specialist Input = [Shared Features (1024+128+64)] + [Shared Logits (Num_Spec)] + [Fourier (128)] + [Raw Coords (2)]
        specialist_in_dim = shared_total_features + num_species
        
        if self.coord_embedder is not None:
            specialist_in_dim += self.coord_embedder.out_dim # +128
        specialist_in_dim += 2 # Raw Coords

        print(f"Specialist Head Input Dim: {specialist_in_dim}")

        for _ in range(self.num_species):
            layers = []
            in_dim = specialist_in_dim 
            
            for h in species_head_dims: 
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.LeakyReLU(0.1))
                # --- CHANGE 4: Standard Dropout Only ---
                layers.append(nn.Dropout(0.4)) 
                in_dim = h
            
            layers.append(nn.Linear(in_dim, 1)) 
            species_heads_mlps.append(nn.Sequential(*layers))
            
        self.specialist_input_norm = nn.BatchNorm1d(specialist_in_dim)
        self.species_heads_mlps = nn.ModuleList(species_heads_mlps)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def set_training_stage(self, stage):
        self.training_mode = stage
        print(f"Model switched to: {stage}")
        
    def train(self, mode=True):
        super().train(mode)
        if mode and self.training_mode == "stage2":
            # Freeze Shared Components
            if self.shared_sat_branch: self.shared_sat_branch.eval()
            if self.shared_clim_branch: self.shared_clim_branch.eval()
            if self.shared_coords_mlp: self.shared_coords_mlp.eval()
            self.shared_head_mlp.eval()
            # Train only Specialists
            self.species_heads_mlps.train()
        return self

    def forward(self, inputs, coordinates=None, alpha=None):
        # --- 1. Input Unpacking ---
        sat_input, clim_input = None, None
        input_idx = 0
        if self.shared_sat_branch: sat_input = inputs[input_idx]; input_idx += 1
        if self.shared_clim_branch: clim_input = inputs[input_idx]; input_idx += 1
        
        # --- 2. Feature Extraction ---
        # We capture outputs in variables first so we can route them differently
        f_sat_shared, f_sat_res = None, None
        f_clim = None
        f_coords_emb = None
        
        # A. Satellite Branch (DUAL PATH)
        if self.shared_sat_branch and sat_input is not None:
            # Expecting tuple: (frozen_features, unfrozen_features)
            f_sat_shared, f_sat_res = self.shared_sat_branch(sat_input)
            
        # B. Climate Branch (SINGLE PATH - Reused)
        if self.shared_clim_branch and clim_input is not None:
            f_clim = self.shared_clim_branch(clim_input)
            
        # C. Coordinates MLP (SINGLE PATH - Reused)
        if self.shared_coords_mlp and coordinates is not None:
            f_coords_emb = self.shared_coords_mlp(coordinates)
        
        # --- 3. Build SHARED Tower Input (Frozen Path) ---
        shared_feats_list = []
        if f_sat_shared is not None: shared_feats_list.append(f_sat_shared)
        if f_clim is not None:       shared_feats_list.append(f_clim)
        if f_coords_emb is not None: shared_feats_list.append(f_coords_emb)
        
        x_shared = torch.cat(shared_feats_list, dim=1)
        
        # --- 4. Shared Head Predictions ---
        shared_logits = self.shared_head_mlp(x_shared)

        # --- 5. Specialist Heads (Unfrozen Path) ---
        if self.training_mode == 'stage2':
            
            # We reconstruct the feature vector specifically for the residuals.
            # CRITICAL: We use 'f_sat_res' (Unfrozen) instead of 'f_sat_shared'
            
            spec_feats_list = []
            
            # 1. The Trainable Satellite Features
            if f_sat_res is not None: 
                spec_feats_list.append(f_sat_res)
            elif f_sat_shared is not None: 
                # Fallback if dual path fails/missing (safety)
                spec_feats_list.append(f_sat_shared)
                
            # 2. Reuse Climate & Coord Embeddings (No split)
            if f_clim is not None:       spec_feats_list.append(f_clim)
            if f_coords_emb is not None: spec_feats_list.append(f_coords_emb)
            
            # 3. Add Shared Logits (Context)
            spec_feats_list.append(shared_logits.detach())
            
            # 4. Add Raw Coordinates (Spatial Context)
            if coordinates is not None:
                spec_feats_list.append(coordinates)
                
            # 5. Add Fourier Features (High freq spatial)
            if self.coord_embedder and coordinates is not None:
                spec_feats_list.append(self.coord_embedder(coordinates))

            # Concatenate and Normalize
            res_features = torch.cat(spec_feats_list, dim=1)
            res_features = self.specialist_input_norm(res_features)

            specific_logits_list = [head(res_features) for head in self.species_heads_mlps]
            specific_logits = torch.cat(specific_logits_list, dim=1)
        else:
            specific_logits = torch.zeros_like(shared_logits)

        final_logits = shared_logits + specific_logits 
        
        # Return tuple (final, shared, specific) for the loss function
        return final_logits, shared_logits, specific_logits, x_shared, {}

    def get_average_gate_weights(self, dataloader, device):
        """
        Calculates the average gate weights PER SPECIES for the 
        INDEPENDENT heads across a dataset.
        
        Args:
            dataloader: The DataLoader (e.g., val_loader) to evaluate on.
            device: The device to run the model on.

        Returns:
            mean_gates: A (num_species, num_modalities) numpy array of average gate weights.
            feature_names: A list of modality names.
        """
        if not self.use_gating:
            print("Gating is not enabled for this model (use_gating=False).")
            return None, None
            
        if not self.use_independent_heads:
            print("Independent heads are not enabled. Cannot get per-species gate weights.")
            return None, None

        self.eval()  # Set model to evaluation mode
        all_species_gates = []
        feature_names = self.feature_names_list # Get feature names from model property

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Calculating Gate Weights", leave=False)
            for batch in pbar:
                # --- Unpack batch ---
                inputs, labels = batch[0], batch[1]
                coords = None
                if len(batch) == 3:
                    coords = batch[2]

                # --- Move to device ---
                if isinstance(inputs, (list, tuple)):
                    inputs = [x.to(device) for x in inputs]
                else:
                    inputs = [inputs.to(device)]
                if coords is not None:
                    coords = coords.to(device).float()
                
                # --- Get model output ---
                # The 5th return value is the debug_info dictionary
                _, _, _, _, debug_info = self.forward(inputs, coordinates=coords)

                # --- Get gates from the debug_info dictionary ---
                independent_gates_list = debug_info.get("independent_head_gates")
                
                if independent_gates_list is not None and isinstance(independent_gates_list, list):
                    # 'independent_gates_list' is a list[num_species] of (Batch_Size, num_modalities)
                    # Stack them to be (num_species, Batch_Size, num_modalities)
                    stacked_gates = torch.stack(independent_gates_list, dim=0)
                    all_species_gates.append(stacked_gates.cpu())
                
        if not all_species_gates:
            print("No gate weights were collected. Check model and dataloader.")
            return None, feature_names

        # --- Calculate Averages ---
        # Concatenate all batches along the batch dimension (dim=1)
        # Shape: (num_species, num_total_samples, num_modalities)
        all_gates = torch.cat(all_species_gates, dim=1) 
        
        # Average across all samples (dim=1)
        # Shape: (num_species, num_modalities)
        mean_gates = all_gates.mean(dim=1) 

        return mean_gates.numpy(), feature_names
    
    
    def get_average_logit_magnitudes(self, dataloader, device):
        """
        Calculates the average logit magnitude PER SPECIES for the 
        shared and independent heads to debug fusion.

        Args:
            dataloader: The DataLoader (e.g., val_loader) to evaluate on.
            device: The device to run the model on.

        Returns:
            (avg_shared_mag, avg_specific_mag): A tuple of numpy arrays, 
                                                each of shape (num_species,).
        """
        if not (self.use_shared_heads and self.use_independent_heads):
            print("Logit magnitude analysis is only for hybrid models.")
            return None, None

        self.eval()
        all_shared_mags = []
        all_specific_mags = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Calculating Per-Species Logit Magnitudes", leave=False)
            for batch in pbar:
                # --- Unpack batch ---
                inputs, labels = batch[0], batch[1]
                coords = None
                if len(batch) == 3:
                    coords = batch[2]

                # --- Move to device ---
                if isinstance(inputs, (list, tuple)):
                    inputs = [x.to(device) for x in inputs]
                else:
                    inputs = [inputs.to(device)]
                if coords is not None:
                    coords = coords.to(device).float()
                
                # --- Get model output ---
                _, _, _, _, debug_info = self.forward(inputs, coordinates=coords)

                if debug_info["shared_logit_magnitude"] is not None:
                    all_shared_mags.append(debug_info["shared_logit_magnitude"])
                if debug_info["specific_logit_magnitude"] is not None:
                    all_specific_mags.append(debug_info["specific_logit_magnitude"])
                
        # --- Calculate Averages ---
        # Stack the list of (num_species,) tensors into a (num_batches, num_species) tensor
        # Then take the mean along the batch dimension (dim=0)
        avg_shared_mag = torch.stack(all_shared_mags).mean(dim=0).cpu().numpy() if all_shared_mags else None
        avg_specific_mag = torch.stack(all_specific_mags).mean(dim=0).cpu().numpy() if all_specific_mags else None

        return avg_shared_mag, avg_specific_mag
    
    def set_training_stage(self, stage):
        self.training_mode = stage
        for m in self.modules():
            if hasattr(m, "training_mode"):
                print(f"Setting training mode of {m.__class__.__name__} to {stage}")
                m.training_mode = stage

def get_dice_bce_loss(
    species_prevalences,
    device,
    dice_weight=0.5,
    bce_weight=0.5,
    smooth=1.0,
    pos_weight_tensor=None,
    eps=1e-8
):
    """
    Dice + BCE loss that correctly supports:
    - prevalence-based positive weights
    - per-sample weights (for AOI weighting)
    """

    # ------------------------------------------------------------------
    # 1. Compute positive-class BCE weights from prevalence (if provided)
    # ------------------------------------------------------------------
    use_pos_weights = False

    if pos_weight_tensor is not None:
        pos_weight_tensor = pos_weight_tensor.to(device)
        use_pos_weights = True

    elif species_prevalences is not None:
        pos_weights = []
        for prev in species_prevalences:
            w = (1.0 - prev) / max(prev, eps)
            pos_weights.append(min(w, 100.0))
        pos_weight_tensor = torch.tensor(pos_weights, device=device, dtype=torch.float32)
        use_pos_weights = True

    else:
        # no weighting for BCE
        pos_weight_tensor = None

    # ------------------------------------------------------------------
    # LOSS FN
    # ------------------------------------------------------------------
    def loss_fn(logits, targets, weights=None):
        """
        weights = per-sample/per-species weight matrix, shape (B, S)
        """

        B, S = targets.shape

        # --------------------------------------------------------------
        # 2. Dice Loss
        # --------------------------------------------------------------
        probs = torch.sigmoid(logits)

        # Dice is computed per species, then averaged
        intersection = (probs * targets).sum(dim=0)
        union = probs.sum(dim=0) + targets.sum(dim=0)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        # --------------------------------------------------------------
        # 3. BCE Loss (per-element)
        # --------------------------------------------------------------
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # shape (B, S)

        # Apply prevalence-based positive weighting
        if use_pos_weights:
            # weight positives only
            # shape broadcast: (1, S) -> (B, S)
            w = 1.0 + (pos_weight_tensor.unsqueeze(0) - 1.0) * targets
            bce = bce * w

        # --------------------------------------------------------------
        # 4. Apply *sample weights*   (AOI=2, outside=1)
        # --------------------------------------------------------------
        if weights is not None:
            # (B,S) elementwise multiplication
            bce = bce * weights

            # Must normalize by the *sum of weights* not number of elements
            norm = weights.sum()
            if norm.item() < 1e-8:
                bce_loss = bce.mean()  # fallback
            else:
                bce_loss = bce.sum() / norm

        else:
            bce_loss = bce.mean()

        # --------------------------------------------------------------
        # 5. Combine Dice + BCE
        # --------------------------------------------------------------
        total = dice_weight * dice_loss + bce_weight * bce_loss
        return total

    return loss_fn



def get_specialist_loss_function_residual(
    species_prevalences,
    device,
    dice_weight=0.5,
    bce_weight=0.5,
    grad_scale_cap=25.0,
    lambda_shared=1.0,
    lambda_specific=1.0,
    training_stage=1,
    eps=1e-8,
    pos_weight_cap=15.0,
    use_focal=False,       # <--- NEW
    focal_gamma=4.0        # <--- NEW
):
    """
    Stage-aware loss: Stage1 (shared+residual), Stage2 (residual only).
    Adds optional Focal Loss for Stage 2 to handle easy negatives.
    """
    # ---- POS WEIGHTS ----
    pos_weight_tensor = None
    if species_prevalences is not None:
        pos_weight_tensor = torch.tensor(
            [min((1-p)/max(p,eps), pos_weight_cap) for p in species_prevalences],
            device=device, dtype=torch.float32
        )

    # ---- GRAD SCALES (Stage 1) ----
    grad_scales = None
    if species_prevalences is not None:
        grad_scales = torch.tensor(
            [min(max((1-p)/max(p,eps),1), grad_scale_cap) for p in species_prevalences],
            device=device, dtype=torch.float32
        )

    # ---- STAGE 1 SHARED LOSS ----
    shared_loss_fn = get_dice_bce_loss(
        species_prevalences, device,
        dice_weight=dice_weight,
        bce_weight=bce_weight,
        pos_weight_tensor=pos_weight_tensor,
        eps=eps
    )

    # ---- STAGE 1 RESIDUAL L1 ----
    def specialist_loss_fn_residual_l1(specific_logits, residual_targets):
        pred = torch.tanh(specific_logits)
        loss = F.l1_loss(pred, residual_targets, reduction='none')
        if grad_scales is not None:
            loss = loss * grad_scales
        return loss.mean()

    # ==========================================
    # ---- STAGE 2: Standard High-Signal BCE ---
    # ==========================================
    def stage2_loss_high_signal(outputs, targets, weights=None):
        shared_logits = outputs[1].detach()
        specific_logits = outputs[2]

        final_logits = shared_logits + specific_logits
        bce = F.binary_cross_entropy_with_logits(final_logits, targets, reduction='none')

        # Apply Class Imbalance Weights
        if pos_weight_tensor is not None:
            w = 1.0 + (pos_weight_tensor.unsqueeze(0) - 1.0) * targets
            bce = bce * w

        # Apply Mask Weights (AoI)
        if weights is not None:
            bce = bce * weights

        dummy = torch.tensor(0.0, device=targets.device)
        return bce.mean(), dummy, bce.detach()

    # ==========================================
    # ---- STAGE 2: Residual Focal Loss ---
    # ==========================================
    def stage2_loss_focal(outputs, targets, weights=None):
        shared_logits = outputs[1].detach()
        specific_logits = outputs[2]
        final_logits = shared_logits + specific_logits
        
        # 1. Compute raw BCE (log probability)
        bce_raw = F.binary_cross_entropy_with_logits(final_logits, targets, reduction='none')
        
        # 2. Compute pt (probability of the true class)
        # Since BCE = -log(pt), then pt = exp(-BCE)
        pt = torch.exp(-bce_raw)
        
        # 3. Compute Focal Modulator: (1 - pt)^gamma
        # If model is sure (pt->1), factor -> 0 (Loss suppressed)
        # If model is wrong (pt->0), factor -> 1 (Loss kept)
        focal_factor = (1.0 - pt) ** focal_gamma
        
        # 4. Apply Class Imbalance Weights (Existing logic)
        if pos_weight_tensor is not None:
            w_class = 1.0 + (pos_weight_tensor.unsqueeze(0) - 1.0) * targets
            bce_raw = bce_raw * w_class
            
        # 5. Apply Mask Weights (AoI)
        if weights is not None:
            bce_raw = bce_raw * weights
            
        # 6. Combine
        final_loss = bce_raw * focal_factor
        
        dummy = torch.tensor(0.0, device=targets.device)
        return final_loss.mean(), dummy, final_loss.detach()

    # ---- SELECT RETURN FUNCTION ----
    if training_stage == 2:
        if use_focal:
            print(f"--- MODE: STAGE 2: Residual Focal Loss (Gamma={focal_gamma}) ---")
            return stage2_loss_focal
        else:
            print("--- MODE: STAGE 2: High-Signal Residual BCE ---")
            return stage2_loss_high_signal

    # ---- RETURN STAGE 1 ----
    def stage1_loss_fn(outputs, targets):
        shared_logits = outputs[1]
        specific_logits = outputs[2]

        loss_shared = shared_loss_fn(shared_logits, targets)

        with torch.no_grad():
            residual_targets = targets - torch.sigmoid(shared_logits)

        loss_specific = specialist_loss_fn_residual_l1(specific_logits, residual_targets)

        total = lambda_shared * loss_shared + lambda_specific * loss_specific
        return total, loss_shared.detach(), loss_specific.detach()

    print("--- MODE: STAGE 1: Combined Shared + Residual ---")
    return stage1_loss_fn



def create_training_mask(reduced_set, num_samples, num_species, device):
    print(f"Building Relevance Mask for {num_samples} samples x {num_species} species...")
    mask = torch.zeros((num_samples, num_species), dtype=torch.float32, device=device)

    for species_idx, item in enumerate(reduced_set):
        if isinstance(item, dict):
            p = item.get('presence_indices', [])
            a = item.get('absence_indices', [])
            valid = np.unique(np.array(list(p) + list(a), dtype=np.int64))
        else:
            valid = np.unique(np.array(list(item), dtype=np.int64))

        valid = valid[valid < num_samples]
        if valid.size > 0:
            mask[torch.from_numpy(valid).long().to(device), species_idx] = 1.0

    return mask



def calculate_pos_weights(prevalences, device, max_weight=15.0):
    """
    Calculates positive class weights to balance rare species.
    Approximates oversampling: rare species get higher gradients.
    """
    if prevalences is None:
        return None
    
    # Convert to tensor if numpy
    if isinstance(prevalences, (list, np.ndarray)):
        prev_tensor = torch.tensor(prevalences, dtype=torch.float32, device=device)
    else:
        prev_tensor = prevalences.to(device)
        
    # Formula: weight = (1 - p) / p
    # Example: p=0.01 => weight=99. p=0.5 => weight=1.
    weights = (1.0 - prev_tensor) / (prev_tensor + 1e-6)
    
    # Clamp to prevent exploding gradients on extremely rare species
    weights = torch.clamp(weights, max=max_weight)
    
    return weights


import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

def simple_train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                       prevalences, train_reduced_set, alpha=0.5, 
                       loss_masking_strategy="none", relevance_mask=None, pos_weights=None):
    
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # --- 1. UNPACKING ---
        inputs = batch[0]
        labels = batch[1]
        indices_batch = batch[2] 
        coords = None
        if len(batch) == 4: coords = batch[3]
        
        if isinstance(inputs, (list, tuple)): inputs = [x.to(device) for x in inputs]
        else: inputs = [inputs.to(device)]
        labels = labels.to(device).float()
        indices_batch = indices_batch.to(device) 
        if coords is not None: coords = coords.to(device).float()
            
        # --- 2. FORWARD PASS ---
        optimizer.zero_grad()
        model_outputs = model(inputs, coords, alpha)
        final_logits = model_outputs[0]
        
        # --- 3. CALCULATE LOSS ---
        if loss_masking_strategy == "aoi_masked" and relevance_mask is not None:
            # === STRATEGY A: MASKED (Virtual Single-Label) ===
            
            # A. Get Mask & Weights
            batch_mask = relevance_mask[indices_batch]
            
            if pos_weights is not None:
                weight_modifier = torch.ones_like(batch_mask)
                weight_modifier = torch.where(labels == 1, pos_weights.unsqueeze(0), weight_modifier)
                final_batch_weights = batch_mask * weight_modifier
            else:
                final_batch_weights = batch_mask

            # B. Compute Loss (With Manual Reduction Control)
            if model.use_shared_heads and model.use_independent_heads:
                # Custom Hybrid Loss (Tuple Input)
                # We rely on the custom criterion supporting 'weights' which we pass the mask into
                # Note: stage2_loss_high_signal returns (mean, 0, raw)
                # We need the raw unreduced loss to normalize it properly ourselves?
                # Actually, your stage2_loss_high_signal calculates .mean() internally.
                # This is risky with masking because the mean includes the zeros.
                # FIX: We passed weights=final_batch_weights, so the 0s are weighted 0.
                # But we still need to correct the denominator.
                
                loss, _, _ = criterion(model_outputs, labels, weights=final_batch_weights)
                
                # Correction for sparse mask (Approximate)
                # Since criterion.mean() divides by (Batch*Species), but we only have (Valid_Pixels)
                avg_valid_weight = final_batch_weights.mean()
                loss = loss / (avg_valid_weight + 1e-6)

            else:
                # Standard BCE (Tensor Input)
                bce_loss = nn.BCEWithLogitsLoss(reduction='none')(final_logits, labels)
                masked_loss = bce_loss * final_batch_weights
                loss = masked_loss.sum() / (labels.size(0) * labels.size(1))

        else:
            # === STRATEGY B: STANDARD MULTI-LABEL (Legacy) ===
            weights = None
            if train_reduced_set is not None and loss_masking_strategy == "none":
                # Legacy manual weighting logic (simplified)
                pass 
            
            # [FIX IS HERE] Correctly switch between Tuple and Tensor input
            if model.use_shared_heads and model.use_independent_heads:
                loss, _, _ = criterion(model_outputs, labels, weights=weights)
            else:
                loss = criterion(final_logits, labels, weights=weights)

        if torch.isnan(loss):
            continue 
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        
        preds = (torch.sigmoid(final_logits) > 0.5).float()
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        pbar.set_postfix({'loss': f'{running_loss / (batch_idx + 1):.4f}'})

    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = (all_preds == all_labels).mean()
    else:
        accuracy, f1 = 0.0, 0.0

    return {'loss': running_loss / len(train_loader), 'accuracy': accuracy, 'f1': f1}


import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os # Import os to join paths
import copy

def calculate_quartile_f1(preds, labels, prevalences):
    """Calculates Macro-F1 for 4 prevalence buckets (Common -> Rare)."""
    if prevalences is None: return {}
    
    # 1. Per-species F1
    f1_per_species = f1_score(labels, preds, average=None, zero_division=0)
    
    # 2. Sort indices by prevalence (High to Low)
    if isinstance(prevalences, torch.Tensor): prevalences = prevalences.detach().cpu().numpy()
    else: prevalences = np.array(prevalences)
    
    sorted_indices = np.argsort(prevalences)[::-1]
    quartiles = np.array_split(sorted_indices, 4)
    
    # 3. Mean F1 per bucket
    results = {}
    for i, indices in enumerate(quartiles):
        key = f'Q{i+1}'
        results[key] = np.mean(f1_per_species[indices]) if len(indices) > 0 else 0.0
    return results

def find_optimal_threshold_per_species_fast(targets_tensor, probs_tensor, device, num_thresholds=100):
    """
    GPU-Accelerated Threshold Search.
    Calculates F1 for 100 thresholds x 60 species x 25k samples in parallel.
    """
    # 1. Create Thresholds Tensor [T, 1, 1]
    # Shape: [100, 1, 1]
    thresholds = torch.linspace(0.01, 0.99, num_thresholds, device=device).view(-1, 1, 1)
    
    # 2. Expand Inputs [1, N, S]
    # Shape: [1, 25000, 60]
    probs_exp = probs_tensor.unsqueeze(0)
    targets_exp = targets_tensor.unsqueeze(0)
    
    # 3. Generate Predictions for ALL thresholds at once
    # Broadcasting creates a massive tensor of booleans: [T, N, S]
    # This represents the predictions for every single threshold choice.
    # Memory usage: ~150MB for 25k samples/60 species (Safe for GPU)
    pred_masks = probs_exp > thresholds
    
    # 4. Calculate TP, FP, FN via Summation over Samples (dim=1)
    # Result Shape: [T, S] (Thresholds x Species)
    tps = (pred_masks & (targets_exp == 1)).sum(dim=1)
    fps = (pred_masks & (targets_exp == 0)).sum(dim=1)
    fns = ((~pred_masks) & (targets_exp == 1)).sum(dim=1)
    
    # 5. Compute F1 for all T x S combinations
    # Shape: [T, S]
    precision = tps / (tps + fps + 1e-6)
    recall = tps / (tps + fns + 1e-6)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # 6. Find Best F1 per Species
    # Max over thresholds (dim=0)
    best_f1s, best_indices = f1_scores.max(dim=0)
    
    # Retrieve the specific threshold values that gave the max
    # We squeeze thresholds back to [T] to index into them
    flat_thresholds = thresholds.squeeze()
    best_thresholds = flat_thresholds[best_indices]
    
    # Return as numpy for storage
    return best_thresholds.cpu().numpy(), best_f1s.cpu().numpy()


def simple_validate_epoch(model, val_loader, criterion, device, ordered_aoi_sets, prevalences, alpha=0.5, specific_thresholds=None):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_indices = [], [], []
    all_shared_gates = []
    all_independent_gates = [[] for _ in range(model.num_species)]

    pbar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in pbar:
            # --- UNPACKING ---
            inputs, labels = batch[0], batch[1]
            coords = None
            if len(batch) == 3: indices_batch = batch[2]
            elif len(batch) == 4: indices_batch = batch[2]; coords = batch[3]
            else: raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            if isinstance(inputs, (list, tuple)): inputs = [x.to(device) for x in inputs]
            else: inputs = [inputs.to(device)]
            labels = labels.to(device).float()
            if coords is not None: coords = coords.to(device).float()

            # --- FORWARD ---
            model_outputs = model(inputs, coords, alpha)
            final_logits = model_outputs[0]

            # --- LOSS ---
            if model.use_shared_heads and model.use_independent_heads:
                loss, _, _ = criterion(model_outputs, labels)
            else:
                loss = criterion(final_logits, labels)
            running_loss += loss.item()

            # --- STORAGE (CPU to avoid OOM) ---
            all_preds.append(final_logits.cpu())
            all_labels.append(labels.cpu())
            all_indices.append(indices_batch.cpu()) 
            
            # Debug Gates
            debug_info = model_outputs[4]
            if debug_info.get("shared_head_gates") is not None:
                all_shared_gates.append(debug_info["shared_head_gates"].cpu())
            if debug_info.get("independent_head_gates") is not None:
                for i, g in enumerate(debug_info["independent_head_gates"]):
                    all_independent_gates[i].append(g.cpu())

    avg_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_metrics = {'loss': avg_loss}
    collected_debug_data = None 
    
    if all_preds:
        # 1. Prepare Tensors (CPU first)
        all_preds_logits_cpu = torch.cat(all_preds)
        all_labels_cpu = torch.cat(all_labels)
        all_indices_cpu = torch.cat(all_indices)
        
        # Calculate Probabilities
        all_preds_probs_cpu = torch.sigmoid(all_preds_logits_cpu)

        # 2. [CRITICAL] Threshold Search
        if specific_thresholds is not None:
            # Use provided thresholds (for Final Report)
            best_thrs = specific_thresholds
            # Re-calculate Max F1 simply for the record (requires search)
            # We assume if specific_thresholds is passed, we care about THOSE metrics, 
            # but let's run search just to populate 'per_species_f1' correctly for tracking.
            try:
                _, max_f1s = find_optimal_threshold_per_species_fast(
                    all_labels_cpu.to(device), all_preds_probs_cpu.to(device), device
                )
            except: max_f1s = np.zeros(all_labels_cpu.shape[1])
        else:
            # Find best thresholds (for Training Epoch)
            try:
                # Move to GPU for calculation -> Move back result
                best_thrs, max_f1s = find_optimal_threshold_per_species_fast(
                    all_labels_cpu.to(device), 
                    all_preds_probs_cpu.to(device), 
                    device
                )
            except Exception as e:
                print(f"Warning: Fast threshold search failed ({e}). Defaulting to 0.5")
                best_thrs = np.full(all_labels_cpu.shape[1], 0.5)
                max_f1s = np.zeros(all_labels_cpu.shape[1])

        # Store Optimization Metrics
        val_metrics['per_species_max_f1'] = max_f1s
        val_metrics['per_species_best_thrs'] = best_thrs
        val_metrics['global_max_f1'] = np.mean(max_f1s)
        val_metrics['per_species_f1'] = max_f1s # Used by Frankenstein Logic

        # 3. Apply Thresholds (Numpy)
        all_probs_np = all_preds_probs_cpu.numpy()
        all_labels_np = all_labels_cpu.numpy()
        all_indices_np = all_indices_cpu.numpy()
        
        # Broadcast comparison: [N, Species] > [1, Species]
        preds_hard = (all_probs_np > best_thrs[None, :]).astype(int)

        # 4. Global Metrics
        val_metrics['accuracy'] = (preds_hard == all_labels_np).mean()
        val_metrics['f1'] = f1_score(all_labels_np, preds_hard, average='macro', zero_division=0)
        val_metrics['quartile_f1'] = calculate_quartile_f1(preds_hard, all_labels_np, prevalences)

        # 5. Per-Species Metrics Loop
        val_index_map = {original_idx: i for i, original_idx in enumerate(all_indices_np)}
        val_original_indices_set = set(val_index_map.keys())

        species_metrics = []
        reduced_species_metrics = []
        
        for i in range(all_probs_np.shape[1]):
            full_labels_sp = all_labels_np[:, i]
            full_preds_sp = preds_hard[:, i] # Using OPTIMAL thresholds
            
            # Full Metrics
            species_metrics.append({
                'precision': precision_score(full_labels_sp, full_preds_sp, zero_division=0),
                'recall': recall_score(full_labels_sp, full_preds_sp, zero_division=0),
                'f1': f1_score(full_labels_sp, full_preds_sp, zero_division=0),
                'accuracy': accuracy_score(full_labels_sp, full_preds_sp),
                'specificity': ((full_labels_sp == 0) & (full_preds_sp == 0)).sum() / max((full_labels_sp == 0).sum(), 1),
                'support': len(full_labels_sp),
                'threshold': best_thrs[i]
            })
            
            # Reduced (AoI) Metrics
            aoi_set = ordered_aoi_sets[i]
            relevant = val_original_indices_set.intersection(aoi_set)
            
            if not relevant:
                reduced_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'specificity': 0.0, 'support': 0}
            else:
                idx_use = [val_index_map[oid] for oid in relevant]
                red_lab = full_labels_sp[idx_use]
                red_pred = full_preds_sp[idx_use]
                
                reduced_metrics = {
                    'precision': precision_score(red_lab, red_pred, zero_division=0),
                    'recall': recall_score(red_lab, red_pred, zero_division=0),
                    'f1': f1_score(red_lab, red_pred, zero_division=0),
                    'accuracy': accuracy_score(red_lab, red_pred),
                    'specificity': ((red_lab == 0) & (red_pred == 0)).sum() / max((red_lab == 0).sum(), 1),
                    'support': len(red_lab)
                }
            reduced_species_metrics.append(reduced_metrics)

        # Aggregates
        reduced_f1s = [m['f1'] for m in reduced_species_metrics if m['support'] > 0]
        val_metrics['reduced_f1'] = np.mean(reduced_f1s) if reduced_f1s else 0.0
        val_metrics['reduced_recall'] = np.mean([m['recall'] for m in reduced_species_metrics if m['support'] > 0]) if reduced_species_metrics else 0.0
        val_metrics['reduced_precision'] = np.mean([m['precision'] for m in reduced_species_metrics if m['support'] > 0]) if reduced_species_metrics else 0.0

        val_metrics['species_metrics'] = species_metrics
        val_metrics['reduced_species_metrics'] = reduced_species_metrics
        
        # Debug Data
        try:
            avg_shared = torch.cat(all_shared_gates, dim=0).mean(dim=0) if all_shared_gates else None
            final_avg_independent = []
            num_sources = len(model.feature_names_list)
            for g_list in all_independent_gates:
                if g_list: final_avg_independent.append(torch.cat(g_list, dim=0).mean(dim=0))
                else: final_avg_independent.append(torch.full((num_sources,), float('nan')))
            avg_indep_matrix = torch.stack(final_avg_independent) if final_avg_independent else None
            collected_debug_data = {"avg_shared_gates": avg_shared, "avg_independent_gates": avg_indep_matrix}
        except:
            collected_debug_data = None

    else:
        val_metrics.update({'accuracy': 0.0, 'f1': 0.0, 'reduced_f1': 0.0})
        val_metrics['quartile_f1'] = {'Q1':0.0, 'Q2':0.0, 'Q3':0.0, 'Q4':0.0}

    return val_metrics, collected_debug_data

def simple_training_loop(model, train_dataset, val_dataset, species_prevalences, val_reduced_set, 
                         train_reduced_set=None, epochs=20, optimizer=None, species_names=None, 
                         loss_fn="default", training_stage=1, 
                         loss_masking_strategy="none", residual_loss="high_signal"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # ====================================================
    # 1. SETUP MASKING & WEIGHTS (Virtual Single-Label)
    # ====================================================
    relevance_mask = None
    pos_weights_tensor = None
    
    if loss_masking_strategy == "aoi_masked":
        print(f">>> CONFIG: Enabling Virtual Single-Label Training (AoI Masking)")
        
        # Safety Check
        if train_reduced_set is None:
            raise ValueError("CRITICAL ERROR: 'aoi_masked' requires 'train_reduced_set'.")
            
        # A. Size Mask correctly (Max Index + 1) to handle global indexing
        if hasattr(train_dataset, 'indices') and len(train_dataset.indices) > 0:
            # Handle numpy array or list
            max_index = int(np.max(np.array(train_dataset.indices)))
            num_samples = max_index + 1
        elif hasattr(train_dataset, '__len__'):
            # Fallback (unsafe if indices are global IDs > len)
            num_samples = len(train_dataset)
        else:
            raise ValueError("Cannot infer dataset size for mask. Ensure dataset has .indices attribute.")
            
        # B. Build Mask
        relevance_mask = create_training_mask(train_reduced_set, num_samples, model.num_species, device)
        
        # Debug Stats
        print(f"    Mask Shape: {relevance_mask.shape}")
        print(f"    Sparsity: {100.0 * (relevance_mask.sum() / relevance_mask.numel()).item():.2f}% active")

        # C. Build Pos Weights
        if species_prevalences is not None:
            print(">>> CONFIG: Enabling Positive Class Weighting (Virtual Oversampling)")
            pos_weights_tensor = calculate_pos_weights(species_prevalences, device)
            
    elif loss_masking_strategy == "none":
        print(f">>> CONFIG: Standard Multi-Label Training")

    # ====================================================
    # 2. DATALOADERS & OPTIMIZER
    # ====================================================
    print(f"Setting up DataLoader... Training data has {len(train_dataset)} samples.")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    val_loader = None
    if(val_dataset is not None):
        print(f"Validation data has {len(val_dataset)} samples. Validation and Early Stopping are ENABLED.")
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    else:
        print(f"No validation dataset provided. Training for full {epochs} epochs. Validation and Early Stopping are DISABLED.")
    
    params = model.parameters()
    if(optimizer is None):
        optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # ====================================================
    # 3. CRITERION SETUP (Focal vs High Signal)
    # ====================================================
    if(loss_fn == "default"): criterion = nn.BCEWithLogitsLoss() 
    elif(loss_fn == "dice_bce"): criterion = get_dice_bce_loss(species_prevalences, device)
    elif(loss_fn == "species_specialist"): criterion = get_specialist_loss_function(species_prevalences, device)
    elif(loss_fn == "species_specialist_residual"): 
        # [NEW] Switch for Focal Loss
        use_focal = (residual_loss == "focal")
        criterion = get_specialist_loss_function_residual(
            species_prevalences, device, 
            training_stage=training_stage, 
            use_focal=use_focal
        )
    elif(loss_fn == "species_specialist_boosting"): criterion = get_specialist_loss_function_boosting(species_prevalences, device)
    else: criterion = nn.BCEWithLogitsLoss()

    # ====================================================
    # 4. FRANKENSTEIN INITIALIZATION
    # ====================================================
    use_frankenstein = getattr(model, 'use_independent_heads', False)
    baseline_species_f1 = None 
    best_heads_weights = [None] * model.num_species
    min_delta = 0.0001 
    patience = 10 

    if use_frankenstein:
        print(f">>> ENABLED: Per-Species Best Epoch Selection (Max-F1 Strategy)")
        print("Establishing Baseline Performance...")
        with torch.no_grad():
            # Run validation to get BASELINE MAX-F1
            baseline_metrics, _ = simple_validate_epoch(
                model, val_loader, criterion, device, 
                ordered_aoi_sets=val_reduced_set, 
                prevalences=species_prevalences, alpha=0.5
            )
        
        # Note: simple_validate_epoch now puts Max-F1 into 'per_species_f1'
        baseline_species_f1 = np.array(baseline_metrics.get('per_species_f1'))
        best_species_f1 = baseline_species_f1.copy()
        
        print("-" * 80)
        print(f"Base model stats (Global Max-F1): {baseline_metrics.get('global_max_f1', 0):.4f}")
        print("-" * 80)
        model.set_training_stage("stage2")
    else:
        print(">>> DISABLED: Per-Species Selection (Shared Head Mode).")
        best_species_f1 = np.full(model.num_species, -1.0)

    # ====================================================
    # 5. TRAINING LOOP
    # ====================================================
    gradient_log = [] 
    best_global_score = -1.0 
    best_model_state = None
    best_val_metrics = None
    patience_counter = 0
    all_train_metrics = []
    all_val_metrics = []
    
    print("Starting training...")
    for epoch in range(epochs):
        # --- TRAIN ---
        train_metrics = simple_train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            prevalences=species_prevalences, train_reduced_set=train_reduced_set, alpha=0.5,
            loss_masking_strategy=loss_masking_strategy, 
            relevance_mask=relevance_mask,               
            pos_weights=pos_weights_tensor               
        )
        all_train_metrics.append(train_metrics)
        
        if(val_loader is not None):
            # --- VALIDATE ---
            val_metrics, epoch_debug_data = simple_validate_epoch(
                model, val_loader, criterion, device, 
                ordered_aoi_sets=val_reduced_set, 
                prevalences=species_prevalences,
                alpha=0.5
            )
            all_val_metrics.append(val_metrics)
            
            # Score for Early Stopping (Use Max-F1 if available, else Standard)
            current_global_score = val_metrics.get('global_max_f1', val_metrics['f1'])
            
            # --- FRANKENSTEIN UPDATE LOGIC (MAX-F1) ---
            frank_improved_count = 0
            if use_frankenstein:
                # This retrieves Max-F1 from the updated validation function
                current_species_f1s = val_metrics.get('per_species_f1') 
                
                if current_species_f1s is not None:
                    for i in range(model.num_species):
                        # Check improvement on POTENTIAL F1
                        if current_species_f1s[i] > (best_species_f1[i] + min_delta):
                            best_species_f1[i] = current_species_f1s[i]
                            best_heads_weights[i] = copy.deepcopy(model.species_heads_mlps[i].state_dict())
                            frank_improved_count += 1

            # --- PRINTING ---
            print("-" * 80)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"TRAIN | Loss: {train_metrics['loss']:.4f}")
            
            quart = val_metrics.get("quartile_f1", {})
            # Print MAX F1s to confirm logic
            print(f"VAL   | Loss: {val_metrics['loss']:.4f} | Global Max-F1: {current_global_score:.4f}")
            print(f"      | Common(Q1): {quart.get('Q1',0):.3f} | Q2: {quart.get('Q2',0):.3f} | Q3: {quart.get('Q3',0):.3f} | Rare(Q4): {quart.get('Q4',0):.3f}")
            
            if use_frankenstein:
                print(f"      | Frankenstein: {frank_improved_count} species updated best Max-F1.")

            # --- EARLY STOPPING ---
            stop_triggered = False
            reset_patience = False

            # 1. Global Best Check (Save Backbone)
            if current_global_score > (best_global_score + 0.0001):
                best_global_score = current_global_score
                best_model_state = model.state_dict().copy()
                best_val_metrics = val_metrics.copy()
                best_debug_data = epoch_debug_data
                print(f"  -> New Best Global Score: {best_global_score:.4f}")
                if not use_frankenstein: reset_patience = True 

            # 2. Frankenstein Patience Check
            if use_frankenstein:
                if frank_improved_count > 0:
                    print(f"  -> Patience Reset (Individual species improved).")
                    reset_patience = True
            
            if reset_patience:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    stop_triggered = True
            
            if stop_triggered:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            print("-" * 80)
        else:
            print(f"Epoch {epoch+1}: Loss {train_metrics['loss']:.4f}")
        
        scheduler.step()

    print("Training finished.")
    
    # Save Artifacts (Debug)
    if best_debug_data is not None and species_names is not None:
        try:
            best_debug_data["species_names"] = species_names
            torch.save(best_debug_data, "best_model_debug_data.pt")
        except: pass

    # ====================================================
    # 6. RESTORE, ASSEMBLE & FINAL REPORT
    # ====================================================
    results = {
        'final_val_metrics': None,
        'all_train_metrics': all_train_metrics,
        'all_val_metrics': all_val_metrics,
        'config': [{'name': 'simple_model'}],
        'optimal_thresholds': None
    }

# --- RESTORE AND ASSEMBLE ---
    if val_loader is not None and best_model_state is not None:
        
        print("\n" + "="*40)
        print("STARTING FRANKENSTEIN ASSEMBLY")
        print("="*40)

        # 1. Load the Global Best Backbone (This contains SOME broken heads)
        model.load_state_dict(best_model_state)
        
        if use_frankenstein:
            improved_count = 0
            reverted_count = 0
            
            # Track stats for verification
            max_gain = -1.0
            max_gainer_idx = -1
            total_gain_sum = 0.0
            
            for i in range(model.num_species):
                # CASE A: We found a better head during training
                if best_heads_weights[i] is not None:
                    model.species_heads_mlps[i].load_state_dict(best_heads_weights[i])
                    improved_count += 1
                    
                    # Calculate the gain for this specific species
                    # (best_species_f1 was updated during the loop, baseline_species_f1 was set at start)
                    gain = best_species_f1[i] - baseline_species_f1[i]
                    total_gain_sum += gain
                    
                    if gain > max_gain:
                        max_gain = gain
                        max_gainer_idx = i
                    
                # CASE B: The head never beat the baseline (The Fix!)
                else:
                    # We must RESET this head to Zero to prevent "Poisoned" weights
                    print(f"   -> Reverting Species {i} to Shared-Only (Zeroing Residual).")
                    for layer in model.species_heads_mlps[i]:
                        if isinstance(layer, nn.Linear):
                            nn.init.zeros_(layer.weight)
                            nn.init.zeros_(layer.bias)
                    reverted_count += 1

            print(f">>> Assembly Complete:")
            print(f"    - Improved Heads Loaded: {improved_count}")
            print(f"    - Reverted to Baseline:  {reverted_count}")
            
            if improved_count > 0:
                species_name_str = species_names[max_gainer_idx] if species_names else f"Index {max_gainer_idx}"
                avg_gain = total_gain_sum / improved_count
                print(f"    - Max Individual Gain:   +{max_gain:.4f} (Species: {species_name_str})")
                print(f"    - Avg Gain (Improvers):  +{avg_gain:.4f}")


        # 3. [NEW] CALCULATE OPTIMAL THRESHOLDS
        # We perform a dry run to find the thresholds that maximize F1 for THIS assembled model
        print(">>> Calculating Optimal Thresholds for Assembled Model...")
        temp_metrics, _ = simple_validate_epoch(
            model, val_loader, criterion, device, 
            ordered_aoi_sets=val_reduced_set, 
            prevalences=species_prevalences, 
            alpha=0.5
        )
        # Extract the thresholds found by the fast search
        optimal_thresholds = temp_metrics['per_species_best_thrs']
        results['optimal_thresholds'] = optimal_thresholds
        
        # 4. [NEW] GENERATE FINAL REPORT USING OPTIMAL THRESHOLDS
        # We pass 'specific_thresholds' to force the metrics to use them
        print(">>> Generating Final Report with Optimal Thresholds...")
        final_val_metrics, _ = simple_validate_epoch(
            model, val_loader, criterion, device, 
            ordered_aoi_sets=val_reduced_set, 
            prevalences=species_prevalences,
            alpha=0.5,
            specific_thresholds=optimal_thresholds # <--- Critical Step
        )

        # 5. Save Results
        final_metrics_report = final_val_metrics.copy()
        if species_names is not None:
            if 'species_metrics' in final_metrics_report:
                final_metrics_report['per_species_metrics'] = {
                    species_names[i]: metrics
                    for i, metrics in enumerate(final_metrics_report['species_metrics'])
                }
            if 'reduced_species_metrics' in final_metrics_report:
                final_metrics_report['per_species_reduced_metrics'] = {
                    species_names[i]: metrics
                    for i, metrics in enumerate(final_metrics_report['reduced_species_metrics'])
                }

        results['final_val_metrics'] = [final_metrics_report]
            
    else:
        results['final_val_metrics'] = "N/A (Trained on full dataset)"
    
    return model, results
import torch
import torch.nn as nn
def build_optimizer_for_specialists(model, lr=1e-3, weight_decay=1e-4):
    print("--- Optimizer Builder (STRICT FREEZING) ---")
    params_to_train = []
    for name, param in model.named_parameters():
        if "species_heads_mlps" in name:
            param.requires_grad = True
            params_to_train.append(param)
            print(f"[TRAINABLE]: {name}")
        else:
            param.requires_grad = False

    # Put shared modules (everything not in species_heads_mlps) into eval() so BN won't update running stats
    # (Assumes that species head modules are in model.species_heads_mlps)
    if hasattr(model, "shared_head_mlp"):
        model.shared_head_mlp.eval()
    # if you have explicit shared branches, set them to eval too
    for attr in ("shared_sat_branch", "shared_clim_branch", "shared_point_branch", "shared_coords_branch"):
        m = getattr(model, attr, None)
        if m is not None:
            m.eval()

    total_trainable = sum(p.numel() for p in params_to_train)
    print(f"Total trainable params: {total_trainable}")
    return optim.Adam(params_to_train, lr=lr, weight_decay=weight_decay)

def build_optimizer_for_dual_path(model, heads_lr=1e-3, backbone_lr=1e-4, weight_decay=1e-4):
    print("--- Optimizer Builder (DUAL PATH SPECIALIST) ---")
    
    # 1. Identify Parameter Groups
    head_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        
        # GROUP A: The Specialist Heads
        if "species_heads_mlps" in name:
            param.requires_grad = True
            head_params.append(param)
            
        # GROUP B: The Residual Backbone (Look for specific naming)
        elif "layer3_residual" in name or "residual_net" in name:
            param.requires_grad = True
            backbone_params.append(param)
            print(f"[TRAINABLE BACKBONE]: {name}")
            
        # GROUP C: Everything else (Shared path, Stem, etc.) -> FROZEN
        else:
            param.requires_grad = False
            
    # 2. Count Params
    n_heads = sum(p.numel() for p in head_params)
    n_back = sum(p.numel() for p in backbone_params)
    print(f"Trainable Params - Heads: {n_heads:,}")
    print(f"Trainable Params - Backbone: {n_back:,}")
    print(f"Total: {n_heads + n_back:,}")

    # 3. Create Optimizer with Differential Learning Rates
    optimizer = torch.optim.Adam([
        {'params': head_params, 'lr': heads_lr},
        {'params': backbone_params, 'lr': backbone_lr} 
    ], weight_decay=weight_decay)
    
    return optimizer

# Usage example:
def create_simple_model_and_train(train_dataset, val_dataset, input_shapes, species_prevalences, species_names, model_name, val_reduced_set, with_latent=False, use_shared_heads=True, use_independent_heads=True, loss_fn="default", use_climate=True, use_satelitte=True, loss_masking_strategy="none", use_fourier_coords=False):
    """
    input_shapes: list of dicts with 'shape' and 'source'
    species_prevalences: list of positive sample rates for each species
    species_names: list of real species codes, e.g., ['RES_S', 'SAB', ...]
    """
    print("Input shapes:")
    for i, shape_info in enumerate(input_shapes):
        print(f"  Branch {i}: {shape_info['source']} with shape {shape_info['shape']}")

    model = HybridMultiSpeciesModel(input_shapes, len(species_prevalences), use_latent=with_latent, use_shared_heads=use_shared_heads, use_independent_heads=use_independent_heads, use_climate=use_climate, use_satelitte=use_satelitte, use_fourier_coords=use_fourier_coords)
    
    trained_model, results = simple_training_loop(model, train_dataset, val_dataset, species_prevalences, val_reduced_set, species_names=species_names, loss_fn=loss_fn, loss_masking_strategy=loss_masking_strategy)
    
    results['config'][0]['name'] = model_name
    
    return trained_model, results


def load_dual_path_backbone(model, checkpoint_path):
    print(f"Loading Stage 1 Backbone from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()

    # Create a new state dict for the Dual Path model
    new_state_dict = model.state_dict()
    
    # Track what we loaded
    loaded_layers = []

    for key, value in state_dict.items():
        # 1. Load Stem, Layer1, Layer2 (Direct match usually, or adjust prefix)
        # Note: Depending on how you saved, keys might start with "shared_sat_branch." or just be the branch
        
        # We look for keys belonging to the sat branch
        if "shared_sat_branch" in key:
            
            # Get the suffix (e.g., "layer1.0.conv1.weight")
            suffix = key.split("shared_sat_branch.")[1]
            
            # CASE A: The Common Trunk (Stem, L1, L2)
            if "stem" in suffix or "layer1" in suffix or "layer2" in suffix:
                target_key = f"shared_sat_branch.{suffix}"
                if target_key in new_state_dict:
                    new_state_dict[target_key] = value
                    loaded_layers.append(target_key)

            # CASE B: The Split (Layer 3)
            # We map the old 'layer3' to BOTH 'layer3_shared' and 'layer3_residual'
            elif "layer3" in suffix:
                # 1. Map to Shared (Frozen)
                shared_key = f"shared_sat_branch.{suffix.replace('layer3', 'layer3_shared')}"
                if shared_key in new_state_dict:
                    new_state_dict[shared_key] = value
                    loaded_layers.append(shared_key)
                
                # 2. Map to Residual (Unfrozen) - THIS IS THE CRITICAL STEP
                res_key = f"shared_sat_branch.{suffix.replace('layer3', 'layer3_residual')}"
                if res_key in new_state_dict:
                    new_state_dict[res_key] = value
                    loaded_layers.append(res_key)
    
    # Load the constructed dictionary into the model
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Successfully loaded {len(loaded_layers)} layers into Dual-Path Backbone.")
    print("-> layer3_shared initialized from Stage 1.")
    print("-> layer3_residual initialized from Stage 1 (Ready to diverge).")


def create_simple_model_and_train_stage_2(
    train_dataset, val_dataset, input_shapes, species_prevalences, species_names,
    model_name, model_args, pretrained_shared_path=None,
    val_reduced_set=None, train_reduced_set=None, loss_fn="species_specialist_residual",
    device=None, loss_masking_strategy="none", learning_rate=0.0005, weight_decay=0.0, residual_loss="high_signal", 
    use_fourier_coords=False
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridMultiSpeciesModel(
        input_shapes=input_shapes, 
        num_species=len(species_names),
        **model_args,
        use_fourier_coords=use_fourier_coords,
    )
    model.to(device)
    model.training_mode = "stage2_warmup"
    # Load Stage 1
    # 2. Load Stage 1 Weights (Robust "Surgical" Loading)
    if pretrained_shared_path:
        print(f"Loading Shared Backbone from {pretrained_shared_path}...")
        checkpoint = torch.load(pretrained_shared_path, map_location=device)
        
        model_dict = model.state_dict()
        
        # Filter out unnecessary keys:
        # 1. Keep keys that exist in both and have matching shapes
        # 2. IGNORE keys with shape mismatches (e.g., species_heads_mlps.0.0.weight)
        pretrained_dict = {
            k: v for k, v in checkpoint.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        # Report what we skipped
        skipped_keys = [k for k in checkpoint.keys() if k not in pretrained_dict]
        if skipped_keys:
            print(f"   -> Skipped loading {len(skipped_keys)} layers due to shape mismatch (Expected behavior for Specialist Heads).")
            # Optional: print first few to confirm they are the heads
            # print(f"      Examples: {skipped_keys[:3]}")

        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        
        # Load the new state dict
        model.load_state_dict(model_dict)
        
        load_dual_path_backbone(model, pretrained_shared_path)

    # Build Optimizer (Strict Freezing)
    # [!!! FIX 5: HIGH LEARNING RATE !!!]
    #optimizer = build_optimizer_for_specialists(model, lr=learning_rate, weight_decay=weight_decay)
    
    optimizer = build_optimizer_for_dual_path(
        model, 
        heads_lr=learning_rate,   # e.g., 0.001 or 0.01 (Your choice)
        backbone_lr=1e-5,         # Keep this SMALL (safe speed)
        weight_decay=weight_decay
    )

    for mlp in model.species_heads_mlps:
        # Re-init the weights to be small
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.0)
        
        # Explicitly set the FINAL projection layer to strict zero
        # This ensures that at Step 0, Final_Logits == Shared_Logits exactly.
        final_layer = mlp[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    # Train
    trained_model, results = simple_training_loop(
        model, train_dataset, val_dataset, species_prevalences,
        val_reduced_set, train_reduced_set=train_reduced_set,
        species_names=species_names, loss_fn=loss_fn,
        optimizer=optimizer, training_stage=2,
        epochs=50, loss_masking_strategy=loss_masking_strategy, residual_loss=residual_loss
    )
    results['config'][0]['name'] = model_name + "_stage2"
    return trained_model, results





# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # Helper: load only shared-related keys from a saved state_dict
# def load_shared_weights_into_model(model: nn.Module, state_dict_path: str, device='cpu'):
#     sd = torch.load(state_dict_path, map_location=device)
#     # If the file is a whole object with keys like 'model_state', try common alternatives:
#     if isinstance(sd, dict) and any(k.startswith('shared_head_mlp') for k in sd.keys()) is False:
#         # sometimes saved as {'model_state_dict': sd} or {'state_dict': sd}
#         for candidate in ['model_state_dict', 'state_dict', 'state']:
#             if candidate in sd and isinstance(sd[candidate], dict):
#                 sd = sd[candidate]
#                 break

#     model_sd = model.state_dict()
#     # Filter keys to only those that belong to shared modules
#     allowed_prefixes = [
#         'shared_head_mlp', 'shared_interaction_layer',
#         'shared_sat_branch', 'shared_clim_branch',
#         'shared_point_branch', 'shared_coords_branch'
#     ]
#     filtered = {k: v for k, v in sd.items() if any(k.startswith(pref) for pref in allowed_prefixes)}
#     if not filtered:
#         print("Warning: no shared keys found in provided state_dict. Check file contents.")
#     # Update model state
#     model_sd.update(filtered)
#     model.load_state_dict(model_sd)
#     print(f"Loaded {len(filtered)} shared keys into model from '{state_dict_path}'")
#     return model

# # Freeze shared tower and head
# def freeze_shared_components(model: nn.Module, freeze_shared_branches=True, freeze_shared_head=True, freeze_interaction=True):
#     # Freeze branch modules
#     if freeze_shared_branches:
#         for name in ['shared_sat_branch', 'shared_clim_branch', 'shared_point_branch', 'shared_coords_branch']:
#             module = getattr(model, name, None)
#             if module is not None:
#                 for p in module.parameters():
#                     p.requires_grad = False
#                 print(f"Froze parameters in {name}")
#     # Freeze shared head MLP
#     if freeze_shared_head and hasattr(model, 'shared_head_mlp') and model.shared_head_mlp is not None:
#         for p in model.shared_head_mlp.parameters():
#             p.requires_grad = False
#         print("Froze parameters in shared_head_mlp")
#     # Freeze interaction layer if present
#     if freeze_interaction and hasattr(model, 'shared_interaction_layer') and model.shared_interaction_layer is not None:
#         for p in model.shared_interaction_layer.parameters():
#             p.requires_grad = False
#         print("Froze parameters in shared_interaction_layer")
        
        


# # Build optimizer only for parameters that require grad
# def build_optimizer_for_specialists(model: nn.Module, lr=1e-3, weight_decay=1e-4):
#     trainable_params = [p for p in model.parameters() if p.requires_grad]
#     if not trainable_params:
#         raise ValueError("No trainable parameters found. Did you accidentally freeze everything?")
#     return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

# # Stage-2 training function
# def train_specialist_stage2(
#     model: nn.Module,
#     pretrained_shared_path: str,
#     train_dataset,
#     val_dataset,
#     species_prevalences,
#     device=None,
#     epochs=10,
#     batch_size=256,
#     val_reduced_set=None,
#     species_names=None,
#     lr=1e-3,
#     weight_decay=1e-4,
#     freeze_shared_branches=True,
#     freeze_shared_head=True,
#     freeze_interaction=True,
#     use_gating=True,
#     verbose=True
# ):
#     """
#     2-stage: load & freeze shared weights, train specialist heads (residual) using full signal.
#     """
#     device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#     model = model.to(device)

#     # 1) Load shared weights
#     if pretrained_shared_path is not None:
#         try:
#             load_shared_weights_into_model(model, pretrained_shared_path, device=str(device))
#         except Exception as e:
#             print(f"Failed to load shared weights: {e}. Continuing without loading.")

#     # 2) Freeze shared components
#     freeze_shared_components(model, freeze_shared_branches=freeze_shared_branches,
#                              freeze_shared_head=freeze_shared_head, freeze_interaction=freeze_interaction)

#     # 3) Ensure species-specific heads and gaters are trainable
#     # By default species_embedding and independent_gaters and species_heads_mlps remain trainable.
#     # If you want to freeze any of those, set requires_grad=False similarly.

#     # 4) DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
#     val_loader = None
#     if val_dataset is not None:
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     # 5) Loss function (residual L1 + gradient scaling)
#     criterion = get_specialist_loss_function_residual(species_prevalences=species_prevalences, device=device)

#     # 6) Optimizer (only params with requires_grad=True)
#     optimizer = build_optimizer_for_specialists(model, lr=lr, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

#     # Optional: Use your existing helper 'simple_validate_epoch' if available, else use fallback evaluator
#     use_simple_validate = 'simple_validate_epoch' in globals()

#     def evaluate_val():
#         model.eval()
#         # If helper exists, use it (it returns metrics, debug_data)
#         if use_simple_validate:
#             try:
#                 metrics, debug = simple_validate_epoch(model, val_loader, criterion, device, val_reduced_set)
#                 return metrics, debug
#             except Exception as e:
#                 print("simple_validate_epoch failed:", e)

#         # FALLBACK: simple eval computing loss and per-batch preds -> approximate F1
#         from sklearn.metrics import f1_score
#         all_targets = []
#         all_preds = []
#         total_loss = 0.0
#         n_batches = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 # Expect batch format (inputs, targets) — adapt if your dataset returns differently
#                 inputs, targets = batch[0], batch[1]
#                 # If inputs is list/tuple of tensors
#                 if isinstance(inputs, (list, tuple)):
#                     inputs = [t.to(device) for t in inputs]
#                 else:
#                     inputs = inputs.to(device)
#                 targets = targets.to(device).float()
#                 outputs = model(inputs)  # returns (final_logits, shared_logits, specific_logits, latent, debug)
#                 if isinstance(outputs, (list, tuple)):
#                     final_logits, shared_logits, specific_logits = outputs[0], outputs[1], outputs[2]
#                 else:
#                     final_logits = outputs
#                 loss_val, _, _ = criterion((final_logits, shared_logits, specific_logits), targets)
#                 total_loss += loss_val.item()
#                 n_batches += 1
#                 probs = torch.sigmoid(final_logits).cpu().numpy()
#                 preds = (probs >= 0.5).astype(int)
#                 all_preds.append(preds.reshape(preds.shape[0], -1))
#                 all_targets.append(targets.cpu().numpy().astype(int))
#         import numpy as np
#         all_preds = np.vstack(all_preds)
#         all_targets = np.vstack(all_targets)
#         # flatten multi-label across species then compute micro f1
#         flat_f1 = f1_score(all_targets.flatten(), all_preds.flatten(), zero_division=0)
#         metrics = {'loss': total_loss / max(1, n_batches), 'f1': flat_f1}
#         return metrics, None

#     # 7) Quick initial validation to get starting F1 (should be ~0.59)
#     if val_loader is not None:
#         metrics0, _ = evaluate_val()
#         if verbose:
#             print(f"Stage-2 Start Val Loss: {metrics0['loss']:.4f}, Val F1: {metrics0['f1']:.4f}")

#     # 8) Training loop (only specialists update)
#     best_f1 = -1.0
#     best_state = None
#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0.0
#         n_batches = 0
#         for batch in train_loader:
#             # adapt to your dataset batch format: assume (inputs, targets)
#             inputs, targets = batch[0], batch[1]
#             if isinstance(inputs, (list, tuple)):
#                 inputs = [t.to(device) for t in inputs]
#             else:
#                 inputs = inputs.to(device)
#             targets = targets.to(device).float()

#             optimizer.zero_grad()
#             outputs = model(inputs)  # (final_logits, shared_logits, specific_logits, latent, debug)
#             final_logits, shared_logits, specific_logits = outputs[0], outputs[1], outputs[2]

#             loss, loss_shared, loss_specific = criterion((final_logits, shared_logits, specific_logits), targets)
#             loss.backward()
#             # (Optional) gradient clipping for specialist gradients
#             torch.nn.utils.clip_grad_value_( [p for p in model.parameters() if p.requires_grad], 5.0)
#             optimizer.step()

#             epoch_loss += loss.item()
#             n_batches += 1

#         scheduler.step()
#         mean_train_loss = epoch_loss / max(1, n_batches)

#         # Validate
#         if val_loader is not None:
#             val_metrics, debug = evaluate_val()
#             if verbose:
#                 print(f"[Stage2][Epoch {epoch+1}/{epochs}] TrainLoss: {mean_train_loss:.4f} | ValLoss: {val_metrics['loss']:.4f} | ValF1: {val_metrics['f1']:.4f}")
#             # track best
#             if val_metrics['f1'] > best_f1 + 1e-4:
#                 best_f1 = val_metrics['f1']
#                 best_state = model.state_dict().copy()
#         else:
#             if verbose:
#                 print(f"[Stage2][Epoch {epoch+1}/{epochs}] TrainLoss: {mean_train_loss:.4f}")

#     if best_state is not None:
#         model.load_state_dict(best_state)
#         if verbose:
#             print(f"Loaded best model from Stage-2 with Val F1: {best_f1:.4f}")

#     return model, {'best_val_f1': best_f1, 'final_train_loss': mean_train_loss}

