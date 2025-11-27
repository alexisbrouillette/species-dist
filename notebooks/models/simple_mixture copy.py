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

# ------------------------------
# Sentinel-2 branch (spatial)
# ------------------------------
class ResNet50SpatialBranch(nn.Module):
    def __init__(self, pretrained=True, out_channels=256):
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

        # Reduce channels before pooling
        self.reduce_channels = nn.Conv2d(1024, out_channels, kernel_size=1)
        
        # Freeze everything by default
        for param in backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the layers you want to fine-tune
        # We unfreeze layer3 and your new reduce_channels layer
        
        print("Freezing ResNet50 up to layer2. Training layer3.")
        for param in self.layer3.parameters():
            param.requires_grad = True
                
        for param in self.reduce_channels.parameters():
                param.requires_grad = True
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)             # (B, 1024, H, W)
        x = self.reduce_channels(x)    # (B, out_channels, H, W)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # global pooling -> (B, out_channels, 1, 1)
        return x.flatten(1)            # (B, out_channels)


# ------------------------------
# Climate branch (spatial)
# ------------------------------
class FlexibleCNNBranchSpatial(nn.Module):
    def __init__(self, input_channels, out_channels=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.conv_layers(x)        # (B, out_channels, H, W)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # global pooling
        return x.flatten(1)            # (B, out_channels)
    

class SqueezeGater(nn.Module):
    """
    A simple "self-gating" module.
    It takes a feature vector (B, F_in) and learns to output a 
    single gate value (B, 1) based *only* on that feature vector.
    Used for the SHARED head (non-species-conditional).
    """
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        if hidden_features is None:
            hidden_features = max(16, in_features // 4) # Heuristic for hidden dim
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x) # (B, 1)

class ConditionalSqueezeGater(nn.Module):
    """
    A "conditional self-gating" module.
    It takes a feature vector (B, F_in) AND a species embedding (B, F_emb)
    and learns to output a single gate value (B, 1).
    Used for the INDEPENDENT heads (per-species).
    """
    def __init__(self, in_features, emb_features, hidden_features=None):
        super().__init__()
        if hidden_features is None:
            hidden_features = max(16, (in_features + emb_features) // 4)
            
        self.mlp = nn.Sequential(
            nn.Linear(in_features + emb_features, hidden_features),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x, species_emb):
        combined = torch.cat([x, species_emb], dim=1)
        return self.mlp(combined) # (B, 1)
    
    
 # If not using gating, use constant gates of 1.0
class ConstantGater(nn.Module):
    def forward(self, x, species_emb):
        return torch.ones((x.size(0), 1), device=x.device)
# ------------------------------
# Hybrid multi-species model
# (MODIFIED)
# ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# (Assuming ResNet50SpatialBranch, FlexibleCNNBranchSpatial, etc., are defined elsewhere)


class HybridMultiSpeciesModel(nn.Module):
    def __init__(
        self,
        input_shapes,
        num_species,
        shared_head_dim=(512, 256, 128,),
        # Specialist heads remain tiny
        species_head_dims=(64, 32), 
        dropout=0.0,
        gate_hidden=128,
        use_latent=True,
        use_gating=False, # Gating is not used in this design
        latent_dim=526,
        use_shared_heads=True,       
        use_independent_heads=True,   
        use_satelitte=True,
        use_climate = True,
        use_interaction_layer=True,
    ):
        super().__init__()
        self.num_species = num_species
        self.dropout = dropout
        self.use_latent = use_latent
        self.latent_dim = latent_dim
        self.use_shared_heads = use_shared_heads
        self.use_independent_heads = use_independent_heads
        self.use_interaction_layer = use_interaction_layer
        self.species_embedding = None # Gating is removed
        self.independent_gaters = None # Gating is removed

        # --------------------------
        # --- 1. SHARED TOWER (Backbone) ---
        # --------------------------
        self.shared_sat_branch = None
        self.shared_clim_branch = None
        self.shared_point_branch = None
        self.shared_coords_branch = None
        shared_total_features = 0
        self.feature_names_list = []

        for input_shape in input_shapes:
            # ... (Your code for building the shared backbone) ...
            source = input_shape['source']
            shape = input_shape['shape']
            
            if source == 'sentinel2' and use_satelitte:
                self.shared_sat_branch = ResNet50SpatialBranch()
                shared_total_features += 256
                self.feature_names_list.append("satellite")
                
            elif source == 'pred_100' and use_climate:
                self.shared_clim_branch = FlexibleCNNBranchSpatial(shape[0])
                shared_total_features += 128
                self.feature_names_list.append("climate")

            elif source == 'point_infos':
                def create_point_branch():
                    return nn.Sequential(
                        nn.Linear(shape[0], 128), nn.LeakyReLU(0.1), nn.BatchNorm1d(128),
                        nn.Linear(128, 256), nn.LeakyReLU(0.1), nn.BatchNorm1d(256),
                        nn.Linear(256, 128), nn.LeakyReLU(0.1))
                self.shared_point_branch = create_point_branch()
                shared_total_features += 128
                self.feature_names_list.append("point")

            elif source == 'coordinates':
                def create_coords_branch():
                    return nn.Sequential(
                        nn.Linear(shape[0], 64), nn.LeakyReLU(0.1), nn.BatchNorm1d(64),
                        nn.Linear(64, 64), nn.LeakyReLU(0.1))
                self.shared_coords_branch = create_coords_branch()
                shared_total_features += 64
                self.feature_names_list.append("coordinates")
        
        print(f"Shared Tower Feature Dim: {shared_total_features}")

        # --------------------------
        # --- 2. SHARED HEAD (MLP) ---
        # --------------------------
        layers = []
        in_dim = shared_total_features
        if self.use_latent:
            layers.append(nn.Linear(in_dim, self.latent_dim))
            layers.append(nn.BatchNorm1d(self.latent_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(self.dropout))
            in_dim = self.latent_dim
        
        if self.use_interaction_layer:
            layers.append(nn.Linear(in_dim, self.num_species))
            self.shared_interaction_layer = nn.Linear(self.num_species, self.num_species)
            in_dim = self.num_species
            
        for h in shared_head_dim:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(self.dropout))
            in_dim = h
        
        layers.append(nn.Linear(in_dim, self.num_species))
        self.shared_head_mlp = nn.Sequential(*layers)
            
        # ---------------------------------------------
        # --- 3. SPECIALIST HEADS (Tiny Correctors) ---
        # ---------------------------------------------
        species_heads_mlps = []
        for _ in range(self.num_species):
            layers = []
            
            # [!!! KEY CHANGE 1 !!!]
            # Input dim is now the raw shared feature dim (e.g., 320)
            in_dim = shared_total_features 
            
            # Specialist heads are still tiny
            for h in species_head_dims: # e.g., (64, 32)
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.BatchNorm1d(h)) # Use BatchNorm here
                layers.append(nn.LeakyReLU(0.1))
                in_dim = h
            
            layers.append(nn.Linear(in_dim, 1)) # Output is 1-dim correction
            species_heads_mlps.append(nn.Sequential(*layers))
            
        self.species_heads_mlps = nn.ModuleList(species_heads_mlps)
            
        self._initialize_weights()
        
        if self.use_interaction_layer and self.shared_interaction_layer is not None:
            nn.init.zeros_(self.shared_interaction_layer.weight)
            nn.init.zeros_(self.shared_interaction_layer.bias)
            
    def _initialize_weights(self):
        # (This function is unchanged)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, coordinates=None, alpha=0.5, return_latent=False):
            # 1. Get raw inputs & process shared backbone
            # (This part is unchanged)
            sat_input, clim_input, point_input = None, None, None
            input_idx = 0
            if self.shared_sat_branch: sat_input = inputs[input_idx]; input_idx += 1
            if self.shared_clim_branch: clim_input = inputs[input_idx]; input_idx += 1
            if self.shared_point_branch: point_input = inputs[input_idx]; input_idx += 1
            
            B_tensor = next(t for t in (sat_input, clim_input, point_input, coordinates) if t is not None)
            B = B_tensor.size(0)
            S = self.num_species

            f_sat_sh = self.shared_sat_branch(sat_input) if self.shared_sat_branch and sat_input is not None else None
            f_clim_sh = self.shared_clim_branch(clim_input) if self.shared_clim_branch and clim_input is not None else None
            f_point_sh = self.shared_point_branch(point_input) if self.shared_point_branch and point_input is not None else None
            f_coords_sh = self.shared_coords_branch(coordinates) if self.shared_coords_branch and coordinates is not None else None
            
            shared_features_list = []
            if f_sat_sh is not None: shared_features_list.append(f_sat_sh)
            if f_clim_sh is not None: shared_features_list.append(f_clim_sh)
            if f_point_sh is not None: shared_features_list.append(f_point_sh)
            if f_coords_sh is not None: shared_features_list.append(f_coords_sh)
            
            if not shared_features_list:
                raise ValueError("No input features available for the model.")
                
            # This is the rich, "un-baked" feature vector (e.g., B, 320)
            x_shared = torch.cat(shared_features_list, dim=1)
            
            # -----------------------------------
            # --- 2. SHARED HEAD PASS ---
            # -----------------------------------
            
            # We still need the shared logits for our baseline
            shared_logits = self.shared_head_mlp(x_shared)
            
            if self.use_interaction_layer and hasattr(self, 'shared_interaction_layer'):
                interaction_shared_logits = self.shared_interaction_layer(shared_logits)
                shared_logits = shared_logits + interaction_shared_logits
            
            # -----------------------------
            # --- 3. SPECIALIST HEADS PASS ---
            # -----------------------------
            
            if self.use_independent_heads:
                specific_logits_list = []
                
                for sp_idx in range(S):
                    # [!!! KEY CHANGE 2 !!!]
                    # We feed the rich, "un-baked" x_shared vector
                    # to the tiny specialist head.
                    residual_correction = self.species_heads_mlps[sp_idx](x_shared)
                    specific_logits_list.append(residual_correction)

                specific_logits = torch.cat(specific_logits_list, dim=1)
            
            else:
                specific_logits = torch.zeros_like(shared_logits)

            # ---- 5. Combine outputs ----
            final_logits = shared_logits + specific_logits 
            
            # Debugging info
            shared_mag = shared_logits.abs().mean(dim=0).detach() if shared_logits is not None else None
            specific_mag = specific_logits.abs().mean(dim=0).detach() if specific_logits is not None else None

            debug_info = {
                "shared_head_gates": None,
                "independent_head_gates": None,
                "feature_names_for_gates": self.feature_names_list,
                "shared_logit_magnitude": shared_mag,    
                "specific_logit_magnitude": specific_mag 
            }
        
            # We return x_shared as the latent vector for consistency,
            # though it's the input, not a hidden state.
            return final_logits, shared_logits, specific_logits, x_shared, debug_info


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

# --- Main Loss Function to combine them ---
def get_specialist_loss_function(
    species_prevalences, device,
    dice_weight=0.5, bce_weight=0.5,
    pos_weight_cap=50.0, eps=1e-8,
    focal_alpha=0.25, focal_gamma=2.0,
    grad_scale_cap=25.0,
    lambda_aux=2.0, lambda_main=1.0): # <-- New lambda weights
    """
    Returns the final hybrid loss function.
    - Applies Dice+BCE to the shared_logits as an AUXILIARY loss.
    - Applies a gradient-scaled FOCAL Loss to the FINAL_logits as the MAIN loss.
    """
    
    # --- Create pos_weight tensor (used by both losses) ---
    pos_weights = []
    for prev in species_prevalences:
        w = (1.0 - prev) / max(prev, eps)
        pos_weights.append(min(w, pos_weight_cap))
    pos_weight_tensor = torch.tensor(pos_weights, device=device, dtype=torch.float32)
    
    # --- Create grad_scales tensor (used by SPECIALIST loss) ---
    grad_scales = []
    for p in species_prevalences:
        scale = max((1.0 - p) / max(p, eps), 1.0)
        grad_scales.append(min(scale, grad_scale_cap))
    grad_scales = torch.tensor(grad_scales, device=device, dtype=torch.float32)

    # 1. Get the loss function for the SHARED head (Auxiliary)
    shared_loss_fn = get_dice_bce_loss(
        species_prevalences, device,
        dice_weight=dice_weight, bce_weight=bce_weight,
        pos_weight_tensor=pos_weight_tensor,
        eps=eps
    )
    
    # --- 2. MATHEMATICALLY CORRECT FOCAL LOSS (for MAIN loss) ---
    def specialist_focal_loss(logits, targets):
        """
        Calculates the per-species, scaled focal loss in a 
        fully vectorized (parallel) and mathematically correct way.
        """
        
        # 1. Calculate raw probabilities
        probs = torch.sigmoid(logits) # (B, S)
        
        # 2. Calculate pt (prob of *correct* class)
        pt = torch.where(targets == 1, probs, 1.0 - probs) # (B, S)
        
        # 3. Calculate Focal Modulator
        focal_modulator = (1.0 - pt)**focal_gamma # (B, S)

        # 4. Calculate Alpha weight
        alpha_weight = torch.where(targets == 1, focal_alpha, 1.0 - focal_alpha) # (B, S)

        # 5. Calculate Class Imbalance weight
        class_weight = 1.0 + (pos_weight_tensor.unsqueeze(0) - 1.0) * targets # (B, S)

        # 6. Calculate the base, unweighted BCE loss
        base_bce = F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            reduction='none'  # Output shape: (B, S)
        )
        
        # 7. Combine all terms multiplicatively
        focal_loss = focal_modulator * alpha_weight * class_weight * base_bce # (B, S)
        
        # 8. Apply per-species gradient scaling
        loss_per_species = focal_loss.mean(dim=0) # Shape: (S,)
        scaled_loss_per_species = loss_per_species * grad_scales # Shape: (S,)
        
        # 9. Final loss
        final_loss = scaled_loss_per_species.mean() # Single number

        return final_loss
    # --- END OF FOCAL LOSS FUNCTION ---
    
    # 3. The main loss_fn to be returned
    def main_loss_fn(outputs, targets):
        """
        Calculates the combined loss using the "Auxiliary Loss" strategy.
        
        Args:
            outputs (tuple): The (final_logits, shared_logits, specific_logits, ...) 
                             tuple from your model's forward pass.
            targets (Tensor): (B, S) ground truth labels.
        """
        
        final_logits = outputs[0]
        shared_logits = outputs[1]
        
        # --- Auxiliary Shared Loss ---
        # Trains the shared head to be a good "baseline"
        loss_shared = shared_loss_fn(shared_logits, targets) 
        
        # --- Main Specialist Loss ---
        # Trains BOTH heads (via final_logits) to fix the "hard" samples
        loss_main = specialist_focal_loss(final_logits, targets)
        
        # 3. Combine
        total_loss = (lambda_aux * loss_shared) + (lambda_main * loss_main)
        
        return total_loss, loss_shared.detach(), loss_main.detach()

    # --- Print/Debug info ---
    print(f"\n--- Specialist Hybrid Loss (AUXILIARY LOSS) ---")
    print(f"Shared Head (lambda_aux={lambda_aux}): Using Dice+BCE")
    print(f"Main Output (lambda_main={lambda_main}): Using Per-Species FOCAL Loss (gamma={focal_gamma}, alpha={focal_alpha})")
    print(f"Pos weights range for all heads: [{pos_weight_tensor.min():.2f}, {pos_weight_tensor.max():.2f}]")
    print(f"Gradient scales for MAIN loss: (cap={grad_scale_cap}) [{grad_scales.min():.2f}, {grad_scales.max():.2f}]")
    
    return main_loss_fn


import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Main Loss Function to combine them ---
# --- Main Loss Function to combine them ---
def get_specialist_loss_function_residual(
    species_prevalences, device,
    dice_weight=0.5, bce_weight=0.5,
    pos_weight_cap=50.0, eps=1e-8,
    grad_scale_cap=25.0,
    lambda_shared=1.0, 
    lambda_specific=1.0,
    weights=None,
    training_stage=1  # <-- This flag is the key
):
    """
    Returns the final hybrid loss function based on the training stage.
    """
    
    # --- Create pos_weight tensor (used by SHARED loss in Stage 1) ---
    pos_weights = []
    # [!!! BUG FIX !!!] We need to check if species_prevalences is None
    if species_prevalences is not None:
        for prev in species_prevalences:
            w = (1.0 - prev) / max(prev, eps)
            pos_weights.append(min(w, pos_weight_cap))
        pos_weight_tensor = torch.tensor(pos_weights, device=device, dtype=torch.float32)
    else:
        pos_weight_tensor = None # No weights needed if None

    
    # --- Create grad_scales tensor (used by SPECIALIST loss in Stage 1) ---
    grad_scales = []
    if species_prevalences is not None: # Also check here
        for p in species_prevalences:
            scale = max((1.0 - p) / max(p, eps), 1.0)
            grad_scales.append(min(scale, grad_scale_cap))
        grad_scales = torch.tensor(grad_scales, device=device, dtype=torch.float32)
    else:
        grad_scales = None # Not used in Stage 2

    # 1. Get the loss function for the SHARED head (Used in Stage 1)
    shared_loss_fn = get_dice_bce_loss(
        species_prevalences, device, # Pass original list
        dice_weight=dice_weight, bce_weight=bce_weight,
        pos_weight_tensor=pos_weight_tensor,
        eps=eps
    )
    
    # --- 2. OLD L1-RESIDUAL LOSS (Used in Stage 1) ---
    def specialist_loss_fn_residual_l1(specific_logits, residual_targets):
        predictions = torch.tanh(specific_logits)
        loss = F.l1_loss(predictions, residual_targets, reduction='none')
        loss_per_species = loss.mean(dim=0)
        scaled_loss_per_species = loss_per_species * grad_scales
        final_loss = scaled_loss_per_species.mean()
        return final_loss
    
    # --- 3. [!!! CORRECTED !!!] Loss for Stage 2 ---
    # We now create a BALANCED, UNWEIGHTED loss
    stage_2_loss_fn = get_dice_bce_loss(
        species_prevalences=None, # [!!! FIX 1 !!!] This makes it UNWEIGHTED
        device=device,
        dice_weight=0.7,  # [!!! FIX 2 !!!] 70% F1-optimization
        bce_weight=0.3    # [!!! FIX 2 !!!] 30% Specificity-protection
    )
    
    # --- 4a. Main loss function for STAGE 1 (Simultaneous Training) ---
    def main_loss_fn_STAGE_1(outputs, targets):
        # (This is your original, L1-residual logic)
        shared_logits = outputs[1]
        specific_logits = outputs[2]
        loss_shared = shared_loss_fn(shared_logits, targets) 
        with torch.no_grad():
            shared_probs = torch.sigmoid(shared_logits)
            residual_targets = targets - shared_probs
        loss_specific = specialist_loss_fn_residual_l1(specific_logits, residual_targets)
        total_loss = (lambda_shared * loss_shared) + (lambda_specific * loss_specific)
        return total_loss, loss_shared.detach(), loss_specific.detach()

   # In your get_specialist_loss_function_residual function,
# modify main_loss_fn_STAGE_2:

    def main_loss_fn_STAGE_2(outputs, targets, weights=None):
        shared_logits = outputs[1]
        specific_logits = outputs[2]
        
        loss_shared = torch.tensor(0.0, device=targets.device)
        
        # 1. Create the final output
        final_logits = shared_logits.detach() + specific_logits
        
        # [!!! NEW: ADAPTIVE WEIGHT CALCULATION !!!]
        with torch.no_grad():
            # a. Get logit magnitude (distance from zero)
            logit_magnitudes = torch.abs(shared_logits.detach()) # Shape: (B, S)
            
            # b. Calculate Inverse-Confidence Weight
            # We add 1.0 to the denominator to prevent massive division by zero/small numbers
            # If magnitude is high (10.0), weight is low (0.1)
            # If magnitude is low (0.0), weight is high (1.0)
            W_adapt = 1.0 / (logit_magnitudes + 1.0) 
            
            # Apply the AOI mask
            if weights is not None:
                W_adapt = W_adapt * weights
                valid_elements = weights.sum()
                mean_W_adapt = W_adapt.sum() / (valid_elements + 1e-8)
            else:
                mean_W_adapt = W_adapt.mean()
    
        # 2. Calculate the BALANCED loss (unweighted Dice+BCE)
        # The gradient flow is only for 'final_logits'
        loss_specific = stage_2_loss_fn(final_logits, targets, weights=weights)
        
        # --- Apply the adaptive weighting to the SCALAR loss ---
        # This ensures that batches with many uncertain samples contribute more to training
        total_loss = loss_specific * mean_W_adapt
        
        return total_loss, loss_shared.detach(), loss_specific.detach()
    
    # --- Print/Debug info ---
    print(f"\n--- Specialist Hybrid Loss ---")
    if training_stage == 2:
        print("--- MODE: STAGE 2 (Specialist Only, BALANCED Dice+BCE) ---")
    else:
        print("--- MODE: STAGE 1 (Simultaneous, L1-Residual) ---")
    
    # --- Return the correct loss function based on the stage ---
    if training_stage == 2:
        return main_loss_fn_STAGE_2
    else:
        return main_loss_fn_STAGE_1


def find_optimal_thresholds_f1(model, val_loader, device, num_species, num_thresholds=50):
    """Find optimal F1 thresholds for each species"""
    model.eval()
    all_probs = []
    all_targets = []
    
    # Collect predictions and targets
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Finding Optimal Thresholds", leave=False)
        for batch in pbar:
            inputs, labels = batch[0], batch[1]
            coords = None
            
            # [!!! BUG FIX: Unpacking was correct, but device transfer was missing !!!]
            if len(batch) == 3:
                # Assumes (inputs, labels, original_idx)
                indices_batch = batch[2]
            elif len(batch) == 4:
                # Assumes (inputs, labels, original_idx, coords)
                indices_batch = batch[2]
                coords = batch[3]
            
            # [!!! BUG 2 FIX: Move inputs to device !!!]
            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = [inputs.to(device)]
                
            if coords is not None:
                coords = coords.to(device).float()
                
            logits, _, _, _, _ = model(inputs, coords)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu())
            
            # [!!! BUG 1 FIX: Use 'labels', not 'targets' !!!]
            all_targets.append(labels.cpu())
    
    all_probs = torch.cat(all_probs).numpy()   # Convert to numpy for sklearn
    all_targets = torch.cat(all_targets).numpy()
    
    optimal_thresholds = []
    f1_scores = []
    
    # Find optimal threshold for each species
    for species_idx in range(num_species):
        best_f1 = -1.0
        best_threshold = 0.5
        
        probs_species = all_probs[:, species_idx]
        targets_species = all_targets[:, species_idx]
        
        # Test different thresholds
        threshold_range = np.linspace(0.1, 0.9, num_thresholds)
        
        for threshold in threshold_range:
            preds = (probs_species > threshold).astype(int)
            
            # Use sklearn's f1_score for simplicity and stability
            f1 = f1_score(targets_species, preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        
        # [!!! BUG 3 FIX: De-indented to run *after* the loop !!!]
        f1_scores.append(best_f1)
    
    return optimal_thresholds, f1_scores



import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

def simple_train_epoch(model, train_loader, criterion, optimizer, device, epoch, prevalences, train_reduced_set, alpha=0.5):
    """
    Model forward must return:
      final_logits, shared_logits, specific_logits, latent, debug_info
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        if len(batch) < 3:
             raise ValueError(f"Unexpected batch format. Expected at least 3 items (inputs, labels, indices). Got {len(batch)}.")

        # --- 1. ROBUST UNPACKING ---
        # Based on your __getitem__, index 2 is ALWAYS original_idx
        inputs = batch[0]
        labels = batch[1]
        indices_batch = batch[2] # Need this for the mask!
        coords = None
        
        if len(batch) == 4:
            coords = batch[3] # Coords is the 4th item
        
        # Move Data to Device
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = [inputs.to(device)]
            
        labels = labels.to(device).float()
        if coords is not None:
            coords = coords.to(device).float()
            
        weights = None
        if train_reduced_set is not None:

            batch_size = labels.shape[0]
            num_species = labels.shape[1]

            # Default weight = 1 (keep everything)
            weights_np = np.ones((batch_size, num_species), dtype=np.float32)

            current_indices = indices_batch.numpy()

            for b in range(batch_size):
                idx = current_indices[b]
                for s in range(num_species):

                    # Inside AOI ⇒ double weight
                    if idx in train_reduced_set[s]:
                        weights_np[b, s] = 2.0

                    # Optional: always ensure positive observations have weight >= 1
                    if labels[b, s] == 1.0:
                        weights_np[b, s] = max(weights_np[b, s], 1.0)

            weights = torch.from_numpy(weights_np).to(device)


        # --- 3. FORWARD PASS ---
        optimizer.zero_grad()

        model_outputs = model(inputs, coords, alpha)
        final_logits = model_outputs[0]
        
        # --- 4. CALC LOSS WITH MASK ---
        if model.use_shared_heads and model.use_independent_heads:
            # HYBRID MODEL (Stage 2)
            # We pass the mask as a keyword argument
            loss, loss_shared, loss_specific = criterion(model_outputs, labels, weights=weights)

        
        elif model.use_shared_heads or model.use_independent_heads:
            # SHARED/INDEPENDENT ONLY
            loss = criterion(final_logits, labels, weights=weights)
        
        else:
            raise ValueError("Model has no heads enabled.")
        
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected at batch {batch_idx}. Skipping batch.")
            continue 
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        # Metrics
        preds = (torch.sigmoid(final_logits) > 0.5).float()
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

        avg_loss_so_far = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})

    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = (all_preds == all_labels).mean()
    else:
        accuracy, f1 = 0.0, 0.0

    avg_loss = running_loss / len(train_loader)
    return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}


import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# --- MODIFIED ---
# Added `ordered_aoi_sets` to the function signature
def simple_validate_epoch(model, val_loader, criterion, device, ordered_aoi_sets, alpha=0.5):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_indices = [], [], []
    all_shared_gates = []
    all_independent_gates = [[] for _ in range(model.num_species)]

    pbar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in pbar:
            
            # [!!! START OF FIX !!!]
            # Corrected unpacking logic
            inputs, labels = batch[0], batch[1]
            coords = None
            
            if len(batch) == 3:
                # Assumes (inputs, labels, original_idx)
                indices_batch = batch[2]
            elif len(batch) == 4:
                # Assumes (inputs, labels, original_idx, coords)
                indices_batch = batch[2]
                coords = batch[3]
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            # [!!! END OF FIX !!!]

            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = [inputs.to(device)]
            labels = labels.to(device).float()
            
            if coords is not None:
                coords = coords.to(device).float()

            model_outputs = model(inputs, coords, alpha)
            final_logits = model_outputs[0]

            # (The rest of your loss calculation logic is correct)
            if model.use_shared_heads and model.use_independent_heads:
                loss, loss_shared, loss_specific = criterion(model_outputs, labels)
            elif model.use_shared_heads or model.use_independent_heads:
                loss = criterion(final_logits, labels)
            else:
                raise ValueError("Model has no heads enabled.")
            
            final_logits = model_outputs[0]
            debug_info = model_outputs[4]
            running_loss += loss.item()

            preds = (torch.sigmoid(final_logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_indices.append(indices_batch.cpu()) 
            
            # (The rest of your function is correct)
            # ... (debug_info processing) ...
    
    # ... (metric calculation) ...
    
    # ... (return statement) ...
    
    # --- (The rest of your function is unchanged) ---
    
    collected_debug_data = None 
    
    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_indices = torch.cat(all_indices).numpy()

        accuracy = (all_preds == all_labels).mean()
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        val_index_map = {original_idx: i for i, original_idx in enumerate(all_indices)}
        val_original_indices_set = set(val_index_map.keys())

        species_metrics = []
        reduced_species_metrics = []
        
        for i in range(all_preds.shape[1]):
            full_labels_sp = all_labels[:, i]
            full_preds_sp = all_preds[:, i]
            
            species_metrics.append({
                'precision': precision_score(full_labels_sp, full_preds_sp, zero_division=0),
                'recall': recall_score(full_labels_sp, full_preds_sp, zero_division=0),
                'f1': f1_score(full_labels_sp, full_preds_sp, zero_division=0),
                'accuracy': accuracy_score(full_labels_sp, full_preds_sp),
                'specificity': ((full_labels_sp == 0) & (full_preds_sp == 0)).sum()
                               / max((full_labels_sp == 0).sum(), 1)
            })
            
            aoi_set_for_species_i = ordered_aoi_sets[i]
            relevant_original_indices = val_original_indices_set.intersection(aoi_set_for_species_i)
            
            if not relevant_original_indices:
                reduced_metrics = {
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                    'accuracy': 0.0, 'specificity': 0.0, 'support': 0
                }
            else:
                val_indices_to_use = [val_index_map[orig_idx] for orig_idx in relevant_original_indices]
                reduced_labels_sp = full_labels_sp[val_indices_to_use]
                reduced_preds_sp = full_preds_sp[val_indices_to_use]
                
                reduced_metrics = {
                    'precision': precision_score(reduced_labels_sp, reduced_preds_sp, zero_division=0),
                    'recall': recall_score(reduced_labels_sp, reduced_preds_sp, zero_division=0),
                    'f1': f1_score(reduced_labels_sp, reduced_preds_sp, zero_division=0),
                    'accuracy': accuracy_score(reduced_labels_sp, reduced_preds_sp),
                    'specificity': ((reduced_labels_sp == 0) & (reduced_preds_sp == 0)).sum()
                                   / max((reduced_labels_sp == 0).sum(), 1),
                    'support': len(reduced_labels_sp)
                }
            
            reduced_species_metrics.append(reduced_metrics)

        try:
            avg_shared_gates = None
            if all_shared_gates:  
                avg_shared_gates = torch.cat(all_shared_gates, dim=0).mean(dim=0)

            avg_independent_gates_matrix = None
            final_avg_independent_gates = []
            num_sources = len(model.feature_names_list) 

            for g_list in all_independent_gates:
                if g_list:
                    final_avg_independent_gates.append(torch.cat(g_list, dim=0).mean(dim=0))
                else:
                    final_avg_independent_gates.append(torch.full((num_sources,), float('nan')))
            
            if final_avg_independent_gates:
                avg_independent_gates_matrix = torch.stack(final_avg_independent_gates)

            interaction_matrix = None
            if hasattr(model, 'shared_interaction_layer') and model.shared_interaction_layer is not None:
                interaction_matrix = model.shared_interaction_layer.weight.data.detach().cpu()
            
            collected_debug_data = {
                "avg_shared_gates": avg_shared_gates,
                "avg_independent_gates": avg_independent_gates_matrix,
                "interaction_matrix": interaction_matrix
            }
        except Exception as e:
            print(f"Warning: Could not process debug data. Error: {e}")
            collected_debug_data = None
    else:
        accuracy, f1 = 0.0, 0.0
        species_metrics, reduced_species_metrics = [], []

    avg_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    reduced_f1_scores = [m['f1'] for m in reduced_species_metrics if m['support'] > 0]
    avg_reduced_f1 = np.mean(reduced_f1_scores) if reduced_f1_scores else 0.0
    
    reduced_recall_scores = [m['recall'] for m in reduced_species_metrics if m['support'] > 0]
    avg_reduced_recall = np.mean(reduced_recall_scores) if reduced_recall_scores else 0.0

    reduced_precision_scores = [m['precision'] for m in reduced_species_metrics if m['support'] > 0]
    avg_reduced_precision = np.mean(reduced_precision_scores) if reduced_precision_scores else 0.0

    
    val_metrics = {
        'loss': avg_loss, 
        'accuracy': accuracy,
        'f1': f1,
        'reduced_f1': avg_reduced_f1,
        'reduced_recall': avg_reduced_recall,
        'reduced_precision': avg_reduced_precision,
        'species_metrics': species_metrics,
        'reduced_species_metrics': reduced_species_metrics
    }
    
    return val_metrics, collected_debug_data

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os # Import os to join paths

# (Assuming all other functions like simple_train_epoch, simple_validate_epoch, etc., are defined)

def simple_training_loop(model, train_dataset, val_dataset, species_prevalences, val_reduced_set, train_reduced_set=None, epochs=100, optimizer=None, species_names=None, loss_fn="default", training_stage=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
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
    
    # (Criterion setup...)
    if(loss_fn == "default"):
        criterion = nn.BCEWithLogitsLoss() 
        print("Using standard BCEWithLogitsLoss.")
    elif(loss_fn == "dice_bce"):
        criterion = get_dice_bce_loss(
            species_prevalences=species_prevalences,
            device=device,
            dice_weight=0.5,
            bce_weight=0.5
        )
    elif(loss_fn == "species_specialist"):
        criterion = get_specialist_loss_function(
            species_prevalences=species_prevalences,
            device=device,
        )
    elif(loss_fn == "species_specialist_residual"):
        criterion = get_specialist_loss_function_residual(
            species_prevalences=species_prevalences,
            device=device,
            training_stage=training_stage
        )
    elif(loss_fn == "species_specialist_boosting"):
        criterion = get_specialist_loss_function_boosting(
            species_prevalences=species_prevalences,
            device=device,
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"Warning: loss_fn '{loss_fn}' not recognized. Using standard BCEWithLogitsLoss.")
    
    # --- GRADIENT LOGGING SETUP ---
    gradient_log = [] 
    
    # [!!! MODIFIED TRACKING VARIABLES !!!]
    best_score = -1.0 # This will now track the AVERAGE of Full and Reduced F1
    best_model_state = None
    best_val_metrics = None
    best_debug_data = None
    patience = 10
    patience_counter = 0
    
    all_train_metrics = []
    all_val_metrics = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        train_metrics = simple_train_epoch(
            model = model,
            train_loader = train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            epoch=epoch,
            prevalences=species_prevalences,
            train_reduced_set=train_reduced_set, # Note: Ensure argument name matches function definition
            alpha=0.5
        )
        all_train_metrics.append(train_metrics)
        
        if(val_loader is not None):
            val_metrics, epoch_debug_data = simple_validate_epoch(model, val_loader, criterion, device, val_reduced_set)
            all_val_metrics.append(val_metrics)
            
            # [!!! NEW SCORING LOGIC !!!]
            full_f1 = val_metrics['f1']
            reduced_f1 = val_metrics['reduced_f1']
            
            # Calculate the combined score (Average)
            current_score = (full_f1 + reduced_f1) / 2.0
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Full F1: {full_f1:.4f} | Red. F1: {reduced_f1:.4f} | "
                  f"Avg Score: {current_score:.4f}") # Print the new metric
            
            # [!!! CHECK AGAINST NEW SCORE !!!]
            if current_score > best_score + 0.001: 
                print(f"  -> New best Avg Score: {current_score:.4f} (was {best_score:.4f}). Saving model.")
                best_score = current_score # Update the best score
                best_model_state = model.state_dict().copy()
                best_val_metrics = val_metrics.copy()
                best_debug_data = epoch_debug_data
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f}")
        
        scheduler.step()

    print("Training finished.")
    
    # --- Save logic (unchanged) ---
    if best_debug_data is not None and species_names is not None:
        try:
            best_debug_data["species_names"] = species_names
            best_debug_data["feature_names"] = model.feature_names_list
            save_path = "best_model_debug_data.pt"
            torch.save(best_debug_data, save_path)
        except Exception as e:
            print(f"Warning: Failed to save debug data. Error: {e}")

    if gradient_log:
        try:
            grad_save_path = "gradient_log.pt"
            torch.save(gradient_log, grad_save_path)
        except Exception as e:
            print(f"Warning: Failed to save gradient log. Error: {e}")
    
    results = {
        'final_val_metrics': None,
        'all_train_metrics': all_train_metrics,
        'all_val_metrics': all_val_metrics,
        'config': [{'name': 'simple_model'}],
        'optimal_thresholds': None
    }

    if val_loader is not None and best_val_metrics is not None:
        if best_model_state is not None:
            print(f"Loading best model state based on Avg Score: {best_score:.4f}")
            model.load_state_dict(best_model_state)
        
        final_metrics_report = best_val_metrics.copy()

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

        try:
            print("Finding optimal thresholds using validation set...")
            optimal_thresholds, f1_scores = find_optimal_thresholds_f1(model, val_loader, device, model.num_species)
            results['optimal_thresholds'] = optimal_thresholds
        except NameError:
            print("Warning: 'find_optimal_thresholds_f1' not defined.")
            
    else:
        print("Training on full dataset complete.")
        results['final_val_metrics'] = "N/A (Trained on full dataset)"
    
    return model, results


# Usage example:
def create_simple_model_and_train(train_dataset, val_dataset, input_shapes, species_prevalences, species_names, model_name, val_reduced_set, with_latent=False, use_shared_heads=True, use_independent_heads=True, use_gating=False, loss_fn="default", use_climate=True, use_satelitte=True, ):
    """
    input_shapes: list of dicts with 'shape' and 'source'
    species_prevalences: list of positive sample rates for each species
    species_names: list of real species codes, e.g., ['RES_S', 'SAB', ...]
    """
    print("Input shapes:")
    for i, shape_info in enumerate(input_shapes):
        print(f"  Branch {i}: {shape_info['source']} with shape {shape_info['shape']}")

    model = HybridMultiSpeciesModel(input_shapes, len(species_prevalences), use_latent=with_latent, use_shared_heads=use_shared_heads, use_independent_heads=use_independent_heads, use_gating=use_gating, use_climate=use_climate, use_satelitte=use_satelitte)
    
    trained_model, results = simple_training_loop(model, train_dataset, val_dataset, species_prevalences, val_reduced_set, species_names=species_names, loss_fn=loss_fn)
    
    results['config'][0]['name'] = model_name
    
    return trained_model, results

import torch
import torch.nn as nn

def build_optimizer_for_specialists(model, lr=1e-3, weight_decay=1e-4):
    """
    Creates an optimizer that trains the specialist heads
    AND fine-tunes the shared MLP's body.
    """
    params_to_train = []
    
    # [!!! NEW: Keep track of shared_head_mlp layers !!!]
    shared_mlp_param_names = [name for name, _ in model.shared_head_mlp.named_parameters()]
    
    # The final layer's parameter names
    # (e.g., 'shared_head_mlp.10.weight', 'shared_head_mlp.10.bias')
    final_layer_param_names = [
        f"shared_head_mlp.{len(model.shared_head_mlp) - 1}.weight",
        f"shared_head_mlp.{len(model.shared_head_mlp) - 1}.bias"
    ]

    print("--- Optimizer Builder ---")
    
    for name, param in model.named_parameters():
        
        # 1. Train all Specialist Heads
        if "species_heads_mlps" in name or \
           "species_embedding" in name or \
           "independent_gaters" in name:
            
            param.requires_grad = True
            params_to_train.append(param)
            print(f"[TRAINABLE]: {name}")
            
        # 2. Train the SHARED MLP BODY
        elif "shared_head_mlp" in name and name not in final_layer_param_names:
            param.requires_grad = True
            params_to_train.append(param)
            print(f"[TRAINABLE (Fine-Tune)]: {name}")

        # 3. Freeze Everything Else
        else:
            param.requires_grad = False
            if name in final_layer_param_names:
                print(f"[FROZEN (Baseline)]: {name}")
            
    print(f"Found {len(params_to_train)} specialist + fine-tuning parameter groups to train.")
    return optim.Adam(params_to_train, lr=lr, weight_decay=weight_decay)

# --- YOUR MODIFIED FUNCTION ---
def create_simple_model_and_train_stage_2(
    train_dataset,
    val_dataset,
    input_shapes,
    species_prevalences,
    species_names,
    model_name,
    model_args,               # Your HybridMultiSpeciesModel class
    pretrained_shared_path=None,
    val_reduced_set=None,
    train_reduced_set=None,
    loss_fn="species_specialist_residual",
    device=None
):
    """
    2-stage training:
    1. Load SHARED MODEL (backbone + head) weights
    2. Freeze SHARED MODEL
    3. Initialize specialist heads to zero
    4. Train specialist heads ONLY
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate full model (shared + independent heads)
    model = HybridMultiSpeciesModel(input_shapes=input_shapes, num_species=len(species_names), **model_args)
    model.to(device)
    
    # [!!! CRITICAL FIX 1: LOADING LOGIC !!!]
    if pretrained_shared_path is not None:
        # Load the *entire* state dict from the Stage 1 "Shared Only" model
        # This includes the backbones AND the shared_head_mlp
        checkpoint = torch.load(pretrained_shared_path, map_location=device)
        
        # strict=False allows us to load the checkpoint even though
        # our new model has extra 'species_heads_mlps'
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded all Stage-1 parameters from '{pretrained_shared_path}'")
    else:
        raise ValueError("stage1_weights_path must be provided for Stage 2 training.")
    
    # [!!! CRITICAL FIX 2: FREEZING LOGIC !!!]
    # Freeze everything *except* the new specialist heads
    # We pass the model to the optimizer builder, which handles this
    print("Freezing Stage-1 parameters and building optimizer for specialist heads...")
    optimizer = build_optimizer_for_specialists(
        model, 
        lr=1e-4, # [!!! SUGGESTION 1: Use a low LR !!!]
        weight_decay=1e-4
    )
    
    # [!!! ADD THIS SNIPPET HERE !!!]
    print("\n--- Verifying Frozen/Trainable Parameters ---")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"[TRAINABLE]: {name}")
        else:
            # Optionally print frozen layers too, but it can be noisy
            # print(f"[FROZEN]: {name}")
            pass
            
    print("---")
    print(f"Total parameters: {total_params}")
    print(f"Trainable (specialist) parameters: {trainable_params}")
    print("---")
    # --- END OF SNIPPET ---
    
    # Zero-initialize independent heads (Your code is perfect)
    print("Zero-initializing specialist head final layers...")
    for mlp in model.species_heads_mlps:
        try:
            final_layer = mlp[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.constant_(final_layer.weight, 0.0)
                if final_layer.bias is not None:
                    nn.init.constant_(final_layer.bias, 0.0)
        except Exception as e:
            print(f"Warning: Could not zero-init a specialist head. {e}")
    
    # [!!! SUGGESTION 2: VERIFY LOSS FUNCTION !!!]
    # I am assuming your `get_specialist_loss_function_residual`
    # is set up to use the simple, UNWEIGHTED BCE loss for Stage 2,
    # as we discussed. If it's using the L1 or weighted loss,
    # it will fail.
    
    # Train the model
    trained_model, results = simple_training_loop(
        model,
        train_dataset,
        val_dataset,
        species_prevalences,
        val_reduced_set,
        train_reduced_set=train_reduced_set,
        species_names=species_names,
        loss_fn=loss_fn,
        optimizer=optimizer,  # Pass in the specialist-only optimizer
        training_stage=2     # Indicate Stage 2 for loss function logic
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

