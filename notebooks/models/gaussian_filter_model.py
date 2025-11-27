import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm 
from torchgeo.models import ResNet50_Weights, resnet50
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ------------------------------
# Gaussian Attention per species - MEMORY EFFICIENT VERSION
# ------------------------------
class GaussianAttention2d(nn.Module):
    def __init__(self, channels, num_species):
        super().__init__()
        self.channels = channels
        self.num_species = num_species

        
        # Fixed mean (center) per species
        self.register_buffer('mean_x', torch.full((num_species,), 0.5))
        self.register_buffer('mean_y', torch.full((num_species,), 0.5))
        # Learnable sigma (radius) per species
        self.sigma_x = nn.Parameter(torch.ones(num_species) * 0.2)
        self.sigma_y = nn.Parameter(torch.ones(num_species) * 0.2)

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, num_species, C, H, W) - attention for all species at once
        """
        B, C, H, W = x.shape
        device = x.device

        # Normalized coordinate grid
        y_coords = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(self.num_species, H, W)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(self.num_species, H, W)

        # Gaussian mask for all species: (num_species, H, W)
        mean_x = self.mean_x.view(-1, 1, 1)
        mean_y = self.mean_y.view(-1, 1, 1)
        sigma_x = self.sigma_x.view(-1, 1, 1)
        sigma_y = self.sigma_y.view(-1, 1, 1)
        
        gauss = torch.exp(-((x_coords - mean_x) ** 2 / (2 * sigma_x**2) +
                            (y_coords - mean_y) ** 2 / (2 * sigma_y**2)))
        # gauss shape: (num_species, H, W)
        
        # Apply attention: (B, C, H, W) * (num_species, 1, H, W) -> (B, num_species, C, H, W)
        gauss = gauss.unsqueeze(1).unsqueeze(0)  # (1, num_species, 1, H, W)
        x_expanded = x.unsqueeze(1)  # (B, 1, C, H, W)
        
        return x_expanded * gauss  # Broadcast: (B, num_species, C, H, W)


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

        # Reduce channels before fusion
        self.reduce_channels = nn.Conv2d(1024, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # B,1024,H,W
        x = self.reduce_channels(x)  # B,out_channels,H,W
        return x


# ------------------------------
# Climate branch (spatial)
# ------------------------------
class FlexibleCNNBranchSpatial(nn.Module):
    def __init__(self, input_channels, out_channels=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_layers(x)  # B,out_channels,H,W


# ------------------------------
# MEMORY EFFICIENT Multi-species model
# ------------------------------
class SimpleMultiSpeciesModel(nn.Module):
    def __init__(self, input_shapes, num_species, spatial_size=16):
        super().__init__()
        self.num_species = num_species
        self.spatial_size = spatial_size

        # Branches
        self.sat_branch = None
        self.clim_branch = None
        self.point_branch = None

        # Channels per branch
        self.sat_channels = 0
        self.clim_channels = 0
        self.point_channels = 0

        for input_shape in input_shapes:
            source = input_shape['source']
            shape = input_shape['shape']

            if source == 'sentinel2':
                self.sat_branch = ResNet50SpatialBranch()
                self.sat_channels = 256

            elif source == 'pred_100':
                self.clim_branch = FlexibleCNNBranchSpatial(shape[0])
                self.clim_channels = 128

            elif source == 'point_infos':
                self.point_branch = nn.Sequential(
                    nn.Linear(shape[0], 32), 
                    nn.ReLU()
                )
                self.point_channels = 32

        # Single Gaussian attention module per branch (handles all species)
        self.sat_attn = GaussianAttention2d(self.sat_channels, num_species) if self.sat_branch else None
        self.clim_attn = GaussianAttention2d(self.clim_channels, num_species) if self.clim_branch else None

        # Calculate total feature dimension
        total_features = 0
        if self.sat_branch:
            total_features += self.sat_channels * spatial_size * spatial_size
        if self.clim_branch:
            total_features += self.clim_channels * spatial_size * spatial_size
        if self.point_branch:
            total_features += self.point_channels

        print(f"Total features per species head: {total_features}")
        print(f"Using spatial size: {spatial_size}x{spatial_size}")

        # Species-specific heads (share same architecture but different weights)
        self.species_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            ) for _ in range(num_species)
        ])

    def forward(self, inputs):
        # Unpack inputs
        sat_input = None
        clim_input = None
        point_input = None
        
        input_idx = 0
        if self.sat_branch is not None:
            sat_input = inputs[input_idx]
            input_idx += 1
        if self.clim_branch is not None:
            clim_input = inputs[input_idx]
            input_idx += 1
        if self.point_branch is not None:
            point_input = inputs[input_idx]

        B = inputs[0].shape[0]

        # Process branches ONCE (not per species!)
        sat_feats_all = None
        clim_feats_all = None
        point_feat = None

        if self.sat_branch is not None:
            sat_feat = self.sat_branch(sat_input)  # (B, C, H, W)
            # Apply attention for ALL species at once
            sat_feats_all = self.sat_attn(sat_feat)  # (B, num_species, C, H, W)
            # Pool each species separately
            B, S, C, H, W = sat_feats_all.shape
            sat_feats_all = sat_feats_all.view(B * S, C, H, W)  # flatten species into batch
            sat_feats_all = F.adaptive_avg_pool2d(sat_feats_all, (self.spatial_size, self.spatial_size))
            sat_feats_all = sat_feats_all.view(B, S, C, self.spatial_size, self.spatial_size)


        if self.clim_branch is not None:
            clim_feat = self.clim_branch(clim_input)  # (B, C, H, W)
            # Apply attention for ALL species at once

            clim_feats_all = self.clim_attn(clim_feat)  # (B, num_species, C, H, W)
            # Pool each species separately
            B, S, C, H, W = clim_feats_all.shape
            clim_feats_all = clim_feats_all.view(B * S, C, H, W)
            clim_feats_all = F.adaptive_avg_pool2d(clim_feats_all, (self.spatial_size, self.spatial_size))
            clim_feats_all = clim_feats_all.view(B, S, C, self.spatial_size, self.spatial_size)


        if self.point_branch is not None:
            point_feat = self.point_branch(point_input)  # (B, point_channels)

        # Per-species heads
        outputs = []
        for i in range(self.num_species):
            features_list = []

            if sat_feats_all is not None:
                features_list.append(sat_feats_all[:, i].flatten(1))

            if clim_feats_all is not None:
                features_list.append(clim_feats_all[:, i].flatten(1))

            if point_feat is not None:
                features_list.append(point_feat)

            combined_features = torch.cat(features_list, dim=1)
            output = self.species_heads[i](combined_features)
            outputs.append(output)

        return torch.cat(outputs, dim=1)

def get_asymmetric_loss(gamma_pos=0, gamma_neg=4, clip=0.05, species_prevalences=None, device=None):
    """
    Asymmetric Loss for precision-oriented training
    gamma_pos: focuses on hard positives
    gamma_neg: penalizes false positives more
    clip: small shift to avoid over-confident negatives
    """
    pos_weights = []
    for prevalence in species_prevalences:
        weight = (1 - prevalence) / max(prevalence, 1e-8)
        pos_weights.append(min(weight, 50.0))
    pos_weight_tensor = torch.tensor(pos_weights, device=device) if device is not None else None
    
    def loss_function(logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_tensor, reduction='none')
        
        # Asymmetric focusing
        pt = probs * targets + (1 - probs) * (1 - targets)
        gamma = gamma_pos * targets + gamma_neg * (1 - targets)
        focal_weight = (1 - pt) ** gamma
        
        # Optional: clip probabilities for negatives
        if clip > 0:
            probs = torch.clamp(probs, min=clip, max=1-clip)
        
        return (ce_loss * focal_weight).mean()
    
    return loss_function


def get_simple_loss_function(species_prevalences, device):
    """Simple loss that handles class imbalance"""
    pos_weights = []
    for prevalence in species_prevalences:
        weight = (1 - prevalence) / max(prevalence, 1e-8)
        pos_weights.append(min(weight, 50.0))
    
    pos_weight_tensor = torch.tensor(pos_weights, device=device)
    
    def loss_function(logits, targets):
        l_bce = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=pos_weight_tensor,
            reduction='mean'
        )
        return l_bce 
    
    return loss_function


def simple_train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    f1_scores = []

    pbar = tqdm(train_loader, desc="Training", leave=False, mininterval=1.0)

    for batch_idx, batch in enumerate(pbar):
        if len(batch) >= 2:
            inputs, labels = batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = [inputs.to(device)]
            
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()

        # Update accuracy counters
        total_correct += (preds == labels).sum().item()
        total_samples += labels.numel()

        # Batch-level F1 (detached to avoid graph growth)
        f1_scores.append(calculate_simple_f1(
            preds.detach().cpu().numpy(),
            labels.detach().cpu().numpy()
        ))

        avg_loss_so_far = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})

    avg_loss = running_loss / len(train_loader)
    accuracy = total_correct / total_samples
    f1 = np.mean(f1_scores)

    return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}



def validate_with_thresholds(model, val_loader, criterion, device, thresholds=None):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0], batch[1]
            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = [inputs.to(device)]
            labels = labels.to(device).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu()
            all_probs.append(probs)
            all_labels.append(labels.cpu())
    
    if all_probs:
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        if thresholds is None:
            thresholds = [0.5] * all_probs.shape[1]
        
        all_preds = np.zeros_like(all_probs)
        for i, thr in enumerate(thresholds):
            all_preds[:, i] = (all_probs[:, i] > thr).astype(float)

        accuracy = (all_preds == all_labels).mean()
        f1 = calculate_simple_f1(all_preds, all_labels)

        species_metrics = []
        for i in range(all_preds.shape[1]):
            species_metrics.append({
                'precision': precision_score(all_labels[:, i], all_preds[:, i], zero_division=0),
                'recall': recall_score(all_labels[:, i], all_preds[:, i], zero_division=0),
                'f1': f1_score(all_labels[:, i], all_preds[:, i], zero_division=0),
                'accuracy': accuracy_score(all_labels[:, i], all_preds[:, i]),
                'specificity': ( ((all_labels[:, i] == 0) & (all_preds[:, i] == 0)).sum() / 
                                 max( (all_labels[:, i] == 0).sum(), 1) )
            })
    else:
        accuracy, f1, species_metrics = 0.0, 0.0, []
    
    avg_loss = running_loss / len(val_loader)
    return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1, 'species_metrics': species_metrics}


def find_optimal_thresholds_f1(model, val_loader, device, num_species, num_thresholds=50):
    """Find optimal F1 thresholds for each species"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = [inputs.to(device)]
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
    
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    
    optimal_thresholds = []
    f1_scores = []
    
    for species_idx in range(num_species):
        best_f1 = 0
        best_threshold = 0.5
        
        probs_species = all_probs[:, species_idx]
        targets_species = all_targets[:, species_idx]
        
        for threshold in torch.linspace(0.1, 0.9, num_thresholds):
            preds = (probs_species > threshold).float()
            
            tp = (preds * targets_species).sum()
            fp = (preds * (1 - targets_species)).sum()
            fn = ((1 - preds) * targets_species).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold.item()
        
        optimal_thresholds.append(best_threshold)
        f1_scores.append(best_f1)
    
    return optimal_thresholds, f1_scores


def calculate_simple_f1(preds, labels):
    """Calculate macro F1 score"""
    f1_scores = []
    for i in range(preds.shape[1]):
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def simple_training_loop(model, train_dataset, val_dataset, species_prevalences, epochs=50, species_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Reduce batch size to prevent OOM
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4,persistent_workers=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4,persistent_workers=True, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #criterion = get_simple_loss_function(species_prevalences, device)
    criterion = get_asymmetric_loss(gamma_pos=1, gamma_neg=3, clip=0.05, species_prevalences=species_prevalences, device = device)
    
    best_f1 = 0
    best_model_state = None
    best_val_metrics = None
    patience = 5
    patience_counter = 0
    
    print("Starting simplified training...")
    
    for epoch in range(epochs):
        train_metrics = simple_train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_with_thresholds(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_f1 * 1.01:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            best_val_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            print(f"No improvement in F1 for epoch {epoch+1}. Patience counter: {patience_counter+1}/{patience}")
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    optimal_thresholds, f1_scores = find_optimal_thresholds_f1(model, val_loader, device, model.num_species)
    final_metrics = validate_with_thresholds(model, val_loader, criterion, device, optimal_thresholds)
    print("Optimal thresholds per species:", optimal_thresholds)
    print("F1 scores at optimal thresholds:", f1_scores)
    
    if species_names is not None and 'species_metrics' in final_metrics:
        per_species_metrics = {
            species_names[i]: metrics
            for i, metrics in enumerate(final_metrics['species_metrics'])
        }
    else:
        per_species_metrics = final_metrics.get('species_metrics', {})
    
    results = {
        'final_val_metrics': [{
            'per_species': per_species_metrics
        }],
        'config': [{'name': 'simple_model'}]
    }
    del train_loader, val_loader
    del optimizer, criterion
    torch.cuda.empty_cache()
    return model, results


def create_simple_model_and_train(train_dataset, val_dataset, input_shapes, species_prevalences, species_names):
    """
    input_shapes: list of dicts with 'shape' and 'source'
    species_prevalences: list of positive sample rates for each species
    species_names: list of real species codes
    """
    print("Input shapes:")
    for i, shape_info in enumerate(input_shapes):
        print(f"  Branch {i}: {shape_info['source']} with shape {shape_info['shape']}")
    
    model = SimpleMultiSpeciesModel(input_shapes, len(species_prevalences), spatial_size=16)
    trained_model, results = simple_training_loop(model, train_dataset, val_dataset, species_prevalences, species_names=species_names)
    return trained_model, results