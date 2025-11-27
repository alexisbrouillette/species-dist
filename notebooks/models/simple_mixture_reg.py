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
# ------------------------------
# Hybrid multi-species model
# ------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Modified Hybrid Model for Regression (Presence Probabilities)
# ------------------------------
class HybridMultiSpeciesModelReg(nn.Module):
    def __init__(
        self,
        input_shapes,
        num_species,
        shared_head_dim=(128, 64),
        species_head_dims=(64, 32),
        dropout=0.0,
        use_gating=False,
        gate_hidden=128
    ):
        super().__init__()
        self.num_species = num_species
        self.use_gating = use_gating
        self.dropout = dropout

        # Branches
        self.sat_branch = None
        self.clim_branch = None
        self.point_branch = None
        self.sat_channels = 0
        self.clim_channels = 0
        self.point_channels = 0

        for input_shape in input_shapes:
            source = input_shape['source']
            shape = input_shape['shape']

            if source == 'sentinel2':
                self.sat_branch = ResNet50SpatialBranch()  # placeholder, your existing branch
                self.sat_channels = 256

            elif source == 'pred_100':
                self.clim_branch = FlexibleCNNBranchSpatial(shape[0])
                self.clim_channels = 128

            elif source == 'point_infos':
                self.point_branch = nn.Sequential(
                    nn.Linear(shape[0], 32),
                    nn.LeakyReLU(0.1),
                )
                self.point_channels = 32

        total_features = self.sat_channels + self.clim_channels + self.point_channels

        # Shared head
        layers = []
        in_dim = total_features
        for h in shared_head_dim:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(self.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.num_species))  # output logits
        self.shared_head = nn.Sequential(*layers)

        # Species-specific heads
        species_heads = []
        for _ in range(self.num_species):
            layers = []
            in_dim = total_features
            for h in species_head_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(self.dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))  # logits
            species_heads.append(nn.Sequential(*layers))
        self.species_heads = nn.ModuleList(species_heads)

        # Optional gating
        if self.use_gating:
            self.species_embedding = nn.Embedding(self.num_species, gate_hidden)
            self.gate_mlp_feature = nn.Sequential(
                nn.Linear(total_features + gate_hidden, gate_hidden),
                nn.LeakyReLU(0.1),
                nn.Linear(gate_hidden, 1)
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def extract_features(self, inputs):
        sat_input, clim_input, point_input = None, None, None
        idx = 0
        if self.sat_branch is not None:
            sat_input = inputs[idx]; idx += 1
        if self.clim_branch is not None:
            clim_input = inputs[idx]; idx += 1
        if self.point_branch is not None:
            point_input = inputs[idx]

        feats = []
        if self.sat_branch: feats.append(self.sat_branch(sat_input))
        if self.clim_branch: feats.append(self.clim_branch(clim_input))
        if self.point_branch: feats.append(self.point_branch(point_input))

        return torch.cat(feats, dim=1)

    def forward(self, inputs, alpha=0.5):
        combined_features = self.extract_features(inputs)

        # Shared head
        shared_logits = self.shared_head(combined_features)

        # Species-specific heads
        specific_logits_list = [h(combined_features) for h in self.species_heads]
        specific_logits = torch.cat(specific_logits_list, dim=1)

        # Blend
        if self.use_gating:
            B = combined_features.size(0)
            feats_exp = combined_features.unsqueeze(1).expand(-1, self.num_species, -1)
            sp_emb = self.species_embedding(torch.arange(self.num_species, device=combined_features.device))
            sp_emb = sp_emb.unsqueeze(0).expand(B, -1, -1)
            cat = torch.cat([feats_exp, sp_emb], dim=2).reshape(B * self.num_species, -1)
            gate_logits = self.gate_mlp_feature(cat).reshape(B, self.num_species)
            alpha_per = torch.sigmoid(gate_logits)
            final_logits = alpha_per * specific_logits + (1.0 - alpha_per) * shared_logits
        else:
            final_logits = alpha * specific_logits + (1 - alpha) * shared_logits

        # Probabilities
        final_probs = torch.sigmoid(final_logits)
        return final_probs, shared_logits, specific_logits


def get_regression_loss(species_prevalences, device, grad_scale_cap=10.0):
    """
    RMSE-style loss with optional per-species scaling.
    Targets should be 0/1 for absence/presence.
    """
    # per-species scaling for rare species
    scales = [min(max(1.0 / max(p, 1e-6), 1.0), grad_scale_cap) for p in species_prevalences]
    scales = torch.tensor(scales, device=device)

    def loss_fn(outputs, targets):
        # outputs: final_probs (B,S)
        loss = ((outputs - targets) ** 2) * scales.unsqueeze(0)
        return loss.mean().sqrt()  # RMSE
    return loss_fn







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

def final_optimal_thresholds(model, val_loader, device, num_species, num_thresholds=50):
    """
    Compute optimal F1 thresholds per species after training is complete.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0], batch[1]
            inputs = [x.to(device) for x in inputs] if isinstance(inputs, (list, tuple)) else [inputs.to(device)]
            labels = labels.to(device).float()

            final_logits, _, _ = model(inputs)
            all_preds.append(torch.sigmoid(final_logits).cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    optimal_thresholds = []
    f1_scores = []

    for i in range(num_species):
        best_f1 = 0
        best_thresh = 0.5
        for t in np.linspace(0.1, 0.9, num_thresholds):
            preds_i = (all_preds[:, i] > t).astype(float)
            tp = ((preds_i == 1) & (all_labels[:, i] == 1)).sum()
            fp = ((preds_i == 1) & (all_labels[:, i] == 0)).sum()
            fn = ((preds_i == 0) & (all_labels[:, i] == 1)).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        optimal_thresholds.append(best_thresh)
        f1_scores.append(best_f1)

    return optimal_thresholds, f1_scores
import csv
import os
import pandas as pd
import os
import pandas as pd
import torch

def log_gradients(model, epoch, prevalences=None, batch_idx=None, filename="./gradients/rare_spec_gradients_log3.csv", rare_threshold=0.1):
    """
    Logs gradient statistics per layer.
    
    Args:
        model: PyTorch model
        epoch: current epoch
        prevalences: list of floats, one per species (used to classify rare vs common)
        batch_idx: optional batch index
        filename: CSV path to store gradients
        rare_threshold: prevalence below which a species is considered rare
    """
    grad_stats = []

    # Ensure directory exists
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    for name, param in model.named_parameters():
        if param.grad is not None:
            # Determine head type
            if 'species_heads' in name:
                head_type = 'species'
                # Extract species index from name: species_heads.<species_idx>...
                species_idx = int(name.split('.')[1])
                rarity = None
                if prevalences is not None:
                    p = prevalences[species_idx]
                    rarity = 'rare' if p < rare_threshold else 'common'
            elif 'shared_head' in name:
                head_type = 'shared'
                species_idx = ''
                rarity = ''
            else:
                head_type = 'other'
                species_idx = ''
                rarity = ''

            grad_stats.append({
                "epoch": epoch,
                "batch": batch_idx,
                "param": name,
                "head_type": head_type,
                "species_idx": species_idx,
                "rarity": rarity,
                "grad_mean": param.grad.abs().mean().item(),
                "grad_std": param.grad.std().item() if param.grad.std() is not None else 0,
                "grad_max": param.grad.abs().max().item()
            })

    df = pd.DataFrame(grad_stats)
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


def train_epoch(model, train_loader, criterion, optimizer, device, alpha=0.5):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in train_loader:
        inputs, labels = batch[0], batch[1]
        inputs = [x.to(device) for x in inputs] if isinstance(inputs, (list, tuple)) else [inputs.to(device)]
        labels = labels.to(device).float()

        optimizer.zero_grad()
        final_logits, _, _ = model(inputs, alpha)
        loss = criterion(final_logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.append(torch.sigmoid(final_logits).detach().cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    preds_bin = (all_preds > 0.5).astype(float)
    f1 = calculate_simple_f1(preds_bin, all_labels)
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    mae = np.mean(np.abs(all_preds - all_labels))

    avg_loss = running_loss / len(train_loader)
    return {'loss': avg_loss, 'f1': f1, 'rmse': rmse, 'mae': mae}

def validate_epoch(model, val_loader, criterion, device, alpha=0.5):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0], batch[1]
            inputs = [x.to(device) for x in inputs] if isinstance(inputs, (list, tuple)) else [inputs.to(device)]
            labels = labels.to(device).float()

            final_logits, _, _ = model(inputs, alpha)
            loss = criterion(final_logits, labels)
            running_loss += loss.item()
            all_preds.append(torch.sigmoid(final_logits).cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Binary metrics using default threshold 0.5
    preds_bin = (all_preds > 0.5).astype(float)
    f1 = calculate_simple_f1(preds_bin, all_labels)

    species_metrics = []
    for i in range(all_preds.shape[1]):
        species_metrics.append({
            'precision': precision_score(all_labels[:, i], preds_bin[:, i], zero_division=0),
            'recall': recall_score(all_labels[:, i], preds_bin[:, i], zero_division=0),
            'f1': f1_score(all_labels[:, i], preds_bin[:, i], zero_division=0),
            'accuracy': accuracy_score(all_labels[:, i], preds_bin[:, i]),
            'specificity': ((all_labels[:, i] == 0) & (preds_bin[:, i] == 0)).sum()
                           / max((all_labels[:, i] == 0).sum(), 1)
        })

    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    mae = np.mean(np.abs(all_preds - all_labels))
    avg_loss = running_loss / len(val_loader)

    return {'loss': avg_loss, 'f1': f1, 'species_metrics': species_metrics, 'rmse': rmse, 'mae': mae}


# ---------------------------
# Full training loop
# ---------------------------
def simple_training_loop(model, train_dataset, val_dataset, species_prevalences, epochs=300, species_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0)
    criterion = get_regression_loss(species_prevalences, device)

    best_f1 = 0
    best_model_state = None
    best_val_metrics = None
    patience = 100
    patience_counter = 0

    all_train_metrics = []
    all_val_metrics = []

    print("Starting training...")

    for epoch in range(epochs):
        # --- TRAIN ---
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        all_train_metrics.append(train_metrics)

        # --- VALIDATE ---
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        all_val_metrics.append(val_metrics)

        print(f"Epoch {epoch+1}: Loss: {train_metrics['loss']:.4f}, "
              f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
              f"Val RMSE: {val_metrics['rmse']:.4f}, Val MAE: {val_metrics['mae']:.4f}")

        # --- Early stopping ---
        if val_metrics['f1'] > best_f1 * 1.01:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            best_val_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # --- Map numeric species keys to names ---
    if species_names is not None and 'species_metrics' in best_val_metrics:
        per_species_metrics = {
            species_names[i]: metrics
            for i, metrics in enumerate(best_val_metrics['species_metrics'])
        }
    else:
        per_species_metrics = best_val_metrics.get('species_metrics', {})

    # --- Compute optimal thresholds only at the end ---
    optimal_thresholds, f1_scores = final_optimal_thresholds(model, val_loader, device, model.num_species)
    print("Optimal thresholds per species:", optimal_thresholds)
    print("F1 scores at optimal thresholds:", f1_scores)

    results = {
        'final_val_metrics': [{
            'per_species': per_species_metrics,
            'optimal_thresholds': optimal_thresholds,
            'f1_at_optimal_thresholds': f1_scores
        }],
        'all_train_metrics': all_train_metrics,
        'all_val_metrics': all_val_metrics,
        'config': [{'name': 'hybrid_model'}]
    }

    return model, results


# Usage example:
def create_simple_model_and_train(train_dataset, val_dataset, input_shapes, species_prevalences, species_names):
    """
    input_shapes: list of dicts with 'shape' and 'source'
    species_prevalences: list of positive sample rates for each species
    species_names: list of real species codes, e.g., ['RES_S', 'SAB', ...]
    """
    print("Input shapes:")
    for i, shape_info in enumerate(input_shapes):
        print(f"  Branch {i}: {shape_info['source']} with shape {shape_info['shape']}")
    
    model = HybridMultiSpeciesModelReg(input_shapes, len(species_prevalences))
    trained_model, results = simple_training_loop(model, train_dataset, val_dataset, species_prevalences, species_names=species_names)
    return trained_model, results
