import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm 
from torchgeo.models import ResNet50_Weights, resnet50
from utils.model_pretraining import transfer_pretrained_weights, ClimatePretrainingCNN

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
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)        # (B, out_channels, H, W)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # global pooling
        return x.flatten(1)            # (B, out_channels)


# ------------------------------
# Multi-species model (no Gaussian, no spatial_size)
# ------------------------------
class SimpleMultiSpeciesModel(nn.Module):
    def __init__(self, input_shapes, num_species):
        super().__init__()
        self.num_species = num_species

        # Branches
        self.sat_branch = None
        self.clim_branch = None
        self.point_branch = None

        # Feature dimensions
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

        # Total features per species head
        total_features = self.sat_channels + self.clim_channels + self.point_channels
        print(f"Total features per species head: {total_features}")

        # Species-specific heads
        self.species_heads = nn.ModuleList([
            # nn.Sequential(
            #     nn.Linear(total_features, 256),
            #     nn.ReLU(),
            #     nn.Dropout(0.3),
            #     nn.Linear(256, 128),
            #     nn.ReLU(),
            #     nn.Dropout(0.3),
            #     nn.Linear(128, 1)
            # ) for _ in range(num_species)
            nn.Sequential(
                nn.Linear(total_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            ) for _ in range(num_species)
        ])

    def forward(self, inputs):
        # Unpack inputs
        sat_input, clim_input, point_input = None, None, None
        input_idx = 0
        if self.sat_branch is not None:
            sat_input = inputs[input_idx]; input_idx += 1
        if self.clim_branch is not None:
            clim_input = inputs[input_idx]; input_idx += 1
        if self.point_branch is not None:
            point_input = inputs[input_idx]

        # Extract branch features
        features_list = []
        if self.sat_branch is not None:
            features_list.append(self.sat_branch(sat_input))   # (B, sat_channels)
        if self.clim_branch is not None:
            features_list.append(self.clim_branch(clim_input)) # (B, clim_channels)
        if self.point_branch is not None:
            features_list.append(self.point_branch(point_input)) # (B, point_channels)

        combined_features = torch.cat(features_list, dim=1)  # (B, total_features)

        # Per-species heads
        outputs = [head(combined_features) for head in self.species_heads]
        return torch.cat(outputs, dim=1)  # (B, num_species)

    def get_loss_function(self, species_prevalences, device):
        return get_simple_loss_function(species_prevalences, device)


def get_simple_loss_function(species_prevalences, device):
    """Simple loss that handles class imbalance"""
    pos_weights = []
    for prevalence in species_prevalences:
        # Higher weight for rare species
        weight = (1 - prevalence) / max(prevalence, 1e-8)
        pos_weights.append(min(weight, 50.0))  # Cap at 50 to avoid instability
    
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
    all_preds, all_labels = [], []
    
    # Add tqdm progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Handle batch format: (inputs, labels, ...)
        if len(batch) >= 2:
            inputs, labels = batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        # Move to device
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
        
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        # Update progress bar with current loss
        avg_loss_so_far = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})
    
    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Calculate simple metrics
        accuracy = (all_preds == all_labels).mean()
        f1 = calculate_simple_f1(all_preds, all_labels)
    else:
        accuracy = 0.0
        f1 = 0.0
    
    avg_loss = running_loss / len(train_loader)
    return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def simple_validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
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
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Global metrics
        accuracy = (all_preds == all_labels).mean()
        f1 = calculate_simple_f1(all_preds, all_labels)

        # Per-species metrics
        species_metrics = []
        for i in range(all_preds.shape[1]):
            species_metrics.append({
                'precision': precision_score(all_labels[:, i], all_preds[:, i], zero_division=0),
                'recall': recall_score(all_labels[:, i], all_preds[:, i], zero_division=0),
                'f1': f1_score(all_labels[:, i], all_preds[:, i], zero_division=0),
                'accuracy': accuracy_score(all_labels[:, i], all_preds[:, i]),
                'specificity': ( (all_labels[:, i] == 0) & (all_preds[:, i] == 0) ).sum() / max( (all_labels[:, i] == 0).sum(), 1)
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
    
    # Collect predictions and targets
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
    
    # Find optimal threshold for each species
    for species_idx in range(num_species):
        best_f1 = 0
        best_threshold = 0.5
        
        probs_species = all_probs[:, species_idx]
        targets_species = all_targets[:, species_idx]
        
        # Test different thresholds
        for threshold in torch.linspace(0.1, 0.9, num_thresholds):
            preds = (probs_species > threshold).float()
            
            # Calculate F1
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
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = get_simple_loss_function(species_prevalences, device)
    
    best_f1 = 0
    best_model_state = None
    best_val_metrics = None
    patience = 5
    patience_counter = 0
    
    print("Starting simplified training...")
    
    for epoch in range(epochs):
        train_metrics = simple_train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = simple_validate_epoch(model, val_loader, criterion, device)
        
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
    
    # Map numeric species keys to real species names
    if species_names is not None and 'species_metrics' in best_val_metrics:
        per_species_metrics = {
            species_names[i]: metrics
            for i, metrics in enumerate(best_val_metrics['species_metrics'])
        }
    else:
        per_species_metrics = best_val_metrics.get('species_metrics', {})

    ##getting optimal thresholds
    optimal_thresholds, f1_scores = find_optimal_thresholds_f1(model, val_loader, device, model.num_species)
    print("Optimal thresholds per species:", optimal_thresholds)
    print("F1 scores at optimal thresholds:", f1_scores)
    
    results = {
        'final_val_metrics': [{
            'per_species': per_species_metrics
        }],
        'config': [{'name': 'simple_model'}]
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
    
    model = SimpleMultiSpeciesModel(input_shapes, len(species_prevalences))
    trained_model, results = simple_training_loop(model, train_dataset, val_dataset, species_prevalences, species_names=species_names)
    return trained_model, results


