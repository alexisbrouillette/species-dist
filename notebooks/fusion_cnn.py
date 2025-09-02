import math

import random

import numpy as np
import tensorflow as tf
import os
import json # Import the json library
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
# ----------------------
# Building Blocks
# ----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchgeo.models import ResNet50_Weights
from torchgeo.models import resnet50

class AsymmetricLoss(nn.Module):
    """ Asymmetric Loss for imbalanced binary classification
    Based on Ridnik et al., 2021.
    Implements:
    - asymmetric focusing (gamma_pos, gamma_neg)
    - optional asymmetric clipping for negatives
    - detached modulation for stability
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=2.0, clip=0.0, eps=1e-8, pos_weight=None):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = eps
        # pos_weight is optional and generally should be small or None with ASL
        self.pos_weight = None
        if pos_weight is not None:
            # allow float or 1D tensor
            self.pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)

    def forward(self, logits, targets):
        # shapes: (N, 1) or (N,) → flatten
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        # probabilities
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        # asymmetric clipping for negatives (optional)
        if self.clip > 0.0:
            # paper-style: add then clamp
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)
        # log-likelihoods
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))
        # base loss (no modulation yet)
        loss = targets * log_pos + (1.0 - targets) * log_neg
        # asymmetric focusing (detach the prob used for the modulating factor)
        with torch.no_grad():
            pt = targets * xs_pos + (1.0 - targets) * xs_neg  # prob of the true class
            gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
            mod_factor = (1.0 - pt).pow(gamma)
        loss = loss * mod_factor  # apply modulation
        # optional positive weighting (use sparingly with ASL)
        if self.pos_weight is not None:
            w = targets * self.pos_weight + (1.0 - targets)
            loss = loss * w
        return -loss.mean()

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # Apply pos_weight if provided
        if self.pos_weight is not None:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

class AttentionPooling(nn.Module):
    """Attention-based pooling instead of max/average pooling"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        weights = self.attention(x)  # (batch_size, 1, height, width)
        weighted = x * weights  # Apply attention weights
        # Global weighted average pooling
        return weighted.sum(dim=[2, 3]) / (weights.sum(dim=[2, 3]) + 1e-8)

class SpatialFourierFeatures(nn.Module):
    """Encode spatial coordinates using Fourier features"""
    def __init__(self, num_features=32):
        super().__init__()
        self.num_features = num_features
        # Create random frequencies for Fourier encoding
        self.register_buffer('frequencies', torch.randn(num_features // 2, 2) * 10)

    def forward(self, coords):
        # coords: (batch_size, 2) - [lat, lon] normalized to [-1, 1]
        # Apply Fourier transform
        projected = 2 * math.pi * coords @ self.frequencies.T  # (batch_size, num_features//2)
        fourier_features = torch.cat([
            torch.sin(projected),
            torch.cos(projected)
        ], dim=-1)  # (batch_size, num_features)
        return fourier_features

class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling layer that handles variable input sizes"""
    def __init__(self, pool_sizes=[1, 2, 4]):
        super().__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        pooled_features = []
        for pool_size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
            pooled = pooled.view(batch_size, -1)
            pooled_features.append(pooled)
        return torch.cat(pooled_features, dim=1)

class FlexibleCNNBranch(nn.Module):
    """CNN branch that handles variable input sizes with multiple pooling options"""
    def __init__(self, in_channels, config):
        super().__init__()
        # Get dropout rate
        dropout_rate = config.get('dropout', 0.3)
        # Progressive channel expansion
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.5)  # Lower dropout in early layers
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.7)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        # Optional deeper layers for better feature extraction
        self.use_deeper = config.get('use_deeper_cnn', False)
        if self.use_deeper:
            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_rate)
            )
            final_channels = 256
        else:
            final_channels = 128
        # Pooling strategy
        self.use_attention = config.get('use_attention', False)
        self.use_spp = config.get('use_spp', True)  # Default to SPP
        if self.use_attention:
            self.attention_pool = AttentionPooling(final_channels)
            self.output_size = final_channels
        elif self.use_spp:
            pool_sizes = config.get('spp_pool_sizes', [1, 2, 4])
            self.spp = SpatialPyramidPooling(pool_sizes)
            self.output_size = final_channels * sum(size * size for size in pool_sizes)
        else:
            # Fallback to adaptive pooling + flatten
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed output size
            self.output_size = final_channels * 16

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.use_deeper:
            x = self.conv4(x)
        # Apply appropriate pooling
        if self.use_attention:
            x = self.attention_pool(x)
        elif self.use_spp:
            x = self.spp(x)
        else:
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
        return x

class TabularFeatureEncoder(nn.Module):
    """Simple MLP to encode tabular features"""
    def __init__(self, input_dim, hidden_dims=[32, 64, 128], dropout=0.3):
        super().__init__()
        if not isinstance(input_dim, int):
            raise ValueError(f"Expected integer input_dim, got {input_dim}")
        print(f"TabularFeatureEncoder input_dim: {input_dim}")
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.output_size = current_dim

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
        return self.encoder(x)

class SimpleMultiInputCNNv2(nn.Module):
    def __init__(self, input_shapes, config):
        """ input_shapes: list of tuples, e.g. [(67,100,100), (9,256,256)]
        config: dict with configuration parameters
        """
        super().__init__()
        self.config = config
        self.branches = nn.ModuleList()
        self.flatten_sizes = []
        # Build CNN branches
        for i, dict in enumerate(input_shapes):
            shape = dict['shape']
            source = dict['source']
            print(f"Building branch {i} for source '{source}' with shape {shape}")
            if source == 'point_infos':  # Tabular branch
                input_dim = shape[0]  # not (c, n)!
                branch = TabularFeatureEncoder(
                    input_dim,
                    config.get('tabular_hidden_dims', [32, 64, 32]),
                    config.get('dropout', 0.3)
                )
            else:
                if source == 'sentinel2':
                    print("Using TorchGeo pretrained ResNet50 for Sentinel-2 branch")
                    weights = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS  # or SENTINEL2_MI_MS_SATLAS
                    model = resnet50(weights=weights)
                    model.fc = nn.Identity()          # remove classifier
                    branch = model
                    branch.output_size = 2048
                else:
                    (c, h, w) = shape
                    branch = FlexibleCNNBranch(c, config)
            self.branches.append(branch)
            self.flatten_sizes.append(branch.output_size)
        # Use the known output size
        total_flatten = sum(self.flatten_sizes)
        # Add spatial features if enabled
        spatial_features = 0
        if config.get('use_spatial_features', False):
            spatial_features = config.get('spatial_feature_dim', 32)
            self.spatial_encoder = SpatialFourierFeatures(spatial_features)
        self.build_classifier(total_flatten + spatial_features, config)

    def build_classifier(self, input_dim, config):
        """Build classifier with optional residual connections"""
        hidden_dims = config.get('hidden_dims', [512, 256, 128])
        dropout_rate = config.get('dropout', 0.3)
        use_residual = config.get('use_residual', True)
        layers = []
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        self.classifier = nn.Sequential(*layers)
        # For class imbalance - you might want to use this in your loss function
        self.class_weights = config.get('class_weights', None)

    def forward(self, inputs, coords=None):
        # inputs: list of tensors, one per branch
        features = [branch(x) for branch, x in zip(self.branches, inputs)]
        merged = torch.cat(features, dim=1)
        # Add spatial features if enabled
        if self.config.get('use_spatial_features', False) and coords is not None:
            spatial_features = self.spatial_encoder(coords)
            merged = torch.cat([merged, spatial_features], dim=1)
        out = self.classifier(merged)
        return out  # raw logits

    def get_loss_function(self, device=None):
        """Get the appropriate loss function based on configuration"""
        pos_weight = torch.tensor(
            [self.config.get('pos_weight', 2.5)],
            dtype=torch.float32,
            device=device if device is not None else "cpu"
        )
        if self.config.get('use_asymmetric_loss', False):
            return AsymmetricLoss(
                gamma_pos=self.config.get('asl_gamma_pos', 0),
                gamma_neg=self.config.get('asl_gamma_neg', 1),
                clip=self.config.get('asl_clip', 0.0),
                pos_weight=pos_weight
            )
        elif self.config.get('use_focal_loss', False):
            return FocalLoss(
                alpha=self.config.get('focal_alpha', 0.5),
                gamma=self.config.get('focal_gamma', 2.0),
                pos_weight=pos_weight
            )
        else:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }


from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds, all_probs, all_labels = [], [], []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (inputs, labels, *coords) in enumerate(pbar):
        # --- move to device ---
        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device).float()
        coords = coords[0].to(device) if coords else None

        # --- forward ---
        optimizer.zero_grad()
        outputs = model(inputs, coords)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # --- accumulate metrics ---
        total_loss += loss.item()
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy())

        correct += (preds == labels.detach().cpu().numpy()).sum()
        total += labels.size(0)

        # --- update tqdm bar ---
        avg_loss = total_loss / (batch_idx + 1)
        avg_acc = correct / total
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc:.4f}"
        })

    # final metrics at epoch end
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(train_loader)
    return metrics



def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:  # with coordinates
                inputs, labels, coords = batch
                inputs = [x.to(device) for x in inputs]
                coords = coords.to(device) if coords is not None else None
            else:  # without coordinates
                inputs, labels = batch
                inputs = [x.to(device) for x in inputs]
                coords = None
                
            labels = labels.to(device).float()
            
            outputs = model(inputs, coords)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(val_loader)
    return metrics


def training_loop(model, training_dataset, validation_dataset, config):
    """Main training loop with configuration support"""
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    batch_size = config.get('batch_size', 64)
    epochs = config.get('epochs', 10)
    learning_rate = config.get('learning_rate', 1e-3)
    
    # Data loaders
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # # Model setup
    # model = model.to(device)
    # if config.get('compile_model', True):
    #     try:
    #         model = torch.compile(model, mode="default")
    #         print("Model compiled successfully")
    #     except:
    #         print("Model compilation failed, using uncompiled model")
    
    # Loss and optimizer
    criterion = model.get_loss_function(device=device)
    
    optimizer_type = config.get('optimizer', 'adam')
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.get('weight_decay', 1e-4))
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=config.get('momentum', 0.9))
    
    # Scheduler
    if config.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
    
    # Metrics tracking
    train_metrics_history = {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': []}
    val_metrics_history = {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': []}
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    print(f"\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        if config.get('use_scheduler', True):
            scheduler.step(val_metrics['loss'])
        
        # Track metrics
        for key in train_metrics_history.keys():
            train_metrics_history[key].append(train_metrics[key])
            val_metrics_history[key].append(val_metrics[key])
        
        # Track best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
        
        # Print progress
        if (epoch + 1) % config.get('print_every', 2) == 0:
            print(f"Epoch [{epoch+1}/{epochs}]:")
            print(f"  Train - Accuracy: {train_metrics['accuracy']:.4f}, Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['roc_auc']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}")
            print(f"  Val   - Accuracy: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    results = {
        'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history,
        'best_val_loss': best_val_loss,
        'best_val_f1': best_val_f1,
        'final_train_metrics': train_metrics,
        'final_val_metrics': val_metrics
    }
    
    return model, results


##################################################
################Data loading######################
##################################################


def load_patches(npy_file, as_float=True, mmap_mode='r'):
    """
    Load uint8 patches with automatic shape detection.
    
    Parameters
    ----------
    npy_file : str
        Path to .npy file
    as_float : bool
        If True, convert to float32 [0,1]. If False, keep as uint8 [0,255]
    mmap_mode : str
        Memory mapping mode ('r', 'r+', None)
    
    Returns
    -------
    np.ndarray
        Loaded patches
    """
    # Try to load metadata first
    metadata_file = npy_file.replace('.npy', '_metadata.json')
    if os.path.exists(metadata_file):
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load with known shape
        data = np.memmap(npy_file, dtype=np.uint8, mode=mmap_mode, 
                        shape=tuple(metadata['shape']))
        print(f"Loaded with shape: {metadata['shape']}")
    else:
        # Fallback: try to infer or load normally
        try:
            # This might work if numpy can infer the shape
            data = np.load(npy_file, mmap_mode=mmap_mode)
        except:
            raise ValueError(f"Cannot determine shape. Please provide metadata file: {metadata_file}")
    
    if as_float:
        if mmap_mode is None:
            # Load into memory and convert
            return data.astype(np.float32) / 255.0
        else:
            # Return a view that converts on access (memory efficient)
            class Float32View:
                def __init__(self, uint8_data):
                    self._data = uint8_data
                    self.shape = uint8_data.shape
                    self.dtype = np.float32
                
                def __getitem__(self, key):
                    return self._data[key].astype(np.float32) / 255.0
                
                def __array__(self):
                    return self._data.astype(np.float32) / 255.0
            
            return Float32View(data)
    
    return data, metadata['shape']

class RandomFlipRotate:
    def __call__(self, x):
        # # Random vertical flip
        # if random.random() > 0.5:
        #     x = torch.flip(x, dims=[1])  # flip height (H)
        # # Random horizontal flip
        # if random.random() > 0.5:
        #     x = torch.flip(x, dims=[2])  # flip width (W)
        # Random rotation by 0, 90, 180, or 270 degrees
        k = random.randint(0, 3)
        x = torch.rot90(x, k, dims=[1, 2])
        return x
    
    
def extract_coordinates_from_locations(processed_species_original_locations, indices):
    """
    Extract normalized coordinates from your location data
    
    Args:
        processed_species_original_locations: Your GeoDataFrame with geometry
        indices: Indices to extract coordinates for
        
    Returns:
        torch.Tensor: Normalized coordinates (lat, lon) in range [-1, 1]
    """
    # Extract coordinates for the given indices
    coords_subset = processed_species_original_locations.iloc[indices]
    
    # Extract x, y coordinates from geometry
    lons = coords_subset.geometry.x.values
    lats = coords_subset.geometry.y.values
    
    # Normalize coordinates to [-1, 1] range
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    
    normalized_lons = 2 * (lons - lon_min) / (lon_max - lon_min) - 1
    normalized_lats = 2 * (lats - lat_min) / (lat_max - lat_min) - 1
    
    # Stack into (n_samples, 2) array
    coords = np.stack([normalized_lats, normalized_lons], axis=1)
    return torch.tensor(coords, dtype=torch.float32)


import torch
import numpy as np

class MultiSourceNpyDatasetWithCoords:
    """
    Enhanced dataset class that includes coordinate support,
    with optional cropping of patches to a smaller ratio.
    """
    def __init__(self, npy_files, targets, processed_species_original_locations, 
                 features_index=None, indices=None, transform=None, 
                 include_coords=False, resize_ratio=1.0, add_point_infos=False):
        self.npy_files = npy_files
        self.targets = targets
        self.features_index = features_index
        self.indices = np.arange(len(targets)) if indices is None else indices
        self.transform = transform
        self.include_coords = include_coords
        self.processed_species_original_locations = processed_species_original_locations
        self.resize_ratio = resize_ratio
        self.add_point_infos = add_point_infos
        self.pred100_index = None  # To be set if needed
        if self.targets.dtype != np.float32:
            self.targets = self.targets.astype(np.float32)

        # Load shapes
        self.shapes = []
        if len(self.npy_files) == 0:
            raise ValueError("You must provide at least one .npy file.")
        else:
            for npy_file in self.npy_files:
                try:
                    path = npy_file['path']
                    print(f"Loading shape from: {path}")
                    patches, shape = load_patches(path, as_float=False, mmap_mode='r')
                    self.shapes.append(shape)
                except (FileNotFoundError, Exception) as e:
                    print(f"Warning: Could not load shape from metadata file {path}: {e}")
                    arr = np.load(path, mmap_mode='r')
                    self.shapes.append(arr.shape)

        # Load arrays as memmap
        self.data_arrays = []
        for f, shape in zip(self.npy_files, self.shapes):
            path = f['path']
            arr = np.memmap(path, dtype=np.uint8, mode="r", shape=shape)
            self.data_arrays.append(arr)

        # Compute effective shapes (account for feature subset and crop ratio)
        self.effective_shapes = []
        for i, shape in enumerate(self.shapes):
            c, h, w = shape[1:]  # (N, C, H, W)
            if npy_files[i]['name'] == 'pred100':
                c = len(self.features_index)
                self.pred100_index = i

            if self.resize_ratio < 1.0:
                h = int(h * self.resize_ratio)
                w = int(w * self.resize_ratio)

            self.effective_shapes.append({
                "shape": (c, h, w),
                "source": npy_files[i]['name']
                })
        if self.add_point_infos:
            point_infos = processed_species_original_locations.drop(columns=['geometry', 'species_list']).values
            point_infos = point_infos.astype(np.float32)

            # Select specific columns if features_index is provided
            if self.features_index is not None:
                # Ensure features_index is a list or array
                if isinstance(self.features_index, int):
                    self.features_index = [self.features_index]
                point_infos = point_infos[:, self.features_index]

            self.point_infos = point_infos
            print(f"Extracted point infos shape: {self.point_infos.shape}")
            self.effective_shapes.append({
                "shape": (self.point_infos.shape[1],),
                "source": "point_infos"
                })  # single-element tuple

        # Coordinates if needed
        if self.include_coords:
            self.coordinates = extract_coordinates_from_locations(
                processed_species_original_locations, self.indices
            )
            print(f"Extracted coordinates shape: {self.coordinates.shape}")

    def __len__(self):
        return len(self.indices)

    def _center_crop(self, patch, new_h, new_w):
        _, h, w = patch.shape
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return patch[:, top:top+new_h, left:left+new_w]

    def __getitem__(self, idx):
        data_items = []

        for i, arr in enumerate(self.data_arrays):
            patch = arr[self.indices[idx]].astype(np.float32)  # (C, H, W)

            # Select subset of features for first array if needed
            if i == self.pred100_index and self.features_index is not None:
                patch = patch[self.features_index]

                patch = patch/ 255.0

            # Apply center crop if ratio < 1.0
            if self.resize_ratio < 1.0:
                _, h, w = patch.shape
                new_h, new_w = int(h * self.resize_ratio), int(w * self.resize_ratio)
                patch = self._center_crop(patch, new_h, new_w)

            tensor = torch.from_numpy(patch)
            data_items.append(tensor)

        # Transform (augmentations etc.)
        if self.transform:
            data_items = [self.transform(x) for x in data_items]
        # Add point infos if needed
        if self.add_point_infos:
            point_info = self.point_infos[idx]
            point_info_tensor = torch.from_numpy(point_info)
            data_items.append(point_info_tensor)

        target_tensor = torch.from_numpy(self.targets[idx])

        if self.include_coords:
            coords = self.coordinates[idx]
            return data_items, target_tensor, coords
        else:
            return data_items, target_tensor

    
