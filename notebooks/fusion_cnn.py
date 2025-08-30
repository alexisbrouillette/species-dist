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


class SimpleMultiInputCNNv2(nn.Module):
    def __init__(self, input_shapes, config):
        """
        input_shapes: list of tuples, e.g. [(67,100,100), (9,256,256)]
        config: dict with configuration parameters
        """
        super().__init__()
        self.config = config
        self.branches = nn.ModuleList()
        self.flatten_sizes = []
        
        # Build CNN branches
        for i, (c, h, w) in enumerate(input_shapes):
            layers = []

            # First conv block
            layers.extend([
                nn.Conv2d(c, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2)  # keep regular pooling here
            ])

            # Second conv block
            layers.extend([
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)  # regular pooling
            ])

            # Optional third conv block
            if config.get('use_deeper_cnn', False):
                layers.extend([
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                ])

            # **Apply attention pooling at the end instead of flattening**
            if config.get('use_attention', False):
                final_channels = 64 if config.get('use_deeper_cnn', False) else 32
                layers.append(AttentionPooling(final_channels))
            else:
                layers.append(nn.Flatten())  # If no attention, flatten as usual

            branch = nn.Sequential(*layers)
            self.branches.append(branch)

            # Compute flatten size dynamically
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                flat_size = branch(dummy)
                if config.get('use_attention', False):
                    flat_size = flat_size.shape[1]  # AttentionPooling outputs [batch, channels]
                else:
                    flat_size = flat_size.shape[1]
                self.flatten_sizes.append(flat_size)

        total_flatten = sum(self.flatten_sizes)
        
        # Add spatial features if enabled
        spatial_features = 0
        if config.get('use_spatial_features', False):
            spatial_features = config.get('spatial_feature_dim', 32)
            self.spatial_encoder = SpatialFourierFeatures(spatial_features)
        
        # Fully connected layers
        fc_layers = []
        input_dim = total_flatten + spatial_features
        
        # Hidden layers
        hidden_dims = config.get('hidden_dims', [256])
        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3))
            ])
            input_dim = hidden_dim
        
        # Output layer
        fc_layers.append(nn.Linear(input_dim, 1))  # Single species for now
        
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, inputs, coords=None):
        # inputs: list of tensors, one per branch
        features = [branch(x) for branch, x in zip(self.branches, inputs)]
        merged = torch.cat(features, dim=1)
        
        # Add spatial features if enabled
        if self.config.get('use_spatial_features', False) and coords is not None:
            spatial_features = self.spatial_encoder(coords)
            merged = torch.cat([merged, spatial_features], dim=1)
        
        out = self.fc(merged)
        return out  # raw logits

    def get_loss_function(self):
        """Get the appropriate loss function based on configuration"""
        if self.config.get('use_focal_loss', False):
            return FocalLoss(
                alpha=self.config.get('focal_alpha', 1.0),
                gamma=self.config.get('focal_gamma', 2.0),
                pos_weight=torch.tensor([self.config.get('pos_weight', 2.5)])
            )
        else:
            pos_weight = self.config.get('pos_weight', 2.5)
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))


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
    
    # Model setup
    model = model.to(device)
    if config.get('compile_model', True):
        try:
            model = torch.compile(model, mode="default")
            print("Model compiled successfully")
        except:
            print("Model compilation failed, using uncompiled model")
    
    # Loss and optimizer
    criterion = model.get_loss_function().to(device)
    
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
            optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
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
            print(f"  Train - Accuracy: {train_metrics['accuracy']:.4f}, Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['roc_auc']:.4f}")
            print(f"  Val   - Accuracy: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")
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


# Example usage and testing scenarios
def run_experiments(input_shapes, training_dataset, validation_dataset):
    """Run multiple experiments with different configurations"""
    
    scenarios = [
        {
            'name': 'Baseline (Max Pooling + BCE)',
            'use_attention': False,
            'use_focal_loss': False,
            'use_spatial_features': False,
            'use_deeper_cnn': False,
        },
        {
            'name': 'Attention Pooling Only',
            'use_attention': True,
            'use_focal_loss': False,
            'use_spatial_features': False,
            'use_deeper_cnn': False,
        },
        {
            'name': 'Focal Loss Only',
            'use_attention': False,
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'use_spatial_features': False,
            'use_deeper_cnn': False,
        },
        {
            'name': 'Attention + Focal Loss',
            'use_attention': True,
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'use_spatial_features': False,
            'use_deeper_cnn': False,
        },
        {
            'name': 'Attention + Focal + Spatial Features',
            'use_attention': True,
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'use_spatial_features': True,
            'spatial_feature_dim': 32,
            'use_deeper_cnn': False,
        },
        {
            'name': 'Full Configuration',
            'use_attention': True,
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'use_spatial_features': True,
            'spatial_feature_dim': 32,
            'use_deeper_cnn': True,
            'hidden_dims': [512, 256],
            'dropout': 0.4,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'gradient_clipping': True,
            'max_grad_norm': 1.0,
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Running: {scenario['name']}")
        print(f"{'='*60}")
        
        # Create model with current configuration
        model = SimpleMultiInputCNNv2(input_shapes, scenario[0])
        
        # Train model
        trained_model, training_results = training_loop(
            model, training_dataset, validation_dataset, scenario
        )
        
        # Store results
        results[scenario['name']] = {
            'config': scenario,
            'results': training_results,
            'model': trained_model
        }
        
        print(f"\nFinal Results for {scenario['name']}:")
        print(f"  Best Val Loss: {training_results['best_val_loss']:.4f}")
        print(f"  Best Val F1: {training_results['best_val_f1']:.4f}")
        print(f"  Final Val AUC: {training_results['final_val_metrics']['roc_auc']:.4f}")
    
    return results

# ----- Multi-input CNN -----
class SimpleMultiInputCNN(nn.Module):
    def __init__(self, input_shapes):
        """
        input_shapes: list of tuples, e.g. [(67,100,100), (9,256,256)]
        """
        super().__init__()
        self.branches = nn.ModuleList()
        self.flatten_sizes = []

        for c, h, w in input_shapes:
            layers = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            self.branches.append(layers)

            # Compute flatten size dynamically
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                flat_size = layers(dummy).shape[1]
                self.flatten_sizes.append(flat_size)

        total_flatten = sum(self.flatten_sizes)
        self.fc = nn.Sequential(
            nn.Linear(total_flatten, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Output logits for BCEWithLogitsLoss
        )

    def forward(self, inputs):
        # inputs: list of tensors, one per branch
        features = [branch(x) for branch, x in zip(self.branches, inputs)]
        merged = torch.cat(features, dim=1)
        out = self.fc(merged)
        return out  # raw logits

# ----------------------
# Loss selector
# ----------------------

def get_loss(config):
    if config.get("use_focal_loss", False):
        return FocalLoss()
    else:
        return nn.BCEWithLogitsLoss()

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
    


class MultiSourceNpyDataset(Dataset):
    def __init__(self, npy_files, targets, features_index=None, indices=None, transform=None):
        self.npy_files = npy_files
        self.targets = targets
        self.features_index = features_index
        self.indices = np.arange(len(targets)) if indices is None else indices
        self.transform = transform

        # --- MODIFIED: Load shapes from metadata files ---
        self.shapes = []
        if len(self.npy_files) == 0:
            raise ValueError("You must provide at least one .npy file.")
        else:
            for npy_file in self.npy_files:
                try:
                    patches, shape = load_patches(npy_file, as_float=False, mmap_mode='r')
                    self.shapes.append(shape)
                except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                    raise ValueError(f"Could not load shape from metadata file {npy_file.replace('.npy', '_metadata.json')}: {e}")

        # --- ORIGINAL LOGIC REMAINS THE SAME ---
        self.data_arrays = []
        for f, shape in zip(self.npy_files, self.shapes):
            # The 'dtype' needs to be handled correctly as the original is uint8
            arr = np.memmap(f, dtype=np.uint8, mode="r", shape=shape)
            self.data_arrays.append(arr)

        # compute effective shapes (after channel selection)
        self.effective_shapes = []
        for i, shape in enumerate(self.shapes):
            c, h, w = shape[1:]  # skip batch dim
            if i == 0 and self.features_index is not None:
                c = len(self.features_index)
            self.effective_shapes.append((c, h, w))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        data_items = []
        for i, arr in enumerate(self.data_arrays):
            patch = arr[idx]
            # Convert from uint8 [0, 255] to float32 [0, 1] on the fly
            patch = patch.astype(np.float32) / 255.0
            if i == 0 and self.features_index is not None:
                patch = patch[self.features_index]
            # Convert to TensorFlow tensor instead of PyTorch
            #patch_tensor = tf.convert_to_tensor(patch, dtype=tf.float32)
            patch_tensor = torch.tensor(patch, dtype=torch.float32)
            data_items.append(patch_tensor)
        #target_tensor = tf.convert_to_tensor(self.targets[idx], dtype=tf.float32)
        # --- Apply transform once (across all modalities if needed) ---
        if self.transform:
            data_items = [self.transform(x) for x in data_items]

        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        return tuple(data_items) + (target_tensor,)

    # The to_tf_dataset and KerasSequenceWrapper classes remain the same
    # as the core data loading logic in __getitem__ is now handled.
    def to_tf_dataset(self, batch_size=32, shuffle=False, prefetch=tf.data.AUTOTUNE):
        """
        Convert to tf.data.Dataset for use with TensorFlow/Keras
        """
        def generator():
            indices = self.indices.copy()
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                data_items = []
                for i, arr in enumerate(self.data_arrays):
                    patch = arr[idx]
                    # Convert from uint8 [0, 255] to float32 [0, 1]
                    patch_float = patch.astype(np.float32) / 255.0
                    if i == 0 and self.features_index is not None:
                        patch_float = patch_float[self.features_index]
                    data_items.append(patch_float)
                
                target = self.targets[idx].astype(np.float32)
                yield tuple(data_items), target

        # Determine output signature
        output_types = []
        output_shapes = []
        
        # For data items
        for shape in self.effective_shapes:
            output_types.append(tf.float32)
            output_shapes.append(tf.TensorShape(shape))
        
        # For targets
        target_shape = tf.TensorShape(self.targets[0].shape) if hasattr(self.targets[0], 'shape') else tf.TensorShape([])
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tuple(tf.TensorSpec(shape=shape, dtype=tf.float32) for shape in output_shapes),
                tf.TensorSpec(shape=target_shape, dtype=tf.float32)
            )
        )
        
        if batch_size:
            dataset = dataset.batch(batch_size)
        
        if prefetch:
            dataset = dataset.prefetch(prefetch)
        
        # Set cardinality to help with progress bar
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(self) // batch_size if batch_size else len(self)))
            
        return dataset
    
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


class MultiSourceNpyDatasetWithCoords:
    """
    Enhanced dataset class that includes coordinate support
    """
    def __init__(self, npy_files, targets, processed_species_original_locations, 
                 features_index=None, indices=None, transform=None, include_coords=False):
        self.npy_files = npy_files
        self.targets = targets
        self.features_index = features_index
        self.indices = np.arange(len(targets)) if indices is None else indices
        self.transform = transform
        self.include_coords = include_coords
        self.processed_species_original_locations = processed_species_original_locations

        # Load shapes from metadata files (keeping your original logic)
        self.shapes = []
        if len(self.npy_files) == 0:
            raise ValueError("You must provide at least one .npy file.")
        else:
            for npy_file in self.npy_files:
                try:
                    # Assuming you have this function from your original code
                    patches, shape = load_patches(npy_file, as_float=False, mmap_mode='r')
                    self.shapes.append(shape)
                except (FileNotFoundError, Exception) as e:
                    print(f"Warning: Could not load shape from metadata file {npy_file}: {e}")
                    # Fallback: try to load a small portion to infer shape
                    try:
                        arr = np.load(npy_file, mmap_mode='r')
                        self.shapes.append(arr.shape)
                        print(f"Inferred shape from direct loading: {arr.shape}")
                    except Exception as e2:
                        raise ValueError(f"Could not load or infer shape for {npy_file}: {e2}")

        # Load data arrays
        self.data_arrays = []
        for f, shape in zip(self.npy_files, self.shapes):
            arr = np.memmap(f, dtype=np.uint8, mode="r", shape=shape)
            self.data_arrays.append(arr)

        # Compute effective shapes (after channel selection)
        self.effective_shapes = []
        for i, shape in enumerate(self.shapes):
            c, h, w = shape[1:]  # skip batch dim
            if i == 0 and self.features_index is not None:
                c = len(self.features_index)
            self.effective_shapes.append((c, h, w))

        # Extract coordinates if needed
        if self.include_coords:
            self.coordinates = extract_coordinates_from_locations(
                processed_species_original_locations, self.indices
            )
            print(f"Extracted coordinates shape: {self.coordinates.shape}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        data_items = []
        
        for i, arr in enumerate(self.data_arrays):
            patch = arr[actual_idx]  # shape should already be (C, H, W) for this branch
            # Convert from uint8 [0, 255] to float32 [0, 1]
            patch = patch.astype(np.float32) / 255.0

            # If selecting a subset of features for the first array
            if i == 0 and self.features_index is not None:
                patch = patch[self.features_index]  # still (C', H, W)

            patch_tensor = torch.tensor(patch, dtype=torch.float32)  # (C, H, W)
            data_items.append(patch_tensor)

        # Apply transform (e.g., augmentations)
        if self.transform:
            data_items = [self.transform(x) for x in data_items]

        target_tensor = torch.tensor(self.targets[actual_idx], dtype=torch.float32)

        # Return with or without coordinates
        if self.include_coords:
            coords = self.coordinates[idx]
            return data_items, target_tensor, coords
        else:
            return data_items, target_tensor

    
