import math

import random
import numpy as np
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
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

from utils.model_pretraining import transfer_pretrained_weights, ClimatePretrainingCNN

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

                if source == 'pred100':
                    (c, h, w) = shape
                    branch = FlexibleCNNBranch(c, config)
                    if(config.get('use_pretrained_climate_cnn', False)):
                        pretrained_model = ClimatePretrainingCNN()
                        pretrained_model.load_state_dict(torch.load("climate_pretrained_cnn.pth", map_location=config['device']))
                        transfer_pretrained_weights(pretrained_model, branch)
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
            print(f"Adding spatial features of dimension {spatial_features}")
        self.build_classifier(total_flatten + spatial_features, config)

    def build_classifier(self, input_dim, config):
        """Build classifier with optional residual connections"""
        hidden_dims = config.get('hidden_dims', [512, 256, 128])

        # Warn if the first layer is significantly smaller than the input
        if hidden_dims[0] < (input_dim / 2):
            print(f"⚠️  Warning: First hidden layer ({hidden_dims[0]}) is less than half the size of the input features ({input_dim}). This may cause an information bottleneck.")
            print(f"Size of each branch output: {input_dim}, flatten_sizes: {self.flatten_sizes}")
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # True negatives and false positives
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'pr_auc': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        'specificity': specificity,
    }


from tqdm import tqdm
import torch.profiler as profiler

def train_epoch(model, train_loader, criterion, optimizer, device, config, profile=True):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds, all_probs, all_labels = [], [], []

    def run_loop(prof=None):
        nonlocal total_loss, correct, total, all_preds, all_probs, all_labels

        pbar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (inputs, labels, *coords) in enumerate(pbar):
            # --- move to device ---
            inputs = [x.to(device, non_blocking=True) for x in inputs]
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

            if prof:
                prof.step()

    if profile:
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            on_trace_ready=profiler.tensorboard_trace_handler("./logdir"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            run_loop(prof)
    else:
        run_loop()

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
    """Main training loop with configuration support and enhanced early stopping"""
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    batch_size = config.get('batch_size', 64)
    max_epochs = config.get('epochs', 50)  # Renamed to max_epochs for clarity
    learning_rate = config.get('learning_rate', 1e-3)
    
    # Enhanced early stopping configuration
    early_stopping_patience = config.get('early_stopping_patience', 10)
    early_stopping_metric = config.get('early_stopping_metric', 'loss')  # 'loss', 'f1', or 'pr_auc'
    early_stopping_delta = config.get('early_stopping_delta', 0.001)
    early_stopping_warmup = config.get('early_stopping_warmup', 5)  # Don't stop in first N epochs
    
    # Data loaders
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True)
    
    # Loss and optimizer
    criterion = model.get_loss_function(device=device)
    
    optimizer_type = config.get('optimizer', 'adam')
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                              weight_decay=config.get('weight_decay', 1e-4))
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                            momentum=config.get('momentum', 0.9))
    
    # Scheduler - more aggressive
    if config.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6
        )
    
    # Metrics tracking
    metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall', 'pr_auc', 'specificity']
    train_metrics_history = {metric: [] for metric in metrics}
    val_metrics_history = {metric: [] for metric in metrics}
    
    # Enhanced early stopping variables
    best_val_metric = float('inf') if early_stopping_metric == 'loss' else 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_weights = None
    best_val_metrics = None
    
    print(f"\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nStarting training for up to {max_epochs} epochs...")
    print(f"Early stopping: monitoring {early_stopping_metric}, patience {early_stopping_patience}")
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        if config.get('use_scheduler', True):
            scheduler.step(val_metrics['loss'])
        
        # Track metrics
        for metric in metrics:
            train_metrics_history[metric].append(train_metrics[metric])
            val_metrics_history[metric].append(val_metrics[metric])
        
        # Get the current value of the metric we're monitoring
        current_metric = val_metrics[early_stopping_metric]
        
        # Check for improvement (different logic for loss vs score metrics)
        improved = False
        if early_stopping_metric == 'loss':
            improved = current_metric < best_val_metric - early_stopping_delta
        else:  # f1, pr_auc, etc.
            improved = current_metric > best_val_metric + early_stopping_delta
        
        # Update best metrics if improved
        if improved:
            best_val_metric = current_metric
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_weights = model.state_dict().copy()
            best_val_metrics = val_metrics.copy()
            improvement_msg = f"✓ New best {early_stopping_metric}: {best_val_metric:.4f}"
        else:
            epochs_no_improve += 1
            improvement_msg = f"✗ No improvement for {epochs_no_improve} epochs"
        
        # Print detailed progress every epoch
        print(f"\nEpoch [{epoch+1}/{max_epochs}]:")
        print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, Loss: {train_metrics['loss']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['pr_auc']:.4f}")
        print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['pr_auc']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Best {early_stopping_metric}: {best_val_metric:.4f} (epoch {best_epoch})")
        print(f"  {improvement_msg}")
        print(f"  Early stopping: {epochs_no_improve}/{early_stopping_patience}")
        
        # Check early stopping condition (only after warmup period)
        if epoch >= early_stopping_warmup and epochs_no_improve >= early_stopping_patience:
            print(f"\n🚨 Early stopping triggered after {epoch+1} epochs!")
            print(f"   Best performance at epoch {best_epoch}:")
            for metric, value in best_val_metrics.items():
                print(f"   {metric}: {value:.4f}")
            break
    
    # Restore the best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\n✅ Loaded best model from epoch {best_epoch}")
    else:
        print(f"\n⚠️  No improvement during training, using final model")
        best_val_metrics = val_metrics
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_val_metrics = validate_epoch(model, val_loader, criterion, device)

    results = {
        'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history,
        'best_val_metric': best_val_metric,
        'best_epoch': best_epoch,
        'final_val_metrics': final_val_metrics,
        'stopped_early': epochs_no_improve >= early_stopping_patience,
        'epochs_completed': epoch + 1
    }
    
    return model, results


