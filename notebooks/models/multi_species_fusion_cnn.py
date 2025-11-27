import math

import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import BatchSampler
from collections import defaultdict
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

from utils.dataset import AutoClassAwareBatchSampler


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=1.0, pos_weight=None, clip=0.0, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        
        # Use precomputed pos_weight if provided, else default to 1.0
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        else:
            self.register_buffer('pos_weight', torch.tensor([1.0]))

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Probabilities
        ps = torch.sigmoid(logits)
        ps_pos = ps
        ps_neg = 1 - ps
        
        # Apply clipping
        if self.clip > 0:
            ps_neg = (ps_neg + self.clip).clamp(max=1.0)
        
        # Log losses
        log_pos = torch.log(ps_pos.clamp(min=self.eps))
        log_neg = torch.log(ps_neg.clamp(min=self.eps))
        
        # Basic loss components
        loss_pos = targets * log_pos
        loss_neg = (1 - targets) * log_neg
        
        # Asymmetric focusing
        with torch.no_grad():
            pt_pos = ps_pos
            pt_neg = ps_neg
            mod_pos = (1 - pt_pos).pow(self.gamma_pos)
            mod_neg = (1 - pt_neg).pow(self.gamma_neg)
        
        # Apply modulation and pos_weight
        loss_pos = loss_pos * mod_pos * self.pos_weight
        loss_neg = loss_neg * mod_neg
        
        loss = loss_pos + loss_neg
        return -loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        else:
            self.register_buffer('pos_weight', torch.tensor([1.0]))

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=self.pos_weight, 
            reduction='none'
        )
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


        
class SimpleMultiInputCNNv2WithLatent(nn.Module):
    def __init__(self, input_shapes, config):
        super().__init__()
        self.config = config
        self.branches = nn.ModuleList()
        self.flatten_sizes = []
        self.num_species = len(config.get('species', []))
        self.use_latent = config.get('use_latent', False)  # Add this flag
        
        # Build CNN branches (your existing code)
        for i, dict in enumerate(input_shapes):
            shape = dict['shape']
            source = dict['source']
            print(f"Building branch {i} for source '{source}' with shape {shape}")
            
            if source == 'point_infos':  # Tabular branch
                input_dim = shape[0]
                branch = TabularFeatureEncoder(
                    input_dim,
                    config.get('tabular_hidden_dims', [32, 64, 32]),
                    config.get('dropout', 0.3)
                )
            else:
                if source == 'sentinel2':
                    print("Using TorchGeo pretrained ResNet50 for Sentinel-2 branch")
                    weights = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS
                    model = resnet50(weights=weights)
                    model.fc = nn.Identity()
                    branch = model
                    branch.output_size = 2048
                else:
                    (c, h, w) = shape
                    branch = FlexibleCNNBranch(c, config)
                    if config.get('use_pretrained_climate_cnn', False) and source == 'pred100':
                        pretrained_model = ClimatePretrainingCNN()
                        pretrained_model.load_state_dict(torch.load("climate_pretrained_cnn.pth", map_location=config['device']))
                        transfer_pretrained_weights(pretrained_model, branch)
            
            self.branches.append(branch)
            self.flatten_sizes.append(branch.output_size)
        
        # Calculate total feature size
        total_flatten = sum(self.flatten_sizes)
        spatial_features = 0
        if config.get('use_spatial_features', False):
            spatial_features = config.get('spatial_feature_dim', 32)
            self.spatial_encoder = SpatialFourierFeatures(spatial_features)
        
        if self.use_latent:
            self.latent_dim = config.get('latent_dim', 5)
            self.kl_weight = config.get('kl_weight', 0.001)
            
            self.latent_encoder = nn.Sequential(
                nn.Linear(total_flatten + spatial_features, 64),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            self.latent_mu = nn.Linear(64, self.latent_dim)
            self.latent_logvar = nn.Linear(64, self.latent_dim)
            
            classifier_input_dim = total_flatten + spatial_features + self.latent_dim
        else:
            classifier_input_dim = total_flatten + spatial_features
        
        self.build_classifier(classifier_input_dim, config)


    def build_classifier(self, input_dim, config):
        """Build classifier with latent features"""
        hidden_dims = config.get('hidden_dims', [512, 256, 128])
        dropout_rate = config.get('dropout', 0.3)
        
        layers = []
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Output layer for all species
        layers.append(nn.Linear(current_dim, self.num_species))
        self.classifier = nn.Sequential(*layers)

    def encode(self, x):
        """Encode features to latent distribution parameters"""
        h = self.latent_encoder(x)
        mu = self.latent_mu(h)
        logvar = self.latent_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs, coords=None):
        # Extract features from all branches
        features = [branch(x) for branch, x in zip(self.branches, inputs)]
        merged = torch.cat(features, dim=1)
        
        # Add spatial features if enabled
        if self.config.get('use_spatial_features', False) and coords is not None:
            spatial_features = self.spatial_encoder(coords)
            merged = torch.cat([merged, spatial_features], dim=1)
        
        if self.use_latent:
            # Latent variable pathway
            mu, logvar = self.encode(merged)
            z = self.reparameterize(mu, logvar)
            
            # Combine original features with latent vector
            combined = torch.cat([merged, z], dim=1)
            
            # Final prediction
            out = self.classifier(combined)
            return out, mu, logvar
        else:
            # Direct prediction without latent components
            out = self.classifier(merged)
            return out, None, None

    def get_loss_function(self, device=None):
        # Compute pos_weight tensor
        if 'pos_weight' in self.config and isinstance(self.config['pos_weight'], list):
            pos_weight = torch.tensor(
                self.config['pos_weight'],
                dtype=torch.float32,
                device=device if device is not None else "cpu"
            )
        else:
            pos_weight = torch.ones(self.num_species, device=device)
        print(f"Using pos_weight: {pos_weight}")
        # Base classification loss
        if self.config.get('use_asymmetric_loss', False):
            classification_criterion = AsymmetricLoss(
                gamma_pos=self.config.get('asl_gamma_pos', 0),
                gamma_neg=self.config.get('asl_gamma_neg', 1),
                clip=self.config.get('asl_clip', 0.0),
                pos_weight=pos_weight
            )
        elif self.config.get('use_focal_loss', False):
            classification_criterion = FocalLoss(
                alpha=self.config.get('focal_alpha', 0.5),
                gamma=self.config.get('focal_gamma', 2.0),
                pos_weight=pos_weight
            )
        else:
            classification_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if not self.use_latent:
            # Wrap BCE/focal/asymmetric in latent-compatible signature
            def wrapped_loss(predicted_logits, true_labels, mu=None, logvar=None, kl_weight=None):
                loss = classification_criterion(predicted_logits, true_labels)
                return loss, loss, torch.tensor(0.0, device=predicted_logits.device)
            return wrapped_loss

        # For latent model, return the original combined loss
        def combined_loss(predicted_logits, true_labels, mu, logvar, kl_weight=None):
            if kl_weight is None:
                kl_weight = self.config.get('kl_weight', 0.001)  # Default value
            
            # Classification loss (mean over batch)
            classification_loss = classification_criterion(predicted_logits, true_labels)
            
            # KL divergence (mean over batch)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            
            # Total loss
            total_loss = classification_loss + kl_weight * kl_loss
            return total_loss, classification_loss.item(), kl_loss.item()
        
        return combined_loss
    





def create_species_configs(species_names, prevalences, loss_type="asymmetric", variant="default"):
    """
    Create one config per species.
    
    Args:
        species_names: list of species names
        prevalences: list of prevalence values (same order as names)
        loss_type: "focal", "asymmetric", or "bce"
        variant: which global strategy to apply ("default", "recall", "precision", "balanced", "conservative")
    
    Returns:
        List of configs, one per species
    """
    configs = []

    for name, prev in zip(species_names, prevalences):
        config = {
            "name": name,
            "prevalence": prev,
            "hidden_dims": [128],
            "pos_weight": (1 - prev) / prev if prev > 0 else 1.0
        }

        if loss_type == "focal":
            if prev < 0.1:
                config.update({"focal_alpha": 0.8, "focal_gamma": 2.0})
            elif prev < 0.3:
                config.update({"focal_alpha": 0.6, "focal_gamma": 2.0})
            elif prev > 0.8:
                config.update({"focal_alpha": 0.1, "focal_gamma": 1.0})
            elif prev > 0.6:
                config.update({"focal_alpha": 0.2, "focal_gamma": 1.5})
            else:
                config.update({"focal_alpha": 0.4, "focal_gamma": 2.0})

        elif loss_type == "asymmetric":
            # Very common species
            if prev >= 0.9:
                config.update({"gamma_pos": 0.0, "gamma_neg": 1.5, "clip": 0.15})
            elif prev >= 0.7:
                config.update({"gamma_pos": 0.0, "gamma_neg": 2.0, "clip": 0.1})
            elif prev >= 0.4:
                config.update({"gamma_pos": 0.2, "gamma_neg": 2.5, "clip": 0.07})
            elif prev >= 0.1:
                config.update({"gamma_pos": 0.5, "gamma_neg": 3.0, "clip": 0.05})
            else:
                # Ultra-rare species: choose based on variant
                if variant == "recall":
                    config.update({"gamma_pos": 2.0, "gamma_neg": 1.0, "clip": 0.05})
                elif variant == "precision":
                    config.update({"gamma_pos": 0.5, "gamma_neg": 3.0, "clip": 0.03})
                elif variant == "balanced":
                    config.update({"gamma_pos": 1.5, "gamma_neg": 2.0, "clip": 0.04})
                elif variant == "conservative":
                    config.update({"gamma_pos": 1.0, "gamma_neg": 2.0, "clip": 0.03,
                                   "target_threshold": 0.55})
                else:  # fallback
                    config.update({"gamma_pos": 1.0, "gamma_neg": 2.0, "clip": 0.05})

        elif loss_type == "bce":
            config.update({"pos_weight": (1 - prev) / prev if prev > 0 else 1.0})

        configs.append(config)

    return configs


class SimpleMultiInputCNNv2(nn.Module):
    def __init__(self, input_shapes, config):
        """ input_shapes: list of tuples, e.g. [(67,100,100), (9,256,256)]
        config: dict with configuration parameters
        """
        super().__init__()
        self.config = config
        self.branches = nn.ModuleList()
        self.flatten_sizes = []
        self.num_species = len(config.get('species', []))  # Add this parameter
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
        layers.append(nn.Linear(current_dim, self.num_species))  # Multi-species output
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
        # pos_weight should now be a tensor with one value per species
        if 'pos_weight' in self.config and isinstance(self.config['pos_weight'], list):
            pos_weight = torch.tensor(
                self.config['pos_weight'],
                dtype=torch.float32,
                device=device if device is not None else "cpu"
            )
        else:
            # Default equal weighting if not specified
            pos_weight = torch.ones(self.num_species, device=device)
            
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


def calculate_metrics_multi_label(y_true, y_pred, y_prob, species_list):
    """Calculate comprehensive metrics for multi-label classification"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    
    # Initialize dictionaries to store metrics for each species
    metrics_per_species = {}
    # Calculate metrics for each species
    for i in range(len(species_list)):
        tn = ((y_true[:, i] == 0) & (y_pred[:, i] == 0)).sum()
        fp = ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics_per_species[f'{species_list[i]}'] = {
            'accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'pr_auc': average_precision_score(y_true[:, i], y_prob[:, i]) if len(np.unique(y_true[:, i])) > 1 else 0.0,
            'specificity': specificity,
        }
    
    # Calculate macro-averaged metrics
    macro_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in metrics_per_species.values()]),
        'f1': np.mean([m['f1'] for m in metrics_per_species.values()]),
        'precision': np.mean([m['precision'] for m in metrics_per_species.values()]),
        'recall': np.mean([m['recall'] for m in metrics_per_species.values()]),
        'pr_auc': np.mean([m['pr_auc'] for m in metrics_per_species.values()]),
        'specificity': np.mean([m['specificity'] for m in metrics_per_species.values()]),
    }
    
    return {
        'macro': macro_metrics,
        'per_species': metrics_per_species
    }

def train_epoch(model, train_loader, criterion, optimizer, device, config, profile=False, kl_weight=None):
    model.train()
    running_loss = 0.0  # Changed from total_loss to running_loss to avoid conflict
    correct = 0
    total = 0
    all_preds, all_probs, all_labels = [], [], []
    
    def run_loop(prof=None):
        nonlocal running_loss, correct, total, all_preds, all_probs, all_labels
        pbar = tqdm(train_loader, desc="Training", leave=False)
    
        for batch_idx, (inputs, labels, *coords) in enumerate(pbar):
            # --- move to device ---
            inputs = [x.to(device, non_blocking=True) for x in inputs]
            labels = labels.to(device).float()
            coords = coords[0].to(device) if coords else None

            # --- forward ---
            optimizer.zero_grad()

            outputs, mu, logvar = model(inputs, coords)
            loss, class_loss, kl_loss = criterion(outputs, labels, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            
            if hasattr(model, 'classifier') and hasattr(model.classifier, 'species_heads'):
                # Calculate gradient norms per species head
                grad_norms = []
                for head in model.classifier.species_heads:
                    total_norm = 0
                    for param in head.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.norm().item()
                            total_norm += param_norm
                    grad_norms.append(total_norm)
                
                # Balance gradients if any norms are computed
                if grad_norms and max(grad_norms) > 0:
                    avg_norm = sum(grad_norms) / len(grad_norms)
                    for i, head in enumerate(model.classifier.species_heads):
                        if grad_norms[i] > avg_norm * 2:  # If gradient norm is twice the average, scale down
                            scale = avg_norm * 2 / grad_norms[i]
                            for param in head.parameters():
                                if param.grad is not None:
                                    param.grad *= scale

            optimizer.step()

            # --- accumulate metrics ---
            running_loss += loss.item()  # Use .item() to get Python number
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.detach().cpu().numpy())

            correct += (preds == labels.detach().cpu().numpy()).sum()
            total += labels.size(0)

            # --- update tqdm bar ---
            avg_loss = running_loss / (batch_idx + 1)
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
    
    # After the training loop, update metrics calculation
    metrics = calculate_metrics_multi_label(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_probs),
        config.get('species', [])
    )
    metrics['macro']['loss'] = running_loss / len(train_loader)  # Use running_loss
    return metrics

def validate_epoch(model, val_loader, criterion, device, config, kl_weight=None):
    """Validate for one epoch - multi-label version"""
    model.eval()
    running_loss = 0  # Changed from total_loss
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
            
            outputs, mu, logvar = model(inputs, coords)
            loss, class_loss, kl_loss = criterion(outputs, labels, mu, logvar, kl_weight=kl_weight)

            running_loss += loss.item()  # Use .item() to get Python number
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics_multi_label(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_probs),
        config.get('species', [])
    )
    metrics['macro']['loss'] = running_loss / len(val_loader)  # Use running_loss
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
    
    annealing_epochs = config.get('annealing_epochs', 10)  # Number of epochs to anneal over
    max_kl_weight = config.get('kl_weight', 0.001)  # Maximum KL weight
    species_configs = config.get('species_configs', [])
    # Data loaders
    if config.get('use_balanced_sampler', False):
        sampler = AutoClassAwareBatchSampler(    
                        training_dataset, 
                        batch_size=batch_size,
                        oversample_factor=3,
                )
        train_loader = DataLoader(training_dataset, batch_size=batch_size, sampler=sampler, 
                                num_workers=2, pin_memory=True, drop_last=True, shuffle=True)
        print("Using balanced sampler for training data")
    else:
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=True, drop_last=True)

    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    print(f"create train_loader with {len(train_loader)} batches")
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
    
    # Scheduler
    if config.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6
        )
    
    # Metrics tracking - add per-species tracking if needed
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
        
        if epoch < annealing_epochs:
            current_kl_weight = max_kl_weight * (epoch / annealing_epochs)
        else:
            current_kl_weight = max_kl_weight
        # Training
        model.train()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config, kl_weight=current_kl_weight)
        train_metrics_macro = train_metrics['macro']
        # Validation
        model.eval()
        with torch.no_grad():
            val_metrics = validate_epoch(model, val_loader, criterion, device, config, kl_weight=current_kl_weight)
            val_metrics_macro = val_metrics['macro']
        # Learning rate scheduling
        if config.get('use_scheduler', True):
            scheduler.step(val_metrics_macro['loss'])
        
        # Track metrics - ensure we're storing Python numbers, not tensors
        for metric in metrics:
            # Convert to Python float if it's a tensor
            train_val = train_metrics_macro[metric]
            val_val = val_metrics_macro[metric]
            
            if torch.is_tensor(train_val):
                train_val = train_val.item() if train_val.numel() == 1 else train_val.cpu().numpy()
            if torch.is_tensor(val_val):
                val_val = val_val.item() if val_val.numel() == 1 else val_val.cpu().numpy()
                
            train_metrics_history[metric].append(float(train_val))
            val_metrics_history[metric].append(float(val_val))
        
        # Get the current value of the metric we're monitoring
        current_metric = val_metrics_macro[early_stopping_metric]
        
        # Convert to Python number if it's a tensor
        if torch.is_tensor(current_metric):
            current_metric = current_metric.item() if current_metric.numel() == 1 else current_metric.cpu().numpy()
        
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
            best_val_metrics = val_metrics_macro.copy()
            improvement_msg = f"✓ New best {early_stopping_metric}: {best_val_metric:.4f}"
        else:
            epochs_no_improve += 1
            improvement_msg = f"✗ No improvement for {epochs_no_improve} epochs"

        # Print detailed progress every epoch
        print(f"\nEpoch [{epoch+1}/{max_epochs}]:")
        print(f"  Train - Acc: {train_metrics_macro['accuracy']:.4f}, Loss: {train_metrics_macro['loss']:.4f}, "
              f"F1: {train_metrics_macro['f1']:.4f}, AUC: {train_metrics_macro['pr_auc']:.4f}")
        print(f"  Val   - Acc: {val_metrics_macro['accuracy']:.4f}, Loss: {val_metrics_macro['loss']:.4f}, "
              f"F1: {val_metrics_macro['f1']:.4f}, AUC: {val_metrics_macro['pr_auc']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Best {early_stopping_metric}: {best_val_metric:.4f} (epoch {best_epoch})")
        print(f"  {improvement_msg}")
        print(f"  Early stopping: {epochs_no_improve}/{early_stopping_patience}")
        
        # Check early stopping condition (only after warmup period)
        if epoch >= early_stopping_warmup and epochs_no_improve >= early_stopping_patience:
            print(f"\n🚨 Early stopping triggered after {epoch+1} epochs!")
            print(f"   Best performance at epoch {best_epoch}:")
            for metric, value in best_val_metrics.items():
                # Convert tensor to Python number if needed
                if torch.is_tensor(value):
                    value = value.item() if value.numel() == 1 else value.cpu().numpy()
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
    tresholds = 0.5
    model.eval()
    with torch.no_grad():
        final_val_metrics = validate_epoch(model, val_loader, criterion, device, config, kl_weight=current_kl_weight)

    # Ensure all metrics are converted to Python types
    for key in train_metrics_history:
        train_metrics_history[key] = [float(x) for x in train_metrics_history[key]]
        val_metrics_history[key] = [float(x) for x in val_metrics_history[key]]
    
    results = {
        'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history,
        'best_val_metric': float(best_val_metric),
        'best_epoch': best_epoch,
        'final_val_metrics': final_val_metrics,
        'stopped_early': epochs_no_improve >= early_stopping_patience,
        'epochs_completed': epoch + 1,
        'config': config  # Store config for reference
    }
    
    return model, results


class SpeciesBalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, species_configs, batch_size=32, num_batches=100, alpha=1.0):
        self.dataset = dataset
        self.species_configs = species_configs
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_species = len(species_configs)
        self.alpha = alpha

        # Group indices by species
        self.species_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, target, _ = dataset[idx]
            for s in range(self.num_species):
                if target[s] == 1:
                    self.species_to_indices[s].append(idx)

        # Compute weights (inverse prevalence ^ alpha)
        self.species_weights = {}
        for s, indices in self.species_to_indices.items():
            prevalence = len(indices) / len(dataset)
            self.species_weights[s] = 1.0 / ((prevalence + 1e-6) ** self.alpha)

        # Normalize
        total = sum(self.species_weights.values())
        for s in self.species_weights:
            self.species_weights[s] /= total

    def __iter__(self):
        species_ids = list(self.species_weights.keys())
        weights = [self.species_weights[s] for s in species_ids]

        for _ in range(self.num_batches):
            batch = set()
            # Step 1: try to cover many species
            chosen_species = np.random.choice(
                species_ids, size=min(self.batch_size, len(species_ids)),
                replace=False, p=weights
            )
            for s in chosen_species:
                idx = np.random.choice(self.species_to_indices[s])
                batch.add(idx)

            # Step 2: if not full, fill with weighted sampling
            while len(batch) < self.batch_size:
                s = np.random.choice(species_ids, p=weights)
                idx = np.random.choice(self.species_to_indices[s])
                batch.add(idx)

            yield list(batch)

    def __len__(self):
        return self.num_batches