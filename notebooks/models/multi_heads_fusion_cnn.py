import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchgeo.models import ResNet50_Weights, resnet50
from utils.multi_heads_loss import SpeciesSpecificLossManager, GatedArchitectureLossManager
from utils.model_pretraining import transfer_pretrained_weights, ClimatePretrainingCNN

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


    
class MixtureExpertClassifier(nn.Module):
    def __init__(self, input_dim, species_configs, dropout=0.3, with_init = False):
        super().__init__()
        self.num_species = len(species_configs)
        self.with_init = with_init
        # Shared expert
        self.shared_expert = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Per-species layers
        self.species_experts = nn.ModuleList()
        self.species_gates = nn.ModuleList()
        self.mixture_weights = nn.ModuleList()
        self.shared_heads = nn.ModuleList()   # <-- store linear heads here
        
        for config in species_configs:
            prevalence = config['prevalence']
            
            # Gate
            gate = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
                nn.Sigmoid()
            )
            self.species_gates.append(gate)
            
            # Expert
            if prevalence < 0.1:
                expert = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout*2),
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
                )
            else:
                expert = nn.Sequential(
                    nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
                )
            self.species_experts.append(expert)
            
            # Mixture
            mixture = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Softmax(dim=-1)
            )
            

            
            self.mixture_weights.append(mixture)
            
            # Shared head for this species
            self.shared_heads.append(nn.Linear(128, 1))  # 128 = shared_expert output dim
    
    def forward(self, x):
        shared_features = self.shared_expert(x)
        
        species_outputs = []
        for i, (gate, expert, mixture_net, shared_head) in enumerate(zip(
            self.species_gates, self.species_experts, self.mixture_weights, self.shared_heads
        )):
            # Species-specific path
            feature_weights = gate(x)
            gated_features = x * feature_weights
            specific_pred = expert(gated_features)
            
            # Shared path
            shared_pred = shared_head(shared_features)
            
            # Mixture
            weights = mixture_net(x)
            mixed_pred = weights[:, 0:1] * shared_pred + weights[:, 1:2] * specific_pred
            species_outputs.append(mixed_pred)
        
        return torch.cat(species_outputs, dim=1)
    
    
    def _initialize_mixture_weights(self, mixture_net, prevalence):
        """Initialize mixture weights to favor shared expert for rare species"""
        with torch.no_grad():
            # Get the final linear layer
            final_layer = mixture_net[2]  # nn.Linear(32, 2)
            
            # Calculate bias: higher shared weight for rare species
            shared_bias = 2.0 * (1.0 - prevalence)  # Rare species get higher bias toward shared
            specific_bias = 2.0 * prevalence        # Common species get higher bias toward specific
            
            # Initialize biases
            final_layer.bias[0] = shared_bias   # Index 0: shared expert
            final_layer.bias[1] = specific_bias # Index 1: specific expert
            
            # Initialize weights to small values so input can modulate the bias
            final_layer.weight.data.normal_(0, 0.01)

    
class SpeciesSpecificClassifier(nn.Module):
    """Classifier with separate heads for each species"""
    def __init__(self, input_dim, species_configs, dropout=0.3):
        super().__init__()
        self.species_configs = species_configs
        self.num_species = len(species_configs)
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Species-specific heads
        self.species_heads = nn.ModuleList()
        for i, config in enumerate(species_configs):
            head_layers = []
            current_dim = 256
            
            # Optional species-specific hidden layers
            hidden_dims = config.get('hidden_dims', [128])
            for hidden_dim in hidden_dims:
                head_layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim
            
            # Final output layer
            head_layers.append(nn.Linear(current_dim, 1))
            self.species_heads.append(nn.Sequential(*head_layers))
    
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Species-specific predictions
        species_outputs = []
        for head in self.species_heads:
            species_outputs.append(head(shared_features))
        
        # Stack outputs: (batch_size, num_species)
        return torch.cat(species_outputs, dim=1)


class SimpleMultiInputCNNv2WithSpeciesHeads(nn.Module):
    """Modified model with species-specific classifier heads"""
    def __init__(self, input_shapes, config):
        super().__init__()
        self.config = config
        self.branches = nn.ModuleList()
        self.flatten_sizes = []
        self.species_configs = config.get('species_configs', [])
        self.num_species = len(self.species_configs)
        self.use_latent = config.get('use_latent', False)
        
        # Build CNN branches (same as before)
        for i, dict in enumerate(input_shapes):
            shape = dict['shape']
            source = dict['source']
            
            if source == 'point_infos':
                input_dim = shape[0]
                branch = TabularFeatureEncoder(
                    input_dim,
                    config.get('tabular_hidden_dims', [128, 64, 32]),
                    config.get('dropout', 0.3)
                )
            else:
                if source == 'sentinel2':
                    weights = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS
                    model = resnet50(weights=weights)
                    model.fc = nn.Identity()
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
        
        # Calculate total feature size
        total_flatten = sum(self.flatten_sizes)
        spatial_features = 0
        if config.get('use_spatial_features', False):
            spatial_features = config.get('spatial_feature_dim', 32)
            self.spatial_encoder = SpatialFourierFeatures(spatial_features)
        
        # Latent encoding (if enabled)
        if self.use_latent:
            self.latent_dim = config.get('latent_dim', 5)
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
        
        classifier_type = config.get('species_classifier', 'species_specific')
        if classifier_type == 'mixture_expert':
            self.classifier = MixtureExpertClassifier(
                classifier_input_dim, 
                self.species_configs,
                config.get('dropout', 0.3)
            )
        else:
            # Species-specific classifier
            self.classifier = SpeciesSpecificClassifier(
                classifier_input_dim, 
                self.species_configs,
                config.get('dropout', 0.3)
            )

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
            combined = torch.cat([merged, z], dim=1)
            out = self.classifier(combined)
            return out, mu, logvar
        else:
            out = self.classifier(merged)
            return out, None, None

    def get_loss_function(self, device=None):
        """Get simplified loss manager"""
        loss_manager = self.config.get('loss_manager', 'gated_architecture')
        if loss_manager == 'gated_architecture':
            loss_manager = GatedArchitectureLossManager(
                self.species_configs, 
                device, 
            )
        else:
            loss_manager = SpeciesSpecificLossManager(
                self.species_configs, 
                device, 
            )
        
        # Simple wrapper
        def loss_function(logits, targets, mu=None, logvar=None, kl_weight=None):
            if not self.use_latent:
                loss, species_losses = loss_manager.compute_loss(logits, targets)
                return loss, loss.item(), torch.tensor(0.0, device=device)
            else:
                classification_loss, _ = loss_manager.compute_loss(logits, targets)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                total_loss = classification_loss + kl_weight * kl_loss
                return total_loss, classification_loss.item(), kl_loss.item()
        
        return loss_function