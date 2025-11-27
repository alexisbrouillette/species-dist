import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchgeo.models import ResNet50_Weights, resnet50
from utils.multi_heads_loss import SpeciesSpecificLossManager, GatedArchitectureLossManager
from utils.model_pretraining import transfer_pretrained_weights, ClimatePretrainingCNN


# ==================== MIXTURE EXPERT CLASSIFIER ====================
class MixtureExpertClassifier_v2(nn.Module):
    def __init__(self, input_dim, species_configs, dropout=0.0, mode="mixture"):
        super().__init__()
        self.num_species = len(species_configs)
        self.input_dim = input_dim
        self.mode = mode
        assert mode in ["mixture", "shared", "specific"], "Invalid mode for MixtureExpertClassifier"
        
        # Shared expert - more robust architecture
        self.shared_expert = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, len(species_configs))
        )
        
        # Per-species layers
        self.species_experts = nn.ModuleList()
        self.species_gates = nn.ModuleList()
        self.mixture_weights = nn.ModuleList()
        
        # Temperature factors for adaptive mixing (learnable)
        self.temperature_factors = nn.ParameterList()
        
        for config in species_configs:
            prevalence = config['prevalence']
            
            # Feature gating mechanism
            gate = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, input_dim),
                nn.Sigmoid()
            )
            self.species_gates.append(gate)
            
            # Prevalence-adaptive expert architecture
            expert = self._build_expert_architecture(input_dim, prevalence, dropout)
            self.species_experts.append(expert)
            
            # Mixture network with prevalence-based initialization
            mixture = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout*0.5),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 2)  # Output logits for 2 experts
            )
            
            # Initialize mixture weights based on prevalence
            
            self._initialize_mixture_weights(mixture, prevalence)
            self.mixture_weights.append(mixture)

            
            # Learnable temperature factor for adaptive mixing
            initial_temp = self._get_initial_temperature(prevalence)
            self.temperature_factors.append(nn.Parameter(torch.tensor(initial_temp)))
        
        # Register temperature factors properly
        for i, temp_param in enumerate(self.temperature_factors):
            self.register_parameter(f'temperature_{i}', temp_param)
    
    def _build_expert_architecture(self, input_dim, prevalence, dropout):
        """Build species-specific expert based on prevalence"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    
    def _initialize_mixture_weights(self, mixture_net, prevalence):
        """Nuanced initialization based on prevalence ranges"""
        with torch.no_grad():
            # Get the final linear layer (index 6 in the sequential)
            final_layer = mixture_net[6]  # nn.Linear(32, 2)
            
            if prevalence < 0.1:  # Ultra rare: strongly favor shared (80/20)
                shared_bias = 3.0
                specific_bias = 0.5
            elif prevalence < 0.25:  # Rare: moderately favor shared (70/30)
                shared_bias = 2.0
                specific_bias = 1.0
            elif prevalence > 0.75:  # Ultra common: favor specific expert (30/70)
                shared_bias = 0.5
                specific_bias = 2.5
            else:  # Average: balanced (50/50)
                shared_bias = 1.5
                specific_bias = 1.5
            
            # Initialize biases
            final_layer.bias[0] = shared_bias   # Shared expert bias
            final_layer.bias[1] = specific_bias # Specific expert bias
            
            # Initialize weights with small values
            final_layer.weight.data.normal_(0, 0.02)
            
            # Also initialize earlier layers
            mixture_net[0].weight.data.normal_(0, 0.02)  # First linear
            mixture_net[4].weight.data.normal_(0, 0.02)  # Second linear
    
    def _get_initial_temperature(self, prevalence):
        """Get initial temperature for adaptive mixing"""
        if prevalence < 0.1:  # Ultra rare: high temperature -> more uniform -> more shared
            return 2.0
        elif prevalence < 0.25:  # Rare: moderate temperature
            return 1.5
        elif prevalence > 0.75:  # Ultra common: low temperature -> more peaked -> more specific
            return 0.5
        else:  # Average: neutral
            return 1.0
    
    def forward(self, x):
        batch_size = x.size(0)
        all_shared_preds = self.shared_expert(x)
        
        species_outputs = []
        mixture_weights_log = []
        shared_usage_log = []
        
        for i, (gate, expert, mixture_net, temp_param) in enumerate(zip(
            self.species_gates, self.species_experts, self.mixture_weights, 
            self.temperature_factors
        )):
            # Species-specific path with feature gating
            feature_weights = gate(x)
            gated_features = x * feature_weights
            specific_pred = expert(gated_features)
            shared_pred = all_shared_preds[:, i:i+1]
            
            
            # Combine predictions
            if(self.mode == "shared"):
                weights = torch.tensor([[1., 0.]], device=x.device).repeat(batch_size, 1)
                mixed_pred = shared_pred
            elif(self.mode == "specific"):
                weights = torch.tensor([[0., 1.]], device=x.device).repeat(batch_size, 1)
                mixed_pred = specific_pred
            else:  # mixture
                mixture_logits = mixture_net(x)
                temperature = torch.clamp(temp_param, min=0.1, max=5.0)
                scaled_logits = mixture_logits / temperature
                weights = F.softmax(scaled_logits, dim=-1)
                # Log for analysis
                mixture_weights_log.append(weights.detach())
                shared_usage_log.append(weights[:, 0].mean().detach())  # Shared expert usage
                
                mixed_pred = weights[:, 0:1] * shared_pred + weights[:, 1:2] * specific_pred
            species_outputs.append(mixed_pred)
        
        # Store for monitoring and analysis
        if len(mixture_weights_log) > 0:
            self.last_mixture_weights = torch.stack(mixture_weights_log)  # [num_species, batch_size, 2]
            self.last_shared_usage = torch.stack(shared_usage_log)
        else:
            self.last_mixture_weights = None
            self.last_shared_usage = None
        
        return torch.cat(species_outputs, dim=1)
    
    def get_mixture_statistics(self):
        """Get statistics about mixture weight usage"""
        if getattr(self, "last_mixture_weights", None) is None:
            return {
                "message": f"No mixture weights available (mode = '{self.mode}')",
                "mean_shared_usage": None,
                "per_species_usage": None,
            }

        # If mixture weights exist, compute stats safely
        mean_shared_usage = self.last_shared_usage.mean().item()
        per_species_usage = self.last_shared_usage.detach().cpu().numpy()

        return {
            "message": f"Mixture weights summary (mode = '{self.mode}')",
            "mean_shared_usage": mean_shared_usage,
            "per_species_usage": per_species_usage,
            "shape": tuple(self.last_mixture_weights.size()),
        }
    
    def get_temperature_values(self):
        """Get current temperature values for all species"""
        return [temp_param.item() for temp_param in self.temperature_factors]
 


class MixtureExpertClassifier(nn.Module):
    def __init__(self, input_dim, species_configs, dropout=0.3, with_init=False, mode="mixture"):
        """
        mode: one of ["mixture", "shared", "specific"]
        """
        super().__init__()
        
        self.num_species = len(species_configs)
        self.with_init = with_init
        self.input_dim = input_dim
        self.mode = mode
        assert mode in ["mixture", "shared", "specific"], "Invalid mode for MixtureExpertClassifier"
        # Shared expert
        self.shared_output_dim = 256
        
        self.shared_expert = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(512, self.shared_output_dim),
            nn.LeakyReLU(0.1),
        )

        # Per-species modules
        self.species_experts = nn.ModuleList()
        self.mixture_weights = nn.ModuleList()
        self.shared_heads = nn.ModuleList()

        for config in species_configs:
            prevalence = float(config.get('prevalence', 0.01))

            # Per-species expert (size depends on prevalence)
            if prevalence < 0.1:
                expert = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout * 2),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 1)
                )
            elif prevalence < 0.25:
                expert = nn.Sequential(
                    nn.Linear(input_dim, 96),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout * 1.5),
                    nn.Linear(96, 48),
                    nn.ReLU(inplace=True),
                    nn.Linear(48, 1)
                )
            else:
                expert = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1)
                )
            self.species_experts.append(expert)

            # Mixture net (used only if mode == "mixture")
            mixture_seq = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 2)
            )

            if self.with_init:
                self._initialize_mixture_weights(mixture_seq, prevalence)

            self.mixture_weights.append(nn.Sequential(mixture_seq, nn.Softmax(dim=1)))

            # Shared head
            self.shared_heads.append(nn.Linear(self.shared_output_dim, 1))

    def _initialize_mixture_weights(self, mixture_seq, prevalence):
        final_layer = mixture_seq[-1]
        if not isinstance(final_layer, nn.Linear):
            return
        with torch.no_grad():
            shared_bias = 3.0 * (1.0 - prevalence)
            specific_bias = 3.0 * prevalence
            final_layer.bias.data.zero_()
            final_layer.bias.data[0] = shared_bias
            final_layer.bias.data[1] = specific_bias
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)

    def forward(self, x):
        shared_features = self.shared_expert(x)  # (batch, shared_output_dim)
        species_outputs = []

        for expert, mixture_net, shared_head in zip(
            self.species_experts, self.mixture_weights, self.shared_heads
        ):
            specific_pred = expert(x)
            shared_pred = shared_head(shared_features)

            if self.mode == "shared":
                pred = shared_pred
            elif self.mode == "specific":
                pred = specific_pred
            else:  # "mixture"
                weights = mixture_net(x)
                w_shared = weights[:, 0:1]
                w_specific = weights[:, 1:2]
                pred = w_shared * shared_pred + w_specific * specific_pred

            species_outputs.append(pred)

        return torch.cat(species_outputs, dim=1)



# ==================== SUPPORTING MODULES ====================
class TabularFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_size = current_dim
    
    def forward(self, x):
        return self.network(x)

class FlexibleCNNBranch(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        # Simplified CNN branch - adapt as needed
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.output_size = 128
    
    def forward(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1)

# ==================== MAIN MODEL ====================
class SimpleMultiInputCNNv2WithMixtureExperts(nn.Module):
    def __init__(self, input_shapes, config):
        super().__init__()
        self.config = config
        self.branches = nn.ModuleList()
        self.flatten_sizes = []
        self.species_configs = config.get('species_configs', [])
        self.num_species = len(self.species_configs)
        self.use_latent = config.get('use_latent', False)
        
        # Build CNN branches (your existing code)
        for i, dict in enumerate(input_shapes):
            shape = dict['shape']
            source = dict['source']
            print(f"Building branch {i} for source '{source}' with shape {shape}")
            
            if source == 'point_infos':  # Tabular branch
                input_dim = shape[0]
                branch = TabularFeatureEncoder(
                    input_dim,
                    config.get('tabular_hidden_dims', [128, 64, 32]),
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
                    # Add pretrained climate CNN loading if needed
                    if config.get('use_pretrained_climate_cnn', False) and source == 'pred100':
                        pretrained_model = ClimatePretrainingCNN()
                        pretrained_model.load_state_dict(torch.load("climate_pretrained_cnn.pth", map_location=config['device']))
                        transfer_pretrained_weights(pretrained_model, branch)
                        pass
            
            self.branches.append(branch)
            self.flatten_sizes.append(branch.output_size)
        
        if config.get('freeze_backbones', False):
            print("Freezing backbone weights")
            for branch in self.branches:
                for param in branch.parameters():
                    param.requires_grad = False
        
        # Calculate total feature size
        total_flatten = sum(self.flatten_sizes)
        spatial_features = 0
        if config.get('use_spatial_features', False):
            spatial_features = config.get('spatial_feature_dim', 32)
        
        # Latent encoding (if enabled)
        if self.use_latent:
            self.latent_dim = config.get('latent_dim', 5)
            self.latent_encoder = nn.Sequential(
                nn.Linear(total_flatten + spatial_features, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Dropout(0.3)
            )
            self.latent_mu = nn.Linear(64, self.latent_dim)
            self.latent_logvar = nn.Linear(64, self.latent_dim)
            classifier_input_dim = total_flatten + spatial_features + self.latent_dim
        else:
            classifier_input_dim = total_flatten + spatial_features
        
        # Build the classifier - now using mixture experts
        self.build_classifier(classifier_input_dim, config)

    def build_classifier(self, input_dim, config):
        """Build classifier with mixture experts or fallback to common heads"""
        classifier_type = config.get('species_classifier', 'mixture_expert')
        
        if classifier_type == 'mixture_expert' and self.species_configs:
            print("Using MixtureExpertClassifier with prevalence-aware initialization")
            self.classifier = MixtureExpertClassifier(
                input_dim, 
                self.species_configs,
                config.get('dropout', 0.3),
                mode = config.get('mode', 'mixture')
            )
        elif classifier_type == 'mixture_expert_with_init':
            self.classifier = MixtureExpertClassifier(
                input_dim, 
                self.species_configs,
                config.get('dropout', 0.3),
                with_init = True
            )
        elif classifier_type == 'mixture_expert_v2':
            self.classifier = MixtureExpertClassifier_v2(
                input_dim, 
                self.species_configs,
                config.get('dropout', 0.3),
                mode = config.get('mode', 'mixture')
            )
        else:
            # Fallback to common heads (your original implementation)
            print("Using common heads classifier")
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
            spatial_features = coords
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
        loss_manager = self.config.get('loss_manager', 'species_specific')
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
            if self.use_latent and mu is not None and logvar is not None:
                classification_loss, _ = loss_manager.compute_loss(logits, targets)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                total_loss = classification_loss + kl_weight * kl_loss
                return total_loss, classification_loss.item(), kl_loss.item()
            else:
                if self.config.get("compute_weighted_loss", False):
                    loss, species_losses = loss_manager.compute_loss_weighted(logits, targets)
                    return loss, loss.item(), torch.tensor(0.0, device=device)
                else:
                    loss, species_losses = loss_manager.compute_loss(logits, targets)
                    return loss, loss.item(), torch.tensor(0.0, device=device)
        
        return loss_function