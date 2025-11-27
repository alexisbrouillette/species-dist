import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveAsymmetricLoss(nn.Module):
    """
    Asymmetric loss with adaptive gamma based on training dynamics
    """
    def __init__(self, gamma_pos=0, gamma_neg=1, clip=0.05, pos_weight=1.0, 
                 adaptation_rate=0.01, target_precision=None, target_recall=None):
        super().__init__()
        self.gamma_pos_init = gamma_pos
        self.gamma_neg_init = gamma_neg
        self.clip = clip
        self.adaptation_rate = adaptation_rate
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.eps = 1e-8  # Add epsilon for numerical stability
        
        # Make gamma parameters learnable
        self.gamma_pos = nn.Parameter(torch.tensor(gamma_pos, dtype=torch.float32))
        self.gamma_neg = nn.Parameter(torch.tensor(gamma_neg, dtype=torch.float32))
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        
        # Track performance for adaptation
        self.running_precision = 0.5
        self.running_recall = 0.5
        
    def forward(self, logits, targets):
        # Add NaN check
        if torch.isnan(logits).any():
            print("WARNING: NaN detected in logits")
            logits = torch.nan_to_num(logits, nan=0.0)
        probs = torch.sigmoid(logits)
        # Add clipping to prevent extreme values
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        # Asymmetric focusing
        p_t = torch.where(targets == 1, probs, 1 - probs)
        
        # Different gamma for positive and negative
        gamma = torch.where(targets == 1, self.gamma_pos, self.gamma_neg)
        
        # Asymmetric focusing weight
        focal_weight = torch.pow(1 - p_t, gamma)
        
        # Apply clipping for numerical stability
        probs_clipped = torch.clamp(probs, min=self.clip, max=1-self.clip)
        
        # Binary cross entropy with focal weight
        bce = -(targets * torch.log(probs_clipped) * self.pos_weight + 
                (1 - targets) * torch.log(1 - probs_clipped))

        loss = focal_weight * bce
        
        if torch.isnan(loss).any():
            print("WARNING: NaN in loss, returning zero loss")
            return torch.tensor(0.0, requires_grad=True, device=loss.device)
            
        return loss.mean()
    
    def update_metrics(self, precision, recall):
        """Update running metrics for dynamic adaptation"""
        self.running_precision = 0.9 * self.running_precision + 0.1 * precision
        self.running_recall = 0.9 * self.running_recall + 0.1 * recall
        
        # Adapt gammas based on performance
        if self.target_precision and precision < self.target_precision:
            # Increase gamma_neg to reduce false positives
            self.gamma_neg.data += self.adaptation_rate
        elif self.target_recall and recall < self.target_recall:
            # Increase gamma_pos to reduce false negatives
            self.gamma_pos.data += self.adaptation_rate


class CalibrationLoss(nn.Module):
    """
    Loss that includes calibration penalty to prevent overconfident predictions
    """
    def __init__(self, base_loss_fn, calibration_weight=0.1, num_bins=10):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.calibration_weight = calibration_weight
        self.num_bins = num_bins
        
    def forward(self, logits, targets):
        base_loss = self.base_loss_fn(logits, targets)
        
        # Expected Calibration Error (ECE) as additional loss
        probs = torch.sigmoid(logits)
        ece = self.compute_ece(probs, targets)
        
        return base_loss + self.calibration_weight * ece
    
    def compute_ece(self, probs, targets):
        """Compute Expected Calibration Error"""
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        ece = 0
        
        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (targets[in_bin]).float().mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece


class ThresholdAwareLoss(nn.Module):
    """
    Loss that explicitly optimizes for a target decision threshold
    """
    def __init__(self, base_loss_fn, target_threshold=0.5, threshold_weight=0.1):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.target_threshold = target_threshold
        self.threshold_weight = threshold_weight
        
    def forward(self, logits, targets):
        base_loss = self.base_loss_fn(logits, targets)
        
        probs = torch.sigmoid(logits)
        
        # Penalize predictions that are on the wrong side of threshold
        threshold_loss = 0
        
        # For positive samples, penalize if prob < threshold
        pos_mask = (targets == 1).float()
        if pos_mask.sum() > 0:
            pos_distance = torch.relu(self.target_threshold - probs) * pos_mask
            threshold_loss += pos_distance.sum() / pos_mask.sum()
        
        # For negative samples, penalize if prob > threshold  
        neg_mask = (targets == 0).float()
        if neg_mask.sum() > 0:
            neg_distance = torch.relu(probs - self.target_threshold) * neg_mask
            threshold_loss += neg_distance.sum() / neg_mask.sum()
        
        return base_loss + self.threshold_weight * threshold_loss
    

class SpeciesSpecificLossManager:
    def __init__(self, species_configs, device, epoch=0):
        self.species_configs = species_configs
        self.device = device
        self.epoch = epoch
        self.loss_functions = self._build_optimal_losses()
        
    def _build_optimal_losses(self):
        loss_functions = []
        
        for config in self.species_configs:
            species_name = config['name']
            prevalence = config.get('prevalence', 0.5)
            pos_weight = config.get('pos_weight', 1.0)

            print(f"\n{species_name}: prevalence={prevalence:.3f}")
            
            # Handle ultra/common species with your existing rules
            if prevalence > 0.95:  # Ultra-common like RES_S
                loss_fn = ThresholdAwareLoss(
                    base_loss_fn=AdaptiveAsymmetricLoss(
                        gamma_pos=2.0,
                        gamma_neg=0.5,
                        pos_weight=1.0,
                        clip=0.05,
                        target_recall=0.7
                    ),
                    target_threshold=0.3,
                    threshold_weight=0.2
                )
                print(f"  → ThresholdAwareLoss: forcing recall improvement")
                
            elif prevalence > 0.7:  # Common species
                loss_fn = CalibrationLoss(
                    base_loss_fn=AdaptiveAsymmetricLoss(
                        gamma_pos=1.5,
                        gamma_neg=1.0,
                        target_recall=0.8
                    ),
                    calibration_weight=0.1
                )
                print(f"  → CalibrationLoss: balanced with calibration")
                
            elif prevalence > 0.3:  # Medium prevalence
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
                print(f"  → BCEWithLogitsLoss: balanced")
                
            elif prevalence > 0.15:  # Rare species
                loss_fn = AdaptiveAsymmetricLoss(
                    gamma_pos=0.3,
                    gamma_neg=4.0,
                    pos_weight=pos_weight * 0.5,
                    target_precision=0.7
                )
                print(f"  → AdaptiveAsymmetricLoss: precision-focused")
                
            else:# Very rare species
                # This is a precision-focused setup, similar to your "Rare" block
                loss_fn = AdaptiveAsymmetricLoss(
                    gamma_pos=0.5,        # Less focus on positives
                    gamma_neg=3.0,        # More focus on penalizing FPs
                    pos_weight=min(25, 1/prevalence), # Use a more standard pos_weight
                    clip=0.01,
                    target_precision=0.5  # Target precision, not recall
                )
                
                print(f"  → CalibrationLoss: conservative fallback")
            
            loss_fn.to(self.device)
            loss_functions.append(loss_fn)
            
        return loss_functions

    def compute_loss_weighted(self, logits, targets):
        """
        Just average BCE across species, no weighting.
        """
        batch_size, num_species = targets.shape
        species_losses = []

        for i, loss_fn in enumerate(self.loss_functions):
            species_logits = logits[:, i].unsqueeze(1)
            species_targets = targets[:, i].unsqueeze(1)

            loss = loss_fn(species_logits, species_targets)
            species_losses.append(loss)

        # Simple mean
        total_loss = torch.stack(species_losses).mean()
        return total_loss, [l.item() for l in species_losses]

    def _get_dynamic_weight(self, prevalence, idx):
        """
        Give higher weights to rare species and lower to common ones.
        Smoothly scales between 0.5 and 3.0 based on prevalence.
        """
        # Clamp prevalence to avoid extreme weights
        prevalence = max(min(prevalence, 0.99), 1e-3)
        # Inverse proportionality with mild scaling
        weight = (1.0 / (prevalence + 1e-4)) ** 0.3
        # Optional: normalize roughly around 1
        weight = max(0.5, min(weight, 3.0))
        return weight

    def compute_loss(self, logits, targets, current_metrics=None):
        """
        Compute loss with optional metric-based adaptation
        curreloss_fn = ThresholdAwareLoss(
    base_loss_fn=AdaptiveAsymmetricLoss(
        gamma_pos=1.5,        # keep rare positives alive
        gamma_neg=1.5,        # balanced
        pos_weight=1/prevalence,  # proportional to rarity
        clip=0.03,
        target_precision=0.5,  # don’t force high precision early
        target_recall=0.7
    ),
    target_threshold=0.4,
    threshold_weight=0.15
)nt_metrics: dict with species names as keys and (precision, recall) tuples as values
        """
        total_loss = 0
        species_losses = []
        
        for i, loss_fn in enumerate(self.loss_functions):
            species_logits = logits[:, i].unsqueeze(1)
            species_targets = targets[:, i].unsqueeze(1)
            
            # Update adaptive losses with current metrics if available
            if current_metrics and hasattr(loss_fn, 'update_metrics'):
                species_name = self.species_configs[i]['name']
                if species_name in current_metrics:
                    precision, recall = current_metrics[species_name]
                    loss_fn.update_metrics(precision, recall)
            
            # Compute weighted loss
            prevalence = self.species_configs[i]['prevalence']
            weight = self._get_dynamic_weight(prevalence, i)
            
            loss = loss_fn(species_logits, species_targets) * weight
            total_loss += loss
            species_losses.append(loss.item())
        
        return total_loss / len(self.loss_functions), species_losses
    
            
class GatedArchitectureLossManager(SpeciesSpecificLossManager):
    """Specialized loss manager optimized for gated architecture"""
    
    def _build_optimal_losses(self):
        loss_functions = []
        
        for config in self.species_configs:
            prevalence = config.get('prevalence', 0.5)
            pos_weight = config.get('pos_weight', 1.0)
            
            # Optimized for gated architecture based on your results
            if prevalence < 0.05:
                loss_fn = CalibrationLoss(
                    base_loss_fn=AdaptiveAsymmetricLoss(
                        gamma_pos=3.0, gamma_neg=0.3,
                        pos_weight=min(100, 5/prevalence),
                        clip=0.01, target_recall=0.9
                    ),
                    calibration_weight=0.2
                )
            elif prevalence < 0.1:
                loss_fn = AdaptiveAsymmetricLoss(
                    gamma_pos=2.5, gamma_neg=0.5,
                    pos_weight=min(50, 3/prevalence),
                    clip=0.02, target_recall=0.85
                )
            elif prevalence < 0.3:
                loss_fn = AdaptiveAsymmetricLoss(
                    gamma_pos=1.5, gamma_neg=1.5,
                    pos_weight=pos_weight, clip=0.03
                )
            else:
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            
            loss_fn.to(self.device)
            loss_functions.append(loss_fn)
            
        return loss_functions      
    
    
