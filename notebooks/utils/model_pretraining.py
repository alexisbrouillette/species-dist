import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from utils.dataset import MultiSourceNpyDatasetWithCoords  # Assuming this is defined elsewhere
#from fusion_cnn import FlexibleCNNBranch, SpatialPooling  # Assuming this is defined elsewhere

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
    
class ClimateMaskingDatasetV2(Dataset):
    """Dataset for climate pretraining with variable masking - works with your existing dataset"""
    
    def __init__(self, base_dataset, climate_idx=0, mask_ratio=0.3, variable_groups=None):
        """
        base_dataset: your MultiSourceNpyDatasetWithCoords instance
        climate_idx: index of climate data in the dataset output (usually 0)
        mask_ratio: fraction of variables to mask (0.2-0.4 works well)
        variable_groups: optional dict grouping related variables for structured masking
        """
        self.base_dataset = base_dataset
        self.climate_idx = climate_idx
        self.mask_ratio = mask_ratio
        
        # Get number of variables from the dataset's effective shapes
        climate_shape = None
        for shape_info in base_dataset.effective_shapes:
            if 'pred100' in shape_info['source'] or 'climate' in shape_info['source']:
                climate_shape = shape_info['shape']
                print(f"Identified climate shape from source '{shape_info['source']}': {climate_shape}")
                break
        
        if climate_shape is None:
            # Fallback - assume first shape is climate
            climate_shape = base_dataset.effective_shapes[0]['shape']
        
        self.n_variables = climate_shape[0]  # Should be 67
        print(f"Climate masking dataset initialized with {self.n_variables} variables")
        
        # Define variable groups for more structured masking
        if variable_groups is None:
            # Adjust groups based on actual number of variables
            max_var = min(66, self.n_variables - 1)  # Ensure we don't exceed available variables
            
            self.variable_groups = {
                'climate': [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(26, min(40, self.n_variables))) + [48, 49, 58] if i < self.n_variables],
                'soil': [i for i in [4, 5, 41, 42, 43, 54, 55, 56] if i < self.n_variables],
                'topography': [i for i in [13, 50, 15, 16] if i < self.n_variables],
                'landcover': [i for i in list(range(7, 12)) + [23, 24, 38, 44, 45, 52, 53, 56, 57, 59, 60, 61, 62, 63, 65, 66] if i < self.n_variables],
                'geology': [i for i in [0, 3, 11, 14, 17, 18, 19, 20, 47, 51] if i < self.n_variables],
                'human': [i for i in [21, 25] if i < self.n_variables],
            }
            # Remove empty groups
            self.variable_groups = {k: v for k, v in self.variable_groups.items() if v}
        else:
            self.variable_groups = variable_groups
            
        print(f"Variable groups: {[(k, len(v)) for k, v in self.variable_groups.items()]}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get data from your base dataset
        data_items = self.base_dataset[idx]
        
        # Extract climate patch (assuming it's at climate_idx)
        if isinstance(data_items, (list, tuple)):
            original = data_items[self.climate_idx]
        else:
            original = data_items
            
        # Make sure it's a tensor and clone it
        if not isinstance(original, torch.Tensor):
            # If it's a list, convert to numpy array first
            if isinstance(original, list):
                original = np.array(original)
            original = torch.from_numpy(original.copy())  # Ensure we have a copy
        else:
            original = original.clone()
        
        masked = original.clone()
        
        # Create mask - which variables to mask
        n_to_mask = max(1, int(self.n_variables * self.mask_ratio))
        
        # Generate mask indices
        if random.random() < 0.5:
            # Random masking
            mask_indices = np.random.choice(self.n_variables, n_to_mask, replace=False)
        else:
            # Group-aware masking
            mask_indices = []
            if self.variable_groups and random.random() < 0.3:
                # 30% chance to mask entire group
                group_name = random.choice(list(self.variable_groups.keys()))
                available_indices = self.variable_groups[group_name]
                n_to_mask_from_group = min(len(available_indices), n_to_mask)
                mask_indices = np.random.choice(available_indices, n_to_mask_from_group, replace=False)
            else:
                # Mixed masking from different groups
                if self.variable_groups:
                    remaining = n_to_mask
                    for group_name, group_indices in self.variable_groups.items():
                        if remaining <= 0:
                            break
                        n_from_group = min(len(group_indices), max(1, remaining // max(1, len(self.variable_groups))))
                        if n_from_group > 0:
                            selected = np.random.choice(group_indices, min(n_from_group, len(group_indices)), replace=False)
                            mask_indices.extend(selected)
                            remaining -= len(selected)
                    # Fill remaining with random selection
                    if remaining > 0:
                        all_indices = set(range(self.n_variables))
                        used_indices = set(mask_indices)
                        available = list(all_indices - used_indices)
                        if available:
                            additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                            mask_indices.extend(additional)
                else:
                    # Fallback to random if no groups
                    mask_indices = np.random.choice(self.n_variables, n_to_mask, replace=False)
        
        # Convert to list and ensure uniqueness
        mask_indices = list(set(mask_indices))[:n_to_mask]
        
        # Apply masking (set to zero)
        mask_tensor = torch.zeros_like(original)
        for idx_to_mask in mask_indices:
            masked[idx_to_mask] = 0  
            mask_tensor[idx_to_mask] = 1
        
        return {
            'masked_input': masked,
            'target': original,
            'mask': mask_tensor
        }
    
class ClimatePretrainingCNN(nn.Module):
    """CNN for climate variable reconstruction pretraining with exact output size"""

    def __init__(self, n_variables=67, config=None):
        super().__init__()
        if config is None:
            config = {}

        dropout_rate = config.get('dropout', 0.3)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_variables, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.5),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.7),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )

        # Decoder using Upsample for exact reconstruction
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.7),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),

            nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False),
            nn.Conv2d(32, n_variables, kernel_size=3, padding=1)
        )

        # Feature extraction (optional)
        self.use_spp = config.get('use_spp', True)
        if self.use_spp:
            pool_sizes = config.get('spp_pool_sizes', [1, 2, 4])
            self.spp = SpatialPyramidPooling(pool_sizes)
            self.output_size = 128 * sum(size * size for size in pool_sizes)
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            self.output_size = 128 * 16

    def forward(self, x, return_reconstruction=True):
        encoded = self.encoder(x)

        if return_reconstruction:
            reconstructed = self.decoder(encoded)
            return reconstructed
        else:
            if self.use_spp:
                features = self.spp(encoded)
            else:
                features = self.adaptive_pool(encoded)
                features = features.view(features.size(0), -1)
            return features


from tqdm import tqdm  # Add this at the top

def pretrain_climate_cnn_v2(base_dataset, config=None, num_epochs=100, climate_idx=0):
    if config is None:
        config = {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'mask_ratio': 0.3,
            'dropout': 0.3,
            'use_spp': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    device = config['device']
    print(f"Using device: {device}")

    dataset = ClimateMaskingDatasetV2(base_dataset, climate_idx=climate_idx, mask_ratio=config['mask_ratio'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    n_variables = dataset.n_variables
    print(f"Initializing ClimatePretrainingCNN with {n_variables} variables")

    model = ClimatePretrainingCNN(n_variables=n_variables, config=config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.train()
    print("Starting training...")

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Use tqdm for batch progress
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for batch in tepoch:
                try:
                    masked_input = batch['masked_input'].to(device, non_blocking=True)
                    target = batch['target'].to(device, non_blocking=True)
                    mask = batch['mask'].to(device, non_blocking=True)

                    optimizer.zero_grad()
                    reconstructed = model(masked_input, return_reconstruction=True)

                    loss = criterion(reconstructed, target)
                    masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)

                    masked_loss.backward()
                    optimizer.step()

                    total_loss += masked_loss.item()
                    num_batches += 1

                    tepoch.set_postfix(loss=masked_loss.item())

                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs} completed, Avg Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return model

def transfer_pretrained_weights(pretrained_model, target_model_branch):
    """
    Transfer pretrained encoder weights to your main fusion model's climate branch
    
    pretrained_model: ClimatePretrainingCNN
    target_model_branch: FlexibleCNNBranch from your fusion model
    """
    pretrained_encoder = pretrained_model.encoder
    
    # Build target encoder from your FlexibleCNNBranch structure
    target_layers = []
    if hasattr(target_model_branch, 'conv1'):
        target_layers.append(target_model_branch.conv1)
    if hasattr(target_model_branch, 'conv2'):
        target_layers.append(target_model_branch.conv2)
    if hasattr(target_model_branch, 'conv3'):
        target_layers.append(target_model_branch.conv3)
    
    target_encoder = nn.Sequential(*target_layers)
    
    # Transfer weights layer by layer
    pretrained_state = pretrained_encoder.state_dict()
    target_state = target_encoder.state_dict()
    
    # Map the weights
    transferred = 0
    for target_key, target_param in target_state.items():
        if target_key in pretrained_state:
            if target_param.shape == pretrained_state[target_key].shape:
                target_param.copy_(pretrained_state[target_key])
                transferred += 1
                print(f"Transferred: {target_key}")
            else:
                print(f"Shape mismatch for {target_key}: {target_param.shape} vs {pretrained_state[target_key].shape}")
    
    print(f"Successfully transferred {transferred} parameter tensors")
    return transferred > 0
    

