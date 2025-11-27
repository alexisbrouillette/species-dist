import torch
import numpy as np
import random
from torch.utils.data import Sampler

import torch.nn.functional as F
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import BoundingBox
from typing import Any
from matplotlib import pyplot as plt

import os
import psutil
import concurrent.futures
import itertools
import tqdm
from sklearn.neighbors import NearestNeighbors

from utils.model_preprocess import create_target_species_df_multiple, create_target_species_df

def get_available_memory_gb():
    """Return available system memory in GB."""
    mem = psutil.virtual_memory()
    return mem.available / (1024**3)

def get_file_size_gb(path):
    """Get size of file in GB."""
    return os.path.getsize(path) / (1024**3)

def load_patches_auto(npy_file, as_float=True):
    """
    Load patches into RAM if memory allows; otherwise use np.memmap (safe for raw binary files).
    """
    import psutil, json, os, numpy as np

    # Try to get shape from metadata or infer it
    metadata_file = npy_file.replace('.npy', '_metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        shape = tuple(metadata['shape'])
    else:
        # Try reading header (works only if true .npy)
        try:
            tmp = np.load(npy_file, mmap_mode='r')
            shape = tmp.shape
            del tmp
        except Exception as e:
            raise ValueError(
                f"Cannot determine shape for {npy_file}. "
                f"Please provide metadata JSON with 'shape'. Error: {e}"
            )

    # Estimate RAM need
    n_bytes = np.prod(shape)  # number of elements
    bytes_per_el = 1  # uint8
    required_gb = (n_bytes * bytes_per_el) / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)

    print(f"[INFO] {os.path.basename(npy_file)}: requires ~{required_gb:.2f} GB "
          f"(available RAM: {available_gb:.1f} GB)")

    # Decision rule
    load_into_memory = required_gb < 0.7 * available_gb

    if load_into_memory:
        print(f"[INFO] Loading {os.path.basename(npy_file)} fully into RAM.")
        data = np.fromfile(npy_file, dtype=np.uint8).reshape(shape)
        mmap_mode = None
    else:
        print(f"[INFO] Using memmap for {os.path.basename(npy_file)}.")
        data = np.memmap(npy_file, dtype=np.uint8, mode='r', shape=shape)
        mmap_mode = 'r'

    # Handle float conversion if requested
    if as_float:
        if mmap_mode is None:
            return data.astype(np.float32) / 255.0, shape
        else:
            class Float32View:
                def __init__(self, uint8_data):
                    self._data = uint8_data
                    self.shape = uint8_data.shape
                    self.dtype = np.float32
                def __getitem__(self, key):
                    return self._data[key].astype(np.float32) / 255.0
                def __array__(self):
                    return self._data.astype(np.float32) / 255.0
            return Float32View(data), shape

    return data, shape



def load_patches(npy_file, as_float=True, mmap_mode='r'):
    """
    Load uint8 patches with automatic shape detection.

    Parameters
    ----------
    npy_file : str
        Path to .npy file
    as_float : bool
        If True, convert to float32 [0,1]. If False, keep as uint8 [0,255]
    mmap_mode : str or None
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
        shape = tuple(metadata['shape'])
        if mmap_mode is None:
            # Load all data into memory
            data = np.fromfile(npy_file, dtype=np.uint8).reshape(shape)
        else:
            data = np.memmap(npy_file, dtype=np.uint8, mode=mmap_mode, shape=shape)
        print(f"Loaded with shape: {metadata['shape']}")
    else:
        # Fallback: try to infer or load normally
        try:
            data = np.load(npy_file, mmap_mode=mmap_mode)
            shape = data.shape
        except:
            raise ValueError(f"Cannot determine shape. Please provide metadata file: {metadata_file}")

    if as_float:
        if mmap_mode is None:
            # Load into memory and convert
            return data.astype(np.float32) / 255.0, shape
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

            return Float32View(data), shape

    return data, shape


def load_patches_subset(path, shape, indices, resize_ratio=1.0, as_float=True):
    arr = np.memmap(path, dtype=np.uint8, mode='r', shape=shape)
    subset_size = len(indices)
    c, h, w = shape[1:]
    
    new_h = int(h * resize_ratio)
    new_w = int(w * resize_ratio)

    # Preallocate target in-memory array
    loaded = np.zeros((subset_size, c, new_h, new_w), dtype=np.uint8)

    for i, idx in enumerate(indices):
        patch = arr[idx]
        if resize_ratio < 1.0:
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            patch = patch[:, top:top+new_h, left:left+new_w]
        loaded[i] = patch

    del arr  # free memmap file handle

    if as_float:
        return loaded.astype(np.float32) / 255.0
    return loaded


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
    coords_subset = processed_species_original_locations
    
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
import psutil
import numpy as np
import os

def log_dataset_memory_info(file_path, shape, indices, resize_ratio, dtype=np.uint8):
    """
    Log memory usage estimation for a given dataset file and selected subset.
    """
    # shape = (N, C, H, W)
    total_patches = shape[0]
    selected_patches = len(indices)
    c, h, w = shape[1:]

    # Adjust size for resize_ratio
    h_resized = int(h * resize_ratio)
    w_resized = int(w * resize_ratio)

    # Each element’s size (e.g. uint8 -> 1 byte, float32 -> 4 bytes)
    element_size = np.dtype(dtype).itemsize

    # Compute full and actual sizes
    full_bytes = np.prod(shape) * element_size
    actual_bytes = selected_patches * c * h_resized * w_resized * element_size

    gb = 1024 ** 3
    full_gb = full_bytes / gb
    actual_gb = actual_bytes / gb

    available_ram = psutil.virtual_memory().available / gb

    # Log summary
    print(f"[INFO] {os.path.basename(file_path)}:")
    print(f"       Original shape: {shape}")
    print(f"       Selected: {selected_patches}/{total_patches} ({100*selected_patches/total_patches:.1f}%)")
    if resize_ratio < 1.0:
        print(f"       Resize ratio: {resize_ratio:.2f} → new size: ({c}, {h_resized}, {w_resized})")
    print(f"       Est. size: {actual_gb:.2f} GB (full file: {full_gb:.2f} GB)")
    print(f"       Available RAM: {available_ram:.1f} GB\n")

    return actual_gb, available_ram

class MultiSourceNpyDatasetWithCoords:
    """
    Enhanced dataset class that includes coordinate support,
    with optional cropping of patches to a smaller ratio.
    """
    def __init__(self, npy_files, targets, processed_species_original_locations, 
                 features_index=None, indices=None, transform=None, 
                 include_coords=False, resize_ratio=1.0, add_point_infos=False,
                 feature_importance_only_tabular = False):
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
        self.feature_importance_only_tabular = feature_importance_only_tabular
        if self.targets.dtype != np.float32:
            self.targets = self.targets.astype(np.float32)
        print(f"Processed species locations shape in dataset: {self.processed_species_original_locations.shape}")
        # Load shapes
        self.shapes = []
        self.data_arrays = []
        if len(self.npy_files) == 0:
            raise ValueError("You must provide at least one .npy file.")
        else:
            for npy_file in self.npy_files:
                path = npy_file['path']
                print(f"[Dataset Init] Loading: {path}")
                try:
                    patches, shape = load_patches_auto(path, as_float=False)

                    log_dataset_memory_info(path, shape, self.indices, self.resize_ratio, dtype=np.uint8)

                    if self.resize_ratio < 1.0 or len(self.indices) < shape[0]:
                        patches = load_patches_subset(
                            path, shape, self.indices, resize_ratio=self.resize_ratio, as_float=False
                        )
                    self.data_arrays.append(patches)
                    self.shapes.append(shape)
                except Exception as e:
                    print(f"[Warning] Could not load {path}: {e}")
                    arr = np.load(path, mmap_mode='r')
                    self.data_arrays.append(arr)
                    self.shapes.append(arr.shape)

        # Compute effective shapes (account for feature subset and crop ratio)
        self.effective_shapes = []
        for i, shape in enumerate(self.shapes):
            c, h, w = shape[1:]  # (N, C, H, W)
            if npy_files[i]['name'] == 'pred100':
                if self.feature_importance_only_tabular == False and self.features_index is not None:
                    c = len(self.features_index)
                else:
                    c = self.data_arrays[i].shape[1]  # number of bands/channels
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
            self.effective_shapes.append({
                "shape": (2,),
                "source": "coordinates"
                })  # single-element tuple
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
            #patch = arr[self.indices[idx]].astype(np.float32)/255  # (C, H, W)
            patch = arr[idx].astype(np.float32)/255
            # Select subset of features for first array if needed
            if self.feature_importance_only_tabular == False and i == self.pred100_index and self.features_index is not None:
                patch = patch[self.features_index]


            # Apply center crop if ratio < 1.0
            if self.resize_ratio < 1.0:
                _, h, w = patch.shape
                new_h, new_w = int(h * self.resize_ratio), int(w * self.resize_ratio)
                patch = self._center_crop(patch, new_h, new_w)

            tensor = torch.from_numpy(patch)
            data_items.append(tensor)

        # Transform (augmentations etc.)
        if self.transform:
            data_items = self.transform(data_items)
        # Add point infos if needed
        if self.add_point_infos:
            point_info = self.point_infos[idx]
            point_info_tensor = torch.from_numpy(point_info)
            data_items.append(point_info_tensor)

        target_tensor = torch.from_numpy(self.targets[idx])
        
        original_idx = self.indices[idx]
        
        if self.include_coords:
            coords = self.coordinates[idx]
            return data_items, target_tensor, original_idx, coords
        else:
            return data_items, target_tensor, original_idx

    
class MultiSourceNpyDatasetNoTargets:
    """
    Enhanced dataset class that includes coordinate support,
    with optional cropping of patches to a smaller ratio.
    Loads all data into memory instead of using memmap.
    """
    def __init__(self, npy_files, processed_species_original_locations, 
                 features_index=None, indices=None, transform=None, resize_ratio=1.0):
        self.npy_files = npy_files
        self.features_index = features_index
        self.indices = np.arange(len(processed_species_original_locations)) if indices is None else indices
        self.transform = transform
        self.processed_species_original_locations = processed_species_original_locations
        self.resize_ratio = resize_ratio
        self.pred100_index = None  # To be set if needed

        # Load shapes and arrays fully into memory
        self.shapes = []
        self.data_arrays = []
        if len(self.npy_files) == 0:
            raise ValueError("You must provide at least one .npy file.")
        else:
            for npy_file in self.npy_files:
                try:
                    path = npy_file['path']
                    print(f"Loading shape and data from: {path}")
                    patches, shape = load_patches(path, as_float=False, mmap_mode=None)
                    self.shapes.append(shape)
                    self.data_arrays.append(patches)
                except (FileNotFoundError, Exception) as e:
                    print(f"Warning: Could not load shape from metadata file {path}: {e}")
                    arr = np.load(path)
                    self.shapes.append(arr.shape)
                    self.data_arrays.append(arr)

        # Compute effective shapes (account for feature subset and crop ratio)
        self.effective_shapes = []
        for i, shape in enumerate(self.shapes):
            c, h, w = shape[1:]  # (N, C, H, W)
            if npy_files[i]['name'] == 'pred100':
                if self.features_index is not None:
                    c = len(self.features_index)
                else:
                    c = self.data_arrays[i].shape[1]  # number of bands/channels
                self.pred100_index = i

            if self.resize_ratio < 1.0:
                h = int(h * self.resize_ratio)
                w = int(w * self.resize_ratio)

            self.effective_shapes.append({
                "shape": (c, h, w),
                "source": npy_files[i]['name']
                })
            
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
            #patch = arr[self.indices[idx]].astype(np.float32)  # (C, H, W)
            patch = arr[idx]

            # Select subset of features for first array if needed
            if i == self.pred100_index and self.features_index is not None:
                patch = patch[self.features_index]
                patch = patch / 255.0

            # Apply center crop if ratio < 1.0
            # if self.resize_ratio < 1.0:
            #     _, h, w = patch.shape
            #     new_h, new_w = int(h * self.resize_ratio), int(w * self.resize_ratio)
            #     patch = self._center_crop(patch, new_h, new_w)

            tensor = torch.from_numpy(patch)
            data_items.append(tensor)
        if len(data_items) == 1:
            return data_items[0]
        else:
            return data_items,
        
######################################################
################AUGMENTATIONS#########################
######################################################
class MultiSourceAugmentation:
    """Proper augmentation that distinguishes between data sources"""
    
    def __init__(self, npy_files, satellite_aug=None, climate_aug=None, tabular_aug=None):
        self.npy_files = npy_files  # Store the file metadata to identify sources
        self.satellite_aug = satellite_aug
        self.climate_aug = climate_aug
        self.tabular_aug = tabular_aug
        
        # Map source names to augmentation types
        self.source_to_aug = {}
        for i, npy_file in enumerate(npy_files):
            source_name = npy_file['name']
            if 'pred100' in source_name.lower() or 'sentinel' in source_name.lower() or 'landsat' in source_name.lower():
                self.source_to_aug[i] = 'satellite'
            elif 'climate' in source_name.lower() or 'weather' in source_name.lower() or 'temp' in source_name.lower():
                self.source_to_aug[i] = 'climate'
            else:
                self.source_to_aug[i] = 'climate'  # Default to climate for non-satellite
    
    def __call__(self, data_items):
        augmented_items = []
        
        for i, tensor in enumerate(data_items):
            # Handle tabular data (1D tensors)
            if tensor.dim() == 1:
                if self.tabular_aug:
                    tensor = self.tabular_aug(tensor)
                augmented_items.append(tensor)
                continue
                
            # Handle image-like data (3D tensors)
            if tensor.dim() == 3:
                source_type = self.source_to_aug.get(i, 'climate')  # Default to climate
                
                if source_type == 'satellite' and self.satellite_aug:
                    tensor = self.satellite_aug(tensor)
                elif source_type == 'climate' and self.climate_aug:
                    tensor = self.climate_aug(tensor)
                
                augmented_items.append(tensor)
                continue
            
            # For any other type, pass through unchanged
            augmented_items.append(tensor)
        
        return augmented_items
    


class SatelliteAugmentation:
    """Augmentations for normalized multispectral satellite data (0-1),
    safe from saturation while preserving variability."""
    
    def __init__(self, apply_geometric=True, apply_spectral=False):
        self.apply_geometric = apply_geometric
        self.apply_spectral = apply_spectral

    # Small safe brightness adjustment
    def adjust_brightness(self, tensor, factor):
        return tensor * factor  # no clamping here yet

    # Small safe contrast adjustment
    def adjust_contrast(self, tensor, factor):
        mean = tensor.mean(dim=(-2, -1), keepdim=True)
        return (tensor - mean) * factor + mean  # no clamping yet

    def __call__(self, tensor):
        """
        tensor: (C, H, W) normalized 0-1
        """
        # ----------- Geometric transformations -----------
        if self.apply_geometric:
            # Horizontal flip
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[-1])
            # Vertical flip
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[-2])
            # 90° rotations
            if torch.rand(1) > 0.5:
                k = torch.randint(0, 4, (1,)).item()
                tensor = torch.rot90(tensor, k, dims=[-2, -1])

        # ----------- Spectral / value-based transformations -----------
        if self.apply_spectral:
            # Safe brightness adjustment (small factor)
            if torch.rand(1) > 0.7:
                brightness_factor = 0.98 + 0.04 * torch.rand(1).item()  # 0.98–1.02
                tensor = self.adjust_brightness(tensor, brightness_factor)

            # Safe contrast adjustment
            if torch.rand(1) > 0.7:
                contrast_factor = 0.98 + 0.04 * torch.rand(1).item()  # 0.98–1.02
                tensor = self.adjust_contrast(tensor, contrast_factor)

            # Add small noise
            if torch.rand(1) > 0.7:
                noise_std = 0.01  # smaller to avoid clipping
                noise = torch.randn_like(tensor) * noise_std
                tensor = tensor + noise

            # Channel-wise small adjustments
            if torch.rand(1) > 0.9 and tensor.shape[0] > 1:  # less frequent
                channels_to_adjust = torch.randperm(tensor.shape[0])[:max(1, tensor.shape[0]//3)]
                for c in channels_to_adjust:
                    factor = 0.98 + 0.04 * torch.rand(1).item()
                    tensor[c] = tensor[c] * factor

        # ----------- Clamp only once at the end -----------
        tensor = torch.clamp(tensor, 0, 1)

        return tensor



class ClimateDataAugmentation:
    """Climate-specific augmentations that work with any number of channels"""
    
    def __init__(self, apply_spatial=True, apply_value=False):
        self.apply_spatial = apply_spatial
        self.apply_value = apply_value
    
    def __call__(self, tensor):
        # tensor shape: (C, H, W) where C are different climate variables
        
        # Spatial transformations
        if self.apply_spatial:
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[-1])  # Horizontal flip
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[-2])  # Vertical flip
            if torch.rand(1) > 0.5:
                k = torch.randint(0, 4, (1,)).item()
                tensor = torch.rot90(tensor, k, dims=[-2, -1])  # 90° rotations
        
        # # Value-based transformations
        # if self.apply_value:
        #     # Small noise proportional to each variable's range
        #     if torch.rand(1) > 0.7:
        #         # Calculate per-channel noise based on variance
        #         noise_levels = tensor.std(dim=(-1, -2), keepdim=True) * 0.05  # 5% of std
        #         noise = torch.randn_like(tensor) * noise_levels
        #         tensor = tensor + noise
            
        #     # Simple smoothing using average pooling (works with any channel count)
        #     if torch.rand(1) > 0.8:
        #         # Use average pooling for smoothing
        #         smoothed = F.avg_pool2d(
        #             tensor.unsqueeze(0), kernel_size=3, padding=1, stride=1
        #         ).squeeze(0)
        #         alpha = 0.3
        #         tensor = (1 - alpha) * tensor + alpha * smoothed
        
        return tensor


class TabularAugmentation:
    """Augmentations for tabular climate data"""
    
    def __init__(self, noise_std=0.05, feature_dropout=0.1):
        self.noise_std = noise_std
        self.feature_dropout = feature_dropout
    
    def __call__(self, tensor):
        # tensor shape: (features,)
        
        # Add small Gaussian noise
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = tensor + noise
        
        # Random feature dropout
        if torch.rand(1) > 0.5:
            mask = torch.rand_like(tensor) > self.feature_dropout
            tensor = tensor * mask.float()
        
        return tensor

def create_dataset(npy_files, target_df, processed_species_original_locations, indices, pred_100_indices, use_transform=False, include_coords=False, resize_ratio=1, add_point_infos = False, feature_importance_only_tabular = False):
    """Create a dataset with validation checks."""
    if use_transform:
        # Create source-specific augmentations
        satellite_aug = SatelliteAugmentation(apply_geometric=True, apply_spectral=False)
        climate_aug = ClimateDataAugmentation(apply_spatial=True, apply_value=True)
        multi_aug = MultiSourceAugmentation(
            npy_files=npy_files,
            satellite_aug=satellite_aug,
            climate_aug=climate_aug,
            tabular_aug=None  # No augmentation for tabular data
        )
        transform = multi_aug
    else:
        transform = None
        
    dataset = MultiSourceNpyDatasetWithCoords(
        npy_files=npy_files,
        targets=target_df.iloc[indices].values,
        processed_species_original_locations=processed_species_original_locations.iloc[indices],
        features_index=pred_100_indices,
        indices=indices,
        transform=transform,
        include_coords=include_coords,
        resize_ratio=resize_ratio,
        add_point_infos=add_point_infos,
        feature_importance_only_tabular = feature_importance_only_tabular
    )

    pos_ratio = np.sum(dataset.targets) / len(dataset.targets)
    print(f"Dataset size: {len(dataset)}, Positives: {np.sum(dataset.targets)}, Positive ratio: {pos_ratio:.4f}")
    return dataset


class InMemoryBandsDataset(RasterDataset):
    def __init__(self, paths: str, all_bands: list[str], transforms=None):
        super().__init__(paths, transforms=transforms)
        self.all_bands = all_bands
        self.bands = all_bands

        self._crs = None  # private storage

        # --- Load all rasters into memory ---
        raster_files = sorted([
            os.path.join(paths, f) for f in os.listdir(paths) if f.endswith(".tif")
        ])
        arrays = []
        for f in raster_files:
            with rasterio.open(f) as src:
                arrays.append(src.read(1))  # read first (and only) band
                if self._crs is None:
                    self._crs = src.crs
                    self.transform = src.transform
                    self._bounds = BoundingBox(*src.bounds, mint=0, maxt=0)
                    self.height = src.height
                    self.width = src.width

        # Shape: (bands, H, W)
        self.data = np.stack(arrays, axis=0).astype(np.float32)

    @property
    def bounds(self):
        return self._bounds

    @property
    def crs(self):
        return self._crs

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query."""
        # Use the R-tree index like the original
        hits = self.index.intersection(tuple(query), objects=True)
        if not hits:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # In-memory version: no file reading, just slice self.data
        # Convert query bounds to pixel indices
        col_start, row_start = ~self.transform * (query.minx, query.maxy)
        col_stop,  row_stop  = ~self.transform * (query.maxx, query.miny)

        # Proper rounding and clipping
        row_start = max(0, int(np.floor(row_start)))
        row_stop  = max(0, int(np.ceil(row_stop)))
        col_start = max(0, int(np.floor(col_start)))
        col_stop  = max(0, int(np.ceil(col_stop)))

        # Clip to dataset dimensions
        row_start = min(row_start, self.height)
        row_stop  = min(row_stop, self.height)
        col_start = min(col_start, self.width)
        col_stop  = min(col_stop, self.width)

        # Slice the in-memory array
        data = self.data[:, row_start:row_stop, col_start:col_stop]
        data = torch.from_numpy(data).to(self.dtype)

        # Build sample dict
        sample = {"crs": self.crs, "bounds": query}
        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data.squeeze(0)

        # Apply transforms if any
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        # can adapt later, for now minimal
        return 1
    
    def plot(
        self,
        sample: dict,
        bands: list[str] | None = None,
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> plt.Figure:
        """
        Plot a sample with the selected bands.

        Args:
            sample: dict returned by __getitem__
            bands: list of band names to visualize
            show_titles: whether to display band titles
            suptitle: optional figure title
        """
        image = sample["image"]  # tensor of shape (bands, H, W)
        image_np = image.numpy()

        # Resolve band indices
        if bands is None:
            band_indices = list(range(image_np.shape[0]))
            bands = self.all_bands
        else:
            band_indices = [self.all_bands.index(b) for b in bands]

        n_bands = len(band_indices)

        # Create figure
        fig, axs = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4))
        if n_bands == 1:
            axs = [axs]

        for i, idx in enumerate(band_indices):
            axs[i].imshow(image_np[idx], cmap="viridis")
            axs[i].axis("off")
            if show_titles:
                axs[i].set_title(bands[i])

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()
        plt.close(fig)  # close to avoid automatic display in notebooks
        return fig

################################################################
###################SAMPLERS#####################################
################################################################
class AutoClassAwareBatchSampler(Sampler):
    """
    Automatically handles class imbalance.
    Each batch samples more frequently from rare classes, without hard thresholds.
    """
    def __init__(self, dataset, batch_size, oversample_factor=3, seed=None):
        """
        Parameters
        ----------
        dataset : Dataset
            Dataset with multi-label targets in dataset.targets (shape: N x num_classes)
        batch_size : int
            Number of samples per batch
        oversample_factor : float
            How much more likely rare classes are sampled (1 = proportional)
        seed : int, optional
            Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.oversample_factor = oversample_factor
        self.seed = seed
        
        # Ensure reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Compute class frequencies
        targets = self._get_targets_array()
        class_counts = targets.sum(axis=0)  # num_samples per class
        class_freqs = class_counts / len(dataset)
        
        # Assign sample weights inversely proportional to rare class prevalence
        # For multi-label, take max weight among all present classes
        sample_weights = []
        for sample_targets in targets:
            # Avoid division by zero for empty classes
            present_classes = np.where(sample_targets > 0)[0]
            if len(present_classes) == 0:
                weight = 1.0
            else:
                class_weights = 1.0 / (class_freqs[present_classes] + 1e-6)
                weight = class_weights.max()  # focus on rarest class in sample
            sample_weights.append(weight)
        
        # Scale weights by oversample_factor
        sample_weights = np.array(sample_weights)
        sample_weights = sample_weights ** oversample_factor
        
        # Normalize weights to sum=1
        sample_weights = sample_weights / sample_weights.sum()
        self.sample_weights = sample_weights
        
        # Precompute number of batches
        self.num_samples = len(dataset)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
    def _get_targets_array(self):
        """
        Return dataset targets as numpy array (N x num_classes)
        """
        if hasattr(self.dataset, 'targets'):
            return np.array(self.dataset.targets)
        else:
            # fallback: iterate dataset (may be slower)
            all_targets = []
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                if isinstance(item, tuple):
                    target = item[1]  # assumes (data, target, ...)
                else:
                    target = item
                all_targets.append(np.array(target))
            return np.vstack(all_targets)
        
    def __iter__(self):
        for _ in range(self.num_batches):
            # Sample batch indices using precomputed weights
            batch_indices = np.random.choice(
                self.num_samples,
                size=self.batch_size,
                replace=True,
                p=self.sample_weights
            )
            for idx in batch_indices:
                yield idx
    
    def __len__(self):
        return self.num_batches * self.batch_size
    
    


def get_balanced_eval_set_v2(df, species_name):
    """
    Selects a balanced evaluation set for a single species.
    
    If the species is "common" (presence > 50%), it returns all presences
    and all absences.
    
    If the species is "rare" (presence <= 50%), it performs a 1:1 balanced
    selection by finding the N absences that are *closest* to any presence point,
    where N is the number of presences.
    """
    
    # 1. Get original indices and separate data
    original_indices = df.index.values
    # Assuming create_target_species_df is in another file as per import
    status = create_target_species_df(species_name=species_name, processed_species=df).values.flatten()
    is_presence = (status == 1)
    is_absence = (status == 0)
    
    num_presences = np.sum(is_presence)
    num_absences = np.sum(is_absence)
    
    presence_ratio = num_presences / len(status) if len(status) > 0 else 0
    
    # --- Fast path for "common" species ---
    if presence_ratio > 0.5:
        presence_original_indices = original_indices[is_presence]
        absence_original_indices = original_indices[is_absence]
        return presence_original_indices, absence_original_indices
    
    # Get the coordinates for each group
    presence_geoms = df.loc[is_presence, 'geometry']
    absence_geoms = df.loc[is_absence, 'geometry']
    
    presence_coords_deg = np.array([[geom.y, geom.x] for geom in presence_geoms])
    absence_coords_deg = np.array([[geom.y, geom.x] for geom in absence_geoms])
    
    # Get the *original* dataframe indices for each group
    presence_original_indices = original_indices[is_presence]
    absence_original_indices = original_indices[is_absence]
    
    # --- Handle Edge Cases ---
    if num_presences == 0:
        return np.array([]), np.array([]) 
    if num_absences == 0:
        return presence_original_indices, np.array([])

    # 2. Convert to Radians (required for Haversine)
    presence_coords_rad = np.radians(presence_coords_deg)
    absence_coords_rad = np.radians(absence_coords_deg)
    
    # --- NEW, SIMPLIFIED LOGIC STARTS HERE ---
    
    # 3. Build the BallTree index on *presence* points
    # We fit on the presences to find the closest one for each absence
    nn_model = NearestNeighbors(n_neighbors=1,  # We only care about the single closest presence
                                algorithm='ball_tree',
                                metric='haversine')
    nn_model.fit(presence_coords_rad)
    
    # 4. Find the distance for *every* absence point to its nearest presence
    # This gives a distance for every absence in the dataset
    distances, _ = nn_model.kneighbors(absence_coords_rad)
    
    # 5. Sort the *indices* of the absence array based on distance
    # distances.flatten() turns the (N_absences, 1) array into (N_absences,)
    # np.argsort returns the *indices* that would sort this distance array
    sorted_absence_indices_by_dist = np.argsort(distances.flatten())
    
    # 6. Select the top N absences, where N is the number of presences
    # This guarantees a 1:1 balance.
    
   
    # We can't select more absences than we have
    num_absences_to_select = int(min(num_presences, num_absences))
    # Get the *local* indices of the closest N absences
    selected_local_absence_indices = sorted_absence_indices_by_dist[:num_absences_to_select]
    # 7. Map these local indices back to the *original* dataframe indices
    selected_absence_original_indices = absence_original_indices[selected_local_absence_indices]
    
    # --- END OF NEW LOGIC ---
        
    return presence_original_indices, selected_absence_original_indices

def process_single_species(species_name, processed_species_original_locations):
    """
    Worker: receives a species name and the GeoDataFrame explicitly.
    """
    #try:
    presence_indices, absence_indices = get_balanced_eval_set_v2(
        processed_species_original_locations,
        species_name=species_name
    )
    return {
        'species': species_name,
        'presence_indices': presence_indices,
        'absence_indices': absence_indices
    }
    # except Exception as e:
    #     print(f"Error processing {species_name}: {str(e)}")
    #     return {
    #         'species': species_name,
    #         'presence_indices': np.array([]),
    #         'absence_indices': np.array([]),
    #         'error': str(e)
    #     }

def get_balanced_eval_set_v2_parallel(df, species_names_list):
    """
    Parallel wrapper that maps process_single_species over species_names_list,
    passing the same `df` to each worker via itertools.repeat.
    """
    N_WORKERS = max(1, (os.cpu_count() or 1) - 2)
    all_indices = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results_generator = executor.map(
            process_single_species,
            species_names_list,
            itertools.repeat(df)   # <-- pass the same df to every call
        )
        all_indices = list(tqdm.tqdm(results_generator, total=len(species_names_list)))
    return all_indices