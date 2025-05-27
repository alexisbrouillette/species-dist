import rasterio
import matplotlib.pyplot as plt
import numpy as np

import rasterio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_sentinel_tiles(filepaths, titles=None):
    """
    Visualize multiple Sentinel-2 GeoTIFF files with proper RGB rendering 
    and histogram stretching for better visualization.
    
    Args:
        filepaths (list): List of file paths to Sentinel-2 GeoTIFF files.
        titles (list, optional): List of titles corresponding to each image.
    """
    num_images = len(filepaths)
    cols = min(3, num_images)  # Limit columns to 3 for better layout
    rows = (num_images + cols - 1) // cols  # Compute necessary rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)  # Flatten for easier iteration

    def stretch_img(img):
        """Histogram stretching for better contrast."""
        p2 = np.percentile(img[img > 0], 2) if np.any(img > 0) else 0
        p98 = np.percentile(img[img > 0], 98) if np.any(img > 0) else 1
        return np.clip((img - p2) / (p98 - p2), 0, 1)

    for i, filepath in enumerate(filepaths):
        with rasterio.open(filepath) as src:
            image = src.read()

            if src.count == 3:  # RGB visualization
                stretched = np.stack([stretch_img(image[j]) for j in range(3)], axis=-1)
                axes[i].imshow(stretched)
            else:  # Single-band grayscale visualization
                stretched = stretch_img(image[0])
                axes[i].imshow(stretched, cmap='viridis')  # Use 'viridis' for better visualization
            
            axes[i].set_title(titles[i] if titles else f"Image {i+1}")
            axes[i].axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig, axes


def visualize_sentinel_tile(filepath, colormap='viridis'):
    """
    Visualize a Sentinel-2 GeoTIFF file with proper RGB rendering
    and histogram stretching for better visualization
    """
    with rasterio.open(filepath) as src:
        # Print the shape to understand our data
        print(f"Number of bands: {src.count}")
        print(f"Image shape: {src.height} x {src.width}")
        
        # Read the data
        image = src.read()
        print(f"Array shape: {image.shape}")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Perform histogram stretching for better visualization
        def stretch_img(img):
            p2 = np.percentile(img[img > 0], 2)  # Ignore zero values
            p98 = np.percentile(img[img > 0], 98)
            return np.clip((img - p2) / (p98 - p2), 0, 1)
        
        # If we have 3 bands, assume RGB
        if src.count == 3:
            # Stretch each band
            stretched = np.zeros_like(image, dtype=np.float32)
            for i in range(src.count):
                stretched[i] = stretch_img(image[i])
            
            # Rearrange to height x width x bands
            rgb = np.moveaxis(stretched, 0, -1)
            
            # Plot
            im = ax.imshow(rgb)
        else:
            # If not 3 bands, just show the first band
            stretched = stretch_img(image[0])
            im = ax.imshow(stretched, cmap=colormap)
        
        ax.set_title("Sentinel-2 Image", fontsize=12)
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('Pixel Value')
        
        # Add metadata
        if src.crs:
            plt.figtext(
                0.1, 0.02,
                f"CRS: {src.crs.to_string()}\n"
                f"Image shape: {image.shape}\n"
                f"Resolution: {src.res}\n"
                f"Bounds: {src.bounds}",
                fontsize=8
            )
        
        plt.tight_layout()
        return fig, ax