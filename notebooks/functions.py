import h3
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely import Polygon
from shapely.geometry import mapping
import numpy as np

def geo_to_h3(lat, lng, h3_res=5):
  return h3.latlng_to_cell(lat=lat,lng=lng, res=h3_res)

def group_by_h3_cell(df, h3_col='h3_index', species_col='species'):
    geo_df = gpd.GeoDataFrame(data=None, columns=['geometry', h3_col, 'count', species_col])
    
    # Group by h3_cell and aggregate - ensure flat lists
    grouped = df.groupby(h3_col).agg({
        species_col: [
            'nunique', 
            lambda x: x.tolist()  # This creates a flat list directly
        ]
    }).reset_index()

    # Flatten the MultiIndex columns
    grouped.columns = [h3_col, 'species_count', 'species_list']

    print(grouped['species_count'].sum())

    # Convert all items to strings and ensure flat list
    grouped['species_list'] = grouped['species_list'].apply(
        lambda x: [str(item) for item in (x if isinstance(x, (list, np.ndarray)) else [x])]
    )

    # Merge the geometry back into the grouped DataFrame
    grouped = grouped.merge(geo_df[[h3_col]], on=h3_col, how='left')

    # Merge the grouped DataFrame back into the geo_df
    geo_df = pd.concat([geo_df, grouped], ignore_index=True)

    # Final aggregation - flatten all species lists and get unique values
    geo_df = geo_df.groupby(h3_col).agg({
        'species_list': lambda x: list(set(item for sublist in x for item in sublist))
    }).reset_index()
    
    geo_df['species_count'] = geo_df['species_list'].apply(len)
    
    return geo_df

# Get the bounds of the clipped image
def get_clipped_bounds(out_image, out_transform):
    # Get the height and width of the clipped image
    height, width = out_image.shape[-2:]
    
    # Use the affine transform to calculate the bounds
    left, top = rasterio.transform.xy(out_transform, 0, 0)
    right, bottom = rasterio.transform.xy(out_transform, height, width)
    
    return (left, bottom, right, top) 

def extract_raster_values_for_h3(geometry, raster_path):
    with rasterio.open(raster_path) as src:
        # Reproject geometry to match raster CRS
        h3_geometry = gpd.GeoSeries([geometry], crs="EPSG:4326")  # Replace with actual CRS of geometry
        h3_geometry = h3_geometry.to_crs(src.crs)
        
        # Convert to GeoJSON
        geometry_proj = mapping(h3_geometry[0])
        
        # Check overlap
        if not src.bounds.contains(geometry_proj['coordinates'][0][0]):
            raise ValueError("Geometry does not overlap the raster bounds.")
        
        # Mask the raster
        out_image, out_transform = mask(src, [geometry_proj], crop=True)
        return out_image[0], src.transform

def add_geometry(row, col_name='h3_cell'):
  points = h3.cell_to_boundary(row[col_name])
  flipped = tuple(coord[::-1] for coord in points)
  return Polygon(flipped)


def open_tif(file_name, optimize_extent=False, extent=None):
    with rasterio.open(file_name) as src:
        if optimize_extent and extent:
            # Crop the raster to the specified extent
            window = rasterio.windows.from_bounds(*extent, transform=src.transform)
            raster_data = src.read(1, window=window, masked=True)
            affine = src.window_transform(window)
        else:
            raster_data = src.read(1, masked=True)  # read the first band
            affine = src.transform
        
        raster_data = np.where(raster_data == raster_data.min(), np.nan, raster_data)
        bounds = src.bounds
        crs = src.crs
    return raster_data, bounds, crs


