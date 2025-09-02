import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads as wkt_loads
from shapely.geometry import box
import matplotlib.pyplot as plt

def preprocess_data_for_split_generation(processed_species_df, grid_config):
    """
    Simplified preprocessing that numbers patches sequentially for easy manual selection.
    """
    grid_rows = grid_config['grid_rows']
    grid_cols = grid_config['grid_cols']

    # Work on a copy
    df_copy = processed_species_df.copy()

    # Ensure expanded_cells are valid geometry objects
    if not hasattr(df_copy['expanded_cells'].iloc[0], 'geom_type'):
        df_copy['expanded_cells'] = df_copy['expanded_cells'].apply(wkt_loads)

    gdf = gpd.GeoDataFrame(df_copy, geometry='expanded_cells')


    # Store original index
    original_index_name = gdf.index.name if gdf.index.name is not None else '__temp_original_index__'
    if original_index_name == '__temp_original_index__':
        gdf = gdf.reset_index().rename(columns={'index': original_index_name})
    else:
        gdf = gdf.reset_index()

    # Compute bounding box
    minx, miny, maxx, maxy = gdf.total_bounds
    buffer_factor = 0.001
    width = maxx - minx
    height = maxy - miny
    minx_buffered = minx - width * buffer_factor
    miny_buffered = miny - height * buffer_factor
    maxx_buffered = maxx + width * buffer_factor
    maxy_buffered = maxy + height * buffer_factor

    # Create grid patches
    dx = (maxx_buffered - minx_buffered) / grid_cols
    dy = (maxy_buffered - miny_buffered) / grid_rows

    patches = []
    patch_ids_list = []
    sequential_ids = []
    patch_number = 0
    
    patches = []
    patch_ids_list = []
    sequential_ids = []

    for j in range(grid_rows):      # rows = y direction (bottom → top)
        for i in range(grid_cols):  # cols = x direction (left → right)
            x0 = minx_buffered + i * dx
            y0 = miny_buffered + j * dy
            x1 = x0 + dx
            y1 = y0 + dy
            patch = box(x0, y0, x1, y1)
            patch_number = j * grid_cols + i   # <-- consistent numbering

            patches.append(patch)
            patch_ids_list.append(f"{i}_{j}")
            sequential_ids.append(patch_number)

    patch_gdf = gpd.GeoDataFrame({
        'patch_id': patch_ids_list,
        'sequential_id': sequential_ids,
        'geometry': patches
    }, crs=gdf.crs)

    # Spatial join
    gdf_joined = gpd.sjoin(gdf, patch_gdf, how='left', predicate='intersects')
    gdf_with_patches = gdf_joined.drop_duplicates(subset=[original_index_name], keep='first')

    # Clean up index
    if original_index_name == '__temp_original_index__':
        gdf_with_patches = gdf_with_patches.drop(columns=[original_index_name])
    else:
        gdf_with_patches = gdf_with_patches.set_index(original_index_name)

    
    print("Grid layout with sequential patch numbers:")
    print("+" + "-" * (grid_cols * 4 - 1) + "+")

    for j in reversed(range(grid_rows)):   # print top → bottom
        row = "|"
        for i in range(grid_cols):
            patch_number = j * grid_cols + i
            row += f" {patch_number:2}|"
        print(row)
        print("+" + "-" * (grid_cols * 4 - 1) + "+")
    
    print(f"\nTotal patches: {grid_cols} columns x {grid_rows} rows = {grid_cols * grid_rows} patches")
    print("Patches are numbered sequentially from 0 to", grid_cols * grid_rows - 1)
    
    return gdf_with_patches, patch_gdf


def get_train_val_indices(processed_species_df, grid_config, validation_patch_numbers):
    """
    Get train/validation indices based on manually specified validation patch numbers.
    
    Args:
        processed_species_df: Your original DataFrame
        grid_config: Dictionary with 'grid_rows' and 'grid_cols'
        validation_patch_numbers: List of patch numbers to use for validation (e.g., [0, 1, 4, 5])
    
    Returns:
        tuple: (train_indices, val_indices, gdf_with_patches)
    """
    grid_rows = grid_config['grid_rows']
    grid_cols = grid_config['grid_cols']

    # Work on a copy
    df_copy = processed_species_df.copy()

    # Ensure expanded_cells are valid geometry objects
    if not hasattr(df_copy['geometry'].iloc[0], 'geom_type'):
        df_copy['geometry'] = df_copy['geometry'].apply(wkt_loads)

    gdf = gpd.GeoDataFrame(df_copy, geometry='geometry')

    # Store original index
    original_index_name = gdf.index.name if gdf.index.name is not None else 'original_index'
    if original_index_name == 'original_index':
        gdf = gdf.reset_index().rename(columns={'index': original_index_name})
    else:
        gdf = gdf.reset_index()

    # Compute bounding box
    minx, miny, maxx, maxy = gdf.total_bounds
    buffer_factor = 0.001
    width = maxx - minx
    height = maxy - miny
    minx_buffered = minx - width * buffer_factor
    miny_buffered = miny - height * buffer_factor
    maxx_buffered = maxx + width * buffer_factor
    maxy_buffered = maxy + height * buffer_factor

    # Create grid patches
    dx = (maxx_buffered - minx_buffered) / grid_cols
    dy = (maxy_buffered - miny_buffered) / grid_rows

    patches = []
    patch_ids_list = []
    sequential_ids = []

    for j in range(grid_rows):      # rows = y direction (bottom → top)
        for i in range(grid_cols):  # cols = x direction (left → right)
            x0 = minx_buffered + i * dx
            y0 = miny_buffered + j * dy
            x1 = x0 + dx
            y1 = y0 + dy
            patch = box(x0, y0, x1, y1)
            patch_number = j * grid_cols + i   # <-- consistent numbering

            patches.append(patch)
            patch_ids_list.append(f"{i}_{j}")
            sequential_ids.append(patch_number)

    patch_gdf = gpd.GeoDataFrame({
        'patch_id': patch_ids_list,
        'sequential_id': sequential_ids,
        'geometry': patches
    }, crs=gdf.crs)


    # Spatial join
    gdf_joined = gpd.sjoin(gdf, patch_gdf, how='left', predicate='intersects')
    gdf_with_patches = gdf_joined.drop_duplicates(subset=[original_index_name], keep='first')

    # Get validation patch IDs from sequential numbers
    validation_patch_ids = patch_gdf.loc[patch_gdf['sequential_id'].isin(validation_patch_numbers), 'patch_id'].tolist()
    
    # Create masks
    val_mask = gdf_with_patches['patch_id'].isin(validation_patch_ids)
    train_mask = ~val_mask
    
    # Get original indices
    train_indices = gdf_with_patches.loc[train_mask, original_index_name].values
    val_indices = gdf_with_patches.loc[val_mask, original_index_name].values
    
    print("Grid layout with sequential patch numbers:")
    print("+" + "-" * (grid_cols * 4 - 1) + "+")

    for j in reversed(range(grid_rows)):   # print top → bottom
        row = "|"
        for i in range(grid_cols):
            patch_number = j * grid_cols + i
            marker = "V" if patch_number in validation_patch_numbers else " "
            row += f"{marker}{patch_number:2}|"
        print(row)
        print("+" + "-" * (grid_cols * 4 - 1) + "+")
    
    return train_indices, val_indices, gdf_with_patches



def plot_split(train_indices, val_indices, processed_species):
    training_geom = processed_species.iloc[train_indices].geometry
    val_geom = processed_species.iloc[val_indices].geometry

    fig, ax = plt.subplots(figsize=(8, 8))
    training_geom.plot(ax=ax, color='blue', markersize=5, label='Training')
    val_geom.plot(ax=ax, color='orange', markersize=5, label='Validation')
    ax.set_title("Training vs Validation Points")
    ax.legend()
    plt.show()