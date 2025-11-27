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


def get_train_val_indices_blocked(processed_species_df, cell_size, n_splits=0.2, buffer_cells=0, random_state=342, n_lat_bands=4):
    """
    Spatial blocked cross-validation with latitudinal stratification
    (ensures validation cells are present in both north and south).

    Args:
        processed_species_df: GeoDataFrame with 'geometry' column (points/polygons).
        cell_size: grid cell size (in CRS units).
        n_splits: int = number of folds, OR float = fraction of cells for validation.
        buffer_cells: number of neighboring cells around validation to exclude from training.
        random_state: RNG seed.
        n_lat_bands: how many north-south bands to enforce (default 4).
    Returns:
        train_indices, val_indices, gdf_with_cells
    """
    rng = np.random.default_rng(random_state)

    # Ensure geometry is valid shapely objects
    df_copy = processed_species_df.copy()
    if not hasattr(df_copy['geometry'].iloc[0], 'geom_type'):
        df_copy['geometry'] = df_copy['geometry'].apply(wkt_loads)
    gdf = gpd.GeoDataFrame(df_copy, geometry='geometry')

    # Store original index
    original_index_name = gdf.index.name if gdf.index.name is not None else 'original_index'
    if original_index_name == 'original_index':
        gdf = gdf.reset_index().rename(columns={'index': original_index_name})
    else:
        gdf = gdf.reset_index()

    # Bounding box
    minx, miny, maxx, maxy = gdf.total_bounds

    # Create grid
    x_coords = np.arange(minx, maxx + cell_size, cell_size)
    y_coords = np.arange(miny, maxy + cell_size, cell_size)

    patches = []
    ids, rows, cols = [], [], []
    for j, y in enumerate(y_coords[:-1]):
        for i, x in enumerate(x_coords[:-1]):
            patch = box(x, y, x + cell_size, y + cell_size)
            patches.append(patch)
            ids.append(f"{i}_{j}")
            rows.append(j)
            cols.append(i)

    patch_gdf = gpd.GeoDataFrame({
        'patch_id': ids,
        'row': rows,
        'col': cols,
        'geometry': patches
    }, crs=gdf.crs)

    # Assign points to cells
    gdf_joined = gpd.sjoin(gdf, patch_gdf.reset_index(), how="left", predicate="intersects")
    gdf_with_cells = gdf_joined.drop_duplicates(subset=[original_index_name], keep="first")

    # Drop empty cells
    used_cells = gdf_with_cells['patch_id'].dropna().unique()
    patch_gdf = patch_gdf[patch_gdf['patch_id'].isin(used_cells)]

    # --- Latitudinal bands ---
    patch_gdf["lat_band"] = pd.qcut(patch_gdf["row"], q=n_lat_bands, labels=False, duplicates="drop")

    val_cells = []
    for band in patch_gdf["lat_band"].unique():
        cells_in_band = patch_gdf.loc[patch_gdf["lat_band"] == band, "patch_id"].tolist()
        rng.shuffle(cells_in_band)

        if isinstance(n_splits, float):  # fraction
            n_val = max(1, int(len(cells_in_band) * n_splits))
        else:  # folds
            n_val = max(1, len(cells_in_band) // n_splits)

        val_cells.extend(cells_in_band[:n_val])

    val_cells = set(val_cells)

    # Buffer (optional)
    if buffer_cells > 0:
        val_coords = patch_gdf[patch_gdf['patch_id'].isin(val_cells)][['row', 'col']]
        buffer_ids = []
        for _, (r, c) in val_coords.iterrows():
            neighbors = patch_gdf[
                (patch_gdf['row'].between(r-buffer_cells, r+buffer_cells)) &
                (patch_gdf['col'].between(c-buffer_cells, c+buffer_cells))
            ]['patch_id'].tolist()
            buffer_ids.extend(neighbors)
        val_cells.update(buffer_ids)

    # Masks
    val_mask = gdf_with_cells['patch_id'].isin(val_cells)
    train_mask = ~val_mask

    train_indices = gdf_with_cells.loc[train_mask, original_index_name].values
    val_indices = gdf_with_cells.loc[val_mask, original_index_name].values

    print(f"Total points: {len(gdf_with_cells)} | Train: {len(train_indices)} | Val: {len(val_indices)}")

    return train_indices, val_indices, gdf_with_cells


def plot_split(train_indices, val_indices, processed_species):
    training_geom = processed_species.iloc[train_indices].geometry
    val_geom = processed_species.iloc[val_indices].geometry

    fig, ax = plt.subplots(figsize=(8, 8))
    training_geom.plot(ax=ax, color='blue', markersize=5, label='Training')
    val_geom.plot(ax=ax, color='orange', markersize=5, label='Validation')
    ax.set_title("Training vs Validation Points")
    ax.legend()
    plt.show()