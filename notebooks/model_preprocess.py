import shapely
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import geopandas as gpd
import pandas as pd


def create_target_species_df(species_name, processed_species):
    df_present = processed_species[['species_list', 'geometry']].copy()
    # Use single brackets to get the Series and apply to each list directly
    df_present[species_name] = df_present['species_list'].apply(lambda x: 1 if species_name in x else 0)
    return df_present[species_name].to_frame()



#####################################################
##############Feature importance#####################
#####################################################

def merge_data_inputs(df, location_zonal_stats):
    # Create a simple index key instead of using geometry
    if "gid" not in df.columns:
        df = df.reset_index().rename(columns={'index': 'gid'})
    if "gid" not in location_zonal_stats.columns:
        location_zonal_stats = location_zonal_stats.reset_index().rename(columns={'index': 'gid'})
    
    merged_df = df.merge(location_zonal_stats, on="gid", how="left")

    # Clean up duplicates
    cols_to_drop = [c for c in ['species_list_x', 'species_list_y', 'h3_index'] if c in merged_df.columns]
    merged_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Handle NaNs
    if merged_df.isnull().values.any():
        print(f"Warning: Found {merged_df.isnull().sum().sum()} NaN values, replacing with 0.")
        merged_df = merged_df.fillna(0)
    merged_df.drop(columns=['gid'], inplace=True, errors="ignore")
    return merged_df



def get_feature_importance_for_species(input_processed_species, target, cont_zonal_stats_path, target_df, treshold=None):
    model = XGBClassifier(random_state=42)

    # Load df
    location_zonal_stats = pd.read_csv(cont_zonal_stats_path)
    location_zonal_stats_numeric_columns = location_zonal_stats.select_dtypes(include=['float64', 'int64']).columns
    location_zonal_stats[location_zonal_stats_numeric_columns] = location_zonal_stats[location_zonal_stats_numeric_columns].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
    )

    location_zonal_stats['geometry'] = location_zonal_stats['geometry'].apply(shapely.wkt.loads)
    # Convert to GeoDataFrame
    location_zonal_stats = gpd.GeoDataFrame(location_zonal_stats, geometry='geometry', crs='EPSG:4326')
    location_zonal_stats.to_crs(input_processed_species.crs, inplace=True)
    merge_key = 'geometry'
    location_zonal_stats = location_zonal_stats.merge(
        input_processed_species[[merge_key, 'species_list']],
        on=merge_key,
        how='left'
    )
    location_zonal_stats.rename(columns={'species_list_x': 'species_list'}, inplace=True)
    if 'species_list_y' in location_zonal_stats.columns:
        location_zonal_stats.drop(columns=['species_list_y'], inplace=True)
    #drop nan
    #location_zonal_stats = location_zonal_stats.dropna(subset=['species_list'])

    merged_df_without_region = merge_data_inputs(target_df, location_zonal_stats)

    excluded_from_features = [target, 'h3_index', 'geometry', 'geometry_x', 'geometry_y', 'species_list']
    final_feature_cols = []
    for col in merged_df_without_region.columns:
        if col not in excluded_from_features:
            if pd.api.types.is_numeric_dtype(merged_df_without_region[col]) or pd.api.types.is_bool_dtype(merged_df_without_region[col]):
                final_feature_cols.append(col)
            else:
                print(f"Skipping non-numeric/non-boolean column '{col}' with dtype '{merged_df_without_region[col].dtype}' from features.")
    X = merged_df_without_region[final_feature_cols]
    y = merged_df_without_region[target]
    model.fit(X, y)
    selector = SelectFromModel(model, prefit=True, threshold='median')
    selected_mask = selector.get_support()
    
    importances = model.feature_importances_
    full_importance_df = pd.DataFrame({
        'feature_name': X.columns,
        'importance': importances
    })
    feature_importance_df = full_importance_df[selected_mask].copy()

    feature_importance_df['feature_name'] = feature_importance_df['feature_name'].str.replace('_value', '', regex=False)
    if treshold is not None:
        feature_importance_df = feature_importance_df[feature_importance_df['importance'] > treshold]
    return feature_importance_df.sort_values(by='importance', ascending=False)
