from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import numpy as np
import pandas as pd
from utils.model_preprocess import create_target_species_df, get_feature_importance_for_species
from utils.splitting import get_train_val_indices_blocked
import xgboost as xgb

import pandas as pd
import numpy as np
import os
from shapely.geometry import Point
import torch
from torchvision import transforms
from models.fusion_cnn import SimpleMultiInputCNNv2, training_loop
from utils.dataset import MultiSourceNpyDatasetWithCoords, RandomFlipRotate

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import xgboost as xgb

def evaluate_xgb_binary(model_params, X_train, y_train, X_valid, y_valid, species_name=None):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_valid, label=y_valid)

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    model_params["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0

    booster = xgb.train(
        model_params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Predict
    y_pred_val = booster.predict(xgb.DMatrix(X_valid))
    y_pred_train = booster.predict(xgb.DMatrix(X_train))

    val_preds = (y_pred_val > 0.5).astype(int)
    train_preds = (y_pred_train > 0.5).astype(int)

    metrics = {
        "species": species_name,
        "train_precision": precision_score(y_train, train_preds, zero_division=0),
        "train_recall": recall_score(y_train, train_preds, zero_division=0),
        "train_f1": f1_score(y_train, train_preds, zero_division=0),
        "train_accuracy": accuracy_score(y_train, train_preds),
        "train_specificity": ((y_train == 0) & (train_preds == 0)).sum() / max((y_train == 0).sum(), 1),
        "val_precision": precision_score(y_valid, val_preds, zero_division=0),
        "val_recall": recall_score(y_valid, val_preds, zero_division=0),
        "val_f1": f1_score(y_valid, val_preds, zero_division=0),
        "val_accuracy": accuracy_score(y_valid, val_preds),
        "val_specificity": ((y_valid == 0) & (val_preds == 0)).sum() / max((y_valid == 0).sum(), 1),
        "val_roc_auc": roc_auc_score(y_valid, y_pred_val) if len(np.unique(y_valid)) > 1 else np.nan,
    }
    return pd.DataFrame([metrics])





def evaluate_xgb_for_species(target_species_arr, train_indices, val_indices, processed_species, model_params):
    results = []
    for target_species in target_species_arr:
        
        target_df = create_target_species_df(target_species, processed_species=processed_species)
        pred_100_by_locations = pd.read_csv("./data/saved_df/processed_species_original_locations_point_data.csv")
        feature_importance = get_feature_importance_for_species(
            processed_species, 
            target=target_species, 
            cont_zonal_stats_path="./data/saved_df/processed_species_original_locations_point_data.csv",
            target_df=target_df
        )
        final_dataset = pred_100_by_locations[feature_importance['feature_name'].values]

        # train_indices, val_indices, gdf_with_patches = get_train_val_indices_blocked(
        #     processed_species, 
        #     cell_size=cell_size,
        #     n_splits=0.2,
        # )

        target_binary = target_df[target_species]
        X_train = final_dataset.iloc[train_indices]
        y_train = target_binary.iloc[train_indices]
        X_valid = final_dataset.iloc[val_indices]
        y_valid = target_binary.iloc[val_indices]

        results_df = evaluate_xgb_binary(
            model_params = model_params,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid
        )
        # Add metadata columns for split and species
        results_df['species'] = target_species
        results.append(results_df)

    # Concatenate all results into a single DataFrame
    final_results_df = pd.concat(results, ignore_index=True)
    return final_results_df


def get_pred_100_band_names():
    # folder = "../data/data_layers/predictors_100_QC_normalized"
    # band_names = [f.split('.')[0] for f in os.listdir(folder) if f.endswith('.tif')]
    # band_names.sort()
    # return band_names
    band_names = ['alluvion',
        'annual_precipitation_amount',
        'annual_range_of_air_temperature',
        'barren',
        'bulk_density',
        'clay',
        'combined_mask',
        'coniferous',
        'cropland',
        'deciduous',
        'depot',
        'distance_to_roads',
        'eau_peu_profonde',
        'elevation',
        'eolien',
        'geomflat',
        'geomfootslope',
        'glaciaire',
        'glaciolacustre',
        'human_modification',
        'indifferencie',
        'isothermality',
        'lacustre',
        'lai',
        'marais',
        'marecage',
        'mean_annual_air_temperature',
        'mean_daily_maximum_air_temperature_of_the_warmest_month',
        'mean_daily_mean_air_temperatures_of_the_coldest_quarter',
        'mean_daily_mean_air_temperatures_of_the_driest_quarter',
        'mean_daily_mean_air_temperatures_of_the_warmest_quarter',
        'mean_daily_mean_air_temperatures_of_the_wettest_quarter',
        'mean_daily_minimum_air_temperature_of_the_coldest_month',
        'mean_diurnal_air_temperature_range',
        'mean_monthly_precipitation_amount_of_the_coldest_quarter',
        'mean_monthly_precipitation_amount_of_the_driest_quarter',
        'mean_monthly_precipitation_amount_of_the_warmest_quarter',
        'mean_monthly_precipitation_amount_of_the_wettest_quarter',
        'mixed',
        'ndvi',
        'nitrogen',
        'organic_carbon_density',
        'organique',
        'ph',
        'polar_grass',
        'prairie_humide',
        'precipitation_amount_of_the_driest_month',
        'precipitation_amount_of_the_wettest_month',
        'precipitation_seasonality',
        'quaternaire',
        'roche',
        'ruggedness',
        'sand',
        'silt',
        'soil_organic_carbon',
        'taiga',
        'temperate_grass',
        'temperate_shrub',
        'temperature_seasonality',
        'till',
        'tourbiere_boisee',
        'tourbiere_indifferenciee',
        'tourbiere_minerotrophe',
        'tourbiere_ombrotrophe',
        'urban',
        'water',
        'wetland']
    return band_names

# def compute_pos_weight(targets):
#     """Compute pos_weight dynamically for binary classification."""
#     targets = np.squeeze(targets)
#     pos_count = np.sum(targets)
#     neg_count = len(targets) - pos_count
#     return neg_count / pos_count if pos_count > 0 else 1.0

def compute_pos_weight(targets_df):
    """Calculate positive weights for each species based on class imbalance"""
    num_species = targets_df.shape[1]
    positive_counts = [0] * num_species
    total_samples = len(targets_df)
    
    pos_weights = []
    for i_species in range(num_species):
        count = targets_df.iloc[:, i_species].sum()
        pos_count = count
        neg_count = total_samples - pos_count
        pos_weights.append(neg_count / (count + 1e-8))  # Avoid division by zero
    return pos_weights



def save_results(metrics_df, results_file="results.csv"):
    """Append results to CSV, creating if it doesn't exist."""
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    metrics_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

def evaluate_fusion_cnn(cell_sizes, target_species_arr, processed_species_original_locations, configs, results_file="results.csv"):
    """Evaluate CNN across species, splits, and configurations."""
    # Initialize results DataFrame
    all_results = pd.DataFrame()
    print("Starting evaluation...")
    # Get predictor band names
    pred_100_band_names = get_pred_100_band_names()

    for target_species in target_species_arr:
        print(f"\nProcessing species: {target_species}")
        
        # Create target DataFrame
        target_df = create_target_species_df(target_species, processed_species_original_locations)
        if target_df.empty:
            print(f"Warning: No data for {target_species}")
            continue
        
        # Get feature importance
        pred_100_by_locations = pd.read_csv("./data/saved_df/processed_species_original_locations_point_data.csv")
        feature_importance = get_feature_importance_for_species(
            processed_species_original_locations,
            target=target_species,
            target_df=target_df,
            cont_zonal_stats_path="./data/saved_df/processed_species_original_locations_point_data.csv"
        )
        feature_importance_names = feature_importance['feature_name'].values
        pred_100_indices = [pred_100_band_names.index(band) for band in feature_importance_names if band in pred_100_band_names]
        print(f"Got {len(pred_100_indices)} feature importances for {target_species}")

        for cell_size in cell_sizes:
            print(f"\nCell size: {cell_size} degrees")

            # Option 1: Grid-based split
            train_indices, val_indices, gdf_with_patches = get_train_val_indices_blocked(
                processed_species_original_locations, 
                cell_size=cell_size,
                n_splits=0.2,
            )
            
            # Validate indices
            if len(np.intersect1d(train_indices, val_indices)) > 0:
                raise ValueError("Train and validation indices overlap")
            if max(train_indices) >= len(target_df) or max(val_indices) >= len(target_df):
                raise ValueError(f"Indices out of bounds: max {len(target_df)-1}")
            
            for config in configs:
                if(config.get("use_satelite", False) == False):
                    npy_files = [
                        {
                            "path": "../scratch/data/npy_data/pred100_patches.npy",
                            'name': 'pred_100',
                        },
                    ]
                else:
                    npy_files = [
                        {
                            "path": "../scratch/data/npy_data/sentinel2_patches.npy",
                            'name': 'sentinel2',
                        },
                        {
                            "path": "../scratch/data/npy_data/pred100_patches.npy",
                            'name': 'pred_100',
                        },
                    ]
                training_dataset = create_dataset(
                    npy_files=npy_files,
                    target_df=target_df,
                    processed_species_original_locations=processed_species_original_locations,
                    indices=train_indices,
                    pred_100_indices=pred_100_indices,
                    transform=transforms.Compose([
                        RandomFlipRotate(),
                    ]),
                    include_coords=True,
                    resize_ratio=config.get('resize_ratio', 1),
                    add_point_infos=config.get('add_point_infos', False),
                    feature_importance_only_tabular = config.get('use_pretrained_climate_cnn', False)
                )
                validation_dataset = create_dataset(
                    npy_files=npy_files,
                    target_df=target_df,
                    processed_species_original_locations=processed_species_original_locations,
                    indices=val_indices,
                    pred_100_indices=pred_100_indices,
                    include_coords=True,
                    resize_ratio=config.get('resize_ratio', 1),
                    add_point_infos=config.get('add_point_infos', False),
                    feature_importance_only_tabular = config.get('use_pretrained_climate_cnn', False)
                )

                input_shapes = training_dataset.effective_shapes
            
                print(f"\nTesting config: {config['name']}")
                config['pos_weight'] = compute_pos_weight(training_dataset.targets, config.get('num_species', 1))
                print(f"Computed pos_weight: {config['pos_weight']:.2f}")

                # Create and train model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("Using device:", device)  # <-- Add this
                model = SimpleMultiInputCNNv2(input_shapes, config).to(device)
                trained_model, training_results = training_loop(
                    model, training_dataset, validation_dataset, config
                )

                # Collect metrics
                metrics_row = {
                    'species': target_species,
                    'cell_size': str(cell_size),
                    'config_name': config['name'],
                    'train_positive_ratio': np.sum(training_dataset.targets) / len(training_dataset),
                    'val_positive_ratio': np.sum(validation_dataset.targets) / len(validation_dataset)
                }
                metrics_row.update({f"val_{k}": v for k, v in training_results['final_val_metrics'].items()})
                #metrics_row.update({f"train_{k}": v for k, v in training_results['final_train_metrics'].items()})
                
                # Append to results
                new_metrics_df = pd.DataFrame([metrics_row])
                save_results(new_metrics_df, results_file)

                all_results = pd.concat([all_results, pd.DataFrame([training_results])], ignore_index=True)

    return all_results