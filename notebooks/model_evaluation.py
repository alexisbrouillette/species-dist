from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import numpy as np
import pandas as pd
from model_preprocess import create_target_species_df, get_feature_importance_for_species
from splitting import get_train_val_indices
import xgboost as xgb
def evaluate_model(model_params, X_train, y_train, X_valid, y_valid):

    train_positive_ratio = np.mean(y_train == 1)
    val_positive_ratio = np.mean(y_valid == 1)
    # Fit the model to the training data
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_valid, label=y_valid)

    # compute scale_pos_weight dynamically
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    model_params["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = xgb.train(
        model_params,
        dtrain=dtrain,
        num_boost_round=1000,  # High number, early stopping will cut it
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False   # <--- no more spam
    )

    # Predictions
    y_pred_val = model.predict(xgb.DMatrix(X_valid))
    y_pred_train = model.predict(xgb.DMatrix(X_train))

    # Round predictions for classification metrics
    val_final = np.round(y_pred_val)
    train_final = np.round(y_pred_train)

    # Validation metrics
    val_accuracy = accuracy_score(y_valid, val_final)
    val_f1 = f1_score(y_valid, val_final)
    val_roc_auc = roc_auc_score(y_valid, y_pred_val)  # use probabilities
    val_recall = recall_score(y_valid, val_final)

    # Training accuracy (manual, since Booster has no .score())
    train_accuracy = accuracy_score(y_train, train_final)
    train_f1 = f1_score(y_train, train_final)
    train_roc_auc = roc_auc_score(y_train, y_pred_train)
    train_recall = recall_score(y_train, train_final)

    res_df = pd.DataFrame({
        'train_accuracy': [train_accuracy],
        'train_f1': [train_f1],
        'train_roc_auc': [train_roc_auc],
        'train_recall': [train_recall],
        'val_accuracy': [val_accuracy],
        'val_f1': [val_f1],
        'val_roc_auc': [val_roc_auc],
        'val_recall': [val_recall],
        'train_positive_ratio' : train_positive_ratio,
        'val_positive_ratio' : val_positive_ratio,
        'train/val_nb_ratio': len(y_train) / len(y_valid)
    })
    return res_df



def evaluate_xgb_for_species(target_species_arr, splits, grid_config, processed_species, model_params):
    results = []
    for target_species in target_species_arr:
        target_df = create_target_species_df(target_species, processed_species=processed_species)
        pred_100_by_locations = pd.read_csv("../data/saved_df/processed_species_original_locations_point_data.csv")
        feature_importance = get_feature_importance_for_species(
            processed_species, 
            target=target_species, 
            cont_zonal_stats_path="../data/saved_df/processed_species_original_locations_point_data.csv",
            target_df=target_df
        )
        final_dataset = pred_100_by_locations[feature_importance['feature_name'].values]

        for split in splits:
            train_indices, val_indices, patches_gdf = get_train_val_indices(
                processed_species, 
                grid_config, 
                validation_patch_numbers=split
            )

            target_binary = target_df[target_species]
            X_train = final_dataset.iloc[train_indices]
            y_train = target_binary.iloc[train_indices]
            X_valid = final_dataset.iloc[val_indices]
            y_valid = target_binary.iloc[val_indices]

            results_df = evaluate_model(
                model_params = model_params,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid
            )
            # Add metadata columns for split and species
            results_df['species'] = target_species
            results_df['split'] = str(split)
            results.append(results_df)

    # Concatenate all results into a single DataFrame
    final_results_df = pd.concat(results, ignore_index=True)
    return final_results_df


import pandas as pd
import numpy as np
import os
from shapely.geometry import Point
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from fusion_cnn import MultiSourceNpyDatasetWithCoords, RandomFlipRotate, SimpleMultiInputCNNv2, training_loop
from model_preprocess import get_feature_importance_for_species, create_target_species_df
from splitting import get_train_val_indices

def get_pred_100_band_names():
    folder = "../data/data_layers/predictors_100_QC_normalized"
    band_names = [f.split('.')[0] for f in os.listdir(folder) if f.endswith('.tif')]
    band_names.sort()
    return band_names

def compute_pos_weight(targets):
    """Compute pos_weight dynamically for binary classification."""
    targets = np.squeeze(targets)
    pos_count = np.sum(targets)
    neg_count = len(targets) - pos_count
    return neg_count / pos_count if pos_count > 0 else 1.0

def create_dataset(npy_files, target_df, processed_species_original_locations, indices, pred_100_indices, transform=None, include_coords=False, resize_ratio=1, add_point_infos = False):
    """Create a dataset with validation checks."""
    dataset = MultiSourceNpyDatasetWithCoords(
        npy_files=npy_files,
        targets=target_df.iloc[indices].values,
        processed_species_original_locations=processed_species_original_locations.iloc[indices],
        features_index=pred_100_indices,
        indices=indices,
        transform=transform,
        include_coords=include_coords,
        resize_ratio=resize_ratio,
        add_point_infos=add_point_infos
    )

    pos_ratio = np.sum(dataset.targets) / len(dataset.targets)
    print(f"Dataset size: {len(dataset)}, Positives: {np.sum(dataset.targets)}, Positive ratio: {pos_ratio:.4f}")
    return dataset

def save_results(metrics_df, results_file="results.csv"):
    """Append results to CSV, creating if it doesn't exist."""
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    metrics_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

def evaluate_fusion_cnn(splits, target_species_arr, processed_species_original_locations, configs, results_file="results.csv"):
    """Evaluate CNN across species, splits, and configurations."""
    # Initialize results DataFrame
    all_results = pd.DataFrame()
    
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
        pred_100_by_locations = pd.read_csv("../data/saved_df/processed_species_original_locations_point_data.csv")
        feature_importance = get_feature_importance_for_species(
            processed_species_original_locations,
            target=target_species,
            target_df=target_df,
            cont_zonal_stats_path="../data/saved_df/processed_species_original_locations_point_data.csv"
        )
        feature_importance_names = feature_importance['feature_name'].values
        pred_100_indices = [pred_100_band_names.index(band) for band in feature_importance_names if band in pred_100_band_names]
        print(f"Got {len(pred_100_indices)} feature importances for {target_species}")

        for split_idx, validation_patch_numbers in enumerate(splits):
            print(f"\nProcessing split {split_idx + 1}: {validation_patch_numbers}")
            my_grid_configuration = {'grid_rows': 6, 'grid_cols': 6}

            # Option 1: Grid-based split
            train_indices, val_indices, patches = get_train_val_indices(
                processed_species_original_locations,
                my_grid_configuration,
                validation_patch_numbers=validation_patch_numbers
            )
            
            # Option 2: Stratified split (uncomment to use)
            # train_indices, val_indices = train_test_split(
            #     np.arange(len(target_df)),
            #     test_size=0.22,
            #     stratify=target_df.values,
            #     random_state=42 + split_idx
            # )
            
            # Validate indices
            if len(np.intersect1d(train_indices, val_indices)) > 0:
                raise ValueError("Train and validation indices overlap")
            if max(train_indices) >= len(target_df) or max(val_indices) >= len(target_df):
                raise ValueError(f"Indices out of bounds: max {len(target_df)-1}")
            
            for config in configs:
                if(config.get("use_satelite", False) == False):
                    npy_files = [
                        {
                            "path": "../data/npy_data/pred100_patches.npy",
                            'name': 'pred_100',
                        },
                    ]
                else:
                    npy_files = [
                        {
                            "path": "../data/npy_data/sentinel2_patches.npy",
                            'name': 'sentinel2',
                        },
                        {
                            "path": "../data/npy_data/pred100_patches.npy",
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
                    include_coords=False,
                    resize_ratio=config.get('resize_ratio', 1),
                    add_point_infos=config.get('add_point_infos', False)
                )
                validation_dataset = create_dataset(
                    npy_files=npy_files,
                    target_df=target_df,
                    processed_species_original_locations=processed_species_original_locations,
                    indices=val_indices,
                    pred_100_indices=pred_100_indices,
                    include_coords=False,
                    resize_ratio=config.get('resize_ratio', 1),
                    add_point_infos=config.get('add_point_infos', False)
                )

                input_shapes = training_dataset.effective_shapes
            
                print(f"\nTesting config: {config['name']}")
                config['pos_weight'] = compute_pos_weight(training_dataset.targets)
                print(f"Computed pos_weight: {config['pos_weight']:.2f}")

                # Create and train model
                model = SimpleMultiInputCNNv2(input_shapes, config)
                trained_model, training_results = training_loop(
                    model, training_dataset, validation_dataset, config
                )

                # Collect metrics
                metrics_row = {
                    'species': target_species,
                    'split': split_idx + 1,
                    'validation_patches': str(validation_patch_numbers),
                    'config_name': config['name'],
                    'train_positive_ratio': np.sum(training_dataset.targets) / len(training_dataset),
                    'val_positive_ratio': np.sum(validation_dataset.targets) / len(validation_dataset)
                }
                metrics_row.update({f"val_{k}": v for k, v in training_results['final_val_metrics'].items()})
                metrics_row.update({f"train_{k}": v for k, v in training_results['final_train_metrics'].items()})
                
                # Append to results
                new_metrics_df = pd.DataFrame([metrics_row])
                all_results = pd.concat([all_results, new_metrics_df], ignore_index=True)
                
                # Save after each run
                save_results(all_results, results_file)

    return all_results