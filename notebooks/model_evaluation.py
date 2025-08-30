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

    res_df = pd.DataFrame({
        'val_accuracy': [val_accuracy],
        'train_accuracy': [train_accuracy],
        'val_f1': [val_f1],
        'val_roc_auc': [val_roc_auc],
        'val_recall': [val_recall],
        'train_positive_ratio' : train_positive_ratio,
        'val_positive_ratio' : val_positive_ratio,
        'train/val_nb_ratio': len(y_train) / len(y_valid)
    })
    return res_df



def evaluate_xgb_for_species(target_species_arr, splits, grid_config, model, processed_species, model_params):
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