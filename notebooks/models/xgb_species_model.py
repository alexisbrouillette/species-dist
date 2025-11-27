import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from torchgeo.models import ResNet50_Weights, resnet50
from model_pretraining import transfer_pretrained_weights, ClimatePretrainingCNN

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb

# ---------------- Feature extractor ----------------
class FlexibleCNNBranch(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.output_size = 128

    def forward(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1)

class SimpleMultiSpeciesFeatureExtractor(nn.Module):
    def __init__(self, input_shapes):
        super().__init__()
        self.branches = nn.ModuleList()
        branch_output_sizes = []

        for input_shape in input_shapes:
            source = input_shape['source']
            shape = input_shape['shape']
            branch_output_size = None

            if source == 'point_infos':
                branch = nn.Sequential(
                    nn.Linear(shape[0], 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                branch_output_size = 32

            elif source == 'sentinel2':
                weights = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS
                model = resnet50(weights=weights)
                model.fc = nn.Identity()
                branch = model
                branch_output_size = 2048

            elif source == 'pred_100':
                (c, _, _) = shape
                branch = FlexibleCNNBranch(c)
                pretrained_model = ClimatePretrainingCNN()
                pretrained_model.load_state_dict(torch.load("climate_pretrained_cnn.pth", map_location="cpu"))
                transfer_pretrained_weights(pretrained_model, branch)
                branch_output_size = branch.output_size

            branch_output_sizes.append(branch_output_size)
            self.branches.append(branch)

        total_features = sum(branch_output_sizes)

        self.fusion = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.output_size = 256

    def forward(self, inputs):
        branch_outputs = []
        for branch, x in zip(self.branches, inputs):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            branch_outputs.append(branch(x))
        combined = torch.cat(branch_outputs, dim=1)
        fused = self.fusion(combined)
        return fused

# ---------------- Boosting wrapper ----------------

def compute_pos_weight(y):
    # y: binary array of shape [num_samples]
    num_pos = y.sum()
    num_neg = len(y) - num_pos
    return max(num_neg / max(num_pos, 1), 1.0)  # at least 1

class BoostedHeadsClassifier:
    def __init__(self, species_names, prevalences, num_rounds=200, learning_rate=0.05, max_depth=6):
        self.species_names = species_names
        self.prevalences = prevalences
        self.models = {}
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        for i, sp in enumerate(self.species_names):
            y_train_sp = y_train[:, i]
            dtrain = xgb.DMatrix(X_train, label=y_train_sp)

            # Validation for early stopping (if provided)
            evals = []
            if X_val is not None and y_val is not None:
                y_val_sp = y_val[:, i]
                dval = xgb.DMatrix(X_val, label=y_val_sp)
                evals = [(dval, "eval")]

            # compute pos_weight from prevalence
            pos_weights = [(1 - p) / max(p, 1e-6) for p in species_prevalences]  # per-species

            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "eta": self.learning_rate,
                "max_depth": self.max_depth,
                "scale_pos_weight": scale_pos_weight,
                "verbosity": 0,
            }

            booster = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=pos_weights[i],  # one booster per species
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.models[sp] = booster

    def predict(self, X):
        preds = []
        dtest = xgb.DMatrix(X)
        for sp in self.species_names:
            preds.append(self.models[sp].predict(dtest))
        return np.stack(preds, axis=1)


# ---------------- Training procedure ----------------
def extract_features(dataloader, feature_extractor, device):
    feature_extractor.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Handle flexible batch format
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = [inputs.to(device)]

            labels = labels.cpu().numpy()
            feats = feature_extractor(inputs)

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels)

    return np.concatenate(all_features), np.concatenate(all_labels)


from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
from sklearn.metrics import f1_score

# def find_optimal_threshold(y_true, y_prob, num_thresholds=100):
#     """Find the threshold that maximizes F1 for a single species."""
#     best_thr, best_f1 = 0.5, -1
#     for thr in np.linspace(0, 1, num_thresholds):
#         y_pred = (y_prob >= thr).astype(int)
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         if f1 > best_f1:
#             best_f1, best_thr = f1, thr
#     return best_thr, best_f1
def find_optimal_threshold(y_true, y_prob, min_recall=0.3, num_thresholds=100):
    best_thr, best_prec = 0.5, 0.0
    for thr in np.linspace(0, 1, num_thresholds):
        y_pred = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        if rec >= min_recall and prec > best_prec:
            best_prec, best_thr = prec, thr
    return best_thr, best_prec

def evaluate_species_models(booster, X_val, y_val, species_names, use_optimal_threshold=True):
    preds = booster.predict(X_val)

    results = {}
    thresholds = {}

    for i, sp in enumerate(species_names):
        y_true = y_val[:, i]
        y_prob = preds[:, i]

        if use_optimal_threshold:
            thr, _ = find_optimal_threshold(y_true, y_prob)
        else:
            thr = 0.5
        thresholds[sp] = thr

        y_pred = (y_prob >= thr).astype(int)
        results[sp] = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "threshold": thr
        }

    return results, thresholds


def train_with_boosting(train_dataset, val_dataset, input_shapes, num_species, species_prevalences, species_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = SimpleMultiSpeciesFeatureExtractor(input_shapes).to(device)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 1. Extract features
    X_train, y_train = extract_features(train_loader, feature_extractor, device)
    X_val, y_val = extract_features(val_loader, feature_extractor, device)

    # 2. Train boosting models
    booster = BoostedHeadsClassifier(species_names, species_prevalences)
    booster.fit(X_train, y_train, X_val, y_val)

    # 3. Evaluate
    xgb_results, thresholds = evaluate_species_models(booster, X_val, y_val, species_names, use_optimal_threshold=True)

    return feature_extractor, booster, xgb_results, thresholds


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_results_by_rarity(results, species_prevalences, num_groups=4):
    """
    results: dict {species_name: metrics_dict}
    species_prevalences: list of prevalence values in [0,1] per species
    num_groups: how many rarity groups to split into (default=4 quartiles)
    """
    species_names = list(results.keys())
    metrics = ["precision", "recall", "f1", "accuracy"]

    # Put everything into a DataFrame
    df = pd.DataFrame([
        {
            "species": species_names[i],
            "prevalence": species_prevalences[i],
            **results[species_names[i]]
        }
        for i in range(len(species_names))
    ])

    # Assign rarity groups
    df["rarity_group"] = pd.qcut(df["prevalence"], num_groups, labels=False, duplicates="drop")

    # Aggregate metrics per group
    group_means = df.groupby("rarity_group")[metrics].mean()

    # Plot
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4), sharey=True)
    for j, metric in enumerate(metrics):
        axes[j].bar(group_means.index.astype(str), group_means[metric])
        axes[j].set_title(f"Avg {metric} by rarity group")
        axes[j].set_xlabel("Rarity group (low=rare, high=common)")
        axes[j].set_ylabel(metric)

    plt.tight_layout()
    plt.show()

    return df, group_means
