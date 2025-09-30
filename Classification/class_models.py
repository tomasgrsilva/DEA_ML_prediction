import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

from common.helper_functions_f import evaluate_on_test_molecules, plot_mlp_epoch_curves
from common.mlflow.mlflow_logger import log_model_results

from sklearn.pipeline import Pipeline

# Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from common.feature_engineering.df_eng import build_model_dataframe

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED=42
SEED_VC=52 #62

# Models that need scaling (distance/gradient-based)
NEEDS_SCALING = {
    LogisticRegression, SVC, KNeighborsClassifier, MLPClassifier, VotingClassifier
}

def make_estimator(model_class, params, seed=42):
    """Wrap scale-sensitive models in StandardScaler; also set seeds if available."""
    p = dict(params)  # don't mutate caller dict
    try:
        if "random_state" in model_class().get_params():
            p.setdefault("random_state", seed)
        if "random_seed" in model_class().get_params():  # e.g., imbalanced/others
            p.setdefault("random_seed", seed)
    except Exception:
        pass

    if model_class in NEEDS_SCALING:
        # One scaler on the input is fine for LR/SVC/KNN/MLP and also harmless for tree models inside Voting
        return Pipeline([("scaler", StandardScaler()), ("model", model_class(**p))])
    return model_class(**p)

# ========== Load & Prepare Data ==========
TEST_MOLECULES = [6, 13, 24, 50, 59, 81, 89, 106, 108, 113]
df = build_model_dataframe(task="c", use_halogen_positions=False)
df_train = df[~df["molecule_id"].isin(TEST_MOLECULES)].copy()  # Training dataset excludes testing molecules
df_test = df[df["molecule_id"].isin(TEST_MOLECULES)].copy()  # Testing dataset with only the selected molecules

#===== Feature Selection =====
#X = df_train.drop(columns=["energy_range", "molecule_id", "name", "result_id"])
X = df_train.drop(columns=["energy_range", "molecule_id", "name", "result_id", "lowest_bde", "th_energy"])

y = df_train["energy_range"]
molecule_ids = df_train["molecule_id"]
molecule_names = df_train["name"]

#"th_energy",
#"lowest_bde",
#"mean_energy",

#X_test = df_test.drop(columns=["energy_range", "molecule_id", "name", "result_id"])
X_test = df_test.drop(columns=["energy_range", "molecule_id", "name", "result_id", "lowest_bde", "th_energy"])

y_test = df_test["energy_range"]
test_names = df_test["name"]

# Label encode y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ========== Define Best Hyperparameters ==========
model_configs = {
    1: {
        "name": "XGBoost",
        "model": XGBClassifier,
        "params": {
            "n_estimators": 60,
            "max_depth": 7,
            "learning_rate": 0.041816969808085865,
            "subsample": 0.950186431236384,
            "colsample_bytree": 0.9754283794668885,
            "eval_metric": "mlogloss",
            "random_state": SEED
        }
    },
    2: {
        "name": "Logistic Regression",
        "model": LogisticRegression,
        "params": {
            "C": 9.831566814514062,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 500,
            "random_state": SEED
        }
    },
    3: {
        "name": "SVC",
        "model": SVC,
        "params": {
            "C": 7.597214746555025,
            "kernel": "rbf",
            "gamma": "auto",
            "class_weight": "balanced",
            "probability": True,
            "random_state": SEED
        }
    },
    4: {
        "name": "KNN",
        "model": KNeighborsClassifier,
        "params": {
            "n_neighbors": 5,
            "weights": "distance",
            "p": 2
        }
    },
    5: {
        "name": "MLP",
        "model": MLPClassifier,
        "params": {
            "hidden_layer_sizes": (160, 32), # (100, 50)
            "activation": "tanh", #relu
            "solver": "lbfgs", #adam
            "alpha": 0.0018040297972195072, #0.09533658318528786
            "max_iter": 1426, #500
            'tol': 0.0004893624195592564,
            "random_state": SEED
        }
    },
    6: {
        "name": "Balanced RF",
        "model": BalancedRandomForestClassifier,
        "params": {
            "n_estimators": 52,
            "max_depth": 6,
            "criterion": "gini",
            "random_state": SEED
        }

    },
    7: {
        "name": "VotingClassifier",
        "model": "VotingClassifier",  # Special case to handle manually
        "params": {}  # Will define base models manually
    },
    8: {
        "name": "Random Forest",
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": 449, #449
            "max_depth": 35, #35
            "min_samples_split": 4, #4
            "min_samples_leaf": 1, #1
            "max_features": None,
            "random_state": SEED
        }
    }
}
# ========== Run LOOCV with Predefined Parameters ==========
def run_selected_model(selection: int):
    config = model_configs[selection]
    print(f"\n===== Running {config['name']} with Best Params =====")

    def build_voting_params(seed_inner):
        """Factory that returns params dict for a fresh VotingClassifier."""
        clf1 = LogisticRegression(
            C=9.831566814514062, penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=500, random_state=seed_inner
        )
        clf2 = XGBClassifier(
            n_estimators=60, max_depth=7, learning_rate=0.041816969808085865,
            subsample=0.950186431236384, colsample_bytree=0.9754283794668885,
            eval_metric="mlogloss", random_state=seed_inner
        )
        clf3 = SVC(
            C=7.597214746555025, kernel="rbf", gamma="auto",
            class_weight="balanced", probability=True, random_state=seed_inner
        )
        return {"estimators": [('lr', clf1), ('xgb', clf2), ('svc', clf3)], "voting": "soft"}

    # pull model class + params (for VC we’ll create fresh ones per fold)
    model_class = VotingClassifier if config["name"] == "VotingClassifier" else config["model"]
    best_params_template = build_voting_params(SEED_VC) if config["name"] == "VotingClassifier" else config["params"]

    loo = LeaveOneOut()
    predictions, actuals, molecule_list, molecule_name_list = [], [], [], []

    # --- LOOCV (always use raw X; scaling handled by make_estimator when needed) ---
    for train_idx, test_idx in loo.split(X.values):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[test_idx]

        # fresh Voting params each fold; otherwise reuse template
        if config["name"] == "VotingClassifier":
            best_params = build_voting_params(SEED_VC)
        else:
            best_params = best_params_template

        est = make_estimator(model_class, best_params, seed=SEED)
        est.fit(X_train, y_train)

        pred = est.predict(X_val)[0]
        predictions.append(pred)
        actuals.append(y_val[0])
        molecule_list.append(molecule_ids.iloc[test_idx].values[0])
        molecule_name_list.append(molecule_names.iloc[test_idx].values[0])

    # Decode to original labels for reporting/metrics
    predictions_decoded = label_encoder.inverse_transform(predictions)
    actuals_decoded = label_encoder.inverse_transform(actuals)

    # --- Metrics ---
    accuracy = accuracy_score(actuals_decoded, predictions_decoded)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actuals_decoded, predictions_decoded, average="weighted", zero_division=0
    )
    conf_matrix = confusion_matrix(actuals_decoded, predictions_decoded, labels=label_encoder.classes_)

    # --- Results table ---
    results_df = pd.DataFrame({
        "Molecule ID": molecule_list,
        "Molecule Name": molecule_name_list,
        "Actual Energy Range": actuals_decoded,
        "Predicted Energy Range": predictions_decoded
    })

    print("\nLOOCV Predictions:")
    print(results_df.to_string(index=False))
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:\n", pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix - {config['name']}")
    plt.show()

    # ======= Train final model on full data for logging (mirror CV path) ========
    if config["name"] == "VotingClassifier":
        final_params = build_voting_params(SEED_VC)
    else:
        final_params = best_params_template

    final_model = make_estimator(model_class, final_params, seed=SEED)
    final_model.fit(X, y_encoded)

    # Show epoch curves if this is the MLP
    if config["name"] == "MLP":
        plot_mlp_epoch_curves(final_model, title="MLPClassifier training")

    # ======= Evaluate on held-out molecules =======
    # No external scaler: the Pipeline (if any) scales internally.
    y_pred = evaluate_on_test_molecules(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        molecule_names=test_names,
        scaler=None,
        task_type="classification",
        label_encoder=label_encoder,
        model_name=config["name"],
        scale_input=False
    )

    # ========== Log to MLflow ==========
    log_model_results(
        model=final_model,
        X_train=X,
        y_train=y,                       # original labels for readability
        val_predictions=predictions,     # encoded ints (OK)
        y_val=actuals,                   # encoded ints (OK)
        test_predictions=y_pred,         # decoded by evaluate_on_test_molecules
        y_test=y_test,                   # original labels
        params=final_params,
        experiment_name=f"Classification_{config['name']}",
        save_dir="../results",
        tracking_uri="file:///C:/Users/tomas/PycharmProjects/DEA/mlruns",
        task_type="classification",
        model_type="sklearn",
        registered_model_name=f"{config['name']}_Energy_Classifier",
        use_mlflow=False
    )

# ========== User Selection ==========
selection = int(input("Select model to run:\n1 - XGBoost\n2 - Logistic Regression\n3 - SVC\n4 - KNN\n5 - MLP\n6 - Balanced RF\n7 - VotingClassifier\n8 - Random Forest\n"))
if selection in model_configs:
    run_selected_model(selection)
else:
    print("Invalid selection.")
