import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
import optuna.visualization as vis
from sklearn.pipeline import Pipeline
from optuna.samplers import TPESampler

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

SEED = 42

# ---------- Helper: which models need scaling + unified constructor ----------
NEEDS_SCALING = {LogisticRegression, SVC, KNeighborsClassifier, MLPClassifier}

def make_estimator(model_class, params, seed=SEED):
    """
    Return an estimator ready to fit.
    - If the model is scale-sensitive, returns Pipeline(StandardScaler -> model).
    - Adds random_state/random_seed if supported and not already provided.
    """
    p = dict(params)  # don't mutate caller dict
    try:
        est0 = model_class()
        est_params = est0.get_params()
        if "random_state" in est_params:
            p.setdefault("random_state", seed)
        if "random_seed" in est_params:  # e.g., some libs
            p.setdefault("random_seed", seed)
    except Exception:
        pass

    if model_class in NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()),
                         ("model", model_class(**p))])
    return model_class(**p)

# ========== Load & Prepare Data ==========
EXCLUDED_MOLECULES = [6, 13, 24, 50, 59, 81, 89, 106, 108, 113]
df = build_model_dataframe(task="c", use_halogen_positions=False)
df = df[~df["molecule_id"].isin(EXCLUDED_MOLECULES)]

#X = df.drop(columns=["energy_range", "molecule_id", "name", "result_id"])
X = df.drop(columns=["energy_range", "molecule_id", "name", "result_id", "th_energy", "lowest_bde"])

y = df["energy_range"]
molecule_ids = df["molecule_id"]
molecule_names = df["name"]

# Label encode y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ========== Optuna Objective Functions ==========
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "eval_metric": "mlogloss",
    }
    est = make_estimator(XGBClassifier, params, seed=SEED)
    scores = cross_val_score(est, X, y_encoded, cv=5, scoring="accuracy")
    return scores.mean()

def objective_logreg(trial):
    params = {
        "C": trial.suggest_float("C", 1e-3, 10, log=True),
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "max_iter": 500,
    }
    est = make_estimator(LogisticRegression, params, seed=SEED)
    scores = cross_val_score(est, X, y_encoded, cv=5, scoring="accuracy")
    return scores.mean()

def objective_svc(trial):
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    params = {
        "kernel": kernel,
        "C": trial.suggest_float("C", 0.1, 30.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "probability": True,
    }
    if kernel == "rbf":
        params["gamma"] = trial.suggest_float("gamma", 1e-3, 3e-1, log=True)  # ~1/n_features ≈ 0.07 in range
    else:
        params["gamma"] = "scale"
    est = make_estimator(SVC, params, seed=SEED)
    scores = cross_val_score(est, X, y_encoded, cv=5, scoring="accuracy", n_jobs=-1)
    return scores.mean()

def objective_knn(trial):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 35),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
        "algorithm": "brute",  # stability/determinism
    }
    est = make_estimator(KNeighborsClassifier, params, seed=SEED)
    scores = cross_val_score(est, X, y_encoded, cv=5, scoring="accuracy", n_jobs=-1)
    return scores.mean()

def objective_mlp(trial):
    # ----- architecture sampled with scalars (storage-safe) -----
    n_layers = trial.suggest_int("n_layers", 1, 3)
    w1 = trial.suggest_categorical("width1", [32, 64, 100, 128, 160])
    w2 = trial.suggest_categorical("width2", [16, 32, 50, 64, 128]) if n_layers >= 2 else None
    w3 = trial.suggest_categorical("width3", [8, 16, 32, 64]) if n_layers == 3 else None
    hidden_layer_sizes = (w1,) if n_layers == 1 else (w1, w2) if n_layers == 2 else (w1, w2, w3)

    # store the resolved tuple so we can rebuild later
    trial.set_user_attr("hidden_layer_sizes", hidden_layer_sizes)

    # ----- solver / optimization -----
    #solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])
    solver = trial.suggest_categorical("solver", ["adam"])


    # common params
    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "max_iter": trial.suggest_int("max_iter", 800, 3000),
        "random_state": SEED,
        "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True),  # a bit looser tol helps convergence
    }

    if solver == "adam":
        params.update({
            "solver": "adam",
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
            "early_stopping": True,
            "n_iter_no_change": trial.suggest_int("n_iter_no_change", 15, 40),
            "validation_fraction": 0.1,  # shows validation_scores_ per epoch
        })
    else:
        # lbfgs ignores early_stopping; often converges quickly on small datasets
        params.update({"solver": "lbfgs"})

    est = make_estimator(MLPClassifier, params, seed=SEED)
    scores = cross_val_score(est, X, y_encoded, cv=5, scoring="accuracy", n_jobs=-1)
    return scores.mean()

def objective_brf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
    }
    est = make_estimator(BalancedRandomForestClassifier, params, seed=SEED)  # trees: no scaling
    scores = cross_val_score(est, X, y_encoded, cv=5, scoring="accuracy", n_jobs=-1)
    return scores.mean()

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    est = make_estimator(RandomForestClassifier, params, seed=SEED)  # trees: no scaling
    scores = cross_val_score(est, X, y_encoded, cv=5, scoring="accuracy", n_jobs=-1)
    return scores.mean()

# ========== Run Optuna and LOO for each model ==========
def run_model(model_name, objective_fn, model_class):
    print(f"\n===== {model_name} =====")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective_fn, n_trials=500, timeout=300)

    # Show Optuna visualizations
    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_slice(study).show()

    best_params = study.best_params

    if model_class is MLPClassifier:
        hls = study.best_trial.user_attrs["hidden_layer_sizes"]
        # drop structural keys and inject resolved tuple
        best_params = {k: v for k, v in best_params.items() if k not in ("n_layers", "width1", "width2", "width3")}
        best_params["hidden_layer_sizes"] = hls

    print("Best params:", best_params)

    # LOO CV with best params
    loo = LeaveOneOut()
    predictions, actuals, molecule_list, molecule_name_list = [], [], [], []

    for train_idx, test_idx in loo.split(X.values):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[test_idx]

        est = make_estimator(model_class, best_params, seed=SEED)
        est.fit(X_train, y_train)
        pred = est.predict(X_val)[0]

        predictions.append(pred)
        actuals.append(y_val[0])
        molecule_list.append(molecule_ids.iloc[test_idx].values[0])
        molecule_name_list.append(molecule_names.iloc[test_idx].values[0])

    # Decode integer labels back to original class names
    predictions_decoded = label_encoder.inverse_transform(predictions)
    actuals_decoded = label_encoder.inverse_transform(actuals)

    accuracy = accuracy_score(actuals_decoded, predictions_decoded)
    precision, recall, f1, _ = precision_recall_fscore_support(actuals_decoded, predictions_decoded, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(actuals_decoded, predictions_decoded, labels=["0-1eV", "1-2eV", "2eV+"])

    results_df = pd.DataFrame({
        "Molecule ID": molecule_list,
        "Molecule Name": molecule_name_list,
        "Actual Energy Range": actuals_decoded,
        "Predicted Energy Range": predictions_decoded
    })
    print("\nLOOCV Predictions:")
    print(results_df.to_string(index=False))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", pd.DataFrame(conf_matrix, index=["0-1eV", "1-2eV", "2eV+"], columns=["0-1eV", "1-2eV", "2eV+"]))
    print("Best params:", best_params)

# Run models
name_model = int(input("Please select which model you would like to try:\n1 - XGBoost\n2 - Logistic Regression\n3 - SVC\n4 - KNN\n5 - MLP\n6 - Balanced RF\n7 - Random Forest\n"))
if name_model == 1:
    run_model("XGBoost", objective_xgb, XGBClassifier)
elif name_model == 2:
    run_model("Logistic Regression", objective_logreg, LogisticRegression)
elif name_model == 3:
    run_model("SVC", objective_svc, SVC)
elif name_model == 4:
    run_model("KNN", objective_knn, KNeighborsClassifier)
elif name_model == 5:
    run_model("MLP", objective_mlp, MLPClassifier)
elif name_model == 6:
    run_model("Balanced RF", objective_brf, BalancedRandomForestClassifier)
elif name_model == 7:
    run_model("Random Forest", objective_rf, RandomForestClassifier)

