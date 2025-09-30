import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
import optuna.visualization as vis

# Models
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline


from common.feature_engineering.df_eng import build_model_dataframe

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42

# Models that need scaling
NEEDS_SCALING = {SVR, KNeighborsRegressor, ElasticNet, MLPRegressor}

def make_estimator(model_class, params):
    """Return an estimator; wrap with StandardScaler if scaling is needed."""
    # Ensure a random seed for models that support it
    if "random_state" in model_class().get_params().keys():
        params.setdefault("random_state", SEED)
    if "random_seed" in model_class().get_params().keys():  # CatBoost
        params.setdefault("random_seed", SEED)

    if model_class in NEEDS_SCALING:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model_class(**params))
        ])
    return model_class(**params)


# ========== Load & Prepare Data ==========
EXCLUDED_MOLECULES = [6, 13, 24, 50, 59, 81, 89, 106, 108, 113]
df = build_model_dataframe(task="r", use_halogen_positions=True)
df = df[~df["molecule_id"].isin(EXCLUDED_MOLECULES)]

#X = df.drop(columns=["peak", "molecule_id", "name", "result_id"])
X = df.drop(columns=["peak", "molecule_id", "name", "result_id", "lowest_bde", "Br5", "Br6", "Br7", "Br8", "Br9", "Br10", "Cl6", "I3", "F5", "F7", "F8", "F9", "F10", "F12", "F14", "F16", "F17", "F18", "F20"])

y = df["peak"]
molecule_ids = df["molecule_id"]
molecule_names = df["name"]

# ========== Optuna Objective Functions ==========
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }
    est = make_estimator(XGBRegressor, params)  # returns bare model (no scaling)
    scores = cross_val_score(est, X, y, cv=5, scoring="neg_mean_absolute_error")
    return scores.mean()

def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 2),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
    }
    est = make_estimator(lgb.LGBMRegressor, params)
    scores = cross_val_score(est, X, y, cv=5, scoring="neg_mean_absolute_error")
    return scores.mean()

def objective_elastic(trial):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }
    est = make_estimator(ElasticNet, params)
    scores = cross_val_score(est, X, y, cv=5, scoring="neg_mean_absolute_error")
    return scores.mean()

def objective_svr(trial):
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    params = {
        "kernel": kernel,
        "C": trial.suggest_float("C", 0.1, 50.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 0.5),
    }
    if kernel == "rbf":
        # 30–34 standardized features -> gamma around ~1/n_features ≈ 0.03
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
    est = make_estimator(SVR, params)
    scores = cross_val_score(est, X, y, cv=5, scoring="neg_mean_absolute_error")
    return scores.mean()

def objective_knn(trial):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
        "algorithm": "brute",  # stability
    }
    est = make_estimator(KNeighborsRegressor, params)
    scores = cross_val_score(est, X, y, cv=5, scoring="neg_mean_absolute_error")
    return scores.mean()

def objective_catboost(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 400),
        "depth": trial.suggest_int("depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "rsm": trial.suggest_float("rsm", 0.5, 0.9),  # feature subsampling helpful with 30+ feats
        "verbose": 0,
    }
    est = make_estimator(CatBoostRegressor, params)
    scores = cross_val_score(est, X, y, cv=5, scoring="neg_mean_absolute_error")
    return scores.mean()

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 4, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": SEED,
        "n_jobs": -1,
    }
    est = make_estimator(RandomForestRegressor, params)
    scores = cross_val_score(est, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    return scores.mean()


def objective_mlp(trial):
    # ----- architecture sampled with scalars (storage-safe) -----
    n_layers = trial.suggest_int("n_layers", 1, 3)
    w1 = trial.suggest_categorical("width1", [8, 16, 32, 50, 64, 100, 128, 160, 192, 256])
    w2 = trial.suggest_categorical("width2", [8, 16, 32, 50, 64, 128, 160, 176]) if n_layers >= 2 else None
    w3 = trial.suggest_categorical("width3", [8, 16, 32]) if n_layers == 3 else None
    hidden_layer_sizes = (w1,) if n_layers == 1 else (w1, w2) if n_layers == 2 else (w1, w2, w3)

    # store the resolved tuple so we can rebuild later
    trial.set_user_attr("hidden_layer_sizes", hidden_layer_sizes)

    # ----- solver / optimization -----
    solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])

    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "max_iter": trial.suggest_int("max_iter", 1200, 3000),
        "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True),  # slightly looser can reduce warnings
        "random_state": SEED,
    }

    if solver == "adam":
        params.update({
            "solver": "adam",
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
            "early_stopping": True,
            "n_iter_no_change": trial.suggest_int("n_iter_no_change", 20, 40),
            "validation_fraction": 0.15,
        })
    else:
        # lbfgs converges fast on small data; no early_stopping/loss_curve_
        params.update({"solver": "lbfgs"})

    est = make_estimator(MLPRegressor, params)
    scores = cross_val_score(est, X, y, cv=5, scoring="r2")
    #scores = cross_val_score(est, X, y, cv=5, scoring="neg_mean_absolute_error")
    #scores = cross_val_score(est, X, y, cv=5, scoring="neg_root_mean_squared_error")
    return scores.mean()


# ========== Optuna Runner with LOOCV ==========
def run_model(model_name, objective_fn, model_class):
    print(f"\n===== {model_name} =====")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective_fn, n_trials=1000, timeout=300)

    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_slice(study).show()

    best_params = study.best_params
    if model_class is MLPRegressor:
        hls = study.best_trial.user_attrs["hidden_layer_sizes"]
        # drop structural keys and inject resolved tuple
        best_params = {k: v for k, v in best_params.items() if k not in ("n_layers", "width1", "width2", "width3")}
        best_params["hidden_layer_sizes"] = hls

    print("Best params:", best_params)

    loo = LeaveOneOut()
    predictions, actuals, molecule_list, molecule_name_list = [], [], [], []
    for train_idx, test_idx in loo.split(X.values):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        est = make_estimator(model_class, best_params)
        est.fit(X_train, y_train)

        pred = est.predict(X_val)[0]
        pred = max(pred, 0)  # your non-negative clamp

        predictions.append(pred)
        actuals.append(y_val.values[0])
        molecule_list.append(molecule_ids.iloc[test_idx].values[0])
        molecule_name_list.append(molecule_names.iloc[test_idx].values[0])

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)

    results_df = pd.DataFrame({
        "Molecule ID": molecule_list,
        "Molecule Name": molecule_name_list,
        "Actual Peak": actuals,
        "Predicted Peak": predictions
    })

    print("\nLOOCV Predictions:")
    print(results_df.to_string(index=False))
    print(f"\nFinal Metrics:\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}")
    print("Best params:", best_params)

# ========== Model Selection ==========
model_choice = int(input("Select regression model to tune:\n1 - KNN\n2 - SVR\n3 - XGBoost\n4 - LightGBM\n5 - ElasticNet\n6 - CatBoost\n7 - Random Forest\n8 - MLP\n"))

if model_choice == 1:
    run_model("KNN", objective_knn, KNeighborsRegressor)
elif model_choice == 2:
    run_model("SVR", objective_svr, SVR)
elif model_choice == 3:
    run_model("XGBoost", objective_xgb, XGBRegressor)
elif model_choice == 4:
    run_model("LightGBM", objective_lgb, lgb.LGBMRegressor)
elif model_choice == 5:
    run_model("ElasticNet", objective_elastic, ElasticNet)
elif model_choice == 6:
    run_model("CatBoost", objective_catboost, CatBoostRegressor)
elif model_choice == 7:
    run_model("Random Forest", objective_rf, RandomForestRegressor)
elif model_choice == 8:
    run_model("MLP", objective_mlp, MLPRegressor)
else:
    print("Invalid model selection.")
