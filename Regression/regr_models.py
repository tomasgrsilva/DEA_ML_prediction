import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from common.mlflow.mlflow_logger import log_model_results
import matplotlib.pyplot as plt
import numpy as np
from common.feature_engineering.df_eng import build_model_dataframe
from common.helper_functions_f import plot_predictions, evaluate_on_test_molecules, plot_mlp_epoch_curves
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

SEED=42

# Models that need scaling
NEEDS_SCALING = {SVR, KNeighborsRegressor, ElasticNet, MLPRegressor}

def make_estimator(model_class, params, seed=42):
    """Wrap scale-sensitive models in StandardScaler; also set seeds if available."""
    # copy so we don't mutate the original dict
    p = dict(params)
    # add seeds if supported
    try:
        if "random_state" in model_class().get_params():
            p.setdefault("random_state", seed)
        if "random_seed" in model_class().get_params():  # CatBoost
            p.setdefault("random_seed", seed)
    except Exception:
        pass

    if model_class in NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()),
                         ("model", model_class(**p))])
    return model_class(**p)

# ========== Prepare Data ==========
TEST_MOLECULES = [6, 13, 24, 50, 59, 81, 89, 106, 108, 113]
df = build_model_dataframe(use_halogen_positions=True)
df_train = df[~df["molecule_id"].isin(TEST_MOLECULES)].copy()  # Training dataset excludes testing molecules
df_test = df[df["molecule_id"].isin(TEST_MOLECULES)].copy()  # Testing dataset with only the selected molecules

#===== Feature Selection =====
#X = df_train.drop(columns=["peak", "molecule_id", "name", "result_id"])
X = df_train.drop(columns=["peak", "molecule_id", "name", "result_id", "lowest_bde", "Br5", "Br6", "Br7", "Br8", "Br9", "Br10", "Cl6", "I3", "F5", "F7", "F8", "F9", "F10", "F12", "F14", "F16", "F17", "F18", "F20"])

y = df_train["peak"]
molecule_ids = df_train["molecule_id"]
molecule_names = df_train["name"]

#X_test = df_test.drop(columns=["peak", "molecule_id", "name", "result_id"])
X_test = df_test.drop(columns=["peak", "molecule_id", "name", "result_id", "lowest_bde", "Br5", "Br6", "Br7", "Br8", "Br9", "Br10", "Cl6", "I3", "F5", "F7", "F8", "F9", "F10", "F12", "F14", "F16", "F17", "F18", "F20"])

y_test = df_test["peak"]
test_names = df_test["name"]


# ========== Model Configurations ==========
model_configs = {
    1: {
        "name": "KNN",
        "model": KNeighborsRegressor,
        "params": {"n_neighbors": 5, "weights": "distance", "p": 2}
    },
    2: {
        "name": "SVR",
        "model": SVR,
        "params": {"C": 0.112719689024738, "epsilon": 0.28519625205604515,"kernel": "linear", "gamma": "scale"}
    },
    3: {
        "name": "XGBoost",
        "model": XGBRegressor,
        "params": {"n_estimators": 195, "max_depth": 3, "learning_rate": 0.16893122701935406, "subsample": 0.5954612794496502, "random_state": SEED}
    },
    4: {
        "name": "LGBM",
        "model": lgb.LGBMRegressor,
        "params": {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.198889418220904, "random_state": SEED}
    },
    5: {
        "name": "ElasticNet",
        "model": ElasticNet,
        "params": {"alpha": 0.07300777463647719, "l1_ratio": 0.014981799315990519, "random_state": SEED}
    },
    6: {
        "name": "CatBoost",
        "model": CatBoostRegressor,
        "params": {"iterations": 291, "depth": 3, "learning_rate": 0.28158844316752973, "verbose": 0, "random_seed": SEED}
    },
    7: {
        "name": "Random Forest",
        "model": RandomForestRegressor,
        "params": {
            "n_estimators": 514, #954
            "max_depth": 23, #44
            "min_samples_split": 4,
            "min_samples_leaf": 2, #1
            "max_features": None,
            "random_state": SEED
        }
    },
    8: {
        "name": "MLP",
        "model": MLPRegressor,
        "params": {
            "hidden_layer_sizes": (192, 32, 32),  # from Optuna best_params
            "activation": "logistic",  # sigmoid
            "solver": "adam",
            "alpha": 0.00013799505730715503,
            "learning_rate_init": 0.0008965906299676472,
            "learning_rate": "adaptive",
            "max_iter": 1673,
            "early_stopping": True,  # valid with solver="adam"
            "n_iter_no_change": 40,
            "tol": 0.0009180718000535249,
            "random_state": SEED
        }
    }
    }



# ========== LOOCV Runner ==========
def run_selected_model(selection: int):
    config = model_configs[selection]
    print(f"\n===== Running {config['name']} with Best Params =====")

    model_class = config["model"]
    best_params = config["params"]

    loo = LeaveOneOut()
    predictions, actuals, molecule_list, molecule_name_list = [], [], [], []

    for train_idx, test_idx in loo.split(X.values):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        est = make_estimator(model_class, best_params, seed=SEED)
        est.fit(X_train, y_train)

        pred = est.predict(X_val)[0]
        pred = max(pred, 0)  # ReLU clamp if you want to keep it

        predictions.append(pred)
        actuals.append(y_val.values[0])
        # FIX: ensure scalar, not 1-row Series
        molecule_list.append(molecule_ids.iloc[test_idx].values[0])
        molecule_name_list.append(molecule_names.iloc[test_idx].values[0])

    # Metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)


    results_df = pd.DataFrame({
        "Molecule ID": molecule_list,
        "Molecule Name": molecule_name_list,
        "Actual Peak": actuals,
        "Predicted Peak": predictions
    })

    print("\nLOOCV Predictions:")
    print(results_df.to_string(index=False))
    print(f"\nFinal Metrics:\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.4f}")

    # ======= Train final model on full data for logging ========
    final_model = make_estimator(model_class, best_params, seed=SEED)
    final_model.fit(X, y)

    # Plot epoch curves if this is the MLP
    if config["name"] == "MLP":
        plot_mlp_epoch_curves(final_model, title="MLPRegressor training")


    # ======= PLOT LOO CV =======
    plot_predictions(actuals, predictions, mae, rmse, r2, model_name=config['name'])

    # Evaluate and retrieve predictions
    y_pred = evaluate_on_test_molecules(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        molecule_names=test_names,
        scaler=None,
        task_type="regression",  # or "classification"
        model_name=config["name"],
        scale_input=False
    )

    # ========== Log to MLflow ==========
    # Log with MLflow
    log_model_results(
        model=final_model,
        X_train=X,
        y_train=y,
        val_predictions=predictions,  # from LOOCV
        y_val=actuals,  # from LOOCV
        test_predictions=y_pred,
        y_test=y_test,
        params=best_params,
        experiment_name=f"Regression_{config['name']}",
        save_dir="../results",
        tracking_uri="file:///C:/Users/tomas/PycharmProjects/DEA/mlruns",
        task_type="regression",
        model_type="sklearn",
        registered_model_name=f"{config['name']}_Energy_Regressor",
        use_mlflow=True
    )

# ========== User Selection ==========
selection = int(input("Select model to run:\n1 - KNN\n2 - SVR\n3 - XGBOOST\n4 - LGBM\n5 - Elastic Net\n6 - Cat Boost\n7 - Random Forest\n8 - MLP\n"))
if selection in model_configs:
    run_selected_model(selection)
else:
    print("Invalid selection.")
