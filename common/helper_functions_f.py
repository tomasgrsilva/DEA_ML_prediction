import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    precision_recall_fscore_support
)

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

def plot_predictions(y_true, y_pred, mae, rmse, r2, model_name="Model"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor='k')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} – Predicted vs. True")
    plt.suptitle(
        f"MAE: {mae:.4f} | "
        f"RMSE: {rmse:.4f} | "
        f"R2: {r2:.4f}", fontsize=10, y=0.97
    )
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_on_test_molecules(
        model,
        X_test,
        y_test,
        molecule_names,
        scaler=None,
        task_type="classification",
        label_encoder=None,
        model_name="Model",
        scale_input=True):
    # guard
    if X_test is None or (hasattr(X_test, "empty") and X_test.empty):
        print("No excluded data to test.")
        return None

    # --- Decide how to scale ---
    # If the model is a Pipeline and has an internal scaler step, let it handle scaling.
    has_internal_scaler = hasattr(model, "named_steps") and ("scaler" in getattr(model, "named_steps", {}))

    if has_internal_scaler:
        X_eval = X_test  # pipeline will scale internally
    elif scale_input and scaler is not None:
        X_eval = scaler.transform(X_test)  # external scaling (legacy path)
    else:
        X_eval = X_test  # no scaling

    if task_type == "classification":
        y_pred_encoded = model.predict(X_eval)
        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(np.asarray(y_pred_encoded).ravel())
        else:
            y_pred = np.asarray(y_pred_encoded).ravel()

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        if label_encoder is not None:
            labels = label_encoder.classes_
        else:
            labels = np.unique(np.concatenate([np.asarray(y_test), y_pred]))
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

        print("\n=== Evaluation on EXCLUDED Molecules (Classification) ===")
        print(pd.DataFrame({
            "Molecule": getattr(molecule_names, "values", molecule_names),
            "Actual":   getattr(y_test, "values", y_test),
            "Predicted": y_pred
        }).to_string(index=False))

        print("Testing metrics:")
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
        plt.figure(figsize=(6, 6))
        disp.plot(cmap="Oranges", values_format='d')
        plt.title(f"Confusion Matrix - {model_name} on Excluded Molecules")
        plt.show()

        return y_pred

    elif task_type == "regression":
        y_pred = model.predict(X_eval)
        y_pred = np.maximum(np.asarray(y_pred).ravel(), 0)

        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        #mape = mean_absolute_percentage_error(y_test, y_pred)

        print("\n=== Evaluation on EXCLUDED Molecules (Regression) ===")
        print(pd.DataFrame({
            "Molecule": getattr(molecule_names, "values", molecule_names),
            "Actual":   getattr(y_test, "values", y_test),
            "Predicted": y_pred
        }).to_string(index=False))

        print(f"\nMSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        # print(f"MAPE: {mape:.4f}")

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
        lo, hi = np.min(y_test), np.max(y_test)
        plt.plot([lo, hi], [lo, hi], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{model_name} on Test Molecules")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return y_pred

    else:
        raise ValueError("Invalid task_type. Must be 'classification' or 'regression'.")



def plot_mlp_epoch_curves(est, title=None):
    """
    Plot per-epoch training curves for sklearn MLP models.

    Works with:
      - MLPRegressor (training loss + validation R^2 when early_stopping=True)
      - MLPClassifier (training log-loss + validation accuracy when early_stopping=True)
      - Either model inside a Pipeline (looks for the MLP as the last step, or any step named 'model').

    Parameters
    ----------
    est : estimator or Pipeline
        Fitted estimator (MLPRegressor/MLPClassifier) or Pipeline containing one.
        Must have been trained with solver='adam' or 'sgd'. ('lbfgs' does not expose loss_curve_.)
    title : str, optional
        Plot title; if None a sensible default is used.
    """
    # --- unwrap the inner MLP if this is a Pipeline ---
    mlp = None
    if isinstance(est, Pipeline):
        # prefer a step actually being an MLP, else try 'model', else last step
        for _, step in est.named_steps.items():
            if isinstance(step, (MLPRegressor, MLPClassifier)):
                mlp = step
                break
        if mlp is None:
            mlp = est.named_steps.get("model", est.steps[-1][1])
    else:
        mlp = est

    if not isinstance(mlp, (MLPRegressor, MLPClassifier)):
        print("plot_mlp_epoch_curves: estimator is not an MLPRegressor/MLPClassifier, skipping.")
        return

    # --- pull curves from sklearn ---
    loss_curve = getattr(mlp, "loss_curve_", None)            # train loss per epoch
    val_scores = getattr(mlp, "validation_scores_", None)     # R^2 (reg) or accuracy (cls) per epoch if early_stopping=True
    n_iter = getattr(mlp, "n_iter_", None)

    if loss_curve is None:
        print("No loss_curve_ found. Use solver='adam' or 'sgd' (lbfgs doesn't expose it).")
        return

    is_classifier = isinstance(mlp, MLPClassifier)
    y_label_train = "Loss (log-loss)" if is_classifier else "Loss (squared-loss)"
    y_label_val   = "Validation accuracy" if is_classifier else "Validation R²"
    default_title = "MLPClassifier training" if is_classifier else "MLPRegressor training"

    epochs = np.arange(1, len(loss_curve) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, loss_curve, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel(y_label_train)
    ttl = title or default_title
    if n_iter is not None:
        ttl += f" (n_iter_={n_iter})"
    plt.title(ttl)
    plt.grid(True)

    # second y-axis for validation metric if available
    if val_scores is not None:
        ax2 = plt.twinx()
        ax2.plot(epochs, val_scores, linestyle="--", label=y_label_val)
        ax2.set_ylabel(y_label_val)
        ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
