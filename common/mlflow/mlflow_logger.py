import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_squared_error, mean_absolute_error
)
import os
import shutil
import numpy as np

def log_confusion_matrix(y_true, y_pred, save_dir, name="confusion_matrix", cmap=plt.cm.Blues):
    class_names = ["0–1 eV", "1–2 eV", "2 eV+"]
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    ax = disp.plot(cmap=cmap).ax_
    ax.set_xlabel("Predicted Resonance Energy Range", fontsize=12)
    ax.set_ylabel("Ground truth Resonance Energy Range", fontsize=12)
    # Define save paths
    png_path = os.path.join(save_dir, f"{name}.png")
    eps_path = os.path.join(save_dir, f"{name}.eps")

    # Save both formats
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(eps_path, format="eps", dpi=600, bbox_inches="tight")
    plt.close()
    return png_path, eps_path


def log_pred_vs_true_plot(y_true, y_pred, save_dir, name="pred_vs_ground_truth"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor='k', s=80)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Ground truth Peak Energy (eV)")
    plt.ylabel("Predicted Peak Energy (eV)")
    plt.title(f"{name.replace('_', ' ').title()} Resonance Peak Energy Plot")
    plt.grid(True)
    plt.tight_layout()

    # Save both PNG and EPS
    png_path = os.path.join(save_dir, f"{name}.png")
    eps_path = os.path.join(save_dir, f"{name}.eps")

    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(eps_path, format="eps", dpi=600, bbox_inches="tight")
    plt.close()
    return png_path, eps_path


def log_model_results(
    *,
    model,
    X_train,
    y_train,
    val_predictions=None,
    y_val=None,
    test_predictions=None,
    y_test=None,
    params: dict,
    experiment_name: str,
    save_dir: str = "./mlruns_artifacts",
    tracking_uri: str = "file:///C:/Users/tomas/PycharmProjects/DEA/mlruns",
    input_example=None,
    task_type: str = "regression",
    model_type: str = "sklearn",
    registered_model_name: str = None,
    use_mlflow: bool = True,
    log_feature_importances: bool = True,
    training_plot_png=None,
    training_plot_eps=None
):
    if not use_mlflow:
        return

    os.makedirs(save_dir, exist_ok=True)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Feature names
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns
        else:
            feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        mlflow.log_param("features_used", ", ".join(feature_names))
        with open(os.path.join(save_dir, "features_used.txt"), "w") as f:
            f.write("\n".join(feature_names))
        mlflow.log_artifact(os.path.join(save_dir, "features_used.txt"))


        # Classification-specific metrics and plots
        if task_type == "classification":
            if val_predictions is not None and y_val is not None:
                mlflow.log_metric("val_accuracy", accuracy_score(y_val, val_predictions))
                mlflow.log_metric("val_precision", precision_score(y_val, val_predictions, average="weighted", zero_division=0))
                mlflow.log_metric("val_recall", recall_score(y_val, val_predictions, average="weighted", zero_division=0))
                mlflow.log_metric("val_f1", f1_score(y_val, val_predictions, average="weighted"))
                val_cm_png, val_cm_eps = log_confusion_matrix(y_val, val_predictions, save_dir, "val_confusion_matrix", cmap=plt.cm.Blues)
                mlflow.log_artifact(val_cm_png)
                mlflow.log_artifact(val_cm_eps)

            if test_predictions is not None and y_test is not None:
                mlflow.log_metric("test_accuracy", accuracy_score(y_test, test_predictions))
                mlflow.log_metric("test_precision", precision_score(y_test, test_predictions, average="weighted", zero_division=0))
                mlflow.log_metric("test_recall", recall_score(y_test, test_predictions, average="weighted", zero_division=0))
                mlflow.log_metric("test_f1", f1_score(y_test, test_predictions, average="weighted"))
                test_cm_png, test_cm_eps = log_confusion_matrix(y_test, test_predictions, save_dir, "test_confusion_matrix", cmap=plt.cm.Oranges)
                mlflow.log_artifact(test_cm_png)
                mlflow.log_artifact(test_cm_eps)

        # Regression-specific plots
        elif task_type == "regression":
            if val_predictions is not None and y_val is not None:
                mlflow.log_metric("val_mae", mean_absolute_error(y_val, val_predictions))
                mlflow.log_metric("val_r2", r2_score(y_val, val_predictions))
                mlflow.log_metric("val_rmse", np.sqrt(mean_squared_error(y_val, val_predictions)))
                mlflow.log_metric("val_mse", mean_squared_error(y_val, val_predictions))
                val_plot_png, val_plot_eps = log_pred_vs_true_plot(y_val, val_predictions, save_dir, "validation_pred_vs_true")
                mlflow.log_artifact(val_plot_png)
                mlflow.log_artifact(val_plot_eps)

            if test_predictions is not None and y_test is not None:
                mlflow.log_metric("test_mae", mean_absolute_error(y_test, test_predictions))
                mlflow.log_metric("test_r2", r2_score(y_test, test_predictions))
                mlflow.log_metric("test_rmse", np.sqrt(mean_squared_error(y_test, test_predictions)))
                mlflow.log_metric("test_mse", mean_squared_error(y_test, test_predictions))
                test_plot_png, test_plot_eps = log_pred_vs_true_plot(y_test, test_predictions, save_dir, "test_pred_vs_true")
                mlflow.log_artifact(test_plot_png)
                mlflow.log_artifact(test_plot_eps)

        # Log tags
        mlflow.set_tag("Task", task_type)
        mlflow.set_tag("Model Type", model_type)

        # Validation predictions
        if y_val is not None and val_predictions is not None:
            results_df = pd.DataFrame({
                "Actual": y_val,
                "Predicted": val_predictions
            })
            results_file = os.path.join(save_dir, "val_predictions.csv")
            results_df.to_csv(results_file, index=False)
            mlflow.log_artifact(results_file)

        # Test predictions
        if y_test is not None and test_predictions is not None:
            results_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": test_predictions
            })
            results_file = os.path.join(save_dir, "test_predictions.csv")
            results_df.to_csv(results_file, index=False)
            mlflow.log_artifact(results_file)

        # Feature importances
        if log_feature_importances and model_type == "sklearn" and hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            })
            fi_file = os.path.join(save_dir, "feature_importances.csv")
            fi_df.to_csv(fi_file, index=False)
            mlflow.log_artifact(fi_file)

        if training_plot_png and training_plot_eps is not None:
            mlflow.log_artifact(training_plot_png, artifact_path="training_plot_png")
            mlflow.log_artifact(training_plot_eps, artifact_path="training_plot_eps")

        # Log the model
        if model_type == "sklearn":
            if input_example is None:
                input_example = X_train[:1]
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=registered_model_name
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    # Cleanup
    if os.path.exists(save_dir):
        try:
            shutil.rmtree(save_dir)
        except Exception as e:
            print(f"Warning: Failed to delete {save_dir}. Reason: {e}")

    print("MLflow run completed. Check MLflow UI for details.")
