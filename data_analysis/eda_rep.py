import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

def run_eda(df: pd.DataFrame, target_col: str, save_dir: str = "eda_report", task_type: str = "regression"):
    os.makedirs(save_dir, exist_ok=True)

    # --- Basic Info ---
    with open(os.path.join(save_dir, "dataset_info.txt"), "w") as f:
        df.info(buf=f)
        f.write("\n\nDescribe:\n")
        f.write(df.describe().to_string())

    # --- Missing Values ---
    missing_summary = df.isnull().sum()
    missing_summary = pd.DataFrame({
        'MissingCount': missing_summary,
        'MissingPct': (missing_summary / len(df)) * 100
    })
    missing_summary[missing_summary.MissingCount > 0].to_csv(os.path.join(save_dir, "missing_summary.csv"))

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Value Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "missing_values.png"))
    plt.close()

    # --- Duplicates ---
    num_duplicates = df.duplicated().sum()
    with open(os.path.join(save_dir, "duplicates.txt"), "w") as f:
        f.write(f"Number of duplicate rows: {num_duplicates}")

    # --- Feature Types ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors="ignore")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.drop(target_col, errors="ignore")

    # --- Numeric Distributions ---
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"hist_{col}.png"))
        plt.close()

    # --- Categorical Distributions ---
    for col in cat_cols:
        plt.figure(figsize=(10, 4))
        df[col].value_counts().head(20).plot(kind='bar')
        plt.title(f"Value Counts: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"value_counts_{col}.png"))
        plt.close()

    # --- Target Distribution ---
    plt.figure()
    if task_type == "regression":
        sns.histplot(df[target_col], kde=True, bins=30)
    else:
        sns.countplot(x=target_col, data=df)
        class_counts = df[target_col].value_counts()
        class_perc = df[target_col].value_counts(normalize=True) * 100
        class_summary = pd.DataFrame({'Count': class_counts, 'Percentage': class_perc})
        class_summary.to_csv(os.path.join(save_dir, "class_distribution.csv"))
    plt.title(f"Target Distribution: {target_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "target_distribution.png"))
    plt.savefig(os.path.join(save_dir, "target_distribution.eps"), format="eps")  # EPS save
    plt.close()

    # --- Correlation Matrix ---
    if task_type == "regression":
        corr = df[numeric_cols.to_list() + [target_col]].corr()
    else:
        corr = df[numeric_cols.to_list()].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_matrix.png"))
    plt.savefig(os.path.join(save_dir, "correlation_matrix.eps"), format="eps")  # EPS save
    plt.close()

    # --- Highly Correlated Features ---
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [(col, row, upper_tri.loc[row, col])
                 for col in upper_tri.columns
                 for row in upper_tri.index
                 if abs(upper_tri.loc[row, col]) > 0.95]
    with open(os.path.join(save_dir, "highly_correlated_pairs.txt"), "w") as f:
        for a, b, val in high_corr:
            f.write(f"{a} - {b}: {val:.2f}\n")

    # --- Low Variance Features ---
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df[numeric_cols].fillna(0))
    low_var_cols = [col for col, keep in zip(numeric_cols, selector.get_support()) if not keep]
    with open(os.path.join(save_dir, "low_variance_features.txt"), "w") as f:
        f.write("\n".join(low_var_cols))

    # --- Skewness & Kurtosis ---
    skew_kurt = df[numeric_cols].agg(['skew', 'kurtosis']).T
    skew_kurt.to_csv(os.path.join(save_dir, "skewness_kurtosis.csv"))

    # --- Outlier Detection (IQR method) ---
    outliers = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[col] = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
    pd.Series(outliers).sort_values(ascending=False).to_csv(os.path.join(save_dir, "outlier_counts.csv"))

    # --- Feature vs Target Plots (Regression only) ---
    if task_type == "regression":
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col], y=df[target_col])
            plt.title(f"{col} vs {target_col}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{col}_vs_target.png"))
            plt.savefig(os.path.join(save_dir, f"{col}_vs_target.eps"), format="eps")  # EPS save
            plt.close()

    # --- Feature Importance (Tree-Based) ---
    X = df[numeric_cols].fillna(0)
    y = df[target_col]
    if task_type == "classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    model = RandomForestClassifier(random_state=42) if task_type == "classification" else RandomForestRegressor(random_state=42)
    model.fit(X, y)
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    importances.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importances.head(20))
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top_features.png"))
    plt.close()

    # --- PCA (2D) ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    if task_type == "classification":
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set2")
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
        plt.colorbar(label=target_col)
    plt.title("PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_projection.png"))
    plt.close()

    # --- PCA with Plotly (optional) ---
    try:
        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_pca[target_col] = y
        fig = px.scatter(df_pca, x="PC1", y="PC2", color=target_col,
                         title="PCA Projection (Interactive)", width=800, height=600)
        fig.write_html(os.path.join(save_dir, "pca_projection_plotly.html"))
    except Exception as e:
        print("Plotly PCA failed:", e)

    print(f"EDA report saved to: {save_dir}")
