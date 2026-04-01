"""
Step 4: PCA and Feature Analysis
=================================
Performs PCA, correlation analysis, VIF checks, and factor analysis on the
14 stylometric features to understand the feature space structure.

"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDE_COLS = ["text", "label", "source", "generator"]


def load_and_prepare(input_path: str, exclude_generators: list = None):
    """Load feature CSV and prepare feature matrix."""
    df = pd.read_csv(input_path)

    # Optional: filter out specific generators
    if exclude_generators:
        for gen in exclude_generators:
            df = df[df["generator"] != gen]

    # Remove problematic columns if present
    for col in ["sentence_complexity"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    return df, feature_cols, X_std


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(X_std, feature_cols, output_dir):
    """Run PCA and save explained variance plot and loadings."""
    pca = PCA()
    X_pca = pca.fit_transform(X_std)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # Print variance thresholds
    for thresh in [0.8, 0.9, 0.95]:
        k = np.argmax(cum_var >= thresh) + 1
        print(f"  {int(thresh*100)}% variance explained by {k} components")

    # Effective rank
    eigvals = pca.explained_variance_
    p = eigvals / eigvals.sum()
    p = p[p > 0]
    eff_rank = np.exp(-np.sum(p * np.log(p)))
    print(f"  Effective rank: {eff_rank:.2f}")

    # Cumulative variance plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
    plt.axhline(0.9, linestyle="--", color="gray", label="90% variance")
    plt.axhline(0.95, linestyle="--", color="black", label="95% variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA on Stylometric Feature Matrix")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_explained_variance.png"), dpi=150)
    plt.close()

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(feature_cols))],
        index=feature_cols,
    )
    loadings.to_csv(os.path.join(output_dir, "loadings.csv"))

    return pca, X_pca, loadings


def correlation_analysis(df, feature_cols, output_dir, threshold=0.70):
    """Compute and visualize feature correlation matrix."""
    X = df[feature_cols].copy()
    X = X.loc[:, X.std() > 0]

    corr = X.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close()

    # High-correlation pairs
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > threshold:
                pairs.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 4)))

    if pairs:
        print(f"\n  High-correlation pairs (|r| > {threshold}):")
        for f1, f2, r in pairs:
            print(f"    {f1} <-> {f2}: {r}")

    return corr


def vif_analysis(df, feature_cols, output_dir):
    """Compute Variance Inflation Factors."""
    X = df[feature_cols].copy()
    X = X.loc[:, X.std() > 0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False)
    vif_data.to_csv(os.path.join(output_dir, "vif_table.csv"), index=False)

    print("\n  VIF analysis:")
    for _, row in vif_data.iterrows():
        flag = " (!)" if row["VIF"] > 10 else ""
        print(f"    {row['feature']:25s} VIF={row['VIF']:.2f}{flag}")

    return vif_data


def pca_scatter_plots(df, X_pca, output_dir):
    """Generate 2D and 3D PCA scatter plots (human vs LLM)."""
    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]
    df_plot["PC3"] = X_pca[:, 2]

    # 2D scatter
    plt.figure(figsize=(8, 6))
    for label in df_plot["label"].unique():
        subset = df_plot[df_plot["label"] == label]
        plt.scatter(subset["PC1"], subset["PC2"], alpha=0.4, label=label)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("Human vs LLM in Latent Stylistic Space")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter_2d.png"), dpi=150)
    plt.close()

    # 3D scatter
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for label in df_plot["label"].unique():
        subset = df_plot[df_plot["label"] == label]
        ax.scatter(subset["PC1"], subset["PC2"], subset["PC3"], alpha=0.4, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Human vs LLM in 3D Stylistic Space")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter_3d.png"), dpi=150)
    plt.close()

    # Generator clustering
    gen_means = df_plot.groupby("generator")[["PC1", "PC2"]].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=gen_means, x="PC1", y="PC2", hue="generator", s=120)
    plt.title("Generator Clustering in Stylistic Space")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generator_clustering.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCA and feature analysis")
    parser.add_argument("--input", required=True, help="Path to models_generations_with_features.csv")
    parser.add_argument("--output_dir", default="./pca_results", help="Output directory")
    parser.add_argument("--exclude_generators", nargs="*", default=[], help="Generators to exclude")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    df, feature_cols, X_std = load_and_prepare(args.input, args.exclude_generators)
    print(f"  Shape: {df.shape}, Features: {len(feature_cols)}")

    print("\nRunning PCA...")
    pca, X_pca, loadings = run_pca(X_std, feature_cols, args.output_dir)

    print("\nCorrelation analysis...")
    correlation_analysis(df, feature_cols, args.output_dir)

    print("\nVIF analysis...")
    vif_analysis(df, feature_cols, args.output_dir)

    print("\nGenerating scatter plots...")
    pca_scatter_plots(df, X_pca, args.output_dir)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
