"""
Complete Transfer Learning Analysis System for AI Text Detection
================================================================

This module provides a complete pipeline for analyzing transfer learning
in AI-generated text detection across different sources and generators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random
import warnings
import os
import argparse
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ACL format settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 11

# ACL column widths (inches)
#SINGLE_COL_WIDTH = 3.25
#DOUBLE_COL_WIDTH = 6.75

SINGLE_COL_WIDTH = 7
DOUBLE_COL_WIDTH = 12

# Paper results auto-save directory.
# All analysis functions write CSVs here automatically when this is set.
# Set to None to disable all auto-saving.
PAPER_RESULTS_DIR: str = "./paper_results"

# Internal accumulator — collects results across calls so the master
# key-numbers CSV can be built once all required pieces are available.
_paper_cache: Dict = {}

# ============================================================================
# PAPER RESULTS AUTO-SAVE HELPERS
# ============================================================================

def _csv_save(df: pd.DataFrame, subdir: str, filename: str,
              index: bool = True) -> None:
    """Write df to PAPER_RESULTS_DIR/subdir/filename. No-op if saving disabled."""
    if not PAPER_RESULTS_DIR:
        return
    dest = os.path.join(PAPER_RESULTS_DIR, subdir)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, filename)
    df.to_csv(path, index=index)
    print(f"  [csv] {subdir}/{filename}  ({df.shape[0]}r × {df.shape[1]}c)")


def _aggregate_matrices(results_dict: Dict, key: str) -> pd.DataFrame:
    """Stack per-condition accuracy matrices and return element-wise mean."""
    mats = [results_dict[k][key].astype(float) for k in results_dict]
    return pd.concat(mats).groupby(level=0).mean().round(4)


def _overlap_summary(overlap_dict: Dict, condition_col: str) -> pd.DataFrame:
    """Compute off-diagonal Jaccard statistics for each condition."""
    rows = []
    for cond, mat in overlap_dict.items():
        m = mat.astype(float)
        mask = ~np.eye(m.shape[0], dtype=bool)
        off = m.values[mask]
        off = off[~np.isnan(off)]
        if len(off) == 0:
            continue
        rows.append({
            condition_col:  cond,
            "mean_jaccard": round(float(np.mean(off)), 4),
            "std_jaccard":  round(float(np.std(off)),  4),
            "min_jaccard":  round(float(np.min(off)),  4),
            "max_jaccard":  round(float(np.max(off)),  4),
            "n_pairs":      int(len(off)),
        })
    return pd.DataFrame(rows).sort_values("mean_jaccard", ascending=False)


def _try_save_master_key_numbers() -> None:
    """Build master/19_key_numbers_for_paper.csv when all required pieces exist.

    Required _paper_cache keys:
        global_result, aggregated_df,
        source_summary, source_overlap_summary,
        gen_summary,    gen_overlap_summary
    Optionally enriched by: stability_df, tier_dict
    """
    required = [
        "global_result", "aggregated_df",
        "source_summary", "source_overlap_summary",
        "gen_summary",    "gen_overlap_summary",
    ]
    if not all(k in _paper_cache for k in required):
        return

    gr     = _paper_cache["global_result"]
    agg    = _paper_cache["aggregated_df"]
    ssumm  = _paper_cache["source_summary"]
    ssovlp = _paper_cache["source_overlap_summary"]
    gsumm  = _paper_cache["gen_summary"]
    gsovlp = _paper_cache["gen_overlap_summary"]

    kn: List[Dict] = []

    def _kn(section: str, metric: str, value: float, note: str = "") -> None:
        try:
            v = round(float(value), 4)
        except (TypeError, ValueError):
            v = float("nan")
        kn.append({"section": section, "metric": metric, "value": v, "note": note})

    # RQ1
    _kn("RQ1", "global_accuracy",    gr["accuracy"],  "balanced, all data")
    _kn("RQ1", "global_std",         gr["std"],       "std across CV runs")
    _kn("RQ1", "global_n_per_class", gr["n_human"],   "n_human == n_ai after balancing")
    _kn("RQ1", "domain_acc_mean",    agg["accuracy"].mean())
    _kn("RQ1", "domain_acc_std",     agg["accuracy"].std())
    _kn("RQ1", "domain_acc_min",     agg["accuracy"].min(),
        note=str(agg.loc[agg["accuracy"].idxmin(), "source"]))
    _kn("RQ1", "domain_acc_max",     agg["accuracy"].max(),
        note=str(agg.loc[agg["accuracy"].idxmax(), "source"]))
    _kn("RQ1", "domain_acc_range",   agg["accuracy"].max() - agg["accuracy"].min())

    # RQ2
    _kn("RQ2", "source_full_acc_mean",     ssumm["Avg Acc (Full)"].mean())
    _kn("RQ2", "source_same_cond_mean",    ssumm["Same Condition"].mean())
    _kn("RQ2", "source_cross_cond_mean",   ssumm["Cross Condition"].mean())
    _kn("RQ2", "source_transfer_gap_mean", ssumm["Transfer Gap"].mean())
    _kn("RQ2", "source_transfer_gap_max",  ssumm["Transfer Gap"].max(),
        note=str(ssumm.loc[ssumm["Transfer Gap"].idxmax(), "Name"]))
    _kn("RQ2", "source_top5_acc_mean",     ssumm["Avg Acc (Top-5)"].mean())
    _kn("RQ2", "source_avg_drop_pp",       ssumm["Avg Drop"].mean() * 100)
    _kn("RQ2", "source_overlap_mean",
        ssovlp["mean_jaccard"].mean() if len(ssovlp) > 0 else float("nan"))

    # RQ3
    _kn("RQ3", "gen_full_acc_mean",     gsumm["Avg Acc (Full)"].mean())
    _kn("RQ3", "gen_same_cond_mean",    gsumm["Same Condition"].mean())
    _kn("RQ3", "gen_cross_cond_mean",   gsumm["Cross Condition"].mean())
    _kn("RQ3", "gen_transfer_gap_mean", gsumm["Transfer Gap"].mean())
    _kn("RQ3", "gen_transfer_gap_max",  gsumm["Transfer Gap"].max(),
        note=str(gsumm.loc[gsumm["Transfer Gap"].idxmax(), "Name"]))
    _kn("RQ3", "gen_top5_acc_mean",     gsumm["Avg Acc (Top-5)"].mean())
    _kn("RQ3", "gen_avg_drop_pp",       gsumm["Avg Drop"].mean() * 100)
    _kn("RQ3", "gen_overlap_mean",
        gsovlp["mean_jaccard"].mean() if len(gsovlp) > 0 else float("nan"))
    _kn("RQ3", "gap_rq3_minus_rq2",
        gsumm["Transfer Gap"].mean() - ssumm["Transfer Gap"].mean())

    # RQ4
    full_acc = ssumm["Avg Acc (Full)"].mean()
    top5_acc = ssumm["Avg Acc (Top-5)"].mean()
    _kn("RQ4", "top5_acc_mean",       top5_acc)
    _kn("RQ4", "top5_retention_pct",
        (top5_acc / full_acc * 100) if full_acc > 0 else float("nan"))
    _kn("RQ4", "top5_drop_pp",        (full_acc - top5_acc) * 100)

    # RQ5
    if "stability_df" in _paper_cache and "tier_dict" in _paper_cache:
        stab = _paper_cache["stability_df"]
        td   = _paper_cache["tier_dict"]
        if "tier" not in stab.columns:
            tier_lookup = {}
            for lbl, feats in td.items():
                num = {"tier1_universal": 1,
                       "tier2_context_sensitive": 2,
                       "tier3_specialized": 3}[lbl]
                for f in feats:
                    tier_lookup[f] = num
            stab = stab.copy()
            stab["tier"] = stab["feature"].map(tier_lookup)
        for tier_num in [1, 2, 3]:
            t = stab[stab["tier"] == tier_num]
            _kn("RQ5", f"n_tier{tier_num}_features", len(t))
            if tier_num == 1 and len(t) > 0:
                best = t.iloc[0]
                _kn("RQ5", "best_feature_top5_freq", best["top5_frequency"],
                    note=str(best["feature"]))
                _kn("RQ5", "best_feature_cv",        best["cv_importance"],
                    note=str(best["feature"]))
                _kn("RQ5", "tier1_mean_top5_freq",   t["top5_frequency"].mean())
                _kn("RQ5", "tier1_mean_cv",          t["cv_importance"].mean())

    kn_df = pd.DataFrame(kn)
    _csv_save(kn_df, "master", "19_key_numbers_for_paper.csv", index=False)
    print("  [csv] master/19_key_numbers_for_paper.csv  ← all headline numbers")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_columns(df: pd.DataFrame) -> pd.Index:
    """Extract feature column names from dataframe."""
    return df.drop(columns=["generator", "text", "source", "label"]).columns


def resolve_feature_cols(df: pd.DataFrame,
                         feature_subset: Optional[List[str]] = None) -> pd.Index:
    """Return the feature columns to use, with optional subset validation.

    Args:
        df: The full dataframe.
        feature_subset: Optional list of column names to restrict to.
                        If None, all non-metadata columns are used.

    Returns:
        pd.Index of feature column names.

    Raises:
        ValueError: If any name in feature_subset is not a valid feature column.
    """
    all_cols = get_feature_columns(df)
    if feature_subset is None:
        return all_cols
    invalid = [f for f in feature_subset if f not in all_cols]
    if invalid:
        raise ValueError(
            f"Unknown feature(s): {invalid}. "
            f"Available features: {list(all_cols)}"
        )
    return pd.Index(feature_subset)


def apply_default_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the *same* dataset filters used in the original notebook cells.

    - Remove 'summarization' source
    - Remove generators 'phi4' and 'mistral'
    - Drop 'sentence_complexity' if present
    """
    df = df.copy()
    if "source" in df.columns:
        df = df[df["source"] != "summarization"]
    if "generator" in df.columns:
        df = df[df["generator"] != "phi4"]
        df = df[df["generator"] != "mistral"]
    df = df.drop(columns=["sentence_complexity"], errors="ignore")
    return df


def balance_classes(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Balance dataset by downsampling to minority class size."""
    min_count = df[label_col].value_counts().min()
    return df.groupby(label_col).sample(n=min_count, random_state=SEED)


def get_top_k_features(importance: np.ndarray, feature_cols: pd.Index, k: int = 5) -> List[str]:
    """Return top k features based on importance scores."""
    sorted_pairs = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    return [f for f, _ in sorted_pairs[:k]]


def compute_jaccard_overlap(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_single_model(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> Tuple[LogisticRegression, StandardScaler]:
    """Train a single logistic regression model with scaling."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed, solver="liblinear")
    clf.fit(X_scaled, y)
    
    return clf, scaler


def train_with_cross_validation(X: pd.DataFrame, y: pd.Series,
                                n_runs: int = 5, test_size: float = 0.2) -> Tuple[float, np.ndarray]:
    """Train model multiple times and return average *test* accuracy and feature importance.

    Fixes a critical bug in the original implementation: previously it evaluated on the
    training split (optimistically biased). Here, each run uses a stratified holdout
    test split and reports accuracy on that held-out set.

    Notes:
    - Scaling is fit on the train split only (no leakage).
    - Feature importance is the mean absolute coefficient magnitude across runs.
    """
    accuracies: List[float] = []
    importances: List[np.ndarray] = []

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=SEED + run
        )

        clf, scaler = train_single_model(X_train, y_train, seed=SEED + run)
        # Evaluate on held-out test (NOT on train)
        acc = evaluate_model(clf, scaler, X_test, y_test)
        accuracies.append(acc)

        importances.append(np.abs(clf.coef_[0]))

    return float(np.mean(accuracies)), np.mean(importances, axis=0)


def evaluate_model(clf: LogisticRegression, scaler: StandardScaler, 
                   X: pd.DataFrame, y: pd.Series) -> float:
    """Evaluate model accuracy on test set."""
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    return accuracy_score(y, y_pred)


# ============================================================================
# OVERLAP ANALYSIS
# ============================================================================

def compute_overlap_matrix(top_features_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """Compute pairwise Jaccard overlap matrix for top features."""
    keys = list(top_features_dict.keys())
    matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)
    
    for a in keys:
        for b in keys:
            set_a, set_b = set(top_features_dict[a]), set(top_features_dict[b])
            matrix.loc[a, b] = compute_jaccard_overlap(set_a, set_b)
    
    return matrix


# ============================================================================
# SOURCE TRANSFER ANALYSIS
# ============================================================================

def analyze_source_transfer(df: pd.DataFrame, generator: str,
                            top_k: int = 5, n_runs: int = 5,
                            feature_subset: Optional[List[str]] = None) -> Tuple[Dict, pd.DataFrame]:
    """
    Analyze transfer across sources for a given generator.
    Train on one source, test on all sources.

    Args:
        df: DataFrame with columns [generator, source, label, text, features...]
        generator: Name of generator to analyze
        top_k: Number of top features to use
        n_runs: Number of cross-validation runs
        feature_subset: Optional list of feature names to restrict to.
                        If None, all available features are used.

    Returns:
        results: Dict with 'full', 'top5', 'drop' accuracy matrices
        overlap_matrix: Feature overlap matrix
    """
    df_gen = df[df["generator"].isin(["human", generator])]
    sources = df["source"].unique()
    feature_cols = resolve_feature_cols(df, feature_subset)
    
    full_matrix = pd.DataFrame(index=sources, columns=sources, dtype=float)
    top5_matrix = pd.DataFrame(index=sources, columns=sources, dtype=float)
    top_features_by_source = {}
    
    for train_source in sources:
        df_train = df_gen[df_gen["source"] == train_source]
        
        if df_train["label"].nunique() < 2:
            continue
        
        # Train models
        df_train_bal = balance_classes(df_train)
        X_train = df_train_bal[feature_cols]
        y_train = df_train_bal["label"]
        
        _, importance = train_with_cross_validation(X_train, y_train, n_runs)
        top_features = get_top_k_features(importance, feature_cols, top_k)
        top_features_by_source[train_source] = top_features
        
        clf_full, scaler_full = train_single_model(X_train, y_train)
        clf_top, scaler_top = train_single_model(X_train[top_features], y_train)
        
        # Test on all sources
        for test_source in sources:
            df_test = df_gen[df_gen["source"] == test_source]
            
            if df_test["label"].nunique() < 2:
                continue
            
            df_test_bal = balance_classes(df_test)
            X_test_full = df_test_bal[feature_cols]
            X_test_top = df_test_bal[top_features]
            y_test = df_test_bal["label"]
            
            full_matrix.loc[train_source, test_source] = evaluate_model(clf_full, scaler_full, X_test_full, y_test)
            top5_matrix.loc[train_source, test_source] = evaluate_model(clf_top, scaler_top, X_test_top, y_test)
    
    drop_matrix = full_matrix - top5_matrix
    overlap_matrix = compute_overlap_matrix(top_features_by_source)
    
    results = {
        "full": full_matrix,
        "top5": top5_matrix,
        "drop": drop_matrix
    }
    
    return results, overlap_matrix


# ============================================================================
# GENERATOR TRANSFER ANALYSIS
# ============================================================================

def analyze_generator_transfer(df: pd.DataFrame, source: str,
                               generators: List[str], top_k: int = 5,
                               n_runs: int = 5,
                               feature_subset: Optional[List[str]] = None) -> Tuple[Dict, pd.DataFrame]:
    """
    Analyze transfer across generators for a given source.
    Train on one generator, test on all generators.

    Args:
        df: DataFrame with columns [generator, source, label, text, features...]
        source: Name of source domain to analyze
        generators: List of generator names
        top_k: Number of top features to use
        n_runs: Number of cross-validation runs
        feature_subset: Optional list of feature names to restrict to.
                        If None, all available features are used.

    Returns:
        results: Dict with 'full', 'top5', 'drop' accuracy matrices
        overlap_matrix: Feature overlap matrix
    """
    df_src = df[df["source"] == source]
    feature_cols = resolve_feature_cols(df, feature_subset)
    
    full_matrix = pd.DataFrame(index=generators, columns=generators, dtype=float)
    top5_matrix = pd.DataFrame(index=generators, columns=generators, dtype=float)
    top_features_by_generator = {}
    
    for train_gen in generators:
        df_train = df_src[df_src["generator"].isin(["human", train_gen])]
        
        if df_train["label"].nunique() < 2:
            continue
        
        # Train models
        df_train_bal = balance_classes(df_train)
        X_train = df_train_bal[feature_cols]
        y_train = df_train_bal["label"]
        
        _, importance = train_with_cross_validation(X_train, y_train, n_runs)
        top_features = get_top_k_features(importance, feature_cols, top_k)
        top_features_by_generator[train_gen] = top_features
        
        clf_full, scaler_full = train_single_model(X_train, y_train)
        clf_top, scaler_top = train_single_model(X_train[top_features], y_train)
        
        # Test on all generators
        for test_gen in generators:
            df_test = df_src[df_src["generator"].isin(["human", test_gen])]
            
            if df_test["label"].nunique() < 2:
                continue
            
            df_test_bal = balance_classes(df_test)
            X_test_full = df_test_bal[feature_cols]
            X_test_top = df_test_bal[top_features]
            y_test = df_test_bal["label"]
            
            full_matrix.loc[train_gen, test_gen] = evaluate_model(clf_full, scaler_full, X_test_full, y_test)
            top5_matrix.loc[train_gen, test_gen] = evaluate_model(clf_top, scaler_top, X_test_top, y_test)
    
    drop_matrix = full_matrix - top5_matrix
    overlap_matrix = compute_overlap_matrix(top_features_by_generator)
    
    results = {
        "full": full_matrix,
        "top5": top5_matrix,
        "drop": drop_matrix
    }
    
    return results, overlap_matrix


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def run_all_source_transfer(df: pd.DataFrame, generators: List[str],
                            top_k: int = 5, n_runs: int = 5,
                            feature_subset: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
    """
    Run source transfer analysis for all generators.

    Args:
        df: Full dataset
        generators: List of generator names (excluding 'human')
        top_k: Number of top features
        n_runs: Number of CV runs
        feature_subset: Optional list of feature names to restrict to.

    Returns:
        results_dict: Dict mapping generator -> results
        overlap_dict: Dict mapping generator -> overlap matrix
    """
    results_dict = {}
    overlap_dict = {}

    for i, generator in enumerate(generators, 1):
        print(f"[{i}/{len(generators)}] Analyzing source transfer for {generator}...")
        results, overlap = analyze_source_transfer(df, generator, top_k, n_runs, feature_subset)
        results_dict[generator] = results
        overlap_dict[generator] = overlap

    return results_dict, overlap_dict


def run_all_generator_transfer(df: pd.DataFrame, generators: List[str],
                               top_k: int = 5, n_runs: int = 5,
                               feature_subset: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
    """
    Run generator transfer analysis for all sources.

    Args:
        df: Full dataset
        generators: List of generator names (excluding 'human')
        top_k: Number of top features
        n_runs: Number of CV runs
        feature_subset: Optional list of feature names to restrict to.

    Returns:
        results_dict: Dict mapping source -> results
        overlap_dict: Dict mapping source -> overlap matrix
    """
    sources = df["source"].unique()
    results_dict = {}
    overlap_dict = {}

    for i, source in enumerate(sources, 1):
        print(f"[{i}/{len(sources)}] Analyzing generator transfer for {source}...")
        results, overlap = analyze_generator_transfer(df, source, generators, top_k, n_runs, feature_subset)
        results_dict[source] = results
        overlap_dict[source] = overlap

    return results_dict, overlap_dict


# ============================================================================
# GLOBAL AND AGGREGATED LLM vs HUMAN ANALYSES
# ============================================================================

def run_global_llm_vs_human(df: pd.DataFrame,
                             feature_subset: Optional[List[str]] = None,
                             n_runs: int = 5) -> Dict:
    """Run one logistic regression on the entire balanced dataset: all LLMs vs Human.

    Sampling strategy
    -----------------
    AI class  : stratified by (generator × source) — each LLM-domain cell
                contributes the same number of samples so no single LLM or
                domain dominates the positive class.
    Human class: stratified by source — each domain contributes equally.
    Final step : if one class still has more rows, downsample it to match
                 the other so label=0 and label=1 are exactly balanced.

    The best-performing model across all CV runs is saved to
    PAPER_RESULTS_DIR/models/global_best_model.joblib for later inference.

    Args:
        df: Full filtered dataframe (label=0 human, label=1 AI).
        feature_subset: Optional list of feature names. None → all features.
        n_runs: Number of stratified CV runs.

    Returns:
        dict with keys:
            accuracy     – mean held-out test accuracy across runs
            std          – std of accuracy across runs
            importance   – mean absolute coefficient per feature (ndarray)
            feature_cols – list of feature names used
            top_features – top-10 features by importance
            n_human      – balanced human sample count
            n_ai         – balanced AI sample count
            best_model   – {"clf", "scaler", "accuracy"} for the top run
    """
    feature_cols = resolve_feature_cols(df, feature_subset)

    # ── Stratified sampling for AI class (generator × source) ────────────────
    df_ai = df[df["label"] == 1]
    ai_groups = df_ai.groupby(["generator", "source"], group_keys=False)
    ai_min = ai_groups.size().min()
    df_ai_bal = ai_groups.apply(
        lambda g: g.sample(n=ai_min, random_state=SEED)
    ).reset_index(drop=True)

    # ── Stratified sampling for human class (source) ──────────────────────────
    df_human = df[df["label"] == 0]
    human_groups = df_human.groupby("source", group_keys=False)
    human_min = human_groups.size().min()
    df_human_bal = human_groups.apply(
        lambda g: g.sample(n=human_min, random_state=SEED)
    ).reset_index(drop=True)

    # ── Final class balance ───────────────────────────────────────────────────
    target = min(len(df_ai_bal), len(df_human_bal))
    if len(df_ai_bal) > target:
        df_ai_bal = df_ai_bal.sample(n=target, random_state=SEED)
    if len(df_human_bal) > target:
        df_human_bal = df_human_bal.sample(n=target, random_state=SEED)

    df_bal = pd.concat([df_ai_bal, df_human_bal]).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)

    n_human = int((df_bal["label"] == 0).sum())
    n_ai    = int((df_bal["label"] == 1).sum())
    print(f"  Balanced dataset: n_human={n_human}, n_ai={n_ai}")
    print(f"  AI samples per (generator×source) cell: {ai_min}")
    print(f"  Human samples per source: {human_min}")

    X = df_bal[feature_cols]
    y = df_bal["label"]

    accs: List[float] = []
    importances: List[np.ndarray] = []
    best_acc     = -1.0
    best_clf     = None
    best_scaler  = None
    best_X_train = None
    best_X_test  = None
    best_y_train = None
    best_y_test  = None

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED + run
        )
        clf, scaler = train_single_model(X_train, y_train, seed=SEED + run)
        acc = evaluate_model(clf, scaler, X_test, y_test)
        accs.append(acc)
        importances.append(np.abs(clf.coef_[0]))

        if acc > best_acc:
            best_acc     = acc
            best_clf     = clf
            best_scaler  = scaler
            best_X_train = X_train.copy()
            best_X_test  = X_test.copy()
            best_y_train = y_train.copy()
            best_y_test  = y_test.copy()

    importance   = np.mean(importances, axis=0)
    top_features = get_top_k_features(importance, feature_cols, k=min(10, len(feature_cols)))

    result = {
        "accuracy":     float(np.mean(accs)),
        "std":          float(np.std(accs)),
        "importance":   importance,
        "feature_cols": list(feature_cols),
        "top_features": top_features,
        "n_human":      n_human,
        "n_ai":         n_ai,
        "best_model":   {"clf": best_clf, "scaler": best_scaler, "accuracy": best_acc},
    }

    # ── Auto-save paper CSVs + model ─────────────────────────────────────────
    if PAPER_RESULTS_DIR:
        # Save best model for later inference
        model_dir = os.path.join(PAPER_RESULTS_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "global_best_model.joblib")
        joblib.dump(
            {"clf": best_clf, "scaler": best_scaler, "feature_cols": list(feature_cols)},
            model_path,
        )
        print(f"  [model] models/global_best_model.joblib  (best run acc={best_acc:.4f})")

        # Save train / test splits from the best run
        # Pull full rows from df_bal using the split indices so text/generator/source are included
        meta_cols = [c for c in ["text", "generator", "source"] if c in df_bal.columns]
        col_order = meta_cols + list(feature_cols) + ["label"]
        train_df = df_bal.loc[best_X_train.index, col_order].reset_index(drop=True)
        test_df  = df_bal.loc[best_X_test.index,  col_order].reset_index(drop=True)
        train_df.to_csv(os.path.join(model_dir, "global_best_train.csv"), index=False, encoding="utf-8-sig")
        test_df.to_csv( os.path.join(model_dir, "global_best_test.csv"),  index=False, encoding="utf-8-sig")
        print(f"  [csv] models/global_best_train.csv  ({len(train_df)} rows)")
        print(f"  [csv] models/global_best_test.csv   ({len(test_df)} rows)")

        # 01: one-row global baseline table
        global_csv = pd.DataFrame([{
            "accuracy":        round(result["accuracy"], 4),
            "std":             round(result["std"], 4),
            "best_run_acc":    round(best_acc, 4),
            "n_human":         n_human,
            "n_ai":            n_ai,
            "n_features_used": len(feature_cols),
            "top_features":    ", ".join(top_features),
        }])
        _csv_save(global_csv, "rq1_detection", "01_global_accuracy.csv", index=False)

        # 13: ranked feature importance from the global model
        global_imp = pd.DataFrame({
            "feature":    list(feature_cols),
            "importance": importance.round(5),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        global_imp.insert(0, "rank", range(1, len(global_imp) + 1))
        _csv_save(global_imp, "rq4_features", "13_global_feature_importance.csv", index=False)

        _paper_cache["global_result"] = result
        _try_save_master_key_numbers()

    return result


def run_aggregated_llm_by_domain(df: pd.DataFrame,
                                  feature_subset: Optional[List[str]] = None,
                                  n_runs: int = 5) -> pd.DataFrame:
    """Aggregate all LLMs as one class and run binary classification per source domain.

    For each domain, all AI-generated samples (regardless of generator) are
    treated as label=1, human samples as label=0. The per-domain dataset is
    class-balanced before each experiment.

    Args:
        df: Full filtered dataframe.
        feature_subset: Optional list of feature names. None → all features.
        n_runs: Number of stratified CV runs per domain.

    Returns:
        DataFrame with columns [source, accuracy, std, n_human, n_ai].
    """
    feature_cols = resolve_feature_cols(df, feature_subset)
    sources = df["source"].unique()
    rows = []

    for source in sources:
        df_src = df[df["source"] == source]

        if df_src["label"].nunique() < 2:
            print(f"  Skipping {source}: only one class present.")
            continue

        df_bal = balance_classes(df_src)
        n_human = int((df_bal["label"] == 0).sum())
        n_ai    = int((df_bal["label"] == 1).sum())

        X = df_bal[feature_cols]
        y = df_bal["label"]

        accs: List[float] = []
        for run in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEED + run
            )
            clf, scaler = train_single_model(X_train, y_train, seed=SEED + run)
            accs.append(evaluate_model(clf, scaler, X_test, y_test))

        rows.append({
            "source":   source,
            "accuracy": float(np.mean(accs)),
            "std":      float(np.std(accs)),
            "n_human":  n_human,
            "n_ai":     n_ai,
        })
        print(f"  {source}: acc={np.mean(accs):.3f} ± {np.std(accs):.3f}"
              f"  (n_human={n_human}, n_ai={n_ai})")

    return pd.DataFrame(rows)


def plot_global_and_aggregated_results(global_result: Dict,
                                        aggregated_df: pd.DataFrame,
                                        figsize: tuple = None,
                                        save_path: str = None):
    """Two-panel figure summarising the global and per-domain analyses.

    Left panel:  per-domain accuracy bars (aggregated LLM vs Human),
                 with the global accuracy drawn as a reference line.
    Right panel: feature importance from the global model.
    """
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH, 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Panel 1: per-domain accuracy ---
    df_sorted = aggregated_df.sort_values("accuracy", ascending=False)
    x = np.arange(len(df_sorted))
    ax1.bar(x, df_sorted["accuracy"], yerr=df_sorted["std"], capsize=3,
            color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.axhline(
        global_result["accuracy"], color="crimson", linestyle="--", linewidth=1.5,
        label=f"Global ({global_result['accuracy']:.3f}±{global_result['std']:.3f})"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_sorted["source"], rotation=45, ha="right")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Aggregated LLM vs Human by Domain")
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # --- Panel 2: global model feature importance ---
    feature_cols = global_result["feature_cols"]
    importance   = global_result["importance"]
    sorted_pairs = sorted(zip(feature_cols, importance), key=lambda p: p[1], reverse=True)
    feat_names = [p[0] for p in sorted_pairs]
    feat_vals  = [p[1] for p in sorted_pairs]
    y_pos = np.arange(len(feat_names))
    ax2.barh(y_pos, feat_vals, color="coral", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feat_names, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |Coefficient|")
    ax2.set_title("Global Model Feature Importance")
    ax2.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  → Saved: {save_path}")

    plt.show()
    plt.close()


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def compute_summary_statistics(results_dict: Dict) -> pd.DataFrame:
    """
    Compute summary statistics across all experiments.
    
    Args:
        results_dict: Dictionary of results from transfer analysis
    
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for name, results in results_dict.items():
        full_acc = results["full"].values.flatten()
        top5_acc = results["top5"].values.flatten()
        drop = results["drop"].values.flatten()
        
        # Remove NaN values
        full_acc = full_acc[~np.isnan(full_acc)]
        top5_acc = top5_acc[~np.isnan(top5_acc)]
        drop = drop[~np.isnan(drop)]
        
        # Compute diagonal (same condition) accuracy
        full_diag = np.diag(results["full"].astype(float))
        full_diag = full_diag[~np.isnan(full_diag)]
        
        # Compute off-diagonal (cross condition) accuracy
        mask = ~np.eye(results["full"].shape[0], dtype=bool)
        full_cross = results["full"].values[mask]
        full_cross = full_cross[~np.isnan(full_cross)]
        
        summary_data.append({
            "Name": name,
            "Avg Acc (Full)": np.mean(full_acc),
            "Std Acc (Full)": np.std(full_acc),
            "Avg Acc (Top-5)": np.mean(top5_acc),
            "Std Acc (Top-5)": np.std(top5_acc),
            "Avg Drop": np.mean(drop),
            "Std Drop": np.std(drop),
            "Same Condition": np.mean(full_diag),
            "Cross Condition": np.mean(full_cross),
            "Transfer Gap": np.mean(full_diag) - np.mean(full_cross)
        })
    
    return pd.DataFrame(summary_data)


# ============================================================================
# CSV EXPORT HELPERS
# ============================================================================

def save_transfer_matrices(results_dict: Dict, overlap_dict: Dict, save_dir: str) -> None:
    """Save all matrices (full/top-k/drop + overlap) to CSV for every condition."""
    os.makedirs(save_dir, exist_ok=True)
    for condition, results in results_dict.items():
        cond_dir = os.path.join(save_dir, str(condition))
        os.makedirs(cond_dir, exist_ok=True)

        results["full"].to_csv(os.path.join(cond_dir, "full_accuracy.csv"))
        results["top5"].to_csv(os.path.join(cond_dir, "topk_accuracy.csv"))
        results["drop"].to_csv(os.path.join(cond_dir, "accuracy_drop.csv"))

        if condition in overlap_dict:
            overlap_dict[condition].to_csv(os.path.join(cond_dir, "topk_feature_overlap.csv"))


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_single_heatmap(data: pd.DataFrame, title: str, 
                        xlabel: str = "Test", ylabel: str = "Train",
                        cmap: str = "Blues", center: float = None,
                        figsize: tuple = None, save_path: str = None,
                        annot: bool = True, fmt: str = ".2f",
                        hide_zeros: bool = False):
    """Create a single publication-quality heatmap."""
    if figsize is None:
        figsize = (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 1.2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Clean data
    data = data.astype(float).fillna(0.0)
    
    # Handle zero hiding
    if hide_zeros:
        data_clean = data.round(2)
        data_clean = data_clean.mask(np.isclose(data_clean, 0.0, atol=1e-8), 0.0)
        annot_data = data_clean.apply(
            lambda col: col.map(
                lambda x: "" if np.isclose(x, 0.0, atol=1e-8) else f"{x:.2f}"
            )
        )
        fmt = ""
        annot = annot_data
    
    # Create heatmap
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, ax=ax, 
                center=center, cbar_kws={'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_accuracy_comparison(results_dict: Dict, metric: str = "full",
                             figsize: tuple = None, save_path: str = None):
    """Create a bar plot comparing average accuracies across conditions."""
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH * 0.6, 2.5)
    
    # Extract data
    names = []
    means = []
    stds = []
    
    for name, results in results_dict.items():
        data = results[metric].values.flatten()
        data = data[~np.isnan(data)]
        names.append(name)
        means.append(np.mean(data))
        stds.append(np.std(data))
    
    # Sort by mean
    sorted_idx = np.argsort(means)[::-1]
    names = [names[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=3, 
                  color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Average {metric.capitalize()} Accuracy')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_diagonal_vs_cross(results_dict: Dict, 
                           figsize: tuple = None, save_path: str = None):
    """Compare diagonal (same condition) vs cross-condition transfer."""
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH, 3)
    
    names = []
    diagonal_full = []
    cross_full = []
    diagonal_top5 = []
    cross_top5 = []
    
    for name, results in results_dict.items():
        full_mat = results["full"].astype(float)
        top5_mat = results["top5"].astype(float)
        
        # Diagonal values
        diag_full = np.diag(full_mat)
        diag_top5 = np.diag(top5_mat)
        
        # Off-diagonal values
        mask = ~np.eye(full_mat.shape[0], dtype=bool)
        off_diag_full = full_mat.values[mask]
        off_diag_top5 = top5_mat.values[mask]
        
        # Remove NaN
        diag_full = diag_full[~np.isnan(diag_full)]
        diag_top5 = diag_top5[~np.isnan(diag_top5)]
        off_diag_full = off_diag_full[~np.isnan(off_diag_full)]
        off_diag_top5 = off_diag_top5[~np.isnan(off_diag_top5)]
        
        names.append(name)
        diagonal_full.append(np.mean(diag_full))
        cross_full.append(np.mean(off_diag_full))
        diagonal_top5.append(np.mean(diag_top5))
        cross_top5.append(np.mean(off_diag_top5))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(names))
    width = 0.35
    
    # Full features
    ax1.bar(x - width/2, diagonal_full, width, label='Same Condition', 
            color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, cross_full, width, label='Cross Condition', 
            color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Full Features')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1])
    
    # Top-5 features
    ax2.bar(x - width/2, diagonal_top5, width, label='Same Condition', 
            color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.bar(x + width/2, cross_top5, width, label='Cross Condition', 
            color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Top-5 Features')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_feature_stability(overlap_dict: Dict,
                           figsize: tuple = None, save_path: str = None):
    """Plot distribution of feature overlap across conditions."""
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH * 0.6, 3)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    data_by_condition = []
    labels = []
    
    for name, overlap_mat in overlap_dict.items():
        # Get off-diagonal values
        mask = ~np.eye(overlap_mat.shape[0], dtype=bool)
        off_diag = overlap_mat.values[mask]
        off_diag = off_diag[~np.isnan(off_diag)]
        
        data_by_condition.append(off_diag)
        labels.append(name)
    
    # Create violin plot
    parts = ax.violinplot(data_by_condition, positions=range(len(labels)),
                          showmeans=True, showmedians=True)
    
    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Jaccard Overlap')
    ax.set_title('Feature Overlap Distribution')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_compact_grid(results_dict: Dict, overlap_dict: Dict,
                      conditions: List[str], top_k: int = 5,
                      figsize: tuple = None, save_path: str = None):
    """Create a compact 2x2 grid for each condition."""
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.5)
    
    for condition in conditions:
        if condition not in results_dict:
            print(f"Warning: {condition} not in results")
            continue
        
        results = results_dict[condition]
        overlap = overlap_dict[condition]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{condition}', fontsize=12, y=0.98)
        
        matrices = [
            results["full"].astype(float).fillna(0.0),
            results["top5"].astype(float).fillna(0.0),
            results["drop"].astype(float).fillna(0.0),
            overlap.astype(float).fillna(0.0)
        ]
        
        titles = ['Full Accuracy', f'Top-{top_k} Accuracy', 'Accuracy Drop', f'Top-{top_k} Feature Overlap']
        cmaps = ['Blues', 'Blues', 'RdBu_r', 'Purples']
        centers = [None, None, 0.0, None]
        
        for idx, (ax, data, title, cmap, center) in enumerate(
            zip(axes.flat, matrices, titles, cmaps, centers)):
            
            if idx == 2:  # Drop matrix
                data = data.round(2).mask(np.isclose(data, 0.0, atol=1e-8), 0.0)
                annot_data = data.apply(
                    lambda col: col.map(
                        lambda x: "" if np.isclose(x, 0.0, atol=1e-8) else f"{x:.2f}"
                    )
                )
                sns.heatmap(data, annot=annot_data, fmt="", cmap=cmap, ax=ax,
                           center=center, cbar_kws={'shrink': 0.8},
                           linewidths=0.3, linecolor='white')
            else:
                sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                           center=center, cbar_kws={'shrink': 0.8},
                           linewidths=0.3, linecolor='white')
            
            ax.set_title(title, fontsize=10, pad=5)
            ax.set_xlabel('Test' if idx < 3 else 'Column', fontsize=9)
            ax.set_ylabel('Train' if idx < 3 else 'Row', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            path = f"{save_path}_{condition}.pdf"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"  → Saved: {path}")
        
        plt.show()
        plt.close()


def plot_selected_conditions(results_dict: Dict, overlap_dict: Dict,
                             selected: List[str], top_k: int = 5,
                             figsize: tuple = None, save_path: str = None):
    """Create focused visualization for a subset of conditions."""
    n_conditions = len(selected)
    
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH, n_conditions * 1.8)
    
    fig, axes = plt.subplots(n_conditions, 4, figsize=figsize)
    
    if n_conditions == 1:
        axes = axes.reshape(1, -1)
    
    for i, condition in enumerate(selected):
        if condition not in results_dict:
            print(f"Warning: {condition} not in results")
            continue
        
        results = results_dict[condition]
        overlap = overlap_dict[condition]
        
        matrices = [
            results["full"].astype(float).fillna(0.0),
            results["top5"].astype(float).fillna(0.0),
            results["drop"].astype(float).fillna(0.0),
            overlap.astype(float).fillna(0.0)
        ]
        
        titles = ['Full Acc.', f'Top-{top_k} Acc.', 'Drop', 'Overlap']
        cmaps = ['Blues', 'Blues', 'RdBu_r', 'Purples']
        centers = [None, None, 0.0, None]
        
        for j, (data, title, cmap, center) in enumerate(
            zip(matrices, titles, cmaps, centers)):
            
            ax = axes[i, j]
            
            if j == 2:  # Drop matrix
                data = data.round(2).mask(np.isclose(data, 0.0, atol=1e-8), 0.0)
                annot_data = data.apply(
                    lambda col: col.map(
                        lambda x: "" if np.isclose(x, 0.0, atol=1e-8) else f"{x:.2f}"
                    )
                )
                sns.heatmap(data, annot=annot_data, fmt="", cmap=cmap, ax=ax,
                           center=center, cbar_kws={'shrink': 0.7},
                           linewidths=0.3, linecolor='white')
            else:
                sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                           center=center, cbar_kws={'shrink': 0.7},
                           linewidths=0.3, linecolor='white')
            
            if i == 0:
                ax.set_title(title, fontsize=9, pad=5)
            
            if j == 0:
                ax.set_ylabel(f'{condition}\nTrain', fontsize=8)
            else:
                ax.set_ylabel('')
            
            if i == n_conditions - 1:
                ax.set_xlabel('Test' if j < 3 else 'Col', fontsize=8)
            else:
                ax.set_xlabel('')
            
            ax.tick_params(labelsize=7)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_aggregated_heatmap(results_dict: Dict, metric: str = "full",
                            aggregation: str = "mean",
                            figsize: tuple = None, save_path: str = None):
    """Create a single heatmap showing aggregated results across all conditions."""
    if figsize is None:
        figsize = (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.9)
    
    # Stack all matrices
    all_matrices = []
    for results in results_dict.values():
        mat = results[metric].astype(float)
        all_matrices.append(mat)
    
    # Aggregate
    if aggregation == "mean":
        agg_matrix = pd.concat(all_matrices).groupby(level=0).mean()
    elif aggregation == "median":
        agg_matrix = pd.concat(all_matrices).groupby(level=0).median()
    elif aggregation == "min":
        agg_matrix = pd.concat(all_matrices).groupby(level=0).min()
    elif aggregation == "max":
        agg_matrix = pd.concat(all_matrices).groupby(level=0).max()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    cmap = 'RdBu_r' if metric == 'drop' else 'Blues'
    center = 0.0 if metric == 'drop' else None
    
    plot_single_heatmap(
        agg_matrix,
        title=f'{aggregation.capitalize()} {metric.capitalize()} Accuracy',
        xlabel='Test',
        ylabel='Train',
        cmap=cmap,
        center=center,
        figsize=figsize,
        save_path=save_path,
        hide_zeros=(metric == 'drop')
    )


def plot_summary_table(summary_df: pd.DataFrame, save_path: str = None):
    """Create a publication-quality summary table."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, len(summary_df) * 0.3 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Format numbers
    formatted_df = summary_df.copy()
    for col in summary_df.columns:
        if col != 'Name' and summary_df[col].dtype in [float, np.float64]:
            formatted_df[col] = summary_df[col].apply(lambda x: f"{x:.3f}")
    
    table = ax.table(cellText=formatted_df.values,
                    colLabels=formatted_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(len(formatted_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(formatted_df) + 1):
        for j in range(len(formatted_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# FULL ANALYSIS PIPELINE
# ============================================================================

def plot_full_analysis_overview(results_dict: Dict, overlap_dict: Dict,
                                 analysis_type: str = "source",
                                 save_dir: str = None):
    """
    Create a complete set of figures for a transfer analysis.
    
    Args:
        results_dict: Dictionary of accuracy results
        overlap_dict: Dictionary of overlap matrices
        analysis_type: Type of analysis ("source" or "generator")
        save_dir: Directory to save figures
    
    Returns:
        summary_df: Summary statistics DataFrame
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"\n{'='*70}")
    print(f"Generating {analysis_type.upper()} transfer analysis figures")
    print(f"{'='*70}\n")
    
    # 1. Accuracy comparison
    print("1. Creating accuracy comparison plot...")
    save_path = f"{save_dir}/01_accuracy_comparison.pdf" if save_dir else None
    plot_accuracy_comparison(results_dict, metric="full", save_path=save_path)
    
    # 2. Diagonal vs cross-condition
    print("\n2. Creating diagonal vs cross-condition plot...")
    save_path = f"{save_dir}/02_diagonal_vs_cross.pdf" if save_dir else None
    plot_diagonal_vs_cross(results_dict, save_path=save_path)
    
    # 3. Feature stability
    print("\n3. Creating feature stability plot...")
    save_path = f"{save_dir}/03_feature_stability.pdf" if save_dir else None
    plot_feature_stability(overlap_dict, save_path=save_path)
    
    # 4. Aggregated heatmap
    print("\n4. Creating aggregated heatmap...")
    save_path = f"{save_dir}/04_aggregated_heatmap.pdf" if save_dir else None
    plot_aggregated_heatmap(results_dict, metric="full", save_path=save_path)
    
    # 5. Compact grids for each condition
    print("\n5. Creating compact grids for each condition...")
    conditions = list(results_dict.keys())
    save_path = f"{save_dir}/05_grid" if save_dir else None
    plot_compact_grid(results_dict, overlap_dict, conditions, save_path=save_path)
    
    # 6. Summary table
    print("\n6. Creating summary table...")
    summary_df = compute_summary_statistics(results_dict)
    save_path = f"{save_dir}/06_summary_table.pdf" if save_dir else None
    plot_summary_table(summary_df, save_path=save_path)
    
    # 7. Save summary to CSV
    if save_dir:
        csv_path = f"{save_dir}/summary_statistics.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"  → Saved: {csv_path}")

        # Save all underlying matrices (per condition)
        save_transfer_matrices(results_dict, overlap_dict, os.path.join(save_dir, "matrices_csv"))
        print("  → Saved matrices to: " + os.path.join(save_dir, "matrices_csv"))
    
    print(f"\n{'='*70}")
    print("All figures generated successfully!")
    print(f"{'='*70}\n")
    
    return summary_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Transfer Learning Analysis for AI Text Detection'
    )
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file with detection data')
    parser.add_argument('--output', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top features to use')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of cross-validation runs')
    parser.add_argument('--analysis', type=str, default='both',
                       choices=['source', 'generator', 'both'],
                       help='Type of analysis to run')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                       help='Feature subset to use (default: all features). '
                            'Example: --features entropy burstiness lexical_diversity')
    parser.add_argument('--selected', type=str, nargs='+',
                       help='Selected conditions for focused analysis')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading data from: {args.data}")
    df = pd.read_csv(args.data)
    df = apply_default_filters(df)

    # Validate data
    required_cols = ['generator', 'source', 'label', 'text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Get generators (exclude 'human')
    generators = df['generator'].unique()
    generators = [g for g in generators if g.lower() != 'human']

    feature_subset = args.features  # None means use all features

    print(f"Found {len(generators)} generators: {generators}")
    print(f"Found {len(df['source'].unique())} sources: {list(df['source'].unique())}")
    print(f"Dataset size: {len(df)} samples")
    if feature_subset:
        print(f"Feature subset: {feature_subset}")
    else:
        print(f"Features: all ({len(get_feature_columns(df))} features)")
    print()

    # ------------------------------------------------------------------
    # PART 0: Global LLM vs Human (whole-dataset baseline)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PART 0: GLOBAL LLM vs HUMAN (WHOLE DATASET)")
    print("=" * 70)
    print("Running single balanced logistic regression on all data...")
    global_result = run_global_llm_vs_human(df, feature_subset=feature_subset, n_runs=args.n_runs)
    print(f"  Global accuracy: {global_result['accuracy']:.3f} ± {global_result['std']:.3f}")
    print(f"  Balanced n_human={global_result['n_human']}, n_ai={global_result['n_ai']}")
    print(f"  Top features: {global_result['top_features']}")

    # ------------------------------------------------------------------
    # PART 0b: Aggregated LLM vs Human by domain
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 0b: AGGREGATED LLM vs HUMAN BY DOMAIN")
    print("=" * 70)
    aggregated_df = run_aggregated_llm_by_domain(df, feature_subset=feature_subset, n_runs=args.n_runs)

    # Visualise both together
    save_path = os.path.join(args.output, 'global_and_aggregated.pdf')
    os.makedirs(args.output, exist_ok=True)
    plot_global_and_aggregated_results(global_result, aggregated_df, save_path=save_path)

    # Save aggregated table
    agg_csv = os.path.join(args.output, 'aggregated_by_domain.csv')
    aggregated_df.to_csv(agg_csv, index=False)
    print(f"  → Saved aggregated results: {agg_csv}")

    # Run analyses
    if args.analysis in ['source', 'both']:
        print("\n" + "="*70)
        print("RUNNING SOURCE TRANSFER ANALYSIS")
        print("="*70)
        
        source_results, source_overlap = run_all_source_transfer(
            df, generators, args.top_k, args.n_runs, feature_subset
        )
        
        output_dir = os.path.join(args.output, 'source_transfer')
        source_summary = plot_full_analysis_overview(
            source_results, source_overlap,
            analysis_type="source",
            save_dir=output_dir
        )
        
        # Selected conditions if specified
        if args.selected:
            print(f"\nCreating focused plot for: {args.selected}")
            save_path = os.path.join(output_dir, '07_selected_conditions.pdf')
            plot_selected_conditions(
                source_results, source_overlap, args.selected,
                save_path=save_path
            )
    
    if args.analysis in ['generator', 'both']:
        print("\n" + "="*70)
        print("RUNNING GENERATOR TRANSFER ANALYSIS")
        print("="*70)
        
        gen_results, gen_overlap = run_all_generator_transfer(
            df, generators, args.top_k, args.n_runs, feature_subset
        )
        
        output_dir = os.path.join(args.output, 'generator_transfer')
        gen_summary = plot_full_analysis_overview(
            gen_results, gen_overlap,
            analysis_type="generator",
            save_dir=output_dir
        )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output}/")
    print("\nQuick summary:")
    if args.analysis in ['source', 'both']:
        print(f"\nSource Transfer - Top 3 by accuracy:")
        print(source_summary.nlargest(3, 'Avg Acc (Full)')[['Name', 'Avg Acc (Full)', 'Transfer Gap']])
    if args.analysis in ['generator', 'both']:
        print(f"\nGenerator Transfer - Top 3 by accuracy:")
        print(gen_summary.nlargest(3, 'Avg Acc (Full)')[['Name', 'Avg Acc (Full)', 'Transfer Gap']])

# ============================================================================
# FEATURE STABILITY ANALYSIS
# ============================================================================

def extract_all_feature_importances(df: pd.DataFrame,
                                    generators: List[str],
                                    sources: List[str],
                                    n_runs: int = 5,
                                    feature_subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract feature importance (coefficients) from all (generator, source) pairs.

    Args:
        feature_subset: Optional list of feature names to restrict to.

    Returns:
        DataFrame with columns: [generator, source, feature, importance, rank]
    """
    feature_cols = resolve_feature_cols(df, feature_subset)
    all_results = []
    
    total_pairs = len(generators) * len(sources)
    current = 0
    
    for generator in generators:
        for source in sources:
            current += 1
            print(f"[{current}/{total_pairs}] Processing {generator} on {source}...")
            
            # Get data for this (generator, source) pair
            df_subset = df[
                (df["generator"].isin(["human", generator])) & 
                (df["source"] == source)
            ]
            
            if df_subset["label"].nunique() < 2:
                continue
            
            # Balance and train
            df_balanced = balance_classes(df_subset)
            X = df_balanced[feature_cols]
            y = df_balanced["label"]
            
            # Get feature importance with cross-validation
            _, importance = train_with_cross_validation(X, y, n_runs)
            
            # Get feature ranks
            sorted_idx = np.argsort(importance)[::-1]
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(len(sorted_idx)) + 1
            
            # Store results for each feature
            for feat_idx, feature in enumerate(feature_cols):
                all_results.append({
                    'generator': generator,
                    'source': source,
                    'feature': feature,
                    'importance': importance[feat_idx],
                    'rank': ranks[feat_idx]
                })
    
    return pd.DataFrame(all_results)


def compute_feature_stability_metrics(feature_importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stability metrics for each feature across all conditions.
    
    Metrics:
    - mean_importance: Average absolute coefficient across all conditions
    - std_importance: Standard deviation of coefficients
    - cv_importance: Coefficient of variation (std/mean)
    - mean_rank: Average rank across all conditions
    - std_rank: Standard deviation of rank
    - top5_frequency: % of conditions where feature is in top-5
    - top10_frequency: % of conditions where feature is in top-10
    
    Returns:
        DataFrame with one row per feature
    """
    stability_metrics = []
    
    for feature in feature_importance_df['feature'].unique():
        feature_data = feature_importance_df[feature_importance_df['feature'] == feature]
        
        importance_values = feature_data['importance'].values
        rank_values = feature_data['rank'].values
        
        # Compute metrics
        mean_imp = np.mean(importance_values)
        std_imp = np.std(importance_values)
        cv_imp = std_imp / mean_imp if mean_imp > 0 else np.inf
        
        mean_rank = np.mean(rank_values)
        std_rank = np.std(rank_values)
        
        top5_freq = np.mean(rank_values <= 5)
        top10_freq = np.mean(rank_values <= 10)
        
        stability_metrics.append({
            'feature': feature,
            'mean_importance': mean_imp,
            'std_importance': std_imp,
            'cv_importance': cv_imp,
            'mean_rank': mean_rank,
            'std_rank': std_rank,
            'top5_frequency': top5_freq,
            'top10_frequency': top10_freq,
            'n_conditions': len(feature_data)
        })
    
    stability_df = pd.DataFrame(stability_metrics)
    
    # Sort by mean importance
    stability_df = stability_df.sort_values('mean_importance', ascending=False)
    
    return stability_df


def categorize_features_by_stability(stability_df: pd.DataFrame,
                                     top5_threshold: float = 0.80,
                                     top10_threshold: float = 0.60,
                                     low_cv_threshold: float = 0.5) -> Dict[str, List[str]]:
    """
    Categorize features into tiers based on stability metrics.
    
    Tier 1 (Universal): High importance + high frequency in top-5 + low variation
    Tier 2 (Context-Sensitive): Moderate importance + moderate frequency
    Tier 3 (Specialized): Low frequency or high variation
    
    Returns:
        Dictionary with keys: 'tier1_universal', 'tier2_context_sensitive', 'tier3_specialized'
    """
    # Tier 1: Universal discriminators
    tier1_mask = (
        (stability_df['top5_frequency'] >= top5_threshold) &
        (stability_df['cv_importance'] <= low_cv_threshold)
    )
    tier1_features = stability_df[tier1_mask]['feature'].tolist()
    
    # Tier 3: Specialized (low frequency)
    tier3_mask = stability_df['top10_frequency'] < top10_threshold
    tier3_features = stability_df[tier3_mask]['feature'].tolist()
    
    # Tier 2: Everything else (context-sensitive)
    tier2_features = stability_df[
        ~stability_df['feature'].isin(tier1_features + tier3_features)
    ]['feature'].tolist()
    
    return {
        'tier1_universal': tier1_features,
        'tier2_context_sensitive': tier2_features,
        'tier3_specialized': tier3_features
    }


def analyze_cross_condition_stability(feature_importance_df: pd.DataFrame,
                                      dimension: str = 'generator') -> pd.DataFrame:
    """
    Analyze how feature importance varies across generators or sources.
    
    Args:
        feature_importance_df: Output from extract_all_feature_importances
        dimension: 'generator' or 'source'
    
    Returns:
        DataFrame showing feature importance by dimension
    """
    if dimension not in ['generator', 'source']:
        raise ValueError("dimension must be 'generator' or 'source'")
    
    # Pivot table: features x dimension, values = mean importance
    pivot_df = feature_importance_df.pivot_table(
        index='feature',
        columns=dimension,
        values='importance',
        aggfunc='mean'
    )
    
    # Add variance across dimension
    pivot_df['variance_across'] = pivot_df.var(axis=1)
    pivot_df['mean_across'] = pivot_df.mean(axis=1)
    pivot_df['cv_across'] = pivot_df['variance_across'].pow(0.5) / pivot_df['mean_across']
    
    # Sort by mean importance
    pivot_df = pivot_df.sort_values('mean_across', ascending=False)
    
    return pivot_df


# ============================================================================
# VISUALIZATION FOR FEATURE STABILITY
# ============================================================================

def plot_feature_stability_overview(stability_df: pd.DataFrame,
                                    top_n: int = 20,
                                    figsize: tuple = None,
                                    save_path: str = None):
    """
    Create comprehensive feature stability visualization.
    Shows: importance, variability, and ranking frequency.
    """
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH, 6)
    
    # Select top N features by mean importance
    plot_df = stability_df.head(top_n).copy()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Mean importance with error bars
    ax = axes[0, 0]
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df['mean_importance'], 
            xerr=plot_df['std_importance'],
            color='steelblue', alpha=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Importance')
    ax.set_title('Feature Importance (Mean ± Std)')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 2. Coefficient of Variation
    ax = axes[0, 1]
    colors = ['green' if cv < 0.5 else 'orange' if cv < 1.0 else 'red' 
              for cv in plot_df['cv_importance']]
    ax.barh(y_pos, plot_df['cv_importance'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Coefficient of Variation')
    ax.set_title('Feature Stability (lower = more stable)')
    ax.axvline(0.5, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(1.0, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 3. Top-K Frequency
    ax = axes[1, 0]
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width/2, plot_df['top5_frequency'], width, 
           label='Top-5', color='darkblue', alpha=0.7)
    ax.bar(x + width/2, plot_df['top10_frequency'], width, 
           label='Top-10', color='lightblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['feature'], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Frequency')
    ax.set_title('Top-K Frequency Across Conditions')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    

    # 4. Mean Rank with error bars
    ax = axes[1, 1]
    ax.barh(y_pos, plot_df['mean_rank'], 
            xerr=plot_df['std_rank'],
            color='coral', alpha=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Rank')
    ax.set_title('Average Feature Rank')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_xaxis()  # Lower rank number = better
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()

    
    

def plot_feature_tier_summary(stability_df: pd.DataFrame,
                              tier_dict: Dict[str, List[str]],
                              figsize: tuple = None,
                              save_path: str = None):
    """
    Visualize the three-tier hierarchy of features.
    """
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH * 0.7, 5)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    tiers = [
        ('tier1_universal', 'Universal\nDiscriminators', 'darkgreen'),
        ('tier2_context_sensitive', 'Context-Sensitive\nMarkers', 'orange'),
        ('tier3_specialized', 'Specialized\nIndicators', 'darkred')
    ]
    
    for idx, (tier_key, tier_name, color) in enumerate(tiers):
        ax = axes[idx]
        features = tier_dict[tier_key]
        
        if len(features) == 0:
            ax.text(0.5, 0.5, 'No features', ha='center', va='center')
            ax.set_title(f'{tier_name}\n(n=0)')
            continue
        
        # Get data for these features
        tier_df = stability_df[stability_df['feature'].isin(features)].copy()
        tier_df = tier_df.sort_values('mean_importance', ascending=False)
        
        # Plot
        y_pos = np.arange(len(tier_df))
        ax.barh(y_pos, tier_df['mean_importance'], 
                color=color, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tier_df['feature'], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Mean Importance', fontsize=8)
        ax.set_title(f'{tier_name}\n(n={len(features)})', fontsize=9)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_feature_heatmap_by_condition(cross_condition_df: pd.DataFrame,
                                      dimension: str = 'generator',
                                      top_n: int = 15,
                                      figsize: tuple = None,
                                      save_path: str = None):
    """
    Heatmap showing how top features vary across generators or sources.
    """
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH * 0.8, 6)
    
    # Select top N features and relevant columns
    feature_cols = [col for col in cross_condition_df.columns 
                   if col not in ['variance_across', 'mean_across', 'cv_across']]
    
    plot_df = cross_condition_df.head(top_n)[feature_cols]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(plot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Mean Importance'},
                linewidths=0.5, linecolor='white')
    
    ax.set_title(f'Feature Importance Across {dimension.capitalize()}s\n(Top {top_n} Features)')
    ax.set_xlabel(dimension.capitalize())
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()

'''
def plot_stability_scatter(stability_df: pd.DataFrame,
                           tier_dict: Dict[str, List[str]],
                           figsize: tuple = None,
                           save_path: str = None):
    """
    Scatter plot: Importance vs. Stability (CV) with tier coloring.
    """
    if figsize is None:
        figsize = (SINGLE_COL_WIDTH * 1.3, SINGLE_COL_WIDTH)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Assign tier to each feature
    tier_colors = {
        'tier1_universal': 'darkgreen',
        'tier2_context_sensitive': 'orange',
        'tier3_specialized': 'darkred'
    }
    
    tier_labels = {
        'tier1_universal': 'Universal',
        'tier2_context_sensitive': 'Context-Sensitive',
        'tier3_specialized': 'Specialized'
    }
    
    for tier_key, color in tier_colors.items():
        features = tier_dict[tier_key]
        tier_df = stability_df[stability_df['feature'].isin(features)]
        
        ax.scatter(tier_df['mean_importance'], 
                  tier_df['cv_importance'],
                  c=color, alpha=0.6, s=80,
                  label=f'{tier_labels[tier_key]} (n={len(features)})',
                  edgecolors='black', linewidth=0.5)
    
    # Add reference lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Mean Importance')
    ax.set_ylabel('Coefficient of Variation (Stability)')
    ax.set_title('Feature Importance vs. Stability')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Annotate a few key features
    top_stable = stability_df.nsmallest(3, 'cv_importance')
    for _, row in top_stable.iterrows():
        ax.annotate(row['feature'], 
                   xy=(row['mean_importance'], row['cv_importance']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=7, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()
'''

def plot_stability_scatter(stability_df: pd.DataFrame,
                           tier_dict: Dict[str, List[str]],
                           figsize: tuple = None,
                           save_path: str = None,
                           annotate: bool = True,
                           cv_threshold: float = None,
                           top5_threshold: float = None):
    """
    Scatter plot: CV (x) vs Top-5 frequency (y) with tier coloring.
    Designed to visualize Tier 1 / 2 / 3 separation.
    """
    if figsize is None:
        figsize = (SINGLE_COL_WIDTH * 1.3, SINGLE_COL_WIDTH)

    fig, ax = plt.subplots(figsize=figsize)

    # Default thresholds (data-driven)
    if cv_threshold is None:
        cv_threshold = float(np.nanmedian(stability_df['cv_importance']))
    if top5_threshold is None:
        top5_threshold = float(np.nanmedian(stability_df['top5_frequency']))

    # Tier definitions
    tier_colors = {
        'tier1_universal': 'darkgreen',
        'tier2_context_sensitive': 'orange',
        'tier3_specialized': 'darkred'
    }

    tier_labels = {
        'tier1_universal': 'Universal',
        'tier2_context_sensitive': 'Context-Sensitive',
        'tier3_specialized': 'Specialized'
    }

    # Plot each tier separately
    for tier_key, color in tier_colors.items():
        features = tier_dict.get(tier_key, [])
        tier_df = stability_df[stability_df['feature'].isin(features)]

        if tier_df.empty:
            continue

        ax.scatter(
            tier_df['cv_importance'],
            tier_df['top5_frequency'],
            c=color,
            alpha=0.75,
            s=80,
            label=f'{tier_labels[tier_key]} (n={len(features)})',
            edgecolors='black',
            linewidth=0.5
        )

        # Optional annotations
        if annotate:
            for _, row in tier_df.iterrows():
                ax.annotate(
                    row['feature'],
                    xy=(row['cv_importance'], row['top5_frequency']),
                    xytext=(4, 3),
                    textcoords='offset points',
                    fontsize=7,
                    alpha=0.8
                )

    # Quadrant reference lines (tier intuition)
    ax.axvline(cv_threshold, linestyle='--', alpha=0.6, linewidth=1)
    ax.axhline(top5_threshold, linestyle='--', alpha=0.6, linewidth=1)

    ax.set_xlabel('Coefficient of Variation (Instability ↑)')
    ax.set_ylabel('Top-5 Frequency (Consistency ↑)')
    ax.set_title('Feature Stability Map (CV vs Top-5 Frequency)')
    ax.set_ylim(0, 1)

    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")

    plt.show()
    plt.close()
    
    

def create_stability_summary_table(stability_df: pd.DataFrame,
                                   tier_dict: Dict[str, List[str]],
                                   save_path: str = None) -> pd.DataFrame:
    """
    Create a publication-ready table summarizing tier statistics.
    """
    summary_data = []
    
    for tier_key in ['tier1_universal', 'tier2_context_sensitive', 'tier3_specialized']:
        features = tier_dict[tier_key]
        tier_df = stability_df[stability_df['feature'].isin(features)]
        
        if len(tier_df) == 0:
            continue
        
        summary_data.append({
            'Tier': tier_key.replace('tier', 'Tier ').replace('_', ' ').title(),
            'N Features': len(tier_df),
            'Mean Importance': tier_df['mean_importance'].mean(),
            'Mean CV': tier_df['cv_importance'].mean(),
            'Mean Top-5 Freq': tier_df['top5_frequency'].mean(),
            'Mean Top-10 Freq': tier_df['top10_frequency'].mean(),
            'Top Features': ', '.join(tier_df.head(5)['feature'].tolist())
        })
    
    summary_table = pd.DataFrame(summary_data)
    
    if save_path:
        summary_table.to_csv(save_path, index=False)
        print(f"  → Saved: {save_path}")
    
    return summary_table


# ============================================================================
# FULL STABILITY ANALYSIS PIPELINE
# ============================================================================

def run_complete_stability_analysis(df: pd.DataFrame,
                                   generators: List[str],
                                   sources: List[str],
                                   n_runs: int = 5,
                                   save_dir: str = None,
                                   feature_subset: Optional[List[str]] = None) -> Dict:
    """
    Run complete feature stability analysis pipeline.
    
    Returns:
        Dictionary containing:
        - feature_importance_df: Raw importance data
        - stability_df: Stability metrics
        - tier_dict: Feature categorization
        - generator_stability: Cross-generator analysis
        - source_stability: Cross-source analysis
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("\n" + "="*70)
    print("FEATURE STABILITY ANALYSIS")
    print("="*70 + "\n")
    
    # Step 1: Extract all feature importances
    print("Step 1: Extracting feature importances from all conditions...")
    feature_importance_df = extract_all_feature_importances(
        df, generators, sources, n_runs, feature_subset
    )
    
    if save_dir:
        csv_path = f"{save_dir}/01_feature_importances_raw.csv"
        feature_importance_df.to_csv(csv_path, index=False)
        print(f"  → Saved: {csv_path}")
    
    # Step 2: Compute stability metrics
    print("\nStep 2: Computing stability metrics...")
    stability_df = compute_feature_stability_metrics(feature_importance_df)
    
    if save_dir:
        csv_path = f"{save_dir}/02_feature_stability_metrics.csv"
        stability_df.to_csv(csv_path, index=False)
        print(f"  → Saved: {csv_path}")
    
    # Step 3: Categorize features
    print("\nStep 3: Categorizing features into tiers...")
    tier_dict = categorize_features_by_stability(stability_df)
    
    for tier, features in tier_dict.items():
        print(f"  {tier}: {len(features)} features")
        if len(features) <= 10:
            print(f"    → {', '.join(features)}")
    
    # Step 4: Cross-condition analysis
    print("\nStep 4: Analyzing cross-generator stability...")
    generator_stability = analyze_cross_condition_stability(
        feature_importance_df, dimension='generator'
    )
    
    if save_dir:
        csv_path = f"{save_dir}/03_generator_stability.csv"
        generator_stability.to_csv(csv_path)
        print(f"  → Saved: {csv_path}")
    
    print("\nStep 5: Analyzing cross-source stability...")
    source_stability = analyze_cross_condition_stability(
        feature_importance_df, dimension='source'
    )
    
    if save_dir:
        csv_path = f"{save_dir}/04_source_stability.csv"
        source_stability.to_csv(csv_path)
        print(f"  → Saved: {csv_path}")
    
    # Step 6: Create visualizations
    print("\nStep 6: Creating visualizations...")
    
    # 6a. Stability overview
    print("  6a. Feature stability overview...")
    save_path = f"{save_dir}/05_stability_overview.pdf" if save_dir else None
    plot_feature_stability_overview(stability_df, top_n=20, save_path=save_path)
    
    # 6b. Tier summary
    print("  6b. Feature tier summary...")
    save_path = f"{save_dir}/06_tier_summary.pdf" if save_dir else None
    plot_feature_tier_summary(stability_df, tier_dict, save_path=save_path)
    
    # 6c. Generator heatmap
    print("  6c. Generator stability heatmap...")
    save_path = f"{save_dir}/07_generator_heatmap.pdf" if save_dir else None
    plot_feature_heatmap_by_condition(generator_stability, 'generator', save_path=save_path)
    
    # 6d. Source heatmap
    print("  6d. Source stability heatmap...")
    save_path = f"{save_dir}/08_source_heatmap.pdf" if save_dir else None
    plot_feature_heatmap_by_condition(source_stability, 'source', save_path=save_path)
    
    # 6e. Stability scatter
    print("  6e. Stability scatter plot...")
    save_path = f"{save_dir}/09_stability_scatter.pdf" if save_dir else None
    plot_stability_scatter(stability_df, tier_dict, save_path=save_path)
    
    # Step 7: Summary table
    print("\nStep 7: Creating summary table...")
    save_path = f"{save_dir}/10_tier_summary_table.csv" if save_dir else None
    summary_table = create_stability_summary_table(stability_df, tier_dict, save_path)
    print(summary_table)
    
    print("\n" + "="*70)
    print("STABILITY ANALYSIS COMPLETE!")
    print("="*70 + "\n")
    
    return {
        'feature_importance_df': feature_importance_df,
        'stability_df': stability_df,
        'tier_dict': tier_dict,
        'generator_stability': generator_stability,
        'source_stability': source_stability,
        'summary_table': summary_table
    }


if __name__ == "__main__" and False:
    pass
    
    
    # Load data
    
    # ============================================================================
    # COMPLETE ANALYSIS PIPELINE
    # ============================================================================
    
    df = pd.read_csv("/projectnb/ivc-ml/shanzy/AI-human-text/analysis/models_generations_with_features.csv")
    df = df[df['source'] != 'summarization']
    df = df[df['generator'] != 'phi4']
    df = df[df['generator'] != 'mistral']
    
    remove_cols = [
        'sentence_complexity'
    ]
    
    df = df.drop(columns=remove_cols)
    
    generators = [g for g in df['generator'].unique() if g.lower() != 'human']
    sources = df['source'].unique()
    
    print(f"Generators: {generators}")
    print(f"Sources: {sources}")
    print(f"Total samples: {len(df)}\n")
    
 
      
    # ============================================================================
    # PART 1: FEATURE STABILITY ANALYSIS (NEW!)
    # ============================================================================
    print("\n" + "="*70)
    print("PART 1: FEATURE STABILITY ANALYSIS")
    print("="*70)
    
    stability_results = run_complete_stability_analysis(
        # df=df,
        generators=generators,
        sources=sources,
        n_runs=5,
        save_dir="/projectnb/ivc-ml/shanzy/AI-human-text/analysis/figures/feature_stability"
    )
    
    # Extract results
    feature_importance_df = stability_results['feature_importance_df']
    stability_df = stability_results['stability_df']
    tier_dict = stability_results['tier_dict']
    
    # Print key findings
    print("\n" + "="*70)
    print("KEY FINDINGS: Universal Discriminators")
    print("="*70)
    print("\nTier 1 (Universal Discriminators):")
    for feature in tier_dict['tier1_universal'][:10]:
        row = stability_df[stability_df['feature'] == feature].iloc[0]
        print(f"  • {feature}: importance={row['mean_importance']:.3f}, CV={row['cv_importance']:.3f}, top5_freq={row['top5_frequency']:.1%}")
    
    print("\nTier 2 (Context-Sensitive): {} features".format(len(tier_dict['tier2_context_sensitive'])))
    print("Top 5:", tier_dict['tier2_context_sensitive'][:5])
    
    print("\nTier 3 (Specialized): {} features".format(len(tier_dict['tier3_specialized'])))
    
    # ============================================================================
    # PART 2: SOURCE TRANSFER ANALYSIS
    # ============================================================================
    print("\n" + "="*70)
    print("PART 2: SOURCE TRANSFER ANALYSIS")
    print("="*70)
    
    source_results, source_overlap = run_all_source_transfer(
        # df=df,
        generators=generators,
        top_k=5,
        n_runs=5
    )
    
    source_summary = plot_full_analysis_overview(
        results_dict=source_results,
        overlap_dict=source_overlap,
        analysis_type="source",
        save_dir="/projectnb/ivc-ml/shanzy/AI-human-text/analysis/figures/source_transfer"
    )
    
    # ============================================================================
    # PART 3: GENERATOR TRANSFER ANALYSIS
    # ============================================================================
    print("\n" + "="*70)
    print("PART 3: GENERATOR TRANSFER ANALYSIS")
    print("="*70)
    
    gen_results, gen_overlap = run_all_generator_transfer(
        # df=df,
        generators=generators,
        top_k=5,
        n_runs=5
    )
    
    gen_summary = plot_full_analysis_overview(
        results_dict=gen_results,
        overlap_dict=gen_overlap,
        analysis_type="generator",
        save_dir="/projectnb/ivc-ml/shanzy/AI-human-text/analysis/figures/generator_transfer"
    )
    
    # ============================================================================
    # PART 4: INTEGRATED ANALYSIS
    # ============================================================================
    print("\n" + "="*70)
    print("PART 4: INTEGRATED SUMMARY")
    print("="*70)
    
    print("\n📊 Universal Discriminators (work everywhere):")
    print(tier_dict['tier1_universal'])
    
    print("\n📊 Source Transfer Summary:")
    print(source_summary[['Name', 'Avg Acc (Full)', 'Transfer Gap']].head())
    
    print("\n📊 Generator Transfer Summary:")
    print(gen_summary[['Name', 'Avg Acc (Full)', 'Transfer Gap']].head())
    
    print("\n" + "="*70)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: /projectnb/ivc-ml/shanzy/AI-human-text/analysis/figures/")
    
    


# ============================================================================
# DIAGNOSTIC: Examine threshold sensitivity
# ============================================================================

def examine_threshold_sensitivity(stability_df: pd.DataFrame):
    """
    Examine how many features fall into each tier at different thresholds.
    """
    print("\n" + "="*70)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Sort by combined score
    df = stability_df.copy()
    df['combined_score'] = df['top5_frequency'] * (1 / (df['cv_importance'] + 0.1))
    df = df.sort_values('combined_score', ascending=False)
    
    print("\nTop 15 Features by Combined Score:")
    print(df[['feature', 'mean_importance', 'cv_importance', 'top5_frequency', 'top10_frequency']].head(15).to_string())
    
    print("\n" + "-"*70)
    print("Tier Counts at Different Thresholds:")
    print("-"*70)
    
    thresholds = [
        (0.80, 0.50, "Strict (Current)"),
        (0.70, 0.60, "Moderate"),
        (0.60, 0.70, "Relaxed")
    ]
    
    for top5_thresh, cv_thresh, label in thresholds:
        tier1 = df[(df['top5_frequency'] >= top5_thresh) & (df['cv_importance'] <= cv_thresh)]
        tier3 = df[df['top10_frequency'] < 0.60]
        tier2 = df[~df['feature'].isin(tier1['feature'].tolist() + tier3['feature'].tolist())]
        
        print(f"\n{label} (top5≥{top5_thresh}, CV≤{cv_thresh}):")
        print(f"  Tier 1: {len(tier1)} features")
        if len(tier1) <= 5:
            print(f"    → {tier1['feature'].tolist()}")
        print(f"  Tier 2: {len(tier2)} features")
        print(f"  Tier 3: {len(tier3)} features")
    
    # Correlation check
    print("\n" + "-"*70)
    print("Checking for Feature Correlations (Top Universal Candidates):")
    print("-"*70)
    top_candidates = df.head(10)['feature'].tolist()
    print(f"Top 10 candidates: {top_candidates}")
    print("\nNote: If lexical_diversity and repetition_score have identical metrics,")
    print("they may be duplicate/redundant features. Check feature correlation matrix.")


# ============================================================================
# Check if lexical_diversity and repetition_score are redundant
# ============================================================================

def check_feature_correlation(df: pd.DataFrame, feature1: str, feature2: str):
    """Check correlation between two features."""
    from scipy.stats import pearsonr, spearmanr
    
    if feature1 not in df.columns or feature2 not in df.columns:
        print(f"Features not found in dataframe")
        return
    
    # Compute correlation on the full dataset
    corr_pearson, p_pearson = pearsonr(df[feature1], df[feature2])
    corr_spearman, p_spearman = spearmanr(df[feature1], df[feature2])
    
    print(f"\nCorrelation between {feature1} and {feature2}:")
    print(f"  Pearson:  r = {corr_pearson:.4f} (p = {p_pearson:.4e})")
    print(f"  Spearman: ρ = {corr_spearman:.4f} (p = {p_spearman:.4e})")
    
    if abs(corr_pearson) > 0.9:
        print("  ⚠️  WARNING: Very high correlation (>0.9) - features may be redundant!")
    elif abs(corr_pearson) > 0.7:
        print("  ⚠️  High correlation (>0.7) - consider treating as one feature")
    else:
        print("  ✓ Correlation acceptable - features provide distinct information")
    
    return corr_pearson, corr_spearman




def categorize_features_by_percentile(stability_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Tier 1: Top 20th percentile in BOTH top5_frequency AND low CV
    Tier 3: Bottom 40th percentile in top10_frequency
    Tier 2: Everything else
    """
    df = stability_df.copy()
    
    # Calculate percentiles
    top5_p80 = df['top5_frequency'].quantile(0.80)
    cv_p20 = df['cv_importance'].quantile(0.20)
    top10_p40 = df['top10_frequency'].quantile(0.40)
    
    print(f"Percentile-based thresholds:")
    print(f"  Top-5 freq (80th percentile): {top5_p80:.3f}")
    print(f"  CV (20th percentile): {cv_p20:.3f}")
    print(f"  Top-10 freq (40th percentile): {top10_p40:.3f}")
    
    tier1_mask = (df['top5_frequency'] >= top5_p80) & (df['cv_importance'] <= cv_p20)
    tier3_mask = df['top10_frequency'] < top10_p40
    
    tier1_features = df[tier1_mask]['feature'].tolist()
    tier3_features = df[tier3_mask]['feature'].tolist()
    tier2_features = df[~df['feature'].isin(tier1_features + tier3_features)]['feature'].tolist()
    
    return {
        'tier1_universal': tier1_features,
        'tier2_context_sensitive': tier2_features,
        'tier3_specialized': tier3_features
    }


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def categorize_features_by_clustering(stability_df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, List[str]]:
    """
    Use k-means clustering on (top5_frequency, 1/CV) to find natural groupings.
    """
    df = stability_df.copy()
    
    # Features for clustering: high frequency + low CV = universal
    X = df[['top5_frequency', 'cv_importance']].values
    X[:, 1] = 1 / (X[:, 1] + 0.01)  # Invert CV so higher is better
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Find which cluster is "universal" (highest mean top5_frequency, lowest mean CV)
    cluster_stats = df.groupby('cluster').agg({
        'top5_frequency': 'mean',
        'cv_importance': 'mean',
        'feature': 'count'
    }).rename(columns={'feature': 'n_features'})
    
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    # Assign clusters to tiers
    # Tier 1 = highest top5_freq cluster
    # Tier 3 = lowest top5_freq cluster
    # Tier 2 = middle
    cluster_stats['tier_score'] = cluster_stats['top5_frequency'] - cluster_stats['cv_importance']
    cluster_stats = cluster_stats.sort_values('tier_score', ascending=False)
    
    tier_mapping = {
        cluster_stats.index[0]: 'tier1_universal',
        cluster_stats.index[-1]: 'tier3_specialized'
    }
    for c in cluster_stats.index[1:-1]:
        tier_mapping[c] = 'tier2_context_sensitive'
    
    result = {
        'tier1_universal': [],
        'tier2_context_sensitive': [],
        'tier3_specialized': []
    }
    
    for cluster, tier in tier_mapping.items():
        result[tier].extend(df[df['cluster'] == cluster]['feature'].tolist())
    
    return result


def categorize_features_by_composite_score(stability_df: pd.DataFrame,
                                           tier1_pct: float = 0.15,
                                           tier3_pct: float = 0.50) -> Dict[str, List[str]]:
    """
    Calculate composite score = top5_frequency * mean_importance / (CV + 0.1)
    Top 15% → Tier 1
    Bottom 50% → Tier 3
    Middle 35% → Tier 2
    """
    df = stability_df.copy()
    
    # Composite score (higher = more universal)
    df['universality_score'] = (
        df['top5_frequency'] * df['mean_importance'] / (df['cv_importance'] + 0.1)
    )
    
    df = df.sort_values('universality_score', ascending=False)
    
    n = len(df)
    n_tier1 = int(n * tier1_pct)
    n_tier3 = int(n * tier3_pct)
    
    tier1_features = df.iloc[:n_tier1]['feature'].tolist()
    tier3_features = df.iloc[-n_tier3:]['feature'].tolist()
    tier2_features = df.iloc[n_tier1:-n_tier3]['feature'].tolist()
    
    print(f"\nComposite Score Ranking (top 10):")
    print(df[['feature', 'mean_importance', 'cv_importance', 'top5_frequency', 'universality_score']].head(10).to_string())
    
    return {
        'tier1_universal': tier1_features,
        'tier2_context_sensitive': tier2_features,
        'tier3_specialized': tier3_features
    }


if __name__ == "__main__" and False:
    pass
    # Compare tier assignments across methods
    print("\n" + "="*70)
    print("COMPARING TIER ASSIGNMENT METHODS")
    print("="*70)
    
    # Method 1: Fixed thresholds (current)
    tier_fixed = categorize_features_by_stability(stability_df, 
                                                  top5_threshold=0.80, 
                                                  low_cv_threshold=0.5)
    
    # Method 2: Percentile-based
    tier_percentile = categorize_features_by_percentile(stability_df)
    
    # Method 3: Clustering
    tier_cluster = categorize_features_by_clustering(stability_df, n_clusters=3)
    
    # Method 4: Composite score
    tier_composite = categorize_features_by_composite_score(stability_df, 
                                                            tier1_pct=0.15, 
                                                            tier3_pct=0.50)
    
    # Compare Tier 1 assignments
    print("\n" + "="*70)
    print("TIER 1 (UNIVERSAL) COMPARISON:")
    print("="*70)
    print(f"Fixed thresholds:  {tier_fixed['tier1_universal']}")
    print(f"Percentile-based:  {tier_percentile['tier1_universal']}")
    print(f"K-means clustering: {tier_cluster['tier1_universal']}")
    print(f"Composite score:    {tier_composite['tier1_universal']}")
    
    # Find consensus features (appear in ≥3 methods)
    from collections import Counter
    all_tier1 = (tier_fixed['tier1_universal'] + 
                 tier_percentile['tier1_universal'] + 
                 tier_cluster['tier1_universal'] + 
                 tier_composite['tier1_universal'])
    tier1_counts = Counter(all_tier1)
    consensus_tier1 = [feat for feat, count in tier1_counts.items() if count >= 3]
    
    print(f"\n✓ CONSENSUS Tier 1 (appear in ≥3 methods): {consensus_tier1}")
    
    


# ============================================================================
# FEATURE CORRELATION ANALYSIS
# ============================================================================

from scipy.stats import pearsonr, spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

def analyze_feature_correlations(df: pd.DataFrame, 
                                 threshold: float = 0.85,
                                 save_dir: str = None) -> pd.DataFrame:
    """
    Comprehensive feature correlation analysis.
    
    Args:
        df: DataFrame with features
        threshold: Correlation threshold for identifying redundant features
        save_dir: Directory to save results
    
    Returns:
        DataFrame with high-correlation pairs
    """
    feature_cols = get_feature_columns(df)
    
    print("\n" + "="*70)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*70)
    
    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr(method='pearson')
    
    # Find high-correlation pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val)
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('abs_correlation', ascending=False)
    
    print(f"\nTotal feature pairs: {len(feature_cols) * (len(feature_cols) - 1) // 2}")
    print(f"High-correlation pairs (|r| ≥ {threshold}): {len(high_corr_df)}")
    
    if len(high_corr_df) > 0:
        print("\nTop 10 highly correlated pairs:")
        print(high_corr_df.head(10).to_string(index=False))
    else:
        print("\n✓ No high-correlation pairs found!")
    
    # Save results
    if save_dir:
        high_corr_df.to_csv(f"{save_dir}/feature_correlations.csv", index=False)
        corr_matrix.to_csv(f"{save_dir}/correlation_matrix.csv")
        print(f"\n→ Saved correlation results to {save_dir}/")
    
    return corr_matrix, high_corr_df


def plot_correlation_matrix(corr_matrix: pd.DataFrame,
                            figsize: tuple = None,
                            save_path: str = None):
    """
    Create publication-quality correlation matrix heatmap.
    """
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
                ax=ax, annot=False)
    
    ax.set_title('Feature Correlation Matrix', pad=20, fontsize=14)
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_correlation_clusters(corr_matrix: pd.DataFrame,
                              figsize: tuple = None,
                              save_path: str = None):
    """
    Create hierarchical clustering dendrogram based on feature correlations.
    """
    if figsize is None:
        figsize = (DOUBLE_COL_WIDTH, 6)
    
    # Compute distance matrix (1 - |correlation|)
    dist_matrix = 1 - np.abs(corr_matrix)
    
    # Hierarchical clustering
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage = hierarchy.linkage(condensed_dist, method='average')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dendrogram = hierarchy.dendrogram(
        linkage,
        labels=corr_matrix.columns,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=8,
        color_threshold=0.3
    )
    
    ax.set_title('Feature Correlation Clustering\n(Distance = 1 - |correlation|)', 
                 pad=20, fontsize=12)
    ax.set_ylabel('Distance', fontsize=10)
    ax.set_xlabel('Feature', fontsize=10)
    ax.axhline(y=0.15, color='red', linestyle='--', linewidth=1, 
               label='High correlation threshold (|r| = 0.85)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    
    plt.show()
    plt.close()




# ============================================================================
# IDENTIFY REDUNDANT FEATURE GROUPS
# ============================================================================

def identify_redundant_groups(corr_matrix: pd.DataFrame,
                              threshold: float = 0.85) -> List[List[str]]:
    """
    Identify groups of highly correlated features using graph-based clustering.
    
    Returns:
        List of feature groups (each group contains redundant features)
    """
    from collections import defaultdict
    import networkx as nx
    
    # Build adjacency graph
    G = nx.Graph()
    G.add_nodes_from(corr_matrix.columns)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j])
    
    # Find connected components (redundant groups)
    redundant_groups = list(nx.connected_components(G))
    redundant_groups = [list(group) for group in redundant_groups if len(group) > 1]
    
    print("\n" + "="*70)
    print("REDUNDANT FEATURE GROUPS")
    print("="*70)
    print(f"Threshold: |r| ≥ {threshold}")
    print(f"Number of redundant groups: {len(redundant_groups)}")
    
    for idx, group in enumerate(redundant_groups, 1):
        print(f"\nGroup {idx}: {len(group)} features")
        for feature in group:
            print(f"  - {feature}")
        
        # Show pairwise correlations within group
        if len(group) <= 5:
            print("  Pairwise correlations:")
            for i, feat1 in enumerate(group):
                for feat2 in group[i+1:]:
                    corr = corr_matrix.loc[feat1, feat2]
                    print(f"    {feat1} ↔ {feat2}: {corr:.3f}")
    
    return redundant_groups


# ============================================================================
# NOTEBOOK-FRIENDLY HELPERS (NO argparse required)
# ============================================================================

def load_default_csv(path: str) -> pd.DataFrame:
    """Load CSV then apply_default_filters()."""
    df = pd.read_csv(path)
    return apply_default_filters(df)


def get_generators_and_sources(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Get generator list (excluding human) and sources (unique)."""
    generators = [g for g in df["generator"].unique() if str(g).lower() != "human"]
    sources = df["source"].unique()
    return generators, sources


def run_notebook_pipeline(
    df: pd.DataFrame,
    *,
    top_k: int = 5,
    n_runs: int = 5,
    figures_root: Optional[str] = None,
    run_stability: bool = True,
    run_source_transfer: bool = True,
    run_generator_transfer: bool = True,
) -> Dict[str, object]:
    """Run the full pipeline from a Jupyter notebook.

    Parameters
    ----------
    df:
        Filtered dataframe (or call apply_default_filters first).
    top_k:
        K for Top-K model + overlap.
    n_runs:
        Number of repeated stratified holdout runs (used by train_with_cross_validation).
    figures_root:
        If set, saves figures/CSVs to this directory using the same folder structure
        as the original script (feature_stability / source_transfer / generator_transfer).
    run_stability, run_source_transfer, run_generator_transfer:
        Toggle sections.

    Returns
    -------
    dict with results objects for downstream use.
    """
    required_cols = ["generator", "source", "label", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    generators, sources = get_generators_and_sources(df)
    print(f"Generators: {generators}")
    print(f"Sources: {list(sources)}")
    print(f"Total samples: {len(df)}\n")

    out: Dict[str, object] = {}

    if run_stability:
        print("\n" + "="*70)
        print("PART 1: FEATURE STABILITY ANALYSIS")
        print("="*70)
        save_dir = os.path.join(figures_root, "feature_stability") if figures_root else None
        out["stability_results"] = run_complete_stability_analysis(
            df=df,
            generators=generators,
            sources=sources,
            n_runs=n_runs,
            save_dir=save_dir,
        )

    if run_source_transfer:
        print("\n" + "="*70)
        print("PART 2: SOURCE TRANSFER ANALYSIS")
        print("="*70)
        source_results, source_overlap = run_all_source_transfer(
            df=df, generators=generators, top_k=top_k, n_runs=n_runs
        )
        out["source_results"] = source_results
        out["source_overlap"] = source_overlap
        save_dir = os.path.join(figures_root, "source_transfer") if figures_root else None
        out["source_summary"] = plot_full_analysis_overview(
            results_dict=source_results,
            overlap_dict=source_overlap,
            analysis_type="source",
            save_dir=save_dir,
        )

    if run_generator_transfer:
        print("\n" + "="*70)
        print("PART 3: GENERATOR TRANSFER ANALYSIS")
        print("="*70)
        gen_results, gen_overlap = run_all_generator_transfer(
            df=df, generators=generators, top_k=top_k, n_runs=n_runs
        )
        out["gen_results"] = gen_results
        out["gen_overlap"] = gen_overlap
        save_dir = os.path.join(figures_root, "generator_transfer") if figures_root else None
        out["gen_summary"] = plot_full_analysis_overview(
            results_dict=gen_results,
            overlap_dict=gen_overlap,
            analysis_type="generator",
            save_dir=save_dir,
        )

    print("\n" + "="*70)
    print("✅ NOTEBOOK PIPELINE COMPLETE!")
    print("="*70)
    if figures_root:
        print(f"Results saved to: {figures_root}")

    return out
