"""
feature_importance_multi_level.py
==================================
Multi-Level Feature Importance Analysis for AI-Generated Text Detection
-----------------------------------------------------------------------

Four analysis levels
--------------------
  L0  Global    – one model trained on ALL data (all domains, all LLMs vs human)
  L1  Domain    – a SEPARATE model trained & tested WITHIN each domain
                  → answers: "Which features distinguish AI from human FOR domain X?"
  L2  LLM       – a SEPARATE model trained & tested per LLM (all domains combined)
                  → answers: "Which features distinguish LLM G from human?"
  L3  Per-Pair  – a SEPARATE model per (domain × LLM) pair
                  → answers: "What does a specialist model prefer for this exact cell?"

Every level retrains independently.  L1/L2/L3 are NOT the global model applied to
subsets — they are fresh logistic regressions that can learn condition-specific
weights.  Comparing L1/L2 importance to L0 therefore shows what is
domain/LLM-specific vs. universally discriminative.

Two importance signals at every level
--------------------------------------
  coef_importance  mean |logistic-regression coefficient| across n_runs CV splits
  perm_importance  ΔAcc         

"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional
import joblib

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
np.random.seed(SEED)

SAVE_DIR: str = "./feature_importance_results"

# ACL-style plot settings
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.titlesize":  11,
})
SINGLE_COL = 7
DOUBLE_COL = 12

# ============================================================================
# DATA UTILITIES
# ============================================================================

def get_feature_columns(df: pd.DataFrame) -> pd.Index:
    """Return all columns that are not metadata."""
    return df.drop(columns=["generator", "text", "source", "label"],
                   errors="ignore").columns


def apply_default_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Standard dataset filters: remove summarization, phi4, mistral,
    and the sentence_complexity column."""
    df = df.copy()
    if "source" in df.columns:
        df = df[df["source"] != "summarization"]
    if "generator" in df.columns:
        df = df[df["generator"] != "phi4"]
        df = df[df["generator"] != "mistral"]
    df = df.drop(columns=["sentence_complexity"], errors="ignore")
    return df


def stratified_balance(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Balance with double stratification so no single LLM or domain dominates.

    AI class   : equal samples per (generator × source) cell
    Human class: equal samples per source domain
    Final      : downsample the larger class so |AI| == |Human|
    """
    df_ai    = df[df["label"] == 1]
    df_human = df[df["label"] == 0]

    ai_groups = df_ai.groupby(["generator", "source"], group_keys=False)
    ai_min    = ai_groups.size().min()
    df_ai_bal = (ai_groups
                 .apply(lambda g: g.sample(n=ai_min, random_state=seed))
                 .reset_index(drop=True))

    hu_groups = df_human.groupby("source", group_keys=False)
    hu_min    = hu_groups.size().min()
    df_hu_bal = (hu_groups
                 .apply(lambda g: g.sample(n=hu_min, random_state=seed))
                 .reset_index(drop=True))

    target    = min(len(df_ai_bal), len(df_hu_bal))
    df_ai_bal = df_ai_bal.sample(n=target, random_state=seed)
    df_hu_bal = df_hu_bal.sample(n=target, random_state=seed)

    return (pd.concat([df_ai_bal, df_hu_bal])
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True))


def simple_balance(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Downsample majority class to minority class size."""
    min_n = df["label"].value_counts().min()
    return (df.groupby("label")
              .sample(n=min_n, random_state=seed)
              .sample(frac=1, random_state=seed)
              .reset_index(drop=True))


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def train_lr(X_train: pd.DataFrame,
             y_train: pd.Series,
             seed: int = SEED) -> Tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)
    clf    = LogisticRegression(max_iter=5000, class_weight="balanced",
                                random_state=seed, solver="liblinear")
    clf.fit(X_sc, y_train)
    return clf, scaler


def evaluate(clf: LogisticRegression,
             scaler: StandardScaler,
             X_test: pd.DataFrame,
             y_test: pd.Series) -> float:
    return float(accuracy_score(y_test, clf.predict(scaler.transform(X_test))))


# ============================================================================
# PERMUTATION IMPORTANCE
# ============================================================================

def compute_permutation_importance(
        clf: LogisticRegression,
        scaler: StandardScaler,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols,
        n_repeats: int = 10,
        seed: int = SEED) -> Tuple[float, Dict[str, float]]:
    """Permutation importance on a held-out test set.

    For each feature j:
        1. Randomly shuffle column j in the SCALED X_test (n_repeats times).
        2. ΔAcc_j = baseline_acc − accuracy_with_shuffled_j

    Higher ΔAcc_j → feature j is more important:
    destroying it hurts accuracy more.

    Args:
        clf, scaler : trained model and its scaler.
        X_test, y_test : held-out test data (unscaled DataFrames / Series).
        feature_cols   : ordered feature names.
        n_repeats      : shuffle repetitions per feature.
        seed           : RNG seed.

    Returns:
        baseline_acc : float
        perm_imp     : Dict {feature_name: mean_ΔAcc}
    """
    rng  = np.random.RandomState(seed)
    X_sc = scaler.transform(X_test)
    base = float(accuracy_score(y_test, clf.predict(X_sc)))

    perm_imp: Dict[str, float] = {}
    for j, feat in enumerate(feature_cols):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_sc.copy()
            rng.shuffle(X_perm[:, j])
            deltas.append(base - float(accuracy_score(y_test, clf.predict(X_perm))))
        perm_imp[feat] = float(np.mean(deltas))

    return base, perm_imp


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _top_k(imp_dict: Dict[str, float], k: int = 5) -> List[str]:
    return [f for f, _ in sorted(imp_dict.items(),
                                  key=lambda x: x[1], reverse=True)[:k]]


def _imp_to_df(results: Dict[str, Dict], imp_key: str,
               feature_cols) -> pd.DataFrame:
    """features × conditions importance DataFrame."""
    return pd.DataFrame(
        {cond: pd.Series(r[imp_key], index=feature_cols)
         for cond, r in results.items()}
    ).loc[feature_cols]


def _shift_df(level_imp_df: pd.DataFrame,
              global_imp: Dict[str, float]) -> pd.DataFrame:
    """Compute per-condition shift relative to global importance.

    shift[feature][condition] = level_imp - global_imp
    Positive → feature MORE important here than globally.
    Negative → feature LESS important here than globally (possibly brittle).
    """
    global_series = pd.Series(global_imp)
    return level_imp_df.subtract(global_series, axis=0).round(5)


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets: |A∩B| / |A∪B|."""
    union = set_a | set_b
    return round(len(set_a & set_b) / len(union), 4) if union else 0.0


def compute_jaccard_matrix(results: Dict[str, Dict], k: int = 5,
                            imp_key: str = "perm_importance") -> pd.DataFrame:
    """Pairwise Jaccard overlap of top-k features between every pair of conditions.

    A value of 1.0 means both conditions share exactly the same top-k features.
    A value of 0.0 means they share no features in their top-k lists.

    Args:
        results  : Dict {condition_name: result_dict}  (e.g. l1 or l2)
        k        : top-k features to compare
        imp_key  : which importance dict to use ("perm_importance" or "coef_importance")

    Returns:
        Square DataFrame (conditions × conditions) of Jaccard scores.
    """
    conds = list(results.keys())
    top_k_sets = {c: set(_top_k(results[c][imp_key], k=k)) for c in conds}
    matrix = pd.DataFrame(index=conds, columns=conds, dtype=float)
    for a in conds:
        for b in conds:
            matrix.loc[a, b] = _jaccard(top_k_sets[a], top_k_sets[b])
    return matrix


def compute_stability_table(results: Dict[str, Dict], feature_cols,
                             k: int = 5,
                             imp_key: str = "perm_importance") -> pd.DataFrame:
    """For each feature, count how many conditions it appears in the top-k list.

    Columns:
        feature          – feature name
        n_top{k}         – count of conditions where this feature is in top-k
        n_conditions     – total number of conditions evaluated
        top{k}_pct       – n_top{k} / n_conditions
        mean_importance  – mean importance value across all conditions
        std_importance   – std of importance across conditions
        conditions_top{k} – comma-separated list of conditions where it IS top-k

    Returns:
        DataFrame sorted by top{k}_pct descending then mean_importance descending.
    """
    conds = list(results.keys())
    n_conds = len(conds)
    rows = []
    for feat in feature_cols:
        in_top = [c for c in conds
                  if feat in set(_top_k(results[c][imp_key], k=k))]
        vals = [results[c][imp_key].get(feat, 0.0) for c in conds]
        rows.append({
            "feature":             feat,
            f"n_top{k}":           len(in_top),
            "n_conditions":        n_conds,
            f"top{k}_pct":         round(len(in_top) / n_conds, 4),
            "mean_importance":     round(float(np.mean(vals)), 5),
            "std_importance":      round(float(np.std(vals)),  5),
            f"conditions_top{k}":  ", ".join(in_top) if in_top else "—",
        })
    return (pd.DataFrame(rows)
            .sort_values([f"top{k}_pct", "mean_importance"], ascending=False)
            .reset_index(drop=True))


def _save_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, encoding="utf-8-sig", **kwargs)
    print(f"  [saved] {path}  ({df.shape[0]}r × {df.shape[1]}c)")


def _accuracy_row(result: Dict) -> Dict:
    return {
        "level":     result["level"],
        "condition": result["condition"],
        "accuracy":  result["accuracy"],
        "std":       result["std"],
        "top5_coef": ", ".join(_top_k(result["coef_importance"])),
        "top5_perm": ", ".join(_top_k(result["perm_importance"])),
    }


# ============================================================================
# LEVEL 0 – GLOBAL MODEL
# ============================================================================

def run_level0_global(df: pd.DataFrame,
                      feature_cols,
                      n_runs: int = 5,
                      n_perm_repeats: int = 10) -> Dict:
    """Train and evaluate the global LLM-vs-Human logistic regression.

    Uses stratified balancing (equal AI cells per generator×source,
    equal human cells per source).  Runs n_runs CV splits; coef and
    permutation importance are averaged across all runs.

    Returns a result dict with keys:
        level, condition, accuracy, std, n_samples,
        coef_importance, perm_importance, best_bundle
    """
    print("\n" + "=" * 65)
    print("LEVEL 0 — GLOBAL MODEL  (all LLMs vs Human, all domains)")
    print("=" * 65)

    df_bal = stratified_balance(df)
    X      = df_bal[feature_cols]
    y      = df_bal["label"]

    accs, coef_imps, perm_imps = [], [], []
    best_acc, best_bundle = -1.0, None

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED + run)
        clf, scaler = train_lr(X_train, y_train, seed=SEED + run)
        acc         = evaluate(clf, scaler, X_test, y_test)
        accs.append(acc)
        coef_imps.append(np.abs(clf.coef_[0]))
        _, perm = compute_permutation_importance(
            clf, scaler, X_test, y_test, feature_cols, n_perm_repeats)
        perm_imps.append(perm)

        if acc > best_acc:
            best_acc    = acc
            best_bundle = {"clf": clf, "scaler": scaler,
                           "X_train": X_train.copy(), "X_test": X_test.copy(),
                           "y_train": y_train.copy(), "y_test": y_test.copy()}

    mean_coef = dict(zip(feature_cols,
                         np.mean(coef_imps, axis=0).round(5)))
    mean_perm = {f: round(float(np.mean([p[f] for p in perm_imps])), 5)
                 for f in feature_cols}

    result = {
        "level": "L0_global", "condition": "global",
        "accuracy":        round(float(np.mean(accs)), 4),
        "std":             round(float(np.std(accs)),  4),
        "n_samples":       len(df_bal),
        "coef_importance": mean_coef,
        "perm_importance": mean_perm,
        "best_bundle":     best_bundle,
    }

    print(f"  Accuracy  : {result['accuracy']:.4f} ± {result['std']:.4f}")
    print(f"  Top-5 coef: {_top_k(mean_coef)}")
    print(f"  Top-5 perm: {_top_k(mean_perm)}")
    return result


# ============================================================================
# LEVEL 1 – PER-DOMAIN MODEL  (train on domain, test on domain)
# ============================================================================

def run_level1_domain(df: pd.DataFrame,
                      feature_cols,
                      n_runs: int = 5,
                      n_perm_repeats: int = 10) -> Dict[str, Dict]:
    """Train a SEPARATE model per source domain.

    For each domain D:
        Train data : all LLMs vs Human samples WITHIN domain D (balanced)
        Test data  : held-out 20% from domain D

    Coefficient importance and permutation importance both come from a model
    that has only ever seen domain D, so they reflect genuinely domain-specific
    discriminative features — not the global model's opinion about domain D.

    Comparing these to L0 (global) directly answers:
        "Which features are specific to this domain vs. universally useful?"

    Returns:
        Dict {domain_name: result_dict}
    """
    print("\n" + "=" * 65)
    print("LEVEL 1 — PER-DOMAIN MODEL")
    print("=" * 65)
    print("  Design : for each domain, train a FRESH logistic regression on that")
    print("           domain's data only, then test on a held-out 20% of the SAME")
    print("           domain.  Feature importance reflects what is genuinely")
    print("           discriminative WITHIN this domain — not the global model's view.")
    print("  Output : coef + perm importance per domain; Jaccard overlap between")
    print("           domains; per-feature stability (how often in top-5).")
    print("-" * 65)

    domains = sorted(df["source"].unique())
    results: Dict[str, Dict] = {}

    for domain in domains:
        # Filter to this domain only; keep all LLMs + human
        df_domain = df[df["source"] == domain]

        if df_domain["label"].nunique() < 2:
            print(f"  skip {domain}: only one class")
            continue
        if len(df_domain) < 20:
            print(f"  skip {domain}: too few samples ({len(df_domain)})")
            continue

        df_bal = simple_balance(df_domain)
        X      = df_bal[feature_cols]
        y      = df_bal["label"]

        accs, coef_imps, perm_imps = [], [], []

        for run in range(n_runs):
            if len(X) < 10:
                break
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEED + run)
            if y_test.nunique() < 2:
                continue

            clf, scaler = train_lr(X_train, y_train, seed=SEED + run)
            acc         = evaluate(clf, scaler, X_test, y_test)
            accs.append(acc)
            coef_imps.append(np.abs(clf.coef_[0]))
            _, perm = compute_permutation_importance(
                clf, scaler, X_test, y_test, feature_cols, n_perm_repeats)
            perm_imps.append(perm)

        if not accs:
            continue

        mean_coef = dict(zip(feature_cols,
                             np.mean(coef_imps, axis=0).round(5)))
        mean_perm = {f: round(float(np.mean([p[f] for p in perm_imps])), 5)
                     for f in feature_cols}
        results[domain] = {
            "level": "L1_domain", "condition": domain,
            "accuracy":        round(float(np.mean(accs)), 4),
            "std":             round(float(np.std(accs)),  4),
            "n_samples":       len(df_bal),
            "coef_importance": mean_coef,
            "perm_importance": mean_perm,
        }
        print(f"\n  Domain : {domain}")
        print(f"    n (balanced) : {len(df_bal)}")
        print(f"    Accuracy     : {results[domain]['accuracy']:.4f} ± "
              f"{results[domain]['std']:.4f}")
        print(f"    Top-5 coef   : {_top_k(mean_coef)}")
        print(f"    Top-5 perm   : {_top_k(mean_perm)}")

    # ── Post-loop stability summary ──────────────────────────────────────────
    if results:
        print("\n  — Feature stability across domains (perm, top-5) —")
        stab = compute_stability_table(results, list(feature_cols))
        for _, row in stab.iterrows():
            bar = "█" * row["n_top5"]
            print(f"    {row['feature']:22s}  {bar:<6}  "
                  f"{row['n_top5']}/{row['n_conditions']} domains  "
                  f"(mean perm={row['mean_importance']:.4f})")
        print()

    return results


# ============================================================================
# LEVEL 2 – PER-LLM MODEL  (train on LLM vs Human, test on same LLM)
# ============================================================================

def run_level2_llm(df: pd.DataFrame,
                   feature_cols,
                   generators: List[str],
                   n_runs: int = 5,
                   n_perm_repeats: int = 10) -> Dict[str, Dict]:
    """Train a SEPARATE model per LLM generator.

    For each LLM G:
        Train data : LLM G vs Human ACROSS ALL domains (balanced)
        Test data  : held-out 20% from [LLM G + Human]

    Coefficient and permutation importance reflect what genuinely distinguishes
    THIS LLM from human writing, regardless of domain.

    Comparing these to L0 (global) directly answers:
        "Which features are LLM-specific vs. universally useful?"
    Comparing L2 across LLMs answers:
        "Do different LLMs leave different stylometric fingerprints?"

    Returns:
        Dict {llm_name: result_dict}
    """
    print("\n" + "=" * 65)
    print("LEVEL 2 — PER-LLM MODEL")
    print("=" * 65)
    print("  Design : for each LLM G, train a FRESH logistic regression on")
    print("           [LLM G vs Human] samples across all domains, then test on")
    print("           a held-out 20% of that same [LLM G + Human] pool.")
    print("           Feature importance reflects what distinguishes THIS LLM from")
    print("           human writing — not a global model's averaged view.")
    print("  Output : coef + perm importance per LLM; Jaccard overlap between LLMs;")
    print("           per-feature stability (how often in top-5 across LLMs).")
    print("-" * 65)

    results: Dict[str, Dict] = {}

    for llm in sorted(generators):
        # Keep only this LLM + human rows
        df_llm = df[df["generator"].isin([llm, "human"])]

        if df_llm["label"].nunique() < 2:
            print(f"  skip {llm}: only one class")
            continue
        if len(df_llm) < 20:
            print(f"  skip {llm}: too few samples ({len(df_llm)})")
            continue

        # Stratify human by source so no domain dominates the human class
        df_bal = simple_balance(df_llm)
        X      = df_bal[feature_cols]
        y      = df_bal["label"]

        accs, coef_imps, perm_imps = [], [], []

        for run in range(n_runs):
            if len(X) < 10:
                break
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEED + run)
            if y_test.nunique() < 2:
                continue

            clf, scaler = train_lr(X_train, y_train, seed=SEED + run)
            acc         = evaluate(clf, scaler, X_test, y_test)
            accs.append(acc)
            coef_imps.append(np.abs(clf.coef_[0]))
            _, perm = compute_permutation_importance(
                clf, scaler, X_test, y_test, feature_cols, n_perm_repeats)
            perm_imps.append(perm)

        if not accs:
            continue

        mean_coef = dict(zip(feature_cols,
                             np.mean(coef_imps, axis=0).round(5)))
        mean_perm = {f: round(float(np.mean([p[f] for p in perm_imps])), 5)
                     for f in feature_cols}
        results[llm] = {
            "level": "L2_llm", "condition": llm,
            "accuracy":        round(float(np.mean(accs)), 4),
            "std":             round(float(np.std(accs)),  4),
            "n_samples":       len(df_bal),
            "coef_importance": mean_coef,
            "perm_importance": mean_perm,
        }
        print(f"\n  LLM  : {llm}")
        print(f"    n (balanced) : {len(df_bal)}")
        print(f"    Accuracy     : {results[llm]['accuracy']:.4f} ± "
              f"{results[llm]['std']:.4f}")
        print(f"    Top-5 coef   : {_top_k(mean_coef)}")
        print(f"    Top-5 perm   : {_top_k(mean_perm)}")

    # ── Post-loop stability summary ──────────────────────────────────────────
    if results:
        print("\n  — Feature stability across LLMs (perm, top-5) —")
        stab = compute_stability_table(results, list(feature_cols))
        for _, row in stab.iterrows():
            bar = "█" * row["n_top5"]
            print(f"    {row['feature']:22s}  {bar:<8}  "
                  f"{row['n_top5']}/{row['n_conditions']} LLMs  "
                  f"(mean perm={row['mean_importance']:.4f})")
        print()

    return results


# ============================================================================
# LEVEL 3 – PER (DOMAIN, LLM) PAIR
# ============================================================================

def run_level3_pairs(df: pd.DataFrame,
                     feature_cols,
                     generators: List[str],
                     n_runs: int = 5,
                     n_perm_repeats: int = 10) -> Dict[str, Dict]:
    """Train a SEPARATE model for each (domain, LLM) pair.

    Each pair is: [LLM + human] samples within that domain, balanced
    (simple minority-class downsampling).

    Coefficient importance here reflects what the specialised model learned —
    compare to L0 coef_importance to see where a pair-level expert diverges
    from the global model.

    Returns:
        Dict {"domain|llm": result_dict}
    """
    print("\n" + "=" * 65)
    print("LEVEL 3 — PER (DOMAIN, LLM) PAIR")
    print("=" * 65)

    domains = sorted(df["source"].unique())
    results: Dict[str, Dict] = {}

    for domain in domains:
        for llm in sorted(generators):
            key     = f"{domain}|{llm}"
            df_pair = df[
                (df["source"] == domain) &
                (df["generator"].isin([llm, "human"]))
            ]

            if df_pair["label"].nunique() < 2:
                print(f"  skip {key}: only one class present")
                continue
            if len(df_pair) < 20:
                print(f"  skip {key}: too few samples ({len(df_pair)})")
                continue

            df_bal = simple_balance(df_pair)
            X      = df_bal[feature_cols]
            y      = df_bal["label"]

            accs, coef_imps, perm_imps = [], [], []

            for run in range(n_runs):
                if len(X) < 10:
                    break
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y,
                    random_state=SEED + run)
                if y_test.nunique() < 2:
                    continue

                clf, scaler = train_lr(X_train, y_train, seed=SEED + run)
                acc         = evaluate(clf, scaler, X_test, y_test)
                accs.append(acc)
                coef_imps.append(np.abs(clf.coef_[0]))
                _, perm = compute_permutation_importance(
                    clf, scaler, X_test, y_test, feature_cols, n_perm_repeats)
                perm_imps.append(perm)

            if not accs:
                continue

            mean_coef = dict(zip(feature_cols,
                                 np.mean(coef_imps, axis=0).round(5)))
            mean_perm = {f: round(float(np.mean([p[f] for p in perm_imps])), 5)
                         for f in feature_cols}
            results[key] = {
                "level": "L3_pair", "condition": key,
                "domain": domain, "llm": llm,
                "accuracy":  round(float(np.mean(accs)), 4),
                "std":       round(float(np.std(accs)),  4),
                "n_samples": len(df_bal),
                "coef_importance": mean_coef,
                "perm_importance": mean_perm,
            }
            print(f"  {key:35s}  acc={results[key]['accuracy']:.4f} ± "
                  f"{results[key]['std']:.4f}  n={len(df_bal)}"
                  f"\n    Top-5 coef : {_top_k(mean_coef)}"
                  f"\n    Top-5 perm : {_top_k(mean_perm)}")

    return results


# ============================================================================
# FEATURE ROBUSTNESS RANKING
# ============================================================================

def compute_feature_robustness(
        l0: Dict,
        l1: Dict[str, Dict],
        l2: Dict[str, Dict],
        l3: Dict[str, Dict],
        feature_cols) -> pd.DataFrame:
    """Rank features by how consistently important they are across all conditions.

    For each feature:
        global_perm          – perm importance in global model
        mean_perm_all        – mean perm importance across L0 + L1 + L2 + L3
        std_perm_all         – std (higher → more brittle across conditions)
        cv_perm              – coefficient of variation (std / mean)
        min_perm             – worst-case importance (smallest condition value)
        n_top5_appearances   – how often the feature is in top-5 across conditions
        robustness_score     – mean_perm / (1 + cv_perm)  (high mean, low CV)

    Returns:
        DataFrame sorted by robustness_score descending.
    """
    all_conditions: Dict[str, Dict[str, float]] = {"global": l0["perm_importance"]}
    for cond, r in l1.items():
        all_conditions[f"dom_{cond}"] = r["perm_importance"]
    for cond, r in l2.items():
        all_conditions[f"llm_{cond}"] = r["perm_importance"]
    for cond, r in l3.items():
        all_conditions[f"pair_{cond}"] = r["perm_importance"]

    n_conditions = len(all_conditions)
    rows = []
    for feat in feature_cols:
        vals = [v[feat] for v in all_conditions.values() if feat in v]
        if not vals:
            continue
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        cv_v   = std_v / mean_v if mean_v > 1e-9 else float("inf")
        min_v  = float(np.min(vals))

        # Count top-5 appearances across all conditions
        n_top5 = sum(
            1 for imp in all_conditions.values()
            if feat in _top_k(imp, k=5)
        )

        rows.append({
            "feature":            feat,
            "global_perm":        round(l0["perm_importance"].get(feat, 0), 5),
            "mean_perm_all":      round(mean_v, 5),
            "std_perm_all":       round(std_v,  5),
            "cv_perm":            round(cv_v,   4),
            "min_perm":           round(min_v,  5),
            "n_top5_appearances": n_top5,
            "n_conditions":       n_conditions,
            "top5_pct":           round(n_top5 / n_conditions, 4),
            # High mean AND low variability → robust
            "robustness_score":   round(mean_v / (1.0 + cv_v), 5),
        })

    return (pd.DataFrame(rows)
            .sort_values("robustness_score", ascending=False)
            .reset_index(drop=True))


# ============================================================================
# SAVE ALL RESULTS
# ============================================================================

def save_all_results(l0: Dict,
                     l1: Dict[str, Dict],
                     l2: Dict[str, Dict],
                     l3: Dict[str, Dict],
                     feature_cols,
                     save_dir: str = SAVE_DIR) -> None:
    """Write all CSVs to save_dir following the layout in the module docstring.

    Each saved file is printed with a description of what it contains so output
    logs are self-documenting.
    """
    fc = list(feature_cols)

    print("\n" + "=" * 65)
    print("SAVING RESULTS")
    print("=" * 65)

    # ── accuracy summary across all levels ──────────────────────────────────
    print("\n[accuracy_all_levels.csv]")
    print("  One row per condition across all four levels.")
    print("  Columns: level, condition, accuracy (mean), std,")
    print("           top5_coef (top-5 features by |coef|),")
    print("           top5_perm (top-5 features by perm ΔAcc).")
    acc_rows = [_accuracy_row(l0)]
    for r in l1.values():
        acc_rows.append(_accuracy_row(r))
    for r in l2.values():
        acc_rows.append(_accuracy_row(r))
    for r in l3.values():
        acc_rows.append(_accuracy_row(r))
    _save_csv(pd.DataFrame(acc_rows),
              os.path.join(save_dir, "accuracy_all_levels.csv"), index=False)

    # ── L0 ──────────────────────────────────────────────────────────────────
    d = os.path.join(save_dir, "L0_global")
    print(f"\n[L0_global/]  — single model trained on ALL data")
    print("  coef_importance.csv : mean |coefficient| per feature (global model)")
    print("  perm_importance.csv : ΔAcc when each feature is shuffled (global model, full test set)")
    _save_csv(pd.DataFrame([l0["coef_importance"]]).T.rename(columns={0: "importance"}),
              os.path.join(d, "coef_importance.csv"))
    _save_csv(pd.DataFrame([l0["perm_importance"]]).T.rename(columns={0: "importance"}),
              os.path.join(d, "perm_importance.csv"))

    # ── L1 ──────────────────────────────────────────────────────────────────
    d = os.path.join(save_dir, "L1_domain")
    print(f"\n[L1_domain/]  — one SEPARATE model per domain (train & test within domain)")
    print("  accuracy.csv        : accuracy of each domain-specific model")
    print("  coef_importance.csv : features × domains; each column = that domain's model weights")
    print("  perm_importance.csv : features × domains; each value = ΔAcc on that domain's test set")
    print("  perm_shift.csv      : Δ vs L0 global perm importance")
    print("                        positive = more important here than globally (domain-specific signal)")
    print("                        negative = less important here than globally (brittle for this domain)")
    print("  feature_stability.csv : how many domains each feature appears in top-5")
    print("                          high count = robust across domains")
    print("  jaccard_overlap.csv   : pairwise Jaccard of top-5 perm features between domains")
    print("                          1.0 = identical top-5; 0.0 = no shared features")
    _save_csv(pd.DataFrame([_accuracy_row(r) for r in l1.values()]),
              os.path.join(d, "accuracy.csv"), index=False)
    coef1 = _imp_to_df(l1, "coef_importance", fc)
    perm1 = _imp_to_df(l1, "perm_importance", fc)
    _save_csv(coef1, os.path.join(d, "coef_importance.csv"))
    _save_csv(perm1, os.path.join(d, "perm_importance.csv"))
    _save_csv(_shift_df(perm1, l0["perm_importance"]),
              os.path.join(d, "perm_shift.csv"))
    _save_csv(compute_stability_table(l1, fc),
              os.path.join(d, "feature_stability.csv"), index=False)
    _save_csv(compute_jaccard_matrix(l1),
              os.path.join(d, "jaccard_overlap.csv"))

    # ── L2 ──────────────────────────────────────────────────────────────────
    d = os.path.join(save_dir, "L2_llm")
    print(f"\n[L2_llm/]  — one SEPARATE model per LLM (train & test within LLM vs Human)")
    print("  accuracy.csv        : accuracy of each LLM-specific model")
    print("  coef_importance.csv : features × LLMs; each column = that LLM's model weights")
    print("  perm_importance.csv : features × LLMs; each value = ΔAcc on that LLM's test set")
    print("  perm_shift.csv      : Δ vs L0 global perm importance")
    print("                        positive = more important for this LLM than globally")
    print("                        negative = less important for this LLM (may be LLM-agnostic)")
    print("  feature_stability.csv : how many LLMs each feature appears in top-5")
    print("  jaccard_overlap.csv   : pairwise Jaccard of top-5 perm features between LLMs")
    _save_csv(pd.DataFrame([_accuracy_row(r) for r in l2.values()]),
              os.path.join(d, "accuracy.csv"), index=False)
    coef2 = _imp_to_df(l2, "coef_importance", fc)
    perm2 = _imp_to_df(l2, "perm_importance", fc)
    _save_csv(coef2, os.path.join(d, "coef_importance.csv"))
    _save_csv(perm2, os.path.join(d, "perm_importance.csv"))
    _save_csv(_shift_df(perm2, l0["perm_importance"]),
              os.path.join(d, "perm_shift.csv"))
    _save_csv(compute_stability_table(l2, fc),
              os.path.join(d, "feature_stability.csv"), index=False)
    _save_csv(compute_jaccard_matrix(l2),
              os.path.join(d, "jaccard_overlap.csv"))

    # ── L3 ──────────────────────────────────────────────────────────────────
    d = os.path.join(save_dir, "L3_pair")
    print(f"\n[L3_pair/]  — one SEPARATE model per (domain × LLM) pair")
    print("  accuracy.csv        : accuracy of each pair-specific model")
    print("  coef_importance.csv : features × pairs  (pair = 'domain|llm')")
    print("  perm_importance.csv : features × pairs")
    print("  coef_shift.csv      : Δ coef vs L0 global — where specialist departs from generalist")
    print("  perm_shift.csv      : Δ perm vs L0 global")
    _save_csv(pd.DataFrame([_accuracy_row(r) for r in l3.values()]),
              os.path.join(d, "accuracy.csv"), index=False)
    coef3 = _imp_to_df(l3, "coef_importance", fc)
    perm3 = _imp_to_df(l3, "perm_importance", fc)
    _save_csv(coef3, os.path.join(d, "coef_importance.csv"))
    _save_csv(perm3, os.path.join(d, "perm_importance.csv"))
    _save_csv(_shift_df(coef3, l0["coef_importance"]),
              os.path.join(d, "coef_shift.csv"))
    _save_csv(_shift_df(perm3, l0["perm_importance"]),
              os.path.join(d, "perm_shift.csv"))

    # ── cross-level comparison ───────────────────────────────────────────────
    d = os.path.join(save_dir, "comparison")
    print(f"\n[comparison/]  — aggregated views across all levels")
    print("  cross_level_perm_importance.csv : features × all conditions (L0 + L1 + L2)")
    print("                                    each column is a condition-specific model's perm ΔAcc")
    print("  cross_level_coef_importance.csv : features × L0 global + all L3 pairs")
    print("                                    shows how specialist coef weights depart from global")
    print("  feature_robustness_ranking.csv  : one row per feature, ranked by robustness_score")
    print("                                    robustness_score = mean_perm / (1 + CV)")
    print("                                    high score = consistently important, low variance")

    cross_perm = pd.DataFrame({"global": pd.Series(l0["perm_importance"])})
    for cond, r in l1.items():
        cross_perm[f"dom_{cond}"] = pd.Series(r["perm_importance"])
    for cond, r in l2.items():
        cross_perm[f"llm_{cond}"] = pd.Series(r["perm_importance"])
    cross_perm = cross_perm.loc[fc]
    _save_csv(cross_perm, os.path.join(d, "cross_level_perm_importance.csv"))

    cross_coef = pd.DataFrame({"global": pd.Series(l0["coef_importance"])})
    for cond, r in l3.items():
        cross_coef[f"pair_{cond}"] = pd.Series(r["coef_importance"])
    cross_coef = cross_coef.loc[fc]
    _save_csv(cross_coef, os.path.join(d, "cross_level_coef_importance.csv"))

    rob = compute_feature_robustness(l0, l1, l2, l3, fc)
    _save_csv(rob, os.path.join(d, "feature_robustness_ranking.csv"), index=False)

    rob_l0l1l2 = compute_feature_robustness(l0, l1, l2, {}, fc)
    _save_csv(rob_l0l1l2, os.path.join(d, "feature_robustness_ranking_L0L1L2.csv"), index=False)
    print("  feature_robustness_ranking_L0L1L2.csv: same metric using only Global + Domain + LLM conditions (no pairwise)")

    print(f"\n{'=' * 65}")
    print(f"All results saved under: {save_dir}")
    print(f"{'=' * 65}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_accuracy_by_level(l0: Dict,
                           l1: Dict[str, Dict],
                           l2: Dict[str, Dict],
                           l3: Dict[str, Dict],
                           save_path: str = None) -> None:
    """Grouped bar chart: accuracy per condition, coloured by level.

    Shows the global baseline (L0) as a horizontal reference line,
    then per-domain (L1), per-LLM (L2), and per-pair (L3) bars.
    """
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 4),
                             sharey=True, constrained_layout=True)
    global_acc = l0["accuracy"]

    for ax, (level_results, title) in zip(
            axes,
            [(l1, "L1: Per Domain"), (l2, "L2: Per LLM"), (l3, "L3: Per Pair")]):
        conds = list(level_results.keys())
        accs  = [level_results[c]["accuracy"] for c in conds]
        errs  = [level_results[c]["std"]      for c in conds]
        x     = np.arange(len(conds))
        ax.bar(x, accs, yerr=errs, capsize=3,
               color="steelblue", alpha=0.8, edgecolor="k", linewidth=0.5)
        ax.axhline(global_acc, color="crimson", linewidth=1.5, linestyle="--",
                   label=f"Global ({global_acc:.3f})")
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Accuracy Across Analysis Levels", fontweight="bold")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [fig] {save_path}")
    plt.show()


def plot_importance_heatmap(imp_df: pd.DataFrame,
                            title: str,
                            top_n: int = 15,
                            save_path: str = None) -> None:
    """Heatmap: top-N features (rows) × conditions (columns).

    Features are sorted by their mean importance across all columns.
    """
    # Select top-N features by mean importance
    top_feats = (imp_df.mean(axis=1)
                        .sort_values(ascending=False)
                        .head(top_n).index.tolist())
    data = imp_df.loc[top_feats]

    figsize = (max(SINGLE_COL, data.shape[1] * 0.8),
               max(4, top_n * 0.4))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data.astype(float), ax=ax, cmap="YlOrRd",
                linewidths=0.3, linecolor="white",
                annot=(data.shape[1] <= 10), fmt=".3f",
                cbar_kws={"label": "Importance"})
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [fig] {save_path}")
    plt.show()


def plot_perm_shift_heatmap(shift_df: pd.DataFrame,
                            title: str,
                            top_n: int = 15,
                            save_path: str = None) -> None:
    """Heatmap of perm importance shift (Δ vs global).

    Blue → feature less important than globally (brittle under this condition).
    Red  → feature more important than globally.
    """
    top_feats = (shift_df.abs().mean(axis=1)
                           .sort_values(ascending=False)
                           .head(top_n).index.tolist())
    data = shift_df.loc[top_feats]

    figsize = (max(SINGLE_COL, data.shape[1] * 0.8),
               max(4, top_n * 0.4))
    vmax = max(abs(data.values.min()), data.values.max())
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data.astype(float), ax=ax, cmap="RdBu_r",
                center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.3, linecolor="white",
                annot=(data.shape[1] <= 10), fmt=".3f",
                cbar_kws={"label": "ΔPerm (condition − global)"})
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [fig] {save_path}")
    plt.show()


def plot_coef_vs_perm(coef_imp: Dict[str, float],
                      perm_imp: Dict[str, float],
                      condition: str,
                      top_n: int = 10,
                      save_path: str = None) -> None:
    """Scatter plot: coefficient importance vs permutation importance per feature.

    Features on the diagonal → both measures agree.
    Top-right → universally important.
    High coef + low perm → coefficient captures the weight but permuting
    that feature doesn't hurt much (possibly correlated with another feature).
    """
    feats  = list(coef_imp.keys())
    c_vals = [coef_imp[f] for f in feats]
    p_vals = [perm_imp[f] for f in feats]

    # Label top-N by mean of both
    combined = [(f, (c + p) / 2)
                for f, c, p in zip(feats, c_vals, p_vals)]
    top_feats = {f for f, _ in sorted(combined, key=lambda x: x[1],
                                      reverse=True)[:top_n]}

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.9))
    ax.scatter(c_vals, p_vals, alpha=0.6, s=30, color="steelblue",
               edgecolors="white", linewidth=0.3)
    for f, c, p in zip(feats, c_vals, p_vals):
        if f in top_feats:
            ax.annotate(f, (c, p), fontsize=6, ha="left",
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Coefficient Importance  (mean |coef|)")
    ax.set_ylabel("Permutation Importance  (ΔAcc)")
    ax.set_title(f"Coef vs Perm Importance — {condition}", fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [fig] {save_path}")
    plt.show()


def plot_feature_robustness(robustness_df: pd.DataFrame,
                            top_n: int = 20,
                            save_path: str = None) -> None:
    """Horizontal bar chart: top-N features by robustness score.

    Colour encodes CV (coefficient of variation): low CV = dark = stable.
    """
    df = robustness_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, top_n * 0.35))

    norm = plt.Normalize(df["cv_perm"].min(), df["cv_perm"].max())
    colors = plt.cm.RdYlGn_r(norm(df["cv_perm"].values))

    bars = ax.barh(df["feature"][::-1], df["robustness_score"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.4)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("CV of perm importance\n(high = brittle)", fontsize=7)
    ax.set_xlabel("Robustness Score  (mean_perm / (1 + CV))")
    ax.set_title("Feature Robustness Ranking\n"
                 "(top = universally important & stable across conditions)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [fig] {save_path}")
    plt.show()


def plot_top_features_comparison(l0: Dict,
                                 l1: Dict[str, Dict],
                                 l2: Dict[str, Dict],
                                 top_n: int = 8,
                                 imp_type: str = "perm",
                                 save_path: str = None) -> None:
    """Grouped bar chart: top-N global features' importance across L0, L1, L2.

    Lets you visually compare how much each globally-important feature
    shifts across domains and LLMs.
    """
    key = "perm_importance" if imp_type == "perm" else "coef_importance"

    global_top = _top_k(l0[key], k=top_n)
    all_conds  = {"global": l0[key]}
    all_conds.update({f"dom_{k}": v[key] for k, v in l1.items()})
    all_conds.update({f"llm_{k}": v[key] for k, v in l2.items()})

    cond_names = list(all_conds.keys())
    n_conds    = len(cond_names)
    n_feats    = len(global_top)
    x          = np.arange(n_feats)
    width      = 0.8 / n_conds

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 5))
    palette = plt.cm.tab20.colors
    for i, cond in enumerate(cond_names):
        vals = [all_conds[cond].get(f, 0) for f in global_top]
        ax.bar(x + i * width, vals, width, label=cond,
               color=palette[i % len(palette)], alpha=0.85, edgecolor="k",
               linewidth=0.3)

    ax.set_xticks(x + width * (n_conds - 1) / 2)
    ax.set_xticklabels(global_top, rotation=30, ha="right")
    ax.set_ylabel(f"{'Perm' if imp_type == 'perm' else 'Coef'} Importance")
    ax.set_title(
        f"Top-{top_n} Global Features — {'Permutation' if imp_type == 'perm' else 'Coefficient'} "
        f"Importance Across Conditions",
        fontweight="bold")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [fig] {save_path}")
    plt.show()


# ============================================================================
# MASTER PIPELINE
# ============================================================================

def run_full_analysis(df: pd.DataFrame,
                      feature_subset: Optional[List[str]] = None,
                      n_runs: int = 5,
                      n_perm_repeats: int = 10,
                      save_dir: str = SAVE_DIR) -> Dict:
    """Run the complete multi-level feature importance analysis.

    Args:
        df             : filtered DataFrame (apply_default_filters first).
        feature_subset : restrict to these features; None = all features.
        n_runs         : CV splits per level.
        n_perm_repeats : shuffle repeats per feature for permutation importance.
        save_dir       : root directory for CSV and figure output.

    Returns:
        Dict with keys: l0, l1, l2, l3, robustness_df
    """
    feature_cols = get_feature_columns(df)
    if feature_subset is not None:
        invalid = [f for f in feature_subset if f not in feature_cols]
        if invalid:
            raise ValueError(f"Unknown features: {invalid}")
        feature_cols = pd.Index(feature_subset)

    generators = sorted([g for g in df["generator"].unique()
                         if g.lower() != "human"])

    print(f"\nDataset : {len(df)} rows")
    print(f"Features: {len(feature_cols)}")
    print(f"Domains : {sorted(df['source'].unique())}")
    print(f"LLMs    : {generators}")

    # ── Run all levels ───────────────────────────────────────────────────────
    l0 = run_level0_global(df, feature_cols, n_runs, n_perm_repeats)
    l1 = run_level1_domain(df, feature_cols, n_runs, n_perm_repeats)
    l2 = run_level2_llm(   df, feature_cols, generators, n_runs, n_perm_repeats)
    l3 = run_level3_pairs( df, feature_cols, generators, n_runs, n_perm_repeats)

    # ── Save CSVs ────────────────────────────────────────────────────────────
    save_all_results(l0, l1, l2, l3, feature_cols, save_dir)

    # ── Figures ──────────────────────────────────────────────────────────────
    figs_dir = os.path.join(save_dir, "figures")
    print("\nGenerating figures...")

    plot_accuracy_by_level(
        l0, l1, l2, l3,
        save_path=os.path.join(figs_dir, "accuracy_by_level.pdf"))

    # Permutation importance heatmaps
    fc = list(feature_cols)
    perm1 = _imp_to_df(l1, "perm_importance", fc)
    perm2 = _imp_to_df(l2, "perm_importance", fc)

    plot_importance_heatmap(
        perm1,
        "L1 Perm Importance — Domain-Specific Models\n"
        "(each column = model trained & tested within that domain)",
        save_path=os.path.join(figs_dir, "L1_perm_importance_heatmap.pdf"))
    plot_importance_heatmap(
        perm2,
        "L2 Perm Importance — LLM-Specific Models\n"
        "(each column = model trained & tested on that LLM vs Human)",
        save_path=os.path.join(figs_dir, "L2_perm_importance_heatmap.pdf"))

    # Perm shift heatmaps (brittle vs robust)
    plot_perm_shift_heatmap(
        _shift_df(perm1, l0["perm_importance"]),
        "L1 ΔPerm Importance (Domain-Specific − Global)\n"
        "Red = more important within domain  |  Blue = less important (brittle)",
        save_path=os.path.join(figs_dir, "L1_perm_shift_heatmap.pdf"))
    plot_perm_shift_heatmap(
        _shift_df(perm2, l0["perm_importance"]),
        "L2 ΔPerm Importance (LLM-Specific − Global)\n"
        "Red = more important for this LLM  |  Blue = less important (brittle)",
        save_path=os.path.join(figs_dir, "L2_perm_shift_heatmap.pdf"))

    # Coef vs Perm scatter for global model
    plot_coef_vs_perm(
        l0["coef_importance"], l0["perm_importance"], "Global (L0)",
        save_path=os.path.join(figs_dir, "L0_coef_vs_perm_scatter.pdf"))

    # Top-feature comparison across L0 / L1 / L2
    plot_top_features_comparison(
        l0, l1, l2, top_n=8, imp_type="perm",
        save_path=os.path.join(figs_dir, "top_features_perm_comparison.pdf"))

    # Feature robustness ranking
    robustness_df = compute_feature_robustness(l0, l1, l2, l3, fc)
    plot_feature_robustness(
        robustness_df,
        save_path=os.path.join(figs_dir, "feature_robustness_ranking.pdf"))

    print_analysis_summary(l0, l1, l2, l3, robustness_df, save_dir)
    return {"l0": l0, "l1": l1, "l2": l2, "l3": l3,
            "robustness_df": robustness_df}


# ============================================================================
# RESULTS SUMMARY PRINTER
# ============================================================================

def print_analysis_summary(l0: Dict,
                            l1: Dict[str, Dict],
                            l2: Dict[str, Dict],
                            l3: Dict[str, Dict],
                            robustness_df: pd.DataFrame,
                            save_dir: str = SAVE_DIR) -> None:
    """Print a human-readable summary of all key results after the analysis.

    This is the final output block — designed to be readable in a notebook or
    terminal without opening any CSV files.
    """
    sep  = "=" * 65
    sep2 = "-" * 65

    print(f"\n{sep}")
    print("ANALYSIS COMPLETE — RESULTS SUMMARY")
    print(sep)

    # ── L0: Global ───────────────────────────────────────────────────────────
    print("\n▶ L0  GLOBAL MODEL  (all data, all domains, all LLMs vs Human)")
    print(f"   Source : one balanced model on ALL data")
    print(f"   Accuracy    : {l0['accuracy']:.4f} ± {l0['std']:.4f}")
    print(f"   Top-5 coef  : {_top_k(l0['coef_importance'])}")
    print(f"   Top-5 perm  : {_top_k(l0['perm_importance'])}")

    # ── L1: Domain ───────────────────────────────────────────────────────────
    print(f"\n▶ L1  PER-DOMAIN MODELS  (separate model trained within each domain)")
    print(f"   {'Domain':<24}  {'Accuracy':>10}  {'Top-5 perm features'}")
    print(f"   {sep2}")
    for dom, r in sorted(l1.items(), key=lambda x: -x[1]['accuracy']):
        top5 = ", ".join(_top_k(r['perm_importance']))
        print(f"   {dom:<24}  {r['accuracy']:.4f}±{r['std']:.4f}  {top5}")

    if l1:
        accs = [r['accuracy'] for r in l1.values()]
        print(f"\n   Accuracy range : {min(accs):.4f} – {max(accs):.4f}  "
              f"(spread = {max(accs)-min(accs):.4f})")
        stab1 = compute_stability_table(l1,
                    list(next(iter(l1.values()))['perm_importance'].keys()))
        print(f"   Features in top-5 for ALL domains  : "
              f"{list(stab1[stab1['top5_pct']==1.0]['feature'])}")
        print(f"   Features in top-5 for ≥3/5 domains : "
              f"{list(stab1[stab1['n_top5']>=3]['feature'])}")

        jac1 = compute_jaccard_matrix(l1)
        off_diag = jac1.values[~np.eye(len(jac1), dtype=bool)].astype(float)
        print(f"   Jaccard overlap (top-5 perm) between domains:")
        print(f"     mean={off_diag.mean():.3f}  "
              f"min={off_diag.min():.3f}  max={off_diag.max():.3f}")
        print(f"     (1.0 = identical top-5; 0.0 = no shared features)")

    # ── L2: LLM ──────────────────────────────────────────────────────────────
    print(f"\n▶ L2  PER-LLM MODELS  (separate model per LLM vs Human, all domains)")
    print(f"   {'LLM':<26}  {'Accuracy':>10}  {'Top-5 perm features'}")
    print(f"   {sep2}")
    for llm, r in sorted(l2.items(), key=lambda x: -x[1]['accuracy']):
        top5 = ", ".join(_top_k(r['perm_importance']))
        print(f"   {llm:<26}  {r['accuracy']:.4f}±{r['std']:.4f}  {top5}")

    if l2:
        accs = [r['accuracy'] for r in l2.values()]
        print(f"\n   Accuracy range : {min(accs):.4f} – {max(accs):.4f}  "
              f"(spread = {max(accs)-min(accs):.4f})")
        stab2 = compute_stability_table(l2,
                    list(next(iter(l2.values()))['perm_importance'].keys()))
        print(f"   Features in top-5 for ALL LLMs  : "
              f"{list(stab2[stab2['top5_pct']==1.0]['feature'])}")
        print(f"   Features in top-5 for ≥5/7 LLMs : "
              f"{list(stab2[stab2['n_top5']>=5]['feature'])}")

        jac2 = compute_jaccard_matrix(l2)
        off_diag2 = jac2.values[~np.eye(len(jac2), dtype=bool)].astype(float)
        print(f"   Jaccard overlap (top-5 perm) between LLMs:")
        print(f"     mean={off_diag2.mean():.3f}  "
              f"min={off_diag2.min():.3f}  max={off_diag2.max():.3f}")

    # ── L3: Pair ─────────────────────────────────────────────────────────────
    print(f"\n▶ L3  PER-PAIR MODELS  (domain × LLM specialist, {len(l3)} pairs)")
    if l3:
        accs = [r['accuracy'] for r in l3.values()]
        perfect = [k for k, r in l3.items() if r['accuracy'] >= 1.0]
        hardest = sorted(l3.items(), key=lambda x: x[1]['accuracy'])[:3]
        print(f"   Accuracy range  : {min(accs):.4f} – {max(accs):.4f}")
        print(f"   Mean accuracy   : {np.mean(accs):.4f}")
        print(f"   Perfect (1.00)  : {len(perfect)} pairs — {perfect}")
        print(f"   Hardest 3 pairs :")
        for k, r in hardest:
            print(f"     {k:<35}  {r['accuracy']:.4f}")

    # ── Feature robustness ───────────────────────────────────────────────────
    print(f"\n▶ FEATURE ROBUSTNESS RANKING  (across all conditions)")
    print(f"   robustness_score = mean_perm_ΔAcc / (1 + CV)")
    print(f"   High score = consistently important AND low variance across conditions.")
    print(f"\n   {'Rank':<5}  {'Feature':<22}  {'Robustness':>10}  "
          f"{'Mean ΔAcc':>9}  {'CV':>6}  {'Top5_pct':>9}")
    print(f"   {sep2}")
    for i, row in robustness_df.head(10).iterrows():
        tier = ("★★ Tier 1" if row['top5_pct'] >= 0.85
                else "★  Tier 2" if row['top5_pct'] >= 0.50
                else "   Tier 3")
        print(f"   {i+1:<5}  {row['feature']:<22}  "
              f"{row['robustness_score']:>10.5f}  "
              f"{row['mean_perm_all']:>9.5f}  "
              f"{row['cv_perm']:>6.3f}  "
              f"{row['top5_pct']:>8.1%}  {tier}")

    # ── Output files ─────────────────────────────────────────────────────────
    print(f"\n▶ OUTPUT FILES  (all saved under: {save_dir})")
    file_descriptions = [
        ("accuracy_all_levels.csv",              "Accuracy + top-5 features for every condition across all 4 levels"),
        ("L0_global/coef_importance.csv",         "Global model: mean |coefficient| per feature"),
        ("L0_global/perm_importance.csv",         "Global model: ΔAcc when each feature is shuffled"),
        ("L1_domain/accuracy.csv",                "Per-domain model accuracy"),
        ("L1_domain/coef_importance.csv",         "features × domains: domain-specific model coefficient weights"),
        ("L1_domain/perm_importance.csv",         "features × domains: domain-specific model permutation ΔAcc"),
        ("L1_domain/perm_shift.csv",              "Δ(L1 perm − L0 perm): positive=more important within domain"),
        ("L1_domain/feature_stability.csv",       "Per-feature: how many domains it appears in top-5 (stability)"),
        ("L1_domain/jaccard_overlap.csv",         "Pairwise Jaccard of top-5 features between domains"),
        ("L2_llm/accuracy.csv",                   "Per-LLM model accuracy"),
        ("L2_llm/coef_importance.csv",            "features × LLMs: LLM-specific model coefficient weights"),
        ("L2_llm/perm_importance.csv",            "features × LLMs: LLM-specific model permutation ΔAcc"),
        ("L2_llm/perm_shift.csv",                 "Δ(L2 perm − L0 perm): positive=more important for this LLM"),
        ("L2_llm/feature_stability.csv",          "Per-feature: how many LLMs it appears in top-5 (stability)"),
        ("L2_llm/jaccard_overlap.csv",            "Pairwise Jaccard of top-5 features between LLMs"),
        ("L3_pair/accuracy.csv",                  "Per-(domain×LLM) pair model accuracy"),
        ("L3_pair/coef_importance.csv",           "features × pairs: pair-specialist coefficient weights"),
        ("L3_pair/coef_shift.csv",                "Δ(L3 coef − L0 coef): where specialist departs from global"),
        ("L3_pair/perm_shift.csv",                "Δ(L3 perm − L0 perm)"),
        ("comparison/cross_level_perm_importance.csv", "features × all conditions (L0+L1+L2): full picture in one table"),
        ("comparison/cross_level_coef_importance.csv", "features × L0 global + all L3 pairs"),
        ("comparison/feature_robustness_ranking.csv",  "Final robustness ranking: best features to use across all conditions"),
    ]
    for fname, desc in file_descriptions:
        print(f"   {fname}")
        print(f"       → {desc}")

    print(f"\n{sep}")
    print("DONE")
    print(sep)


# ============================================================================
# NOTEBOOK USAGE EXAMPLE
# ============================================================================
#
#   import pandas as pd
#   import feature_importance_multi_level as fim
#
#   df = pd.read_csv("models_generations_with_features.csv", encoding="utf-8")
#   df = fim.apply_default_filters(df)
#
#   # Full analysis (all features, all levels)
#   results = fim.run_full_analysis(df, n_runs=5, n_perm_repeats=10,
#                                   save_dir="./feature_importance_results")
#
#   # Access results
#   print(results["l0"]["accuracy"])
#   print(results["robustness_df"].head(10))
#
#   # Run a single level independently
#   feature_cols = fim.get_feature_columns(df)
#   generators   = [g for g in df["generator"].unique() if g != "human"]
#   l3 = fim.run_level3_pairs(df, feature_cols, generators, n_runs=3)
#
#   # Plot a single coef-vs-perm scatter for one domain
#   fim.plot_coef_vs_perm(
#       results["l1"]["news"]["coef_importance"],
#       results["l1"]["news"]["perm_importance"],
#       condition="news")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Level Feature Importance Analysis")
    parser.add_argument("--data",   required=True,
                        help="Path to CSV dataset")
    parser.add_argument("--output", default=SAVE_DIR,
                        help="Output directory (default: ./feature_importance_results)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="CV splits per level (default: 5)")
    parser.add_argument("--n_perm", type=int, default=10,
                        help="Shuffle repeats for permutation importance (default: 10)")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Optional feature subset (default: all features)")
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding="utf-8")
    df = apply_default_filters(df)

    run_full_analysis(df,
                      feature_subset=args.features,
                      n_runs=args.n_runs,
                      n_perm_repeats=args.n_perm,
                      save_dir=args.output)
