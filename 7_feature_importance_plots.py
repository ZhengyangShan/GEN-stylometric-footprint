"""
feature_importance_plots.py
Generates publication-quality feature importance figures for the AI-human text
detection study.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "./feature_importance_results"
SAVE_DIR = "./feature_importance_results/paper_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
CATEGORIES = {
    "lexical":     (["lexical_diversity", "lexical_density", "percent_long_words"], "#4C9BE8"),
    "character":   (["entropy", "percent_vowels", "percent_consonants", "percent_punctuation"], "#E8744C"),
    "structure":   (["num_words", "num_sentences", "avg_sentence_length", "burstiness"], "#4CBF7A"),
    "readability": (["gunning_fog", "linsear_write", "parse_tree_depth"], "#9B74D6"),
}

# Feature order sorted by robustness (n_top5_appearances descending)
FEAT_ORDER = [
    "entropy", "lexical_diversity", "percent_long_words", "num_words",
    "lexical_density", "burstiness", "percent_punctuation", "num_sentences",
    "percent_consonants", "gunning_fog", "avg_sentence_length",
    "percent_vowels", "parse_tree_depth", "linsear_write",
]

FEAT_LABELS = {
    "entropy":             "Entropy",
    "lexical_diversity":   "Lexical Diversity",
    "percent_long_words":  "% Long Words",
    "num_words":           "Word Count",
    "lexical_density":     "Lexical Density",
    "burstiness":          "Burstiness",
    "percent_punctuation": "% Punctuation",
    "num_sentences":       "Sentence Count",
    "percent_consonants":  "% Consonants",
    "gunning_fog":         "Gunning Fog",
    "avg_sentence_length": "Avg Sent. Length",
    "percent_vowels":      "% Vowels",
    "parse_tree_depth":    "Parse Tree Depth",
    "linsear_write":       "Linsear Write",
}

DOMAIN_LABELS = {
    "arxiv":            "ArXiv",
    "reddit":           "Reddit",
    "story_generation": "Story Gen",
    "wikihow":          "WikiHow",
    "wikipedia":        "Wikipedia",
}

LLM_LABELS = {
    "gemma3-12B_text":  "Gemma-12B",
    "gemma3-27B_text":  "Gemma-27B",
    "gpt-oss20B_text":  "GPT-OSS-20B",
    "gpt-oss120B_text":  "GPT-OSS-120B",
    "llama31-8B_text":  "LLaMA-8B",
    "llama33-70B_text": "LLaMA-70B",
    "qwen-72B_text":    "Qwen-72B",
    "qwen-7B_text":     "Qwen-7B",
}

# Lookup: feature -> category color
def feat_color(feat):
    for cat, (feats, color) in CATEGORIES.items():
        if feat in feats:
            return color
    return "#888888"

# Category legend patches
def category_legend_patches():
    cat_names = {
        "lexical":     "Lexical",
        "character":   "Character",
        "structure":   "Length / Structure",
        "readability": "Readability",
    }
    return [
        mpatches.Patch(color=color, label=cat_names[cat])
        for cat, (_, color) in CATEGORIES.items()
    ]

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def save_fig(fig, filename):
    fig.savefig(os.path.join(SAVE_DIR, filename), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")


# ---------------------------------------------------------------------------
# Figure 1 — Global feature importance (coefficient vs. permutation)
# ---------------------------------------------------------------------------
def fig1_global_feature_importance():
    """Two side-by-side horizontal bar charts: coef importance and perm importance,
    sorted by robustness score (n_top5_appearances), colored by feature category."""

    coef_df = pd.read_csv(
        os.path.join(BASE, "L0_global", "coef_importance.csv"), encoding="utf-8-sig"
    ).rename(columns={"Unnamed: 0": "feature"})
    perm_df = pd.read_csv(
        os.path.join(BASE, "L0_global", "perm_importance.csv"), encoding="utf-8-sig"
    ).rename(columns={"Unnamed: 0": "feature"})

    # Index by feature, reorder by FEAT_ORDER
    coef = coef_df.set_index("feature")["importance"].reindex(FEAT_ORDER)
    perm = perm_df.set_index("feature")["importance"].reindex(FEAT_ORDER)

    y = np.arange(len(FEAT_ORDER))
    labels = [FEAT_LABELS[f] for f in FEAT_ORDER]
    colors = [feat_color(f) for f in FEAT_ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    #fig.suptitle(
        #"Global Feature Importance  —  Global model acc = 92.5%",
        #fontsize=11, fontweight="bold", y=1.01,
    #)

    # Left: coefficient importance
    ax1.barh(y, coef.values, color=colors[1], edgecolor="white", linewidth=0.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=12)
    ax1.invert_yaxis()
    ax1.set_xlabel("Coefficient Importance (|coef|)", fontsize=10)
    #ax1.set_title("(a) Coefficient Importance", fontsize=10)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    coef_max = coef.values.max()
    for yi, val in zip(y, coef.values):
        ax1.text(val + coef_max * 0.02, yi, f"{val:.2f}",
                 va="center", ha="left", fontsize=8)
    ax1.set_xlim(right=coef_max * 1.20)

    # Right: permutation importance
    ax2.barh(y, perm.values, color=colors[1], edgecolor="white", linewidth=0.4)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xlabel("Permutation Importance (ΔAcc)", fontsize=10)
    #ax2.set_title("(b) Permutation Importance", fontsize=10)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    perm_max = perm.values.max()
    for yi, val in zip(y, perm.values):
        ax2.text(val + perm_max * 0.02, yi, f"{val:.3f}",
                 va="center", ha="left", fontsize=8)
    ax2.set_xlim(right=perm_max * 1.22)

    # Shared legend
    #fig.legend(
        #handles=category_legend_patches(),
        #title="Feature Category",
        #loc="lower center",
        #ncol=4,
        #bbox_to_anchor=(0.5, -0.06),
        #frameon=False,
        #fontsize=9,
    #)

    fig.tight_layout()
    save_fig(fig, "fig1_global_feature_importance.pdf")



# ---------------------------------------------------------------------------
# Figure 2 — Feature robustness scatter
# ---------------------------------------------------------------------------
def fig2_feature_robustness():
    """Scatter plot: mean perm importance (x) vs. inverted CV (y = 1/CV so
    stable features appear higher), bubble size = n_top5_appearances.
    Quadrants divided at medians with background shading and labels."""

    rob_df = pd.read_csv(
        os.path.join(BASE, "comparison", "feature_robustness_ranking.csv"),
        encoding="utf-8-sig",
    )
    rob_df = rob_df.set_index("feature")

    feats = FEAT_ORDER
    x_vals = rob_df.loc[feats, "mean_perm_all"].values
    cv_vals = rob_df.loc[feats, "cv_perm"].values
    n_top5 = rob_df.loc[feats, "n_top5_appearances"].values

    # Invert CV so higher y = more stable (lower CV)
    y_vals = 1.0 / cv_vals
    colors = [feat_color(f) for f in feats]
    bubble_sizes = (n_top5 / n_top5.max()) * 800 + 30

    med_x = np.median(x_vals)
    med_y = np.median(y_vals)

    # Manual label offsets (dx, dy) — spread crowded bottom-left cluster
    label_offsets = {
        "entropy":             ( 0.003,  0.04),
        "lexical_diversity":   ( -0.06,  0.05),
        "percent_long_words":  ( -0.01, 0.09),
        "num_words":           ( 0.002, -0.05),
        "lexical_density":     ( 0.01,  0.06),
        # bottom-left cluster — fan outward
        "burstiness":          ( 0.01,  0.02),
        "num_sentences":       ( -0.004,  0.04),
        "gunning_fog":         ( -0.03, 0.15),
        "percent_punctuation": ( -0.01, 0.07),
        "percent_consonants":  (-0.01, -0.08),
        "avg_sentence_length": (-0.03, 0.07),
        "percent_vowels":      (-0.01,  0.07),
        "parse_tree_depth":    (-0.03,  -0.08),
        "linsear_write":       (-0.004, -0.05),
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    # Quadrant shading
    #x_min, x_max = -0.03, x_vals.max() * 1.25
    x_min, x_max = -0.03, 0.17
    #y_min, y_max = y_vals.min() * 0.7, y_vals.max() * 1.25
    y_min, y_max = y_vals.min() * 0.7, 1.6

    # Top-right: high importance, high stability (green tint)
    ax.axhspan(med_y, y_max, xmin=(med_x - x_min) / (x_max - x_min),
               alpha=0.08, color="green", zorder=0)

    # Quadrant dashed lines
    ax.axvline(med_x, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(med_y, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)

    # Quadrant labels
    ax.text(x_max * 0.99, y_max * 0.98, "Core Features",
            ha="right", va="top", fontsize=10, color="darkgreen", style="italic")
    ax.text(x_min + 0.0005, y_max * 0.98, "Stable but\nLow Importance",
            ha="left", va="top", fontsize=10, color="gray", style="italic")
    ax.text(x_max * 0.99, y_min * 1, "Domain-specific",
            ha="right", va="bottom", fontsize=10, color="#888", style="italic")
    ax.text(x_min + 0.0005, y_min * 1, "Unstable &\nLow Importance",
            ha="left", va="bottom", fontsize=10, color="#888", style="italic")

    # Bubbles
    sc = ax.scatter(x_vals, y_vals, s=bubble_sizes, c=colors[1],
                    edgecolors="white", linewidths=0.6, zorder=3, alpha=0.88)

    # Labels
    for i, feat in enumerate(feats):
        dx, dy = label_offsets.get(feat, (0.1, 0.3))
        ax.annotate(
            FEAT_LABELS[feat],
            xy=(x_vals[i], y_vals[i]),
            xytext=(x_vals[i] + dx, y_vals[i] + dy),
            fontsize=8,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax.set_xlabel("Mean Permutation Importance", fontsize=12)
    ax.set_ylabel("Feature Stability Score", fontsize=12)
    #ax.set_title(
        #"Feature Robustness: Importance vs. Stability\n"
        #"(bubble size = # times in top-5 across 48 conditions)",
        #fontsize=10, fontweight="bold",
    #)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Legend: categories + bubble size guide — placed outside plot to the right
    #cat_patches = category_legend_patches()
    cat_patches = []
    for n_label in [5, 20, 35]:
        s = (n_label / n_top5.max()) * 800 + 30   # same formula as plot bubbles
        cat_patches.append(
            plt.scatter([], [], s=s, color=colors[1], alpha=0.5,
                        label=f"n={n_label}")
        )
    ax.legend(handles=cat_patches, title="Top-5 count",
              loc="upper left", fontsize=10, frameon=True, framealpha=0.9,
              edgecolor="#ccc",
              bbox_to_anchor=(1.01, 1.0), bbox_transform=ax.transAxes)
    fig.tight_layout()
    save_fig(fig, "fig2_feature_robustness.pdf")


# ---------------------------------------------------------------------------
# Figure 2b — Feature robustness scatter (L0 + L1 + L2 only, no pairwise)
# ---------------------------------------------------------------------------
def fig2b_feature_robustness_L0L1L2():
    """Same scatter as fig2 but computed over only 13 conditions:
    1 Global + 5 Domain-specific + 7 LLM-specific (pairwise excluded)."""

    rob_df = pd.read_csv(
        os.path.join(BASE, "comparison", "feature_robustness_ranking_L0L1L2.csv"),
        encoding="utf-8-sig",
    )
    rob_df = rob_df.set_index("feature")

    n_cond = int(rob_df["n_conditions"].iloc[0])

    feats = FEAT_ORDER
    x_vals = rob_df.loc[feats, "mean_perm_all"].values
    cv_vals = rob_df.loc[feats, "cv_perm"].values
    n_top5 = rob_df.loc[feats, "n_top5_appearances"].values

    y_vals = 1.0 / cv_vals
    colors = [feat_color(f) for f in feats]
    bubble_sizes = (n_top5 / n_top5.max()) * 800 + 30

    med_x = np.median(x_vals)
    med_y = np.median(y_vals)

    label_offsets = {
        "entropy":             ( 0.003,  0.04),
        "lexical_diversity":   ( -0.01,  0.05),
        "percent_long_words":  ( -0.01, 0.09),
        "num_words":           ( 0.002, -0.05),
        "lexical_density":     ( 0.01,  0.03),
        # bottom-left cluster — fan outward
        "burstiness":          ( -0.01,  0.2),
        "num_sentences":       ( -0.004,  0.04),
        "gunning_fog":         ( -0.02, 0.1),
        "percent_punctuation": ( 0.01, 0.07),
        "percent_consonants":  (-0.01, -0.1),
        "avg_sentence_length": (-0.03, 0.07),
        "percent_vowels":      (-0.01,  -0.07),
        "parse_tree_depth":    (-0.03,  -0.08),
        "linsear_write":       (-0.004, -0.05),
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    #x_min, x_max = -0.005, x_vals.max() * 1.25
    #y_min, y_max = y_vals.min() * 0.7, y_vals.max() * 1.25
    x_min, x_max = -0.03, 0.2
    y_min, y_max = y_vals.min() * 0.7, 2.0

    # Top-right: high importance, high stability (green tint)
    ax.axhspan(med_y, y_max, xmin=(med_x - x_min) / (x_max - x_min),
               alpha=0.08, color="green", zorder=0)

    # Quadrant dashed lines
    ax.axvline(med_x, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(med_y, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)

    # Quadrant labels
    ax.text(x_max * 0.99, y_max * 0.98, "Core Features",
            ha="right", va="top", fontsize=10, color="darkgreen", style="italic")
    ax.text(x_min + 0.0005, y_max * 0.98, "Stable but\nLow Importance",
            ha="left", va="top", fontsize=10, color="gray", style="italic")
    ax.text(x_max * 0.99, y_min * 1, "Domain-specific",
            ha="right", va="bottom", fontsize=10, color="#888", style="italic")
    ax.text(x_min + 0.0005, y_min * 1, "Unstable &\nLow Importance",
            ha="left", va="bottom", fontsize=10, color="#888", style="italic")
    ax.scatter(x_vals, y_vals, s=bubble_sizes, c=colors[1],
               edgecolors="white", linewidths=0.6, zorder=3, alpha=0.88)

    # Labels
    for i, feat in enumerate(feats):
        dx, dy = label_offsets.get(feat, (0.1, 0.3))
        ax.annotate(
            FEAT_LABELS[feat],
            xy=(x_vals[i], y_vals[i]),
            xytext=(x_vals[i] + dx, y_vals[i] + dy),
            fontsize=8,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax.set_xlabel("Mean Permutation Importance", fontsize=12)
    ax.set_ylabel("Feature Stability Score", fontsize=12)
    #ax.set_title(
        #"Feature Robustness: Importance vs. Stability\n"
        #"(bubble size = # times in top-5 across 48 conditions)",
        #fontsize=10, fontweight="bold",
    #)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Legend: categories + bubble size guide — placed outside plot to the right
    #cat_patches = category_legend_patches()
    cat_patches = []
    for n_label in [2, 5, 10]:
        s = (n_label / n_top5.max()) * 800 + 30   # same formula as plot bubbles
        cat_patches.append(
            plt.scatter([], [], s=s, color=colors[1], alpha=0.5,
                        label=f"n={n_label}")
        )
    ax.legend(handles=cat_patches, title="Top-5 count",
              loc="upper left", fontsize=10, frameon=True, framealpha=0.9,
              edgecolor="#ccc",
              bbox_to_anchor=(1.01, 1.0), bbox_transform=ax.transAxes)
    

    fig.tight_layout()
    save_fig(fig, "fig2b_feature_robustness_L0L1L2.pdf")



# ---------------------------------------------------------------------------
# Figure 3 — Domain heatmap (L1)
# ---------------------------------------------------------------------------
def fig3_domain_heatmap():
    """Heatmap of L1 per-domain permutation importance.
    Rows: 14 features (robustness order). Columns: 5 domains.
    Cells annotated with value; top-3 per column marked with *."""

    perm_df = pd.read_csv(
        os.path.join(BASE, "L1_domain", "perm_importance.csv"), encoding="utf-8-sig"
    ).rename(columns={"Unnamed: 0": "feature"}).set_index("feature")

    acc_df = pd.read_csv(
        os.path.join(BASE, "L1_domain", "accuracy.csv"), encoding="utf-8-sig"
    )
    acc_map = dict(zip(acc_df["condition"], acc_df["accuracy"]))

    domains = ["arxiv", "reddit", "story_generation", "wikihow", "wikipedia"]
    col_labels = [DOMAIN_LABELS[d] for d in domains]
    acc_labels = [f"Acc: {acc_map[d]*100:.1f}%" for d in domains]

    mat = perm_df.reindex(FEAT_ORDER)[domains].values  # (14, 5)

    # Mark top-3 per column with *
    star_mask = np.zeros_like(mat, dtype=bool)
    for j in range(mat.shape[1]):
        top3_idx = np.argsort(mat[:, j])[-3:]
        star_mask[top3_idx, j] = True

    # Diverging colormap centred at 0
    vmax = np.abs(mat).max()
    norm = TwoSlopeNorm(vmin=-vmax * 0.4, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", norm=norm)

    # Axes ticks
    row_labels = [FEAT_LABELS[f] for f in FEAT_ORDER]
    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(FEAT_ORDER)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Cell annotations
    for i in range(len(FEAT_ORDER)):
        for j in range(len(domains)):
            val = mat[i, j]
            star = "*" if star_mask[i, j] else ""
            text_color = "black" if abs(val) < vmax * 0.55 else "white"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold" if star else "normal")

    # Move x-axis ticks to top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0)

    # Accuracy annotations: placed just below the last row in data coordinates
    for j, acc_lbl in enumerate(acc_labels):
        ax.text(j, len(FEAT_ORDER) - 0.4, acc_lbl,
                ha="center", va="top", fontsize=8, color="#444",
                clip_on=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Permutation Importance (ΔAcc)", fontsize=9)

    ax.set_title(
        "L1 Per-Domain Feature Importance\n"
        "(*  = top-3 per domain;  ArXiv: num_words spike, entropy near zero)",
        fontsize=10, fontweight="bold", pad=18,
    )

    fig.tight_layout()
    save_fig(fig, "fig3_domain_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 4 — Generator heatmap (L2)
# ---------------------------------------------------------------------------
def fig4_generator_heatmap():
    """Heatmap of L2 per-generator permutation importance.
    Same style as Figure 3; highlights GPT-OSS 20B as outlier."""

    perm_df = pd.read_csv(
        os.path.join(BASE, "L2_llm", "perm_importance.csv"), encoding="utf-8-sig"
    ).rename(columns={"Unnamed: 0": "feature"}).set_index("feature")

    acc_df = pd.read_csv(
        os.path.join(BASE, "L2_llm", "accuracy.csv"), encoding="utf-8-sig"
    )
    acc_map = dict(zip(acc_df["condition"], acc_df["accuracy"]))

    generators = [
        "gemma3-12B_text", "gemma3-27B_text", "gpt-oss20B_text","gpt-oss120B_text",
        "llama31-8B_text", "llama33-70B_text", "qwen-72B_text", "qwen-7B_text",
    ]
    col_labels = [LLM_LABELS[g] for g in generators]
    acc_labels = [f"Acc: {acc_map[g]*100:.1f}%" for g in generators]

    mat = perm_df.reindex(FEAT_ORDER)[generators].values  # (14, 7)

    # Mark top-3 per column with *
    star_mask = np.zeros_like(mat, dtype=bool)
    for j in range(mat.shape[1]):
        top3_idx = np.argsort(mat[:, j])[-3:]
        star_mask[top3_idx, j] = True

    vmax = np.abs(mat).max()
    norm = TwoSlopeNorm(vmin=-vmax * 0.4, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", norm=norm)

    row_labels = [FEAT_LABELS[f] for f in FEAT_ORDER]
    ax.set_xticks(np.arange(len(generators)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(len(FEAT_ORDER)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Cell annotations
    for i in range(len(FEAT_ORDER)):
        for j in range(len(generators)):
            val = mat[i, j]
            star = "*" if star_mask[i, j] else ""
            text_color = "black" if abs(val) < vmax * 0.55 else "white"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold" if star else "normal")

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0)

    # Accuracy annotations: placed just below the last row in data coordinates
    for j, acc_lbl in enumerate(acc_labels):
        ax.text(j, len(FEAT_ORDER) - 0.4, acc_lbl,
                ha="center", va="top", fontsize=8, color="#444",
                clip_on=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Permutation Importance (ΔAcc)", fontsize=9)

    ax.set_title(
        "L2 Per-Generator Feature Importance\n"
        "(*  = top-3 per generator;  GPT-OSS 20B: entropy lower, lexical_diversity dominant)",
        fontsize=10, fontweight="bold", pad=18,
    )

    # Highlight GPT-OSS 20B column with a box
    gpt_idx = generators.index("gpt-oss20B_text")
    rect_col = mpatches.FancyBboxPatch(
        (gpt_idx - 0.5, -0.5), 1.0, len(FEAT_ORDER),
        boxstyle="square,pad=0", linewidth=2,
        edgecolor="navy", facecolor="none", zorder=6,
    )
    ax.add_patch(rect_col)
    ax.text(gpt_idx, -0.85, "Outlier", ha="center", va="bottom",
            fontsize=8, color="navy", fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "fig4_generator_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 3+4 — Domain + Generator heatmaps side by side (combined)
# ---------------------------------------------------------------------------
def fig3_4_combined_heatmap():
    """Side-by-side heatmaps: left = L1 per-domain, right = L2 per-generator.
    Shared y-axis (feature labels on left only), shared colorbar, no titles."""

    # ── Load domain data (fig3) ──────────────────────────────────────────────
    perm_dom = pd.read_csv(
        os.path.join(BASE, "L1_domain", "perm_importance.csv"), encoding="utf-8-sig"
    ).rename(columns={"Unnamed: 0": "feature"}).set_index("feature")

    acc_dom = pd.read_csv(os.path.join(BASE, "L1_domain", "accuracy.csv"), encoding="utf-8-sig")
    acc_map_dom = dict(zip(acc_dom["condition"], acc_dom["accuracy"]))

    domains    = ["arxiv", "reddit", "story_generation", "wikihow", "wikipedia"]
    mat_dom    = perm_dom.reindex(FEAT_ORDER)[domains].values          # (14, 5)
    col_dom    = [DOMAIN_LABELS[d] for d in domains]
    acc_dom_lbl = [f"Acc:{acc_map_dom[d]*100:.1f}" for d in domains]

    # ── Load generator data (fig4) ───────────────────────────────────────────
    perm_gen = pd.read_csv(
        os.path.join(BASE, "L2_llm", "perm_importance.csv"), encoding="utf-8-sig"
    ).rename(columns={"Unnamed: 0": "feature"}).set_index("feature")

    acc_gen = pd.read_csv(os.path.join(BASE, "L2_llm", "accuracy.csv"), encoding="utf-8-sig")
    acc_map_gen = dict(zip(acc_gen["condition"], acc_gen["accuracy"]))

    generators = [
        "gemma3-12B_text", "gemma3-27B_text", "gpt-oss20B_text", "gpt-oss120B_text",
        "llama31-8B_text", "llama33-70B_text", "qwen-72B_text", "qwen-7B_text",
    ]
    mat_gen    = perm_gen.reindex(FEAT_ORDER)[generators].values        # (14, 7)
    col_gen    = [LLM_LABELS[g] for g in generators]
    acc_gen_lbl = [f"Acc:{acc_map_gen[g]*100:.1f}" for g in generators]

    # ── Shared colour norm (both matrices together) ──────────────────────────
    vmax = max(np.abs(mat_dom).max(), np.abs(mat_gen).max())
    norm = TwoSlopeNorm(vmin=-vmax * 0.4, vcenter=0, vmax=vmax)

    # ── Top-3 star masks ─────────────────────────────────────────────────────
    def top3_mask(mat):
        mask = np.zeros_like(mat, dtype=bool)
        for j in range(mat.shape[1]):
            mask[np.argsort(mat[:, j])[-3:], j] = True
        return mask

    star_dom = top3_mask(mat_dom)
    star_gen = top3_mask(mat_gen)

    # ── Layout: width proportional to number of columns ─────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(18, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [len(domains)-0.5, len(generators)+1], "wspace": 0.01},
    )
    row_labels = [FEAT_LABELS[f] for f in FEAT_ORDER]
    y_pos = np.arange(len(FEAT_ORDER))

    # ── Panel (a) — Domain ───────────────────────────────────────────────────
    im1 = ax1.imshow(mat_dom, aspect="auto", cmap="RdYlGn", norm=norm)
    ax1.set_xticks(np.arange(len(domains)))
    ax1.set_xticklabels(col_dom, fontsize=10)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(row_labels, fontsize=12)
    ax1.xaxis.set_ticks_position("top")
    ax1.xaxis.set_label_position("top")
    ax1.tick_params(axis="x", length=0)
    ax1.text(0.5, -0.04, "(a) Per Domain", transform=ax1.transAxes,
             ha="center", va="top", fontsize=11, fontweight="bold")

    for i in range(len(FEAT_ORDER)):
        for j in range(len(domains)):
            val = mat_dom[i, j]
            if round(val, 2) == -0.00:
                val = 0.00
            star = "*" if star_dom[i, j] else ""
            tc = "black" if abs(val) < vmax * 0.55 else "white"
            ax1.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                     fontsize=12, color=tc,
                     fontweight="bold" if star else "normal")

    for j, lbl in enumerate(acc_dom_lbl):
        ax1.text(j, len(FEAT_ORDER) - 0.4, lbl,
                 ha="center", va="top", fontsize=12, color="#444", clip_on=False)

    # ── Panel (b) — Generator ────────────────────────────────────────────────
    im2 = ax2.imshow(mat_gen, aspect="auto", cmap="RdYlGn", norm=norm)
    ax2.set_xticks(np.arange(len(generators)))
    ax2.set_xticklabels(col_gen, fontsize=10)
    ax2.xaxis.set_ticks_position("top")
    ax2.xaxis.set_label_position("top")
    ax2.tick_params(axis="x", length=0)
    ax2.tick_params(axis="y", length=0)
    ax2.text(0.5, -0.04, "(b) Per LLM", transform=ax2.transAxes,
             ha="center", va="top", fontsize=11, fontweight="bold")

    for i in range(len(FEAT_ORDER)):
        for j in range(len(generators)):
            val = mat_gen[i, j]
            if round(val, 2) == -0.00:
                val = 0.00
            star = "*" if star_gen[i, j] else ""
            tc = "black" if abs(val) < vmax * 0.55 else "white"
            ax2.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                     fontsize=12, color=tc,
                     fontweight="bold" if star else "normal")

    for j, lbl in enumerate(acc_gen_lbl):
        ax2.text(j, len(FEAT_ORDER) - 0.4, lbl,
                 ha="center", va="top", fontsize=12, color="#444", clip_on=False)

    # GPT-OSS outlier box
    #gpt_idx = generators.index("gpt-oss20B_text")
    #ax2.add_patch(mpatches.FancyBboxPatch(
        #(gpt_idx - 0.5, -0.5), 1.0, len(FEAT_ORDER),
        #boxstyle="square,pad=0", linewidth=2,
        #edgecolor="navy", facecolor="none", zorder=6,
    #))
    #ax2.text(gpt_idx, -0.85, "Outlier", ha="center", va="bottom",
             #fontsize=8, color="navy", fontweight="bold")

    # ── Shared colorbar ──────────────────────────────────────────────────────
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.75, pad=0.01)
    cbar.set_label("Permutation Importance (ΔAcc)", fontsize=11)

    fig.tight_layout()
    save_fig(fig, "fig3_4_combined_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 5 — Feature rank stability heatmap (all 14 features × 13 conditions)
# ---------------------------------------------------------------------------
def fig5_feature_rank_stability():
    """Rank heatmap: rows = all 14 features (global robustness order),
    columns = 13 conditions (Global | 5 domains | 7 generators).
    Cell color = permutation importance rank within that condition
    (rank 1 = most important = dark green; rank 14 = least = dark red).
    Stable features appear as uniform-color rows; unstable ones show variation.
    Cell text shows the rank number."""

    l0_perm = pd.read_csv(
        os.path.join(BASE, "L0_global", "perm_importance.csv"), encoding="utf-8-sig"
    ).rename(columns={"Unnamed: 0": "feature"}).set_index("feature")["importance"]

    l1_perm = pd.read_csv(
        os.path.join(BASE, "L1_domain", "perm_importance.csv"), encoding="utf-8-sig"
    ).rename(columns={"Unnamed: 0": "feature"}).set_index("feature")

    l2_perm = pd.read_csv(
        os.path.join(BASE, "L2_llm", "perm_importance.csv"), encoding="utf-8-sig"
    ).rename(columns={"Unnamed: 0": "feature"}).set_index("feature")

    domains    = ["arxiv", "reddit", "story_generation", "wikihow", "wikipedia"]
    generators = ["gemma3-12B_text", "gemma3-27B_text", "gpt-oss20B_text", "gpt-oss20B_text",
                  "llama31-8B_text", "llama33-70B_text", "qwen-72B_text", "qwen-7B_text"]

    # Build importance matrix: features × conditions
    cond_data = {"Global": l0_perm.reindex(FEAT_ORDER)}
    for d in domains:
        cond_data[d] = l1_perm[d].reindex(FEAT_ORDER)
    for g in generators:
        cond_data[g] = l2_perm[g].reindex(FEAT_ORDER)

    imp_df  = pd.DataFrame(cond_data, index=FEAT_ORDER)          # (14, 13)
    rank_df = imp_df.rank(axis=0, ascending=False, method="min").astype(int)  # rank 1 = best

    # Column labels
    col_labels = (
        ["Global"]
        + [DOMAIN_LABELS[d] for d in domains]
        + [LLM_LABELS[g].replace("\n", " ") for g in generators]
    )
    all_cols = ["Global"] + domains + generators

    mat = rank_df[all_cols].values  # (14, 13)

    # Colormap: rank 1 (best) → green, rank 14 (worst) → red
    cmap = plt.cm.RdYlGn_r   # reversed so low rank number = green
    norm = plt.Normalize(vmin=1, vmax=len(FEAT_ORDER))

    n_feats, n_conds = mat.shape
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    # Axes ticks
    ax.set_xticks(np.arange(n_conds))
    ax.set_xticklabels(col_labels, fontsize=8.5, rotation=30, ha="left")
    ax.set_yticks(np.arange(n_feats))
    ax.set_yticklabels([FEAT_LABELS[f] for f in FEAT_ORDER], fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0)

    # Cell annotations: rank number
    for i in range(n_feats):
        for j in range(n_conds):
            rank = mat[i, j]
            text_color = "white" if rank <= 2 or rank >= 12 else "black"
            ax.text(j, i, str(rank), ha="center", va="center",
                    fontsize=7.5, color=text_color)

    # Vertical separators between condition groups
    ax.axvline(0.5,  color="white", lw=2.5)   # Global | Domain
    ax.axvline(5.5,  color="white", lw=2.5)   # Domain | Generator

    # Group labels below the column headers (in axes fraction)
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0,   1.08, "Global",       transform=trans, ha="center",
            fontsize=9, fontweight="bold", color="#333")
    ax.text(3,   1.08, "Per Domain",   transform=trans, ha="center",
            fontsize=9, fontweight="bold", color="#333")
    ax.text(9,   1.08, "Per Generator", transform=trans, ha="center",
            fontsize=9, fontweight="bold", color="#333")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Importance Rank  (1 = most important)", fontsize=9)
    cbar.set_ticks([1, 4, 7, 10, 14])

    ax.set_title(
        "Feature Rank Stability Across All Conditions\n"
        "Uniform rows = stable features; varied rows = condition-specific signals",
        fontsize=10, fontweight="bold", pad=36,
    )

    fig.tight_layout()
    save_fig(fig, "fig5_feature_rank_stability.pdf")


# ---------------------------------------------------------------------------
# Figure 6 — Accuracy by condition
# Color = domain, marker shape = LLM, applied consistently across L1/L2/L3
# ---------------------------------------------------------------------------
def fig6_accuracy_by_condition():
    """Strip plot: accuracy across L0/L1/L2/L3.
    Color encodes domain; marker shape encodes LLM.
    L1 points: color only (no LLM dimension).
    L2 points: shape only (no domain dimension), neutral gray fill.
    L3 points: color + shape (domain × LLM pair), jittered.
    No level-wide error bars — individual points tell the story."""

    acc_df = pd.read_csv(
        os.path.join(BASE, "accuracy_all_levels.csv"), encoding="utf-8-sig"
    )

    l0_row  = acc_df[acc_df["level"] == "L0_global"]
    l1_rows = acc_df[acc_df["level"] == "L1_domain"]
    l2_rows = acc_df[acc_df["level"] == "L2_llm"]
    l3_rows = acc_df[acc_df["level"] == "L3_pair"]

    domain_colors = {
        "arxiv":            "#4C9BE8",
        "reddit":           "#E8744C",
        "story_generation": "#4CBF7A",
        "wikihow":          "#F5A623",
        "wikipedia":        "#9B74D6",
    }
    generators = [
        "gemma3-12B_text", "gemma3-27B_text", "gpt-oss20B_text",
        "llama31-8B_text", "llama33-70B_text", "qwen-72B_text", "qwen-7B_text",
    ]
    llm_markers = {g: m for g, m in zip(generators,
                   ["o", "s", "^", "D", "v", "P", "*"])}
    llm_short = {
        "gemma3-12B_text":  "Gemma 12B",
        "gemma3-27B_text":  "Gemma 27B",
        "gpt-oss20B_text":  "GPT-OSS 20B",
        "gpt-oss120B_text":  "GPT-OSS 120B",
        "llama31-8B_text":  "LLaMA 8B",
        "llama33-70B_text": "LLaMA 70B",
        "qwen-72B_text":    "Qwen 72B",
        "qwen-7B_text":     "Qwen 7B",
    }

    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Global baseline
    l0_acc = l0_row["accuracy"].values[0]
    ax.axhline(l0_acc, color="gray", linestyle="--", lw=1.2, alpha=0.7, zorder=2)
    ax.scatter([0], [l0_acc], s=130, color="#333", marker="D", zorder=6,
               label=f"_nolegend_")
    ax.annotate(f"{l0_acc:.3f}", xy=(0, l0_acc),
                xytext=(0.08, l0_acc + 0.0015), fontsize=8, color="#444")

    # L1 — per domain: color by domain, fixed circle marker
    for _, row in l1_rows.iterrows():
        dom   = row["condition"]
        color = domain_colors.get(dom, "#888")
        ax.scatter([1], [row["accuracy"]], s=100, color=color,
                   marker="o", edgecolors="white", linewidths=0.5,
                   zorder=5, alpha=0.95)
        ax.annotate(DOMAIN_LABELS.get(dom, dom),
                    xy=(1, row["accuracy"]), xytext=(1.07, row["accuracy"]),
                    fontsize=7.5, va="center", color="#333")

    # L2 — per LLM: shape by LLM, neutral fill
    for _, row in l2_rows.iterrows():
        gen    = row["condition"]
        marker = llm_markers.get(gen, "o")
        ax.scatter([2], [row["accuracy"]], s=110, color="#888",
                   marker=marker, edgecolors="white", linewidths=0.5,
                   zorder=5, alpha=0.92)
        ax.annotate(llm_short.get(gen, gen),
                    xy=(2, row["accuracy"]), xytext=(2.07, row["accuracy"]),
                    fontsize=7, va="center", color="#333")

    # L3 — per pair: color = domain, shape = LLM, jittered
    for _, row in l3_rows.iterrows():
        parts  = row["condition"].split("|")
        dom, gen = parts[0], parts[1]
        color  = domain_colors.get(dom, "#888")
        marker = llm_markers.get(gen, "o")
        jitter = np.random.uniform(-0.15, 0.15)
        ax.scatter([3 + jitter], [row["accuracy"]], s=55, color=color,
                   marker=marker, edgecolors="white", linewidths=0.4,
                   zorder=4, alpha=0.80)

    # Mean line per level (horizontal tick, no error bar)
    for pos, rows in [(0, l0_row), (1, l1_rows), (2, l2_rows), (3, l3_rows)]:
        m = rows["accuracy"].mean()
        ax.plot([pos - 0.18, pos + 0.18], [m, m],
                color="black", lw=2.0, zorder=7, solid_capstyle="round")

    # ── Legend ────────────────────────────────────────────────────────────
    # Domain color patches
    dom_handles = [
        mpatches.Patch(color=c, label=DOMAIN_LABELS[d])
        for d, c in domain_colors.items()
    ]
    # LLM shape handles (gray fill to match L2 style)
    llm_handles = [
        plt.scatter([], [], s=70, color="#888", marker=llm_markers[g],
                    edgecolors="white", linewidths=0.5, label=llm_short[g])
        for g in generators
    ]
    # Baseline line handle
    baseline_handle = plt.Line2D([0], [0], color="gray", linestyle="--",
                                 lw=1.2, label=f"Global baseline ({l0_acc:.3f})")
    mean_handle = plt.Line2D([0], [0], color="black", lw=2,
                             label="Level mean")

    # Single legend: domain section header + patches, then LLM section header + shapes
    from matplotlib.lines import Line2D
    spacer = Line2D([], [], linestyle="none", label="")
    dom_title  = Line2D([], [], linestyle="none", label="─── Domain (color) ───")
    llm_title  = Line2D([], [], linestyle="none", label="─── LLM (shape) ───")

    all_handles = (
        [dom_title] + dom_handles
        + [spacer, llm_title] + llm_handles
        + [spacer, baseline_handle, mean_handle]
    )
    ax.legend(handles=all_handles,
              loc="lower right", fontsize=7.5, frameon=True,
              framealpha=0.9, edgecolor="#ccc",
              bbox_to_anchor=(1.0, 0.0), bbox_transform=ax.transAxes,
              handlelength=1.2, handletextpad=0.5)

    # Axes
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        ["Global\n(n=all)", "Per Domain\n(n=5)",
         "Per LLM\n(n=8)", "Per Pair\n(n=40)"],
        fontsize=9,
    )
    ax.set_xlim(-0.45, 4.2)
    ax.set_ylim(0.75, 1.015)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_ylabel("Accuracy", fontsize=9)
    #ax.set_title(
        #"Detection Accuracy Across Model Levels and Conditions\n"
        #"Color = domain  ·  Shape = LLM  ·  Bar = level mean",
        #fontsize=10, fontweight="bold",
    #)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    save_fig(fig, "fig6_accuracy_by_condition.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating publication-quality figures...")
    print(f"Output directory: {SAVE_DIR}\n")

    fig1_global_feature_importance()
    fig2_feature_robustness()
    fig3_4_combined_heatmap()
    fig5_feature_rank_stability()
    fig6_accuracy_by_condition()

    print("\nAll figures saved successfully.")
    print(f"  fig1_global_feature_importance.pdf  — RQ1: global feature importance (coef + perm)")
    print(f"  fig2_feature_robustness.pdf         — RQ1: importance vs. stability scatter")
    print(f"  fig3_4_combined_heatmap.pdf         — RQ2: per-domain + per-generator heatmaps (combined)")
    print(f"  fig5_feature_rank_stability.pdf     — RQ2: rank trajectory across all conditions")
    print(f"  fig6_accuracy_by_condition.pdf      — accuracy overview across all levels")
