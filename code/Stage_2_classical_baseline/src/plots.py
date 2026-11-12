"""Figures for the Stage A report section."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def feature_importance_heatmap(matrix, overlap, path):
    fig, axes = plt.subplots(1, 2, figsize=(17, 4.6),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    im = ax.imshow(matrix.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.iat[i, j]
            if v > 0.04:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if v < 0.55 else "k")
    ax.set_title("Permutation importance (AUROC drop), row-normalised")
    plt.colorbar(im, ax=ax, fraction=0.025)

    ax = axes[1]
    im = ax.imshow(overlap.to_numpy(), cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(overlap))); ax.set_yticks(range(len(overlap)))
    ax.set_xticklabels([a[:4] for a in overlap.columns], rotation=45)
    ax.set_yticklabels(overlap.index, fontsize=8)
    for i in range(len(overlap)):
        for j in range(len(overlap)):
            ax.text(j, i, f"{overlap.iat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="w" if overlap.iat[i, j] < 0.6 else "k")
    ax.set_title("Shared-evidence similarity")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def confusion_grid(confusions, path, title):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (a, cm) in zip(axes, confusions.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        fontsize=12, color="w" if cm[i, j] > cm.max() / 2 else "k")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(a, fontsize=10)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def model_comparison(results, path):
    """Per-label F1 and aggregate metrics across the three baselines."""
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
    ax = axes[0]
    w = 0.8 / len(names)
    x = np.arange(len(C.ARTIFACTS))
    for k, n in enumerate(names):
        vals = [results[n]["per_label"][a]["f1"] for a in C.ARTIFACTS]
        ax.bar(x + k * w, vals, w, label=n)
        for xi, v in zip(x + k * w, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(C.ARTIFACTS, rotation=12)
    ax.set_ylim(0, 1.15); ax.set_ylabel("F1"); ax.legend(fontsize=8)
    ax.set_title("Per-label F1 on the test split")

    ax = axes[1]
    keys = ["macro_f1", "exact_match", "hamming_loss"]
    x = np.arange(len(keys))
    for k, n in enumerate(names):
        vals = [results[n][m] for m in keys]
        ax.bar(x + k * w, vals, w, label=n)
        for xi, v in zip(x + k * w, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(["macro-F1", "exact-match", "Hamming loss"])
    ax.set_ylim(0, 1.15); ax.legend(fontsize=8)
    ax.set_title("Aggregate metrics on the test split")
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def difficulty_plots(by_n, per_combo, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.4))
    ax = axes[0]
    ax.bar(by_n["n_artifacts"], by_n["exact_match"], color="#4878a8")
    for _, r in by_n.iterrows():
        ax.text(r.n_artifacts, r.exact_match + 0.012,
                f"{r.exact_match:.3f}\n(n={int(r.n_samples)})", ha="center", fontsize=8)
    ax.set_xlabel("artifacts simultaneously present"); ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(0, 1.15); ax.set_title("Experiment 4: accuracy vs number of artifacts")

    ax = axes[1]
    ax.bar(range(len(per_combo)), per_combo["exact_match"], color="#5a9367")
    ax.set_xticks(range(len(per_combo)))
    ax.set_xticklabels(per_combo["combo"], rotation=90, fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel("exact-match accuracy")
    ax.set_title("Per-combination accuracy (16 classes)")
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
