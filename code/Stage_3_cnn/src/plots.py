"""Figures for the Stage B report section."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def training_curves(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    for name, r in runs.items():
        h = r["train"]["history"]
        axes[0].plot([e["epoch"] for e in h], [e["train_loss"] for e in h], label=name)
        axes[1].plot([e["epoch"] for e in h], [e["val_macro_f1"] for e in h], label=name)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss"); axes[0].set_title("Training loss")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val macro-F1")
    axes[1].set_title("Validation macro-F1")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def arm_comparison(rows, baseline, path):
    """rows: list of dicts with name / macro_f1 / exact_match / regime."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    colours = ["#4878a8" if r["regime"] == "fixed" else "#c44e52" for r in rows]
    for ax, key, title in [(axes[0], "macro_f1", "macro-F1 (test)"),
                           (axes[1], "exact_match", "exact-match (test)")]:
        vals = [r[key] for r in rows]
        ax.bar(x, vals, color=colours)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
        if baseline and key in baseline:
            ax.axhline(baseline[key], color="k", ls="--", lw=1.3,
                       label=f"classical baseline {baseline[key]:.3f}")
            ax.legend(fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0, 1.12); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("blue = fixed window (brightness available)   "
                 "red = per-image window (brightness removed)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def confusion_pair(rows, path):
    """Beam-hardening / scatter confusion across arms -- experiment 3."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    w = 0.38
    a = [r.get("scatter_to_bh", 0.0) for r in rows]
    b = [r.get("bh_to_scatter", 0.0) for r in rows]
    ax.bar(x - w / 2, a, w, label="scatter present -> falsely predicts BH", color="#c44e52")
    ax.bar(x + w / 2, b, w, label="BH present -> falsely predicts scatter", color="#4878a8")
    for xi, v in zip(x - w / 2, a):
        ax.text(xi, v + 0.006, f"{v:.1%}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, b):
        ax.text(xi, v + 0.006, f"{v:.1%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("false-positive rate"); ax.legend(fontsize=8)
    ax.set_title("Experiment 3: beam-hardening / scatter confusion")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
