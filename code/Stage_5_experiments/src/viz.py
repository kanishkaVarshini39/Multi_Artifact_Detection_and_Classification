"""Figures for the five experiments."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def grouped_bars(ax, labels, series, ylabel, title, ylim=(0, 1.15), fmt="{:.3f}"):
    x = np.arange(len(labels)); w = 0.8 / max(len(series), 1)
    for k, (name, vals) in enumerate(series.items()):
        ax.bar(x + k * w, vals, w, label=name)
        for xi, v in zip(x + k * w, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.008, fmt.format(v), ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x + w * (len(series) - 1) / 2)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)


def exp2_figure(artifacts, per_label, aggregates, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.6))
    grouped_bars(axes[0], list(artifacts), per_label, "F1",
                 "Experiment 2: per-label F1 on the test split")
    grouped_bars(axes[1], ["macro-F1", "exact-match", "Hamming loss"], aggregates,
                 "", "Experiment 2: aggregate metrics")
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def confusion_matrices(cms, path, title):
    """cms: {model_name: {artifact: 2x2}}"""
    models = list(cms); arts = list(next(iter(cms.values())))
    fig, axes = plt.subplots(len(models), len(arts),
                             figsize=(3.4 * len(arts), 3.2 * len(models)))
    axes = np.atleast_2d(axes)
    for r, mname in enumerate(models):
        for c, a in enumerate(arts):
            cm = cms[mname][a]
            ax = axes[r, c]
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            fontsize=11, color="w" if cm[i, j] > cm.max() / 2 else "k")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred 0", "pred 1"], fontsize=8)
            ax.set_yticklabels(["true 0", "true 1"], fontsize=8)
            ax.set_title(f"{mname}\n{a}" if r == 0 else a, fontsize=9)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def exp3_figure(rates, path):
    """rates: {model: {'scatter_to_bh': x, 'bh_to_scatter': y}}"""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    names = list(rates); x = np.arange(len(names)); w = 0.38
    a = [rates[n].get("scatter_to_bh", 0) for n in names]
    b = [rates[n].get("bh_to_scatter", 0) for n in names]
    ax.bar(x - w / 2, a, w, label="scatter present → falsely predicts BH", color="#c44e52")
    ax.bar(x + w / 2, b, w, label="BH present → falsely predicts scatter", color="#4878a8")
    for xi, v in zip(x - w / 2, a):
        ax.text(xi, v + 0.004, f"{v:.1%}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, b):
        ax.text(xi, v + 0.004, f"{v:.1%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("false-positive rate")
    ax.set_title("Experiment 3: beam-hardening / scatter confusion")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def exp4_figure(by_count, combos, path):
    fig, axes = plt.subplots(1, 2, figsize=(17, 4.6))
    grouped_bars(axes[0], [str(k) for k in range(5)], by_count, "exact-match",
                 "Experiment 4: accuracy vs number of simultaneous artifacts")
    axes[0].set_xlabel("artifacts present")
    names = list(combos); labels = combos[names[0]]["combo"]
    series = {n: combos[n]["exact_match"].to_numpy() for n in names}
    x = np.arange(len(labels)); w = 0.8 / len(series)
    for k, (n, v) in enumerate(series.items()):
        axes[1].bar(x + k * w, v, w, label=n)
    axes[1].set_xticks(x + w * (len(series) - 1) / 2)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].set_ylim(0, 1.1); axes[1].set_ylabel("exact-match")
    axes[1].set_title("Per-combination accuracy (16 classes)")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def exp5_figure(series, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    grouped_bars(axes[0], ["in-band test", "held-out severity"],
                 {k: v["macro_f1"] for k, v in series.items()},
                 "macro-F1", "Experiment 5: macro-F1")
    grouped_bars(axes[1], ["in-band test", "held-out severity"],
                 {k: v["exact_match"] for k, v in series.items()},
                 "exact-match", "Experiment 5: exact-match")
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def exp1_figure(arms, slices, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    grouped_bars(axes[0], slices, {a: [arms[a]["slices"][s]["macro_f1"] for s in slices]
                                   for a in arms},
                 "macro-F1", "Experiment 1: macro-F1 by test slice")
    grouped_bars(axes[1], slices, {a: [arms[a]["slices"][s]["exact_match"] for s in slices]
                                   for a in arms},
                 "exact-match", "Experiment 1: exact-match by test slice")
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
