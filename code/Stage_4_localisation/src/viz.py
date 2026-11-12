"""Qualitative overlays and quantitative figures for Stage C."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def overlay_grid(images, cams, labels, artifacts, masks, path, n=8, title=""):
    """One row per sample: the slice, then a Grad-CAM per artifact."""
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        ax = axes[r, 0]
        ax.imshow(images[r], cmap="gray")
        on = [a for a, b in zip(artifacts, labels[r]) if b]
        ax.set_title("slice\n" + (", ".join(a[:4] for a in on) or "clean"), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for c, a in enumerate(artifacts):
            ax = axes[r, c + 1]
            ax.imshow(images[r], cmap="gray")
            ax.imshow(cams[r, c], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            m = masks[r].get(a) if isinstance(masks[r], dict) else None
            if m is not None and m.any():
                ax.contour(m, levels=[0.5], colors="w", linewidths=0.7)
            mark = "PRESENT" if labels[r][c] else "absent"
            ax.set_title(f"{a}\n{mark}", fontsize=8,
                         color="k" if labels[r][c] else "0.45")
            ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.985 if title else 1])
    plt.savefig(path, dpi=85, bbox_inches="tight"); plt.close(fig)


def localisation_bars(scores, path):
    """Pointing game / IoU / energy ratio for the artifacts that have masks."""
    names = list(scores)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title, ref in [
            (axes[0], "pointing_game", "Pointing game (peak inside mask)", None),
            (axes[1], "iou", f"IoU at the CAM threshold", None),
            (axes[2], "energy_ratio", "Heat concentration on the artifact", 1.0)]:
        vals = [scores[n].get(key, np.nan) for n in names]
        ax.bar(range(len(names)), vals, color="#4878a8",
               label="Grad-CAM" if key == "pointing_game" else None)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        if key == "pointing_game":
            ch = [scores[n].get("pointing_chance", np.nan) for n in names]
            ax.bar(range(len(names)), ch, color="0.7", width=0.4, label="chance (mask area)")
            for i, v in enumerate(ch):
                ax.text(i, v, f"{v:.2f}", ha="center", va="top", fontsize=8, color="0.25")
            ax.legend(fontsize=8)
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=1.2, label="chance (1.0)")
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=12)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def concentration_bars(conc, artifacts, path):
    """Heat focus where the artifact is present, and how often the map is empty."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
    x = np.arange(len(artifacts))

    ax = axes[0]
    vals = [conc[a].get("concentration_present", np.nan) for a in artifacts]
    ax.bar(x, vals, color="#c44e52")
    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.10, color="k", ls="--", lw=1.2, label="heat spread evenly (0.10)")
    ax.set_xticks(x); ax.set_xticklabels(artifacts, rotation=12)
    ax.set_ylabel("share of heat in hottest 10% of pixels")
    ax.set_title("Heat focus where the artifact IS present")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    w = 0.38
    pres = [conc[a]["empty_fraction_present"] for a in artifacts]
    absn = [conc[a]["empty_fraction_absent"] for a in artifacts]
    ax.bar(x - w / 2, pres, w, label="artifact present", color="#c44e52")
    ax.bar(x + w / 2, absn, w, label="artifact absent", color="#9aa0a6")
    for xi, v in zip(x - w / 2, pres):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, absn):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(artifacts, rotation=12)
    ax.set_ylim(0, 1.1); ax.set_ylabel("fraction of all-zero maps")
    ax.set_title("Grad-CAM finds NO evidence (empty map)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
