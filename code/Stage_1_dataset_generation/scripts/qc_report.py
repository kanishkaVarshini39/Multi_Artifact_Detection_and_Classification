"""
Visual QC of the generated dataset (guidelines S9: "validate synthetic realism
qualitatively before large-scale training").

Writes data/qc_combinations.png and data/qc_composition.png.
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C
from preprocess import normalise
from labels import COMBINATIONS, combo_name


def main():
    rows = list(csv.DictReader(open(C.DATA / "labels.csv")))
    main_rows = [r for r in rows if r["split"] != "test_ood_severity"]
    by_combo = defaultdict(list)
    for r in main_rows:
        by_combo[r["combo"]].append(r)

    # ---------------- figure 1: one example of each of the 16 combinations
    order = sorted(by_combo, key=lambda b: (b.count("1"), b))
    fig, axes = plt.subplots(4, 4, figsize=(15, 15.8))
    for ax, bits in zip(axes.ravel(), order):
        r = sorted(by_combo[bits], key=lambda x: -sum(float(x[f"sphys_{a}"])
                                                      for a in C.ARTIFACTS))[0]
        img = np.load(C.DATA / r["image_path"])
        ax.imshow(normalise(img), cmap="gray", vmin=0, vmax=1)
        sev = ", ".join(f"{a[:4]}={float(r[f'sphys_{a}']):.2f}"
                        for a in C.ARTIFACTS if int(r[f"y_{a}"]))
        ax.set_title(f"{bits}  {combo_name(tuple(int(b) for b in bits))}\n{sev or 'no artifacts'}",
                     fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Project 29 dataset - all 16 label combinations "
                 "(strongest-severity example of each)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(C.DATA / "qc_combinations.png", dpi=85, bbox_inches="tight")
    plt.close(fig)

    # ---------------- figure 2: composition
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    cnt = Counter(r["combo"] for r in main_rows)
    ax = axes[0, 0]
    ax.bar(range(len(order)), [cnt[b] for b in order], color="#4878a8")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_title("samples per label combination (main 2400)"); ax.set_ylabel("count")

    ax = axes[0, 1]
    nart = Counter(int(r["n_artifacts"]) for r in main_rows)
    ax.bar(list(nart), [nart[k] for k in nart], color="#5a9367")
    ax.set_title("samples by number of simultaneous artifacts")
    ax.set_xlabel("artifacts present"); ax.set_ylabel("count")

    ax = axes[0, 2]
    co = np.zeros((4, 4))
    for r in main_rows:
        y = [int(r[f"y_{a}"]) for a in C.ARTIFACTS]
        for i in range(4):
            for j in range(4):
                co[i, j] += y[i] * y[j]
    im = ax.imshow(co, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([a[:4] for a in C.ARTIFACTS]); ax.set_yticklabels(C.ARTIFACTS, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(co[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("label co-occurrence counts"); plt.colorbar(im, ax=ax, fraction=0.046)

    floors = json.loads((C.DATA / "manifest.json").read_text())["severity_floors"]
    for k, a in enumerate(C.ARTIFACTS[:3]):
        ax = axes[1, k]
        tr = [float(r[f"sphys_{a}"]) for r in rows
              if int(r[f"y_{a}"]) and r["severity_band"] == "train"]
        oo = [float(r[f"sphys_{a}"]) for r in rows
              if int(r[f"y_{a}"]) and r["severity_band"] == "ood"]
        ax.hist(tr, bins=28, alpha=0.75, label="train band", color="#4878a8")
        ax.hist(oo, bins=28, alpha=0.75, label="OOD band (held out)", color="#c44e52")
        ax.axvline(floors[a], color="k", ls="--", lw=1.4, label=f"floor={floors[a]:.2f}")
        ax.set_title(f"{a}: physical severity"); ax.set_xlabel("s_phys"); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(C.DATA / "qc_composition.png", dpi=85, bbox_inches="tight")
    print(f"wrote {C.DATA/'qc_combinations.png'}\nwrote {C.DATA/'qc_composition.png'}")


if __name__ == "__main__":
    main()
