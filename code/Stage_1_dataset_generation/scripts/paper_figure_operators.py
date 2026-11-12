"""
Publication figure: each corruption operator in isolation.

Replaces the 85 dpi screen-quality QC panel. Changes that matter for print:
  * 6 columns instead of 8, so each panel is wide enough to resolve rings
  * 300 dpi, and a PDF container so the caption-free panels stay crisp
  * difference maps aligned directly under the image they belong to
  * a shared, symmetric colour scale per difference map, annotated with its range

    python3 scripts/paper_figure_operators.py --out <paper>/figures/fig_operators.pdf
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import config as C
from phantoms import make_phantom
from pipeline import build_projection_cache, simulate
from preprocess import normalise

PHANTOM_ID = 2
SEVERITY = 0.85
SEED = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    ph = make_phantom(PHANTOM_ID, C.MASTER_SEED + 991 * PHANTOM_ID)
    cache = build_projection_cache(ph)

    def run(active):
        labels = {a: (a in active) for a in C.ARTIFACTS}
        sev = {a: SEVERITY for a in C.ARTIFACTS}
        img, _, _ = simulate(ph, cache, labels, sev, np.random.default_rng(SEED))
        return img

    clean = run([])
    panels = [("beam hardening", ["beam_hardening"]), ("ring", ["ring"]),
              ("metal", ["metal"]), ("scatter", ["scatter"])]
    imgs = {name: run(act) for name, act in panels}
    allfour = run(list(C.ARTIFACTS))

    fig, axes = plt.subplots(2, 6, figsize=(12.0, 4.25))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)

    kw = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    axes[0, 0].imshow(ph["label_map"], cmap="viridis", interpolation="nearest")
    axes[0, 0].set_title("material map", fontsize=8.5, pad=3)
    axes[0, 1].imshow(normalise(clean), **kw)
    axes[0, 1].set_title("clean", fontsize=8.5, pad=3)
    for k, (name, _) in enumerate(panels):
        axes[0, k + 2].imshow(normalise(imgs[name]), **kw)
        axes[0, k + 2].set_title(name, fontsize=8.5, pad=3)

    axes[1, 0].imshow(normalise(allfour), **kw)
    axes[1, 0].set_title("all four", fontsize=8.5, pad=3)
    d4 = allfour - clean
    m4 = float(np.abs(d4).max()) or 1.0
    axes[1, 1].imshow(d4, cmap="seismic", vmin=-m4, vmax=m4, interpolation="nearest")
    axes[1, 1].set_title(f"all four $-$ clean ($\\pm${m4:.3f})", fontsize=7.5, pad=3)
    for k, (name, _) in enumerate(panels):
        ax = axes[1, k + 2]
        d = imgs[name] - clean
        m = float(np.abs(d).max()) or 1.0
        ax.imshow(d, cmap="seismic", vmin=-m, vmax=m, interpolation="nearest")
        ax.set_title(f"{name} $-$ clean ($\\pm${m:.3f})", fontsize=7.5, pad=3)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01,
                        wspace=0.06, hspace=0.20)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", format=out.suffix[1:])
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size/1e3:.0f} kB)")
    print(f"wrote {png}  ({png.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
