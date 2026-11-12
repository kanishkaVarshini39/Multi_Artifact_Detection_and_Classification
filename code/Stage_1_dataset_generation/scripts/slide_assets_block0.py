"""Image assets for the video deck, Block 0."""
import argparse, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config as C
from phantoms import make_phantom
from pipeline import build_projection_cache, simulate
from preprocess import normalise
from features import compute

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)

PID = 2
ph = make_phantom(PID, C.MASTER_SEED + 991 * PID)
cache = build_projection_cache(ph)

def run(active, sev, seed=7):
    labels = {a: (a in active) for a in C.ARTIFACTS}
    s = {a: sev for a in C.ARTIFACTS}
    img, _, _ = simulate(ph, cache, labels, s, np.random.default_rng(seed))
    return img

# --- find the severity pair where beam hardening and scatter cup EQUALLY.
# The hook only lands if the two really are hard to tell apart.
clean = run([], 0.0)
base = compute(clean)["cupping_index"]
bh = {s: compute(run(["beam_hardening"], s))["cupping_index"] - base
      for s in np.linspace(0.3, 1.0, 8)}
sc = {s: compute(run(["scatter"], s))["cupping_index"] - base
      for s in np.linspace(0.1, 1.0, 10)}
s_bh = max(bh, key=lambda k: bh[k])                     # strongest BH available
target = bh[s_bh]
s_sc = min(sc, key=lambda k: abs(sc[k] - target))       # scatter that matches it
print(f"matched on cupping: BH s={s_bh:.2f} ({bh[s_bh]:+.4f})  "
      f"scatter s={s_sc:.2f} ({sc[s_sc]:+.4f})")

def save(img, name, cmap="gray"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(normalise(img) if cmap == "gray" else img, cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest")
    ax.axis("off")
    fig.savefig(out / name, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  wrote", name)

save(run(["beam_hardening"], s_bh), "hook_A_beam_hardening.png")
save(run(["scatter"], s_sc),        "hook_B_scatter.png")
save(clean,                         "hook_clean.png")

# --- 4x4 grid of all sixteen states, no labels, tight
import itertools
combos = sorted(itertools.product([0, 1], repeat=4), key=lambda c: (sum(c), c))
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for ax, combo in zip(axes.ravel(), combos):
    act = [a for a, b in zip(C.ARTIFACTS, combo) if b]
    ax.imshow(normalise(run(act, 0.8)), cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("0.75"); sp.set_linewidth(0.8)
plt.subplots_adjust(wspace=0.03, hspace=0.03)
fig.savefig(out / "grid16.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("  wrote grid16.png")
