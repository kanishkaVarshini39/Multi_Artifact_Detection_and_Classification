"""Image assets for the video deck, Block 1 (building the data)."""
import argparse, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config as C
from phantoms import make_phantom
from pipeline import (build_projection_cache, simulate, line_integrals,
                      form_projection, apply_scatter, ring_gains, detect, reconstruct)
from preprocess import normalise

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)

PID = 2
ph = make_phantom(PID, C.MASTER_SEED + 991 * PID)
cache = build_projection_cache(ph)
rng = np.random.default_rng(7)

def panel(data, name, cmap="gray", vmin=None, vmax=None, aspect=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect=aspect or "equal")
    ax.axis("off")
    fig.savefig(out / name, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig); print("  wrote", name)

# ---- the four pipeline stages, as real data
panel(ph["label_map"], "p1_phantom.png", cmap="viridis")

clean_sino = line_integrals(cache, C.E_EFF_KEV)
panel(clean_sino, "p2_sinogram.png", aspect="auto")

# corrupt: ring stripes + metal starvation, both visible in the sinogram
labels = {a: a in ("ring", "metal") for a in C.ARTIFACTS}
sev = {a: 0.9 for a in C.ARTIFACTS}
inten, _ = form_projection(cache, labels, sev, ph, np.random.default_rng(7))
r_eff = float(np.sqrt(max(ph["meta"]["body_area_px"], 1) / np.pi))
gains, idx, amp, nb = ring_gains(inten.shape[0], sev["ring"],
                                 np.random.default_rng(7), r_max=0.88 * r_eff)
inten = inten * gains[:, None]
bad_sino = detect(inten, np.random.default_rng(7))

# Panel 3 shows WHAT WE CHANGED, not the corrupted sinogram itself. The metal
# band otherwise swamps the colour scale and the ring stripes vanish; the
# difference makes the edit legible, which is the point of the slide.
clean_meas = detect(np.exp(-clean_sino), np.random.default_rng(7))
diff = bad_sino - clean_meas
m = float(np.percentile(np.abs(diff), 99.5))
panel(diff, "p3_what_we_break.png", cmap="seismic", vmin=-m, vmax=m, aspect="auto")

recon = reconstruct(bad_sino)
panel(normalise(recon), "p4_reconstruction.png", vmin=0, vmax=1)

# ---- four operators, slide-shaped: 4 columns, image over difference
def run(active, s=0.85, seed=7):
    lab = {a: (a in active) for a in C.ARTIFACTS}
    sv = {a: s for a in C.ARTIFACTS}
    img, _, _ = simulate(ph, cache, lab, sv, np.random.default_rng(seed))
    return img

base = run([])
names = [("beam hardening", "beam_hardening"), ("ring", "ring"),
         ("metal", "metal"), ("scatter", "scatter")]
fig, axes = plt.subplots(2, 4, figsize=(11.6, 6.0))
for k, (lbl, key) in enumerate(names):
    img = run([key])
    axes[0, k].imshow(normalise(img), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, k].set_title(lbl, fontsize=15, color="white", pad=8)
    d = img - base; m = float(np.abs(d).max()) or 1.0
    axes[1, k].imshow(d, cmap="seismic", vmin=-m, vmax=m, interpolation="nearest")
    for r in (0, 1):
        axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
        for sp in axes[r, k].spines.values(): sp.set_visible(False)
fig.patch.set_alpha(0)
plt.subplots_adjust(wspace=0.04, hspace=0.06)
fig.savefig(out / "operators4.png", dpi=240, bbox_inches="tight",
            pad_inches=0.05, transparent=True)
plt.close(fig); print("  wrote operators4.png")
