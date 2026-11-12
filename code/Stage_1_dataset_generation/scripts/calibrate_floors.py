"""
Severity-floor calibration.  [D2]

Problem being solved: if severity is drawn from [0, max], a draw near zero
labels an artifact 'present' in an image indistinguishable from clean.  That is
label noise in a dataset whose entire claim is exact ground truth, and it would
cap recall and corrupt the out-of-range-severity experiment.

Method -- a paired SHIFT test, per phantom:
  1. For each calibration phantom, simulate N_CLEAN clean realisations that
     differ only in the Poisson noise seed.  Their standard deviation is how
     much the index moves under noise alone.
  2. Sweep severity; at each severity measure the index again.
  3. Detected when the index SHIFTS from the phantom's clean mean by more than
     K_SIGMA times that noise spread.
  4. The floor is the smallest severity detected on >= DETECT_FRACTION of
     phantoms.

Calibration runs at C.I0_CALIBRATION, the NOISIEST dose in the sampled range,
so a floor established here holds at every dose the generator will use.

Why a shift rather than an absolute threshold: every index also responds to the
phantom's own structure (inclusion layout moves the cupping index, object edges
move the radial statistics).  An absolute threshold therefore asks the artifact
to out-shout the object, which is the wrong question and -- measured -- made
detectability depend on base material and inclusion layout rather than on
severity.  Comparing a phantom against ITSELF cancels the structure term.

Output: data/severity_floors.json, consumed by generate_dataset.py.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C
from phantoms import make_phantom
from pipeline import build_projection_cache, simulate
from features import compute, DETECTION_INDEX

N_PHANTOMS_CAL = 24
N_CLEAN = 8
N_REPS = 2
S_GRID = np.round(np.linspace(0.02, 1.00, 15), 4)
DETECT_FRACTION = 0.95
K_SIGMA = 4.0           # an artifact must move the index 4 noise-sigmas


def _index(img, artifact):
    return compute(img)[DETECTION_INDEX[artifact]]


def one_phantom(pid):
    seed = C.MASTER_SEED + 991 * pid
    ph = make_phantom(pid, seed)
    cache = build_projection_cache(ph)
    off = {a: False for a in C.ARTIFACTS}
    zero = {a: 0.0 for a in C.ARTIFACTS}

    clean_idx = {a: [] for a in C.ARTIFACTS}
    for r in range(N_CLEAN):
        img, _, _ = simulate(ph, cache, off, zero, np.random.default_rng(seed + 7000 + r),
                             i0=C.I0_CALIBRATION)
        f = compute(img)
        for a in C.ARTIFACTS:
            clean_idx[a].append(f[DETECTION_INDEX[a]])
    mu = {a: float(np.mean(clean_idx[a])) for a in C.ARTIFACTS}
    sd = {a: float(np.std(clean_idx[a], ddof=1)) for a in C.ARTIFACTS}

    detected = {a: np.zeros(len(S_GRID), bool) for a in C.ARTIFACTS}
    for a in C.ARTIFACTS:
        labels = {x: (x == a) for x in C.ARTIFACTS}
        band = K_SIGMA * max(sd[a], 1e-12)
        for i, s in enumerate(S_GRID):
            sev = {x: (float(s) if x == a else 0.0) for x in C.ARTIFACTS}
            shifts = []
            for r in range(N_REPS):
                img, _, _ = simulate(ph, cache, labels, sev,
                                     np.random.default_rng(seed + 31 * i + r),
                                     i0=C.I0_CALIBRATION)
                shifts.append(abs(_index(img, a) - mu[a]))
            detected[a][i] = min(shifts) > band
    return pid, dict(mu=mu, sd=sd), {a: detected[a].tolist() for a in C.ARTIFACTS}


def main():
    pids = list(range(N_PHANTOMS_CAL))
    nproc = max(1, (os.cpu_count() or 4) - 2)
    print(f"calibrating severity floors on {len(pids)} phantoms, {nproc} processes")
    with Pool(nproc) as pool:
        results = pool.map(one_phantom, pids)

    floors, report = {}, {}
    for a in C.ARTIFACTS:
        det = np.array([r[2][a] for r in results])            # (phantoms, grid)
        frac = det.mean(axis=0)
        ok = np.where(frac >= DETECT_FRACTION)[0]
        if ok.size:
            floor = float(S_GRID[ok[0]])
        else:
            floor = float(S_GRID[int(np.argmax(frac))])
            print(f"  WARNING: {a} never reached {DETECT_FRACTION:.0%} detection "
                  f"(best {frac.max():.2f}); using the best grid point")
        floors[a] = floor
        report[a] = dict(index=DETECTION_INDEX[a],
                         grid=[float(x) for x in S_GRID],
                         detect_fraction=[round(float(x), 3) for x in frac],
                         floor=floor)
        bar = "".join("#" if f >= DETECT_FRACTION else ("+" if f >= 0.5 else ".") for f in frac)
        print(f"  {a:15s} floor={floor:.3f}  [{bar}]  index={DETECTION_INDEX[a]}")

    out = dict(method="paired per-phantom shift test; detected when the index "
                      "moves K_SIGMA noise-sigmas from that phantom's clean mean",
               n_phantoms=N_PHANTOMS_CAL, n_clean_realisations=N_CLEAN,
               n_repeats=N_REPS, detect_fraction=DETECT_FRACTION, k_sigma=K_SIGMA,
               i0_calibration=C.I0_CALIBRATION,
               floors=floors, detail=report)
    C.DATA.mkdir(parents=True, exist_ok=True)
    path = C.DATA / "severity_floors.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return floors


if __name__ == "__main__":
    main()
