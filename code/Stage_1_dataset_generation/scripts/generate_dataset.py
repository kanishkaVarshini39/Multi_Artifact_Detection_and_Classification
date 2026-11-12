"""
Generate the Project 29 multi-artifact dataset.

One process per phantom: build the projection cache once, then emit every
config of that phantom from it.

Severity handling  [D2]:
  s_norm  is drawn from a BAND  -- [0.2, 0.8] for train/val/test,
          and [0, 0.2] or [0.8, 1.0] for the supplementary OOD-severity set.
  s_phys = floor + s_norm * (1 - floor)   remaps that band above the calibrated
          detectability floor, so no sample is ever labelled with an artifact
          that is invisible, and the OOD set is a DELIBERATE held-out band
          rather than accidental label noise.

Severity labels  [D3]: computed by labels.severity_from_params from the
injection parameters only.  The handcrafted features written to features.csv
are measured from the image and are the classical baseline's inputs -- a
separate quantity that is never used as a target.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C
import materials as M
from phantoms import make_phantom
from pipeline import build_projection_cache, simulate
from labels import severity_vector, combo_name
from plan import build_plan
from features import compute, FEATURE_NAMES

FLOORS_PATH = C.DATA / "severity_floors.json"


def load_floors():
    if not FLOORS_PATH.exists():
        raise SystemExit("run scripts/calibrate_floors.py first (writes severity_floors.json)")
    return json.loads(FLOORS_PATH.read_text())["floors"]


def draw_severity(rng, band_kind, floor):
    """Returns (s_norm, s_phys). s_phys is what the simulator receives."""
    if band_kind == "train":
        lo, hi = C.S_TRAIN_BAND
    else:
        lo, hi = C.S_OOD_BANDS[int(rng.integers(len(C.S_OOD_BANDS)))]
    s_norm = float(rng.uniform(lo, hi))
    return s_norm, float(floor + s_norm * (1.0 - floor))


# Deterministic split -> integer. Python's str hash is salted per process, so
# using hash() here would silently break the fixed-seed reproducibility the
# guidelines require (S9).
SPLIT_CODE = {"train": 1, "val": 2, "test": 3, "test_ood_severity": 4}

# sample_id MUST carry the split.  A test phantom can draw the same combination
# at the same config index in both `test` and `test_ood_severity`, so without a
# split tag 49 ids collided -- harmless on disk (separate per-split folders) but
# silently mis-joins features to labels for anything keyed on sample_id.
SPLIT_TAG = {"train": "tr", "val": "va", "test": "te", "test_ood_severity": "oo"}


def phantom_job(args):
    pid, rows, floors = args
    seed = C.MASTER_SEED + 991 * pid
    ph = make_phantom(pid, seed)
    cache = build_projection_cache(ph)
    out = []
    for row in rows:
        rng = np.random.default_rng(seed * 7919 + row["config_index"] * 131
                                    + SPLIT_CODE[row["split"]] * 10007)
        combo = row["combo"]
        labels = {a: bool(b) for a, b in zip(C.ARTIFACTS, combo)}
        s_norm, s_phys = {}, {}
        for a in C.ARTIFACTS:
            if labels[a]:
                n, p = draw_severity(rng, row["severity_band"], floors[a])
            else:
                n, p = 0.0, 0.0
            s_norm[a], s_phys[a] = n, p

        # dose nuisance variable, log-uniform over the configured range
        lo, hi = C.I0_RANGE
        i0 = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        img, params, masks = simulate(ph, cache, labels, s_phys, rng, i0=i0)

        bits = "".join(str(int(b)) for b in combo)
        sid = f"{SPLIT_TAG[row['split']]}_p{pid:04d}_c{row['config_index']}_{bits}"
        img_rel = f"images/{row['split']}/{sid}.npy"
        img_path = C.DATA / img_rel
        img_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(img_path, img.astype(np.float32))

        mask_rel = ""
        if masks:
            mask_rel = f"masks/{row['split']}/{sid}.npz"
            mp = C.DATA / mask_rel
            mp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mp, **{k: v.astype(bool) for k, v in masks.items()})

        sev = severity_vector(labels, params, ph["meta"])
        feats = compute(img)

        rec = dict(sample_id=sid, phantom_id=pid, split=row["split"],
                   config_index=row["config_index"], combo=bits,
                   combo_name=combo_name(combo), n_artifacts=int(sum(combo)),
                   severity_band=row["severity_band"],
                   base_material=ph["meta"]["base_material"],
                   i0=round(i0, 1),
                   image_path=img_rel, mask_path=mask_rel)
        for i, a in enumerate(C.ARTIFACTS):
            rec[f"y_{a}"] = int(combo[i])
            rec[f"sev_{a}"] = round(float(sev[i]), 6)
            rec[f"snorm_{a}"] = round(float(s_norm[a]), 6)
            rec[f"sphys_{a}"] = round(float(s_phys[a]), 6)
        rec["params"] = json.dumps({k: v for k, v in params.items() if k != "ring_channels"})
        for k in FEATURE_NAMES:
            rec[f"f_{k}"] = round(float(feats[k]), 6)
        out.append(rec)
    return ph["meta"], out


def main():
    import csv
    from collections import defaultdict

    floors = load_floors()
    plan, summary, splits = build_plan()
    print(f"plan: {summary['n_samples']} samples over {C.N_PHANTOMS} phantoms")
    print(f"floors: { {k: round(v,3) for k,v in floors.items()} }")

    by_phantom = defaultdict(list)
    for r in plan:
        by_phantom[r["phantom_id"]].append(r)
    jobs = [(pid, rows, floors) for pid, rows in sorted(by_phantom.items())]

    nproc = max(1, (os.cpu_count() or 4) - 2)
    print(f"generating with {nproc} processes ...")
    records, metas = [], []
    with Pool(nproc) as pool:
        for i, (meta, recs) in enumerate(pool.imap_unordered(phantom_job, jobs, chunksize=4), 1):
            metas.append(meta)
            records.extend(recs)
            if i % 40 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} phantoms  ({len(records)} samples)")

    records.sort(key=lambda r: (r["split"], r["sample_id"]))
    label_cols = ["sample_id", "phantom_id", "split", "config_index", "combo",
                  "combo_name", "n_artifacts", "severity_band", "base_material", "i0"]
    label_cols += [f"y_{a}" for a in C.ARTIFACTS]
    label_cols += [f"sev_{a}" for a in C.ARTIFACTS]
    label_cols += [f"snorm_{a}" for a in C.ARTIFACTS] + [f"sphys_{a}" for a in C.ARTIFACTS]
    label_cols += ["image_path", "mask_path", "params"]
    feat_cols = ["sample_id", "split"] + [f"f_{k}" for k in FEATURE_NAMES]

    with open(C.DATA / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=label_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    with open(C.DATA / "features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=feat_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    manifest = dict(
        project="Project 29 - Multi-Artifact Detection and Classification",
        master_seed=C.MASTER_SEED, n_phantoms=C.N_PHANTOMS,
        n_samples=len(records), image_shape=[C.N_PIX, C.N_PIX],
        fov_cm=C.FOV_CM, px_cm=C.PX_CM, n_angles=C.N_ANGLES, I0=C.I0,
        i0_range=list(C.I0_RANGE), i0_calibration=C.I0_CALIBRATION,
        energy_bins_kev=C.E_BINS_KEV, e_eff_kev=C.E_EFF_KEV,
        artifacts=list(C.ARTIFACTS), severity_floors=floors,
        train_severity_band=list(C.S_TRAIN_BAND),
        ood_severity_bands=[list(b) for b in C.S_OOD_BANDS],
        splits=summary["split_sizes"], plan_summary=summary,
        phantoms=sorted(metas, key=lambda m: m["phantom_id"]),
    )
    (C.DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(records)} samples")
    print(f"  {C.DATA/'labels.csv'}")
    print(f"  {C.DATA/'features.csv'}")
    print(f"  {C.DATA/'manifest.json'}")


if __name__ == "__main__":
    main()
