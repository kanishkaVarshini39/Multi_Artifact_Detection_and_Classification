"""
Package the project into a self-contained deliverable folder.

Layout produced at --dest:

    <dest>/
      README.md                    what is here and how to use it
      code/                        every script that produces the data
      libraries/                   dependency manifests + offline-install helper
      data_train/                  train split, self-contained
      data_val/                    validation split
      data_test/                   in-band test split
      data_test_ood_severity/      supplementary held-out-severity test set
      metadata/                    manifest, calibration, QC figures, full CSVs

Each data_* folder carries its own labels.csv / features.csv whose paths are
relative to that folder, so a split can be moved or zipped on its own.

Re-running overwrites the destination's generated content.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C

SPLIT_DIRS = {"train": "data_train", "val": "data_val",
              "test": "data_test", "test_ood_severity": "data_test_ood_severity"}

CORE_REQUIREMENTS = """# Core - everything needed to GENERATE and VALIDATE the dataset.
# These are the only libraries the generation pipeline imports.
numpy==2.1.3
scipy==1.15.3
scikit-image==0.25.0
scikit-learn==1.6.1
matplotlib==3.10.0
"""

TRAINING_REQUIREMENTS = """# Additional libraries for the MODELLING stage (spec items 11-14).
# Not needed to generate the dataset.
torch==2.12.0
torchvision==0.27.0
pandas==2.2.3
seaborn==0.13.2
xgboost==3.3.0
grad-cam==1.5.5          # pytorch-grad-cam, for the localisation module (item 5)
"""

SETUP_MD = """# Environment setup

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r libraries/requirements.txt              # generation + validation
pip install -r libraries/requirements-training.txt     # modelling stage
```

`pip-freeze.txt` is a full snapshot of the environment the dataset was built in,
for exact reproduction if anything ever disagrees.

## Offline / archival copy of the libraries

`download_wheels.sh` fetches every wheel into `libraries/wheels/` so the exact
environment can be rebuilt with no network:

```bash
bash libraries/download_wheels.sh
pip install --no-index --find-links libraries/wheels -r libraries/requirements.txt
```

This is not run by default because it downloads a few hundred MB.

## A note on ASTRA Toolbox and TIGRE

The project brief (item 22) and the course guidelines (section 3) recommend
ASTRA or TIGRE for forward projection and reconstruction. **Neither installs on
Apple Silicon:**

* ASTRA 2.5.0 publishes `manylinux_x86_64` and `win_amd64` wheels only; its
  conda channel has `linux-64`, `osx-64` and `win-64` but no `osx-arm64`, and
  the `osx-64` line stopped at 2.0.0 (Intel, would need Rosetta).
* TIGRE requires CUDA and is a source build. `pip install tigre` installs an
  unrelated bioinformatics package.
* FDK is GPU-only in ASTRA even on Linux.

This pipeline therefore uses `scikit-image`'s `radon` / `iradon` (parallel-beam,
CPU). That is sufficient because the target is 2D slices, and all four artifact
mechanisms are projection-domain operators that behave identically under
parallel beam. No GPU is needed to generate the dataset.
"""

DOWNLOAD_SH = """#!/usr/bin/env bash
# Fetch every wheel for an offline / archival rebuild of the environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/wheels"
pip download -d "$HERE/wheels" -r "$HERE/requirements.txt"
pip download -d "$HERE/wheels" -r "$HERE/requirements-training.txt"
echo "wheels saved to $HERE/wheels"
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def export_code(dest):
    code = dest / "code"
    if code.exists():
        shutil.rmtree(code)
    (code / "src").mkdir(parents=True)
    (code / "scripts").mkdir(parents=True)
    shutil.copy2(C.ROOT / "config.py", code / "config.py")
    for f in sorted((C.ROOT / "src").glob("*.py")):
        shutil.copy2(f, code / "src" / f.name)
    for f in sorted((C.ROOT / "scripts").glob("*.py")):
        shutil.copy2(f, code / "scripts" / f.name)
    shutil.copy2(C.ROOT / "README.md", code / "README.md")
    n = len(list((code / "src").glob("*.py"))) + len(list((code / "scripts").glob("*.py"))) + 1
    return n


def export_libraries(dest):
    lib = dest / "libraries"
    lib.mkdir(parents=True, exist_ok=True)
    write(lib / "requirements.txt", CORE_REQUIREMENTS)
    write(lib / "requirements-training.txt", TRAINING_REQUIREMENTS)
    write(lib / "SETUP.md", SETUP_MD)
    write(lib / "download_wheels.sh", DOWNLOAD_SH)
    os.chmod(lib / "download_wheels.sh", 0o755)
    try:
        import subprocess
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=180)
        write(lib / "pip-freeze.txt", out.stdout or "# pip freeze unavailable\n")
    except Exception as exc:                                  # pragma: no cover
        write(lib / "pip-freeze.txt", f"# pip freeze failed: {exc}\n")


def export_split(dest, split, rows, feat_by_id):
    out = dest / SPLIT_DIRS[split]
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    masks_dir = out / "masks"
    label_rows, feature_rows, n_masks = [], [], 0
    for r in rows:
        src_img = C.DATA / r["image_path"]
        dst_name = Path(r["image_path"]).name
        shutil.copy2(src_img, out / "images" / dst_name)
        rec = dict(r)
        rec["image_path"] = f"images/{dst_name}"
        if r["mask_path"]:
            masks_dir.mkdir(exist_ok=True)
            mname = Path(r["mask_path"]).name
            shutil.copy2(C.DATA / r["mask_path"], masks_dir / mname)
            rec["mask_path"] = f"masks/{mname}"
            n_masks += 1
        label_rows.append(rec)
        feature_rows.append(feat_by_id[r["sample_id"]])

    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(label_rows[0]))
        w.writeheader(); w.writerows(label_rows)
    with open(out / "features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feature_rows[0]))
        w.writeheader(); w.writerows(feature_rows)

    combos = Counter(r["combo"] for r in label_rows)
    phantoms = sorted({int(r["phantom_id"]) for r in label_rows})
    summary = dict(split=split, n_samples=len(label_rows), n_masks=n_masks,
                   n_phantoms=len(phantoms), phantom_ids=phantoms,
                   combinations_present=len(combos),
                   per_label_positive={a: sum(int(r[f"y_{a}"]) for r in label_rows)
                                       for a in C.ARTIFACTS})
    write(out / "split_summary.json", json.dumps(summary, indent=2))
    return summary


def export_metadata(dest):
    meta = dest / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    for name in ["manifest.json", "severity_floors.json", "labels.csv", "features.csv",
                 "qc_combinations.png", "qc_composition.png", "qc_operators.png"]:
        src = C.DATA / name
        if src.exists():
            target = {"labels.csv": "labels_all_splits.csv",
                      "features.csv": "features_all_splits.csv"}.get(name, name)
            shutil.copy2(src, meta / target)


README_TMPL = """# Project 29 — Multi-Artifact Detection and Classification

Synthetic CT dataset with multi-label artifact ground truth that is **known by
construction**, plus the complete code that produces it.

Generated {date} · seed `{seed}` · regenerable end to end in about 5 minutes.

## Folders

| Folder | Contents |
|---|---|
| `code/` | every script that generates, calibrates, validates and QCs the data. `code/README.md` is the full technical write-up. |
| `libraries/` | pinned dependency manifests, a full environment snapshot, and `download_wheels.sh` for an offline archival copy. See `libraries/SETUP.md`. |
| `data_train/` | **training split** — {train_n} samples from {train_p} phantoms |
| `data_val/` | validation split — {val_n} samples from {val_p} phantoms |
| `data_test/` | **test split** — {test_n} samples from {test_p} phantoms |
| `data_test_ood_severity/` | supplementary held-out-severity test set — {ood_n} samples (required experiment 5) |
| `metadata/` | manifest with all 400 phantom parameters, severity-floor calibration, QC figures, and the combined CSVs |

The three main splits are **phantom-disjoint**: no base phantom appears in more
than one of train / val / test, so nothing leaks. `data_test_ood_severity`
deliberately reuses the *test* phantoms at severities outside the training band.

## Each data folder is self-contained

```
data_train/
  images/            256x256 float32 .npy, per-pixel linear attenuation mu
  masks/             ring / metal localisation masks (.npz), where applicable
  labels.csv         one row per sample: 4-bit presence vector, 4 severities,
                     dose, phantom id, paths relative to this folder
  features.csv       14 handcrafted features (classical-baseline inputs)
  split_summary.json counts and the phantom ids in this split
```

## Loading a sample

```python
import csv, numpy as np, sys
sys.path.insert(0, "code/src")
from preprocess import normalise            # fixed physical window -> [0,1]

rows = list(csv.DictReader(open("data_train/labels.csv")))
r = rows[0]
img = normalise(np.load("data_train/" + r["image_path"]))      # (256, 256) float32
y = [int(r[f"y_{{a}}"]) for a in
     ("beam_hardening", "ring", "metal", "scatter")]           # the label
sev = [float(r[f"sev_{{a}}"]) for a in
       ("beam_hardening", "ring", "metal", "scatter")]         # severity targets
```

## Regenerating from scratch

```bash
cd code
PYTHONPATH=src python3 scripts/calibrate_floors.py    # ~1.7 min
PYTHONPATH=src python3 scripts/generate_dataset.py    # ~2.3 min
PYTHONPATH=src python3 scripts/validate_dataset.py    # all checks must pass
PYTHONPATH=src python3 scripts/qc_report.py
PYTHONPATH=src python3 scripts/export_package.py --dest <this folder>
```

Everything is seeded, so a rebuild reproduces the dataset bit for bit.

## Read next

`code/README.md` covers the physics of each operator, how the five
specification defects were resolved, the design decisions that were measured
rather than assumed, and the known limitations — including an honest account of
why the classical baseline is stronger than the brief anticipated.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(C.DATA / "labels.csv")))
    feat_by_id = {r["sample_id"]: r for r in csv.DictReader(open(C.DATA / "features.csv"))}
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)

    print(f"exporting to {dest}")
    n_code = export_code(dest)
    print(f"  code/            {n_code} files")
    export_libraries(dest)
    print(f"  libraries/       manifests + setup notes")

    summaries = {}
    for split in ["train", "val", "test", "test_ood_severity"]:
        s = export_split(dest, split, by_split[split], feat_by_id)
        summaries[split] = s
        print(f"  {SPLIT_DIRS[split]:24s} {s['n_samples']:5d} samples, "
              f"{s['n_phantoms']:3d} phantoms, {s['n_masks']:4d} masks, "
              f"{s['combinations_present']}/16 combinations")

    export_metadata(dest)
    print("  metadata/        manifest, calibration, QC figures, combined CSVs")

    import datetime
    write(dest / "README.md", README_TMPL.format(
        date=datetime.date.today().isoformat(), seed=C.MASTER_SEED,
        train_n=summaries["train"]["n_samples"], train_p=summaries["train"]["n_phantoms"],
        val_n=summaries["val"]["n_samples"], val_p=summaries["val"]["n_phantoms"],
        test_n=summaries["test"]["n_samples"], test_p=summaries["test"]["n_phantoms"],
        ood_n=summaries["test_ood_severity"]["n_samples"]))
    print(f"\ndone -> {dest}")


if __name__ == "__main__":
    main()
