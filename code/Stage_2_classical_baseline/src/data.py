"""Load the Project 29 dataset's tabular part (labels + handcrafted features)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def find_dataset():
    """Return (root, layout) for the first dataset location that exists."""
    for cand in C.DATASET_CANDIDATES:
        if (cand / "training data" / "train" / "labels.csv").exists():
            return cand, "submission"
        if (cand / "labels.csv").exists() and (cand / "features.csv").exists():
            return cand, "flat"
        if (cand / "data_train" / "labels.csv").exists():
            return cand, "package"
    raise SystemExit(
        "Could not find the dataset. Looked in:\n  "
        + "\n  ".join(str(p) for p in C.DATASET_CANDIDATES)
        + "\nEdit DATASET_CANDIDATES in config.py to point at your copy."
    )


def _read(root, layout):
    if layout == "flat":
        lab = pd.read_csv(root / "labels.csv")
        feat = pd.read_csv(root / "features.csv")
        return lab.merge(feat, on=["sample_id", "split"], validate="one_to_one")
    frames = []
    for split in C.SPLITS:
        d = root / C.SUBMISSION_SPLIT_DIRS[split] if layout == "submission" \
            else root / f"data_{split}"
        lab = pd.read_csv(d / "labels.csv")
        feat = pd.read_csv(d / "features.csv")
        frames.append(lab.merge(feat, on=["sample_id", "split"], validate="one_to_one"))
    return pd.concat(frames, ignore_index=True)


def load():
    """
    Returns (df, splits) where splits maps split name -> dict with
    X (n, 14) float64, Y (n, 4) int, and meta (the DataFrame rows).
    """
    root, layout = find_dataset()
    df = _read(root, layout)

    fcols = [f"f_{k}" for k in C.FEATURE_NAMES]
    missing = [c for c in fcols if c not in df.columns]
    if missing:
        raise SystemExit(f"dataset is missing feature columns: {missing}")

    splits = {}
    for split in C.SPLITS:
        sub = df[df["split"] == split].reset_index(drop=True)
        if sub.empty:
            continue
        X = np.nan_to_num(sub[fcols].to_numpy(dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        Y = sub[[f"y_{a}" for a in C.ARTIFACTS]].to_numpy(dtype=int)
        splits[split] = dict(X=X, Y=Y, meta=sub)
    return df, splits, root, layout


def describe(splits):
    lines = []
    for name, s in splits.items():
        pos = s["Y"].sum(axis=0)
        lines.append(
            f"  {name:22s} n={len(s['Y']):5d}  phantoms={s['meta']['phantom_id'].nunique():3d}  "
            f"positives=" + "/".join(str(int(p)) for p in pos)
        )
    return "\n".join(lines)
