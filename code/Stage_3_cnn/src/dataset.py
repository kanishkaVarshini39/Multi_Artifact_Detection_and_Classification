"""Dataset: loads the .npy slices, applies a normalisation regime and augmentation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

_RAW_CACHE = {}


def find_dataset():
    for cand in C.DATASET_CANDIDATES:
        if (cand / "training data" / "train" / "labels.csv").exists():
            return cand, "submission"
        if (cand / "labels.csv").exists():
            return cand, "flat"
        if (cand / "data_train" / "labels.csv").exists():
            return cand, "package"
    raise SystemExit("Could not find the dataset; edit DATASET_CANDIDATES in config.py")


def load_table():
    root, layout = find_dataset()
    if layout == "flat":
        df = pd.read_csv(root / "labels.csv")
        df["abs_path"] = df["image_path"].map(lambda p: str(root / p))
    else:
        frames = []
        for split in ("train", "val", "test", "test_ood_severity"):
            d = root / C.SUBMISSION_SPLIT_DIRS[split] if layout == "submission" \
                else root / f"data_{split}"
            sub = pd.read_csv(d / "labels.csv")
            sub["abs_path"] = sub["image_path"].map(lambda p: str(d / p))
            frames.append(sub)
        df = pd.concat(frames, ignore_index=True)
    return df, root, layout


def _window_max():
    """The fixed physical window used by the dataset's own preprocess.normalise."""
    # The window is a property of the shipped dataset, so it is recorded
    # beside it.  Reading preprocess.py directly from here cannot be relied on:
    # it does `import config`, which resolves to THIS stage's config.
    for cand in C.DATASET_CANDIDATES:
        rec = cand / "metadata" / "preprocess_window.json"
        if rec.exists():
            import json
            return float(json.loads(rec.read_text())["window_mu_max"])
    srcs = [C.ROOT.parent / "Stage_1_dataset_generation" / "src"]
    for cand in C.DATASET_CANDIDATES:
        srcs.append(cand.parent / "src" if (cand / "labels.csv").exists()
                    else cand / "code" / "src")
    for src in srcs:
        if (src / "preprocess.py").exists():
            import importlib.util
            sys.path.insert(0, str(src))
            spec = importlib.util.spec_from_file_location("ds_preprocess", src / "preprocess.py")
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                return float(mod.WINDOW_MU_MAX)
            except Exception:
                continue
    # fallback: 1.10 * mu_aluminium at 60 keV in per-pixel units
    return 0.03866


WINDOW_MU_MAX = _window_max()


def load_raw(paths, verbose=True):
    """Cache raw images in RAM once, plus the per-image percentiles."""
    key = (len(paths), paths[0], paths[-1])
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    if verbose:
        print(f"    caching {len(paths)} images ...", end="", flush=True)
    imgs = np.empty((len(paths), C.N_PIX, C.N_PIX), dtype=np.float32)
    for i, p in enumerate(paths):
        imgs[i] = np.load(p)
    lo = np.percentile(imgs.reshape(len(paths), -1), 1.0, axis=1).astype(np.float32)
    hi = np.percentile(imgs.reshape(len(paths), -1), 97.0, axis=1).astype(np.float32)
    _RAW_CACHE[key] = (imgs, lo, hi)
    if verbose:
        print(f" {imgs.nbytes / 1e6:.0f} MB")
    return imgs, lo, hi


def normalise_batch(x, lo, hi, regime):
    """x: (n, H, W) float32 raw attenuation -> [0, 1]."""
    if regime == "fixed":
        return np.clip(x / WINDOW_MU_MAX, 0.0, 1.0)
    span = np.maximum(hi - lo, 1e-9)[:, None, None]
    return np.clip((x - lo[:, None, None]) / span, 0.0, 1.0)


class SliceDataset(Dataset):
    """
    Returns (image 1xHxW, labels 4, severities 4).

    Augmentation is label-preserving by construction: rotation about the image
    centre and flips leave every artifact's identity intact.  Translation is
    deliberately unavailable (see config).
    """

    def __init__(self, df, regime, train=False, seed=C.SEED):
        self.df = df.reset_index(drop=True)
        self.regime = regime
        self.train = train
        self.rng = np.random.default_rng(seed)
        imgs, lo, hi = load_raw(self.df["abs_path"].tolist())
        self.x = normalise_batch(imgs, lo, hi, regime)
        self.y = self.df[[f"y_{a}" for a in C.ARTIFACTS]].to_numpy(np.float32)
        self.sev = self.df[[f"sev_{a}" for a in C.ARTIFACTS]].to_numpy(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x[i]).unsqueeze(0)
        if self.train:
            if C.AUG_HFLIP and self.rng.random() < 0.5:
                img = torch.flip(img, dims=[2])
            if C.AUG_VFLIP and self.rng.random() < 0.5:
                img = torch.flip(img, dims=[1])
            if C.AUG_ROTATE_DEG > 0:
                import torchvision.transforms.functional as TF
                angle = float(self.rng.uniform(-C.AUG_ROTATE_DEG, C.AUG_ROTATE_DEG))
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            if C.AUG_INTENSITY_JITTER > 0:
                img = torch.clamp(img * (1.0 + float(self.rng.normal(
                    0, C.AUG_INTENSITY_JITTER))), 0.0, 1.0)
        return img, torch.from_numpy(self.y[i]), torch.from_numpy(self.sev[i])


def make_splits(regime, seed=C.SEED):
    df, root, layout = load_table()
    out = {}
    for split in ("train", "val", "test", "test_ood_severity"):
        sub = df[df["split"] == split]
        if len(sub):
            out[split] = SliceDataset(sub, regime, train=(split == "train"), seed=seed)
    return out, root, layout
