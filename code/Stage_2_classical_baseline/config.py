"""
Stage A - Classical baseline.  Configuration.

Stage A is purely tabular: it needs only labels.csv and features.csv, never the
.npy images.  That is why it runs in seconds on any machine.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


DATASET_CANDIDATES = [
    ROOT.parent.parent,  # submission
    ROOT.parent / "Stage_1_dataset_generation" / "data",  # flat
    Path.home() / "Documents" / "Project29_MultiArtifact",  # package
]

# 'submission' layout: split name -> folder, relative to the package root.
SUBMISSION_SPLIT_DIRS = {
    "train": Path("training data") / "train",
    "val": Path("training data") / "validation",
    "test": Path("testing data") / "test",
    "test_ood_severity": Path("testing data") / "held_out_severity",
}

SEED = 20261017

ARTIFACTS = ("beam_hardening", "ring", "metal", "scatter")

FEATURE_NAMES = [
    "cupping_index",
    "radial_hp_std",
    "radial_hp_max",
    "radial_hp_kurtosis",
    "ring_score",
    "ring_rms",
    "n_bright_blobs",
    "bright_area_frac",
    "peak_ratio",
    "streak_index",
    "bg_level",
    "obj_mean",
    "obj_std",
    "edge_centre_ratio",
]

SPLITS = ("train", "val", "test", "test_ood_severity")

LEVEL_FEATURES = ["obj_mean", "bg_level"]
SHAPE_FEATURES = [f for f in FEATURE_NAMES if f not in LEVEL_FEATURES]


RF_GRID = dict(
    n_estimators=[300, 600],
    max_depth=[None, 12, 20],
    min_samples_leaf=[1, 2, 4],
)
XGB_GRID = dict(
    n_estimators=[300, 600],
    max_depth=[3, 5, 7],
    learning_rate=[0.05, 0.10],
)
THRESHOLD_GRID = [round(0.02 * i, 2) for i in range(2, 50)]  # 0.04 .. 0.98
