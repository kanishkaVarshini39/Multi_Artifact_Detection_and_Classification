"""
Stage B - Multi-label CNN.  Configuration.

The headline experiment is NOT simply "CNN vs baseline".  Stage A showed that
beam hardening and scatter can be detected from overall image brightness rather
than from their cupping signature, and that this shortcut HIDES the
beam-hardening/scatter confusion the brief asks you to analyse.  So every arm
is run under two normalisation regimes:

  fixed      preprocess.normalise          - keeps the global level (shortcut available)
  per_image  preprocess.normalise_per_image - removes it (the network must use shape)

The per_image regime is the honest test and where experiment 3 has substance.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STAGE_A = ROOT.parent / "Stage_2_classical_baseline"

DATASET_CANDIDATES = [
    ROOT.parent.parent,  # submission
    ROOT.parent / "Stage_1_dataset_generation" / "data",  # flat layout
    Path.home() / "Documents" / "Project29_MultiArtifact",  # package layout
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
N_PIX = 256

# ---------------------------------------------------------------- training (spec item 14)
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
LR_STEP = 30
LR_GAMMA = 0.3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 15  # on validation macro-F1
NUM_WORKERS = 0  # images are cached in RAM; workers add no value

# ---------------------------------------------------------------- augmentation

AUG_ROTATE_DEG = 180.0
AUG_HFLIP = True
AUG_VFLIP = True
AUG_TRANSLATE_PX = 0  # keep at 0 - see above
AUG_INTENSITY_JITTER = 0.0  # keep at 0 under the 'fixed' regime: it would
# erase the very level cue the study is about

# ---------------------------------------------------------------- severity head (optional)
SEVERITY_LOSS_WEIGHT = 0.5  # only used when --severity is passed

THRESHOLD_GRID = [round(0.02 * i, 2) for i in range(2, 50)]

# ---------------------------------------------------------------- experiment grid
REGIMES = ("fixed", "per_image")
INITS = ("pretrained", "scratch")
