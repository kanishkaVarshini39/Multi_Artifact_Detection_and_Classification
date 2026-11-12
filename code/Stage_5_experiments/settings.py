"""
Stage D - the five required experiments (spec item 18).

Named settings.py, not config.py: Stage D imports Stage B's modules as a
library and Stage B's dataset.py does `import config`.  A config.py here could
win the sys.modules race and hand Stage B the wrong configuration.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STAGE_A = ROOT.parent / "Stage_2_classical_baseline"
STAGE_B = ROOT.parent / "Stage_3_cnn"
STAGE_C = ROOT.parent / "Stage_4_localisation"

# Experiment 1 uses the same configuration as the chosen Stage B model, so that
# training COMPOSITION is the only thing that changes.
REGIME = "per_image"
INIT = "pretrained"
REFERENCE_CHECKPOINT = "resnet18_per_image_pretrained.pt"

# Arm C (combined, full) is the Stage B model itself; arms A and B are new.
# 630 = the number of training samples with at most one artifact
# (210 clean + 4 x 105 singles), so arm B is size-matched to arm A.
MATCHED_SIZE = 630

SEED = 20261021
