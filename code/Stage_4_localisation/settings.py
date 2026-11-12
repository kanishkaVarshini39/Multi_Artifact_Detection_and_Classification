"""
Stage C - Grad-CAM and localisation.  Settings.

Named settings.py, NOT config.py, on purpose: Stage C imports Stage B's modules
as a library, and Stage B's dataset.py does `import config`.  If Stage C also
had a config.py it could win the sys.modules race and silently hand Stage B the
wrong configuration.  (Stage B hit the same class of bug with plots.py.)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STAGE_B = ROOT.parent / "Stage_3_cnn"

# Which trained model to explain.  per_image/pretrained is the right default:
# equally accurate as the fixed-window arm but demonstrably NOT reading global
# brightness, so its heat-maps should show real artifact structure.
CHECKPOINT = "resnet18_per_image_pretrained.pt"

# Grad-CAM target layers.  ResNet-18 on a 256x256 input gives
#   layer4 -> 8x8   (standard Grad-CAM choice, very coarse: 32 px per cell)
#   layer3 -> 16x16 (finer, better suited to small metal inserts and thin rings)
TARGET_LAYERS = ("layer4", "layer3")

# Ground-truth masks exist only for ring and metal.  Beam hardening and scatter
# are global artifacts with no localisable region - that asymmetry is itself a
# result, not a gap.
LOCALISABLE = ("ring", "metal")

# A ring mask is a 1.5 px annulus.  Grad-CAM at 8x8 cannot resolve that, so the
# mask is dilated by this many pixels before the overlap metrics.  The tolerance
# is reported, not hidden.
MASK_DILATE_PX = 4

CAM_THRESHOLD = 0.5        # fraction of the CAM maximum, for the IoU metric
BATCH_SIZE = 32
N_QUALITATIVE = 8          # example images in the overlay figure
SEED = 20261017
