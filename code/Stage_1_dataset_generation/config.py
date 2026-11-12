"""
Project 29 - Multi-Artifact Detection and Classification
Single source of truth for every generation knob.

Every number here is a deliberate design decision; the five resolved
specification defects are marked [D1]..[D5] where they bite.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
MASKS = DATA / "masks"

# ---------------------------------------------------------------- reproducibility
MASTER_SEED = 20261008  # guidelines S9: fix and document the seed

# ---------------------------------------------------------------- geometry
N_PIX = 256  # reconstruction grid (spec item 10)
FOV_CM = 12.0  # field of view across N_PIX
PX_CM = FOV_CM / N_PIX  # cm per pixel
N_ANGLES = 360
ANGLE_RANGE_DEG = 180.0  # parallel beam, 180 deg is a full data set
I0 = 2.0e5  # nominal incident photons per detector bin


I0_RANGE = (4.0e4, 8.0e5)
I0_CALIBRATION = 4.0e4


# ---------------------------------------------------------------- spectrum [D4]
#
E_GRID_KEV = [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0]  # tabulation grid
E_BINS_KEV = [20.0, 30.0, 45.0, 60.0, 75.0, 90.0, 110.0]  # simulation bins
E_EFF_KEV = 60.0  # monochromatic reference energy
KVP = 120.0  # tube potential for the Kramers spectrum

# Al filtration in mm maps severity -> beam softness.
# Thick filter = hard beam = weak cupping. Thin filter = soft beam = strong cupping.
BH_FILT_MM_WEAK = 2.0
BH_FILT_MM_STRONG = 0.10

# ---------------------------------------------------------------- phantoms
N_PHANTOMS = 400
OBJ_RADIUS_PX = (72, 104)  # outer radius range
N_HARMONICS = 5  # radial harmonics
HARMONIC_AMP = 0.14
N_INCLUSIONS = (1, 5)  # further inclusions beyond the guaranteed aluminium one
INCL_RADIUS_PX = (8, 26)
ALU_RADIUS_PX = (12, 28)  # the one guaranteed aluminium part

# ---------------------------------------------------------------- corruption ranges

RING_N_CHANNELS = (4, 18)
RING_GAIN_PCT = (2.5, 14.0)

METAL_TAU_MAX = (12.0, 18.0)  # opacity at severity s=1, per phantom
METAL_DENSITY_SCALE = (0.25, 1.0)
METAL_RADIUS_LIMIT_PX = 26  # hard cap when fitting the insert into the body
SCATTER_SPR = (0.05, 0.40)
SCATTER_SIGMA_DET = 70.0
SCATTER_SIGMA_ANG = 3.0

# ---------------------------------------------------------------- severity bands [D2]

S_TRAIN_BAND = (0.20, 0.80)
S_OOD_BANDS = ((0.00, 0.20), (0.80, 1.00))

# ---------------------------------------------------------------- dataset plan [D1]

CONFIGS_PER_PHANTOM = 6
QUOTA_BY_N_ARTIFACTS = {0: 300, 1: 600, 2: 720, 3: 480, 4: 300}
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
N_OOD_CONFIGS_PER_TEST_PHANTOM = 6

ARTIFACTS = ("beam_hardening", "ring", "metal", "scatter")
