"""
Preprocessing (spec item 10): "Normalise all slices to a common intensity range".

Two options are provided; the FIXED PHYSICAL WINDOW is the default.

normalise()  -- fixed window, identical for every image.
    Because this is synthetic data on a known attenuation scale, the correct
    analogue of clinical Hounsfield windowing is available: map a fixed mu
    range to [0, 1].  Deterministic, no per-image statistics, and metal simply
    saturates at 1.0 while the polymer body and aluminium parts keep their
    contrast.

normalise_per_image()  -- robust per-image percentile window.
    Removes the global brightness level, which both beam hardening and scatter
    depress.  Use it for the ablation "does the classifier rely on absolute
    level or on artifact shape?", which is worth a paragraph in the report.

    Its upper percentile is 97, not 99.5, and that choice is measured rather
    than guessed: a metal insert occupies 1-2% of the frame, so a
    99.5th-percentile window lands INSIDE the metal, squeezing the whole body
    into the bottom few percent of the range and flattening any co-occurring
    ring or cupping out of view.  Measured on metal+ring samples, the preserved
    contrast of the non-metal object is 0.058 at p99.5 versus 0.47 at p97.

Raw images are stored in physical units (per-pixel mu); normalisation happens
at load time, so nothing is destroyed on disk and either option stays available.
"""
import numpy as np

import config as C
import materials as M

# Aluminium at the reference energy is the brightest NON-metal material in the
# phantoms, so a window just above it spends the full range on the object and
# lets the steel insert saturate.
WINDOW_MU_MIN = 0.0
WINDOW_MU_MAX = 1.10 * float(M.mu_per_pixel(M.ALU, C.E_EFF_KEV)[0])


def normalise(img):
    """Fixed physical window -> [0, 1]. The default."""
    return np.clip((img - WINDOW_MU_MIN) / (WINDOW_MU_MAX - WINDOW_MU_MIN),
                   0.0, 1.0).astype(np.float32)


def normalise_per_image(img, lo_pct=1.0, hi_pct=97.0):
    """Robust per-image window -> [0, 1]. For the brightness-shortcut ablation."""
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi - lo < 1e-9:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
