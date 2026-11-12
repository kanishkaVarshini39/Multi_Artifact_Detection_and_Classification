"""
Ground-truth labels.  [D3]

The presence vector y in {0,1}^4 is simply a record of which operators the
generator was told to apply -- known by construction, never annotated.

Severity labels are functions of the INJECTION PARAMETERS ONLY.  They are
computed before the image exists and never touch the reconstruction.  The
handcrafted quantities in features.py (cupping index, radial-profile variance,
blob count) are a SEPARATE set, measured from the image, used only as the
classical baseline's inputs.  The two are correlated -- that is what makes the
baseline non-trivial -- but neither is derived from the other.
"""
import itertools
import numpy as np

import config as C
import materials as M

COMBINATIONS = [tuple(c) for c in itertools.product([0, 1], repeat=4)]   # all 16
assert len(COMBINATIONS) == 16


def combo_name(combo):
    on = [a for a, b in zip(C.ARTIFACTS, combo) if b]
    return "clean" if not on else "+".join(on)


# --- normalisers: the theoretical span of each physical parameter --------------
_RING_MAX = (C.RING_GAIN_PCT[1] / 100.0) * np.sqrt(C.RING_N_CHANNELS[1])
_METAL_MU_PX = float(M.mu_per_pixel(M.STEEL, C.E_EFF_KEV)[0])
_METAL_MAX = _METAL_MU_PX * 1.0 * (np.pi * C.METAL_RADIUS_LIMIT_PX ** 2)


def severity_from_params(artifact, params, phantom_meta):
    """
    Physical severity in [0,1], derived ONLY from what was injected.
    Returns 0.0 when the artifact is absent.
    """
    if artifact == "beam_hardening":
        t = params.get("bh_filtration_mm")
        if t is None:
            return 0.0
        lo, hi = C.BH_FILT_MM_WEAK, C.BH_FILT_MM_STRONG
        return float(np.clip((lo - t) / (lo - hi), 0.0, 1.0))

    if artifact == "ring":
        amp = params.get("ring_gain_amp")
        n = params.get("ring_n_channels")
        if amp is None:
            return 0.0
        return float(np.clip(amp * np.sqrt(n) / _RING_MAX, 0.0, 1.0))

    if artifact == "metal":
        scale = params.get("metal_scale")
        area = params.get("metal_area_px")
        if area is None:
            return 0.0
        # total attenuating metal presented to the beam
        return float(np.clip(_METAL_MU_PX * scale * area / _METAL_MAX, 0.0, 1.0))

    if artifact == "scatter":
        spr = params.get("scatter_spr")
        if spr is None:
            return 0.0
        lo, hi = C.SCATTER_SPR
        return float(np.clip((spr - lo) / (hi - lo), 0.0, 1.0))

    raise KeyError(artifact)


def severity_vector(labels, params, phantom_meta):
    return [severity_from_params(a, params, phantom_meta) if labels[a] else 0.0
            for a in C.ARTIFACTS]
