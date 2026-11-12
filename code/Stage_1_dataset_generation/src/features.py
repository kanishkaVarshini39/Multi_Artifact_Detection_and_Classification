"""
Handcrafted image features.  [D3]

CRITICAL: every function here measures the RECONSTRUCTED IMAGE.  None of them
is ever used to build a severity label.  Severity labels come only from the
injection parameters (see labels.severity_from_params).  Keeping the two apart
is what stops the classical Random Forest baseline from being handed its own
target, which would make the spec's CNN-vs-classical comparison meaningless.

These features serve two jobs:
  1. inputs to the classical baseline (spec items 10, 19)
  2. detectability indices for the severity-floor calibration (spec defect D2)
"""
import numpy as np
from scipy import ndimage

import config as C

FEATURE_NAMES = [
    "cupping_index", "radial_hp_std", "radial_hp_max", "radial_hp_kurtosis",
    "ring_score", "ring_rms", "n_bright_blobs", "bright_area_frac", "peak_ratio",
    "streak_index", "bg_level", "obj_mean", "obj_std", "edge_centre_ratio",
]


def _object_mask(img):
    """Crude but stable segmentation of the sample from the background."""
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    c = n / 2.0 - 0.5
    inside = np.hypot(yy - c, xx - c) < (n / 2.0 - 3)
    vals = img[inside]
    thr = 0.5 * (np.percentile(vals, 2) + np.percentile(vals, 85))
    mask = (img > thr) & inside
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))
    mask = ndimage.binary_fill_holes(mask)
    if mask.sum() < 200:
        mask = inside
    return mask, inside


def _polar(arr, n_r=128, n_theta=360):
    """Resample onto a (radius, angle) grid about the rotation centre."""
    n = arr.shape[0]
    c = n / 2.0 - 0.5
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    rr = np.linspace(0.0, n / 2.0 - 2.0, n_r)
    R, T = np.meshgrid(rr, th, indexing="ij")
    yy = c + R * np.sin(T)
    xx = c + R * np.cos(T)
    return ndimage.map_coordinates(arr.astype(float), [yy, xx], order=1, mode="nearest")


def _ring_score(img, mask, scale):
    """
    Angular-coherence ring detector.

    A ring is a thin radial feature present at EVERY angle.  An object edge or
    an inclusion is a thin radial feature present at only SOME angles.  So:
    high-pass along radius, then average over angle -- rings survive the
    average, localised structure is diluted, Poisson noise cancels.

    Returns (peak coherent deviation, coherent RMS), both in units of the
    object's mean level.

    Two earlier attempts failed and are worth recording: a plain radial-profile
    std was dominated by the object BOUNDARY, and dividing the peak by a robust
    MAD cancelled the signal, because with up to 14 rings the MAD is inflated by
    the rings themselves.  The un-normalised coherent amplitude separates
    cleanly (roughly 7x clean at full severity).
    """
    pol = _polar(img) / scale
    polm = _polar(mask.astype(float))
    hp = pol - ndimage.uniform_filter1d(pol, size=7, axis=0, mode="nearest")
    valid = polm > 0.5
    coverage = valid.mean(axis=1)
    use = coverage > 0.70                      # radii lying mostly inside the object
    if use.sum() < 16:
        return 0.0, 0.0
    num = (hp * valid).sum(axis=1)
    den = np.maximum(valid.sum(axis=1), 1.0)
    m = (num / den)[use]
    centred = m - np.median(m)
    return float(np.abs(centred).max()), float(np.sqrt((centred ** 2).mean()))


def _radial_profile(img, mask):
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    c = n / 2.0 - 0.5
    r = np.hypot(yy - c, xx - c).astype(int)
    nbins = n // 2
    r = np.clip(r, 0, nbins - 1)
    w = mask.astype(float)
    num = np.bincount(r.ravel(), weights=(img * w).ravel(), minlength=nbins)
    den = np.bincount(r.ravel(), weights=w.ravel(), minlength=nbins)
    prof = np.where(den > 12, num / np.maximum(den, 1e-9), np.nan)
    return prof


def compute(img):
    """Returns a dict of the 12 handcrafted features."""
    mask, inside = _object_mask(img)
    obj = img[mask]
    bg = img[inside & ~mask]
    obj_mean = float(obj.mean()) if obj.size else 0.0
    obj_std = float(obj.std()) if obj.size else 0.0
    scale = max(abs(obj_mean), 1e-6)

    # --- cupping: inner disc of the object vs its outer shell (beam hardening / scatter)
    dist = ndimage.distance_transform_edt(mask)
    dmax = dist.max() if dist.size else 1.0
    core = mask & (dist > 0.62 * dmax)
    shell = mask & (dist > 0.12 * dmax) & (dist < 0.38 * dmax)
    if core.sum() > 40 and shell.sum() > 40:
        cupping = float((img[shell].mean() - img[core].mean()) / scale)
    else:
        cupping = 0.0

    # --- radial-profile oscillation (rings)
    prof = _radial_profile(img, mask)
    valid = np.isfinite(prof)
    if valid.sum() > 40:
        p = prof[valid]
        smooth = ndimage.uniform_filter1d(p, size=7, mode="nearest")
        hp = (p - smooth) / scale
        hp_std = float(hp.std())
        hp_max = float(np.abs(hp).max())
        hp_kurt = float(((hp - hp.mean()) ** 4).mean() / max(hp.std() ** 4, 1e-18))
    else:
        hp_std = hp_max = hp_kurt = 0.0
    ring_score, ring_rms = _ring_score(img, mask, scale)

    # --- bright compact regions (metal)
    hi_thr = obj_mean + 6.0 * max(obj_std, 1e-6)
    bright = (img > hi_thr) & mask
    lab, nlab = ndimage.label(bright)
    if nlab:
        sizes = np.bincount(lab.ravel())[1:]
        keep = int((sizes >= 6).sum())
    else:
        keep = 0
    bright_frac = float(bright.sum() / max(mask.sum(), 1))
    peak_ratio = float(img.max() / scale)

    # --- streaks: angular high-frequency energy in the object (metal)
    gy, gx = np.gradient(img)
    grad = np.hypot(gy, gx)
    streak = float(np.percentile(grad[mask], 99.5) / scale) if mask.sum() else 0.0

    # --- background and shape of the profile (scatter raises the floor)
    bg_level = float(bg.mean() / scale) if bg.size else 0.0
    if valid.sum() > 40:
        p = prof[valid]
        k = max(len(p) // 5, 2)
        edge_centre = float((p[-k:].mean() - p[:k].mean()) / scale)
    else:
        edge_centre = 0.0

    return dict(cupping_index=cupping, radial_hp_std=hp_std, radial_hp_max=hp_max,
                radial_hp_kurtosis=hp_kurt, ring_score=ring_score, ring_rms=ring_rms,
                n_bright_blobs=float(keep),
                bright_area_frac=bright_frac, peak_ratio=peak_ratio,
                streak_index=streak, bg_level=bg_level, obj_mean=obj_mean,
                obj_std=obj_std, edge_centre_ratio=edge_centre)


def vector(img):
    f = compute(img)
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)


# Which feature is the primary DETECTABILITY index for each artifact.
# Used only by the floor calibration, never as a label.
DETECTION_INDEX = {
    "beam_hardening": "cupping_index",
    "ring": "ring_score",
    "metal": "peak_ratio",
    "scatter": "cupping_index",
}
