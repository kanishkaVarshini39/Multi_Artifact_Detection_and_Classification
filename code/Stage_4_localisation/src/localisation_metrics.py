"""
Turning heat-maps into numbers.

The brief (item 5) only asks for Grad-CAM *visualisations*.  The dataset ships
1,932 ring and metal localisation masks, so the explanations can be scored
instead of merely admired -- which is worth more than a page of pretty overlays.

Three metrics where a mask exists:

  pointing game   does the single hottest pixel land inside the mask?
                  The standard weakly-supervised localisation check, and the
                  most forgiving of Grad-CAM's coarse resolution.
  IoU             overlap after thresholding the map at CAM_THRESHOLD of its
                  maximum.  Harsh here by construction: an 8x8 map upsampled to
                  256x256 cannot trace a 1.5 px ring, so read it as a floor.
  energy ratio    share of total heat inside the mask, divided by the mask's
                  share of the image area.  1.0 = heat spread at random,
                  >1 = concentrated on the artifact.  Resolution-tolerant, so
                  the most informative of the three.

And one metric that needs no mask at all:

  concentration   share of total heat in the hottest 10% of pixels.  Beam
                  hardening and scatter are GLOBAL artifacts with no region to
                  point at, so a well-behaved model should spread heat for them
                  and concentrate it for metal and rings.  Measuring that
                  difference turns "we have no mask" into a testable claim.
"""
import numpy as np
from scipy import ndimage


def dilate(mask, px):
    if px <= 0:
        return mask
    return ndimage.binary_dilation(mask, ndimage.generate_binary_structure(2, 2),
                                   iterations=int(px))


def pointing_hit(cam, mask):
    idx = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(mask[idx])


def iou(cam, mask, thresh_frac=0.5):
    binary = cam >= (thresh_frac * cam.max() if cam.max() > 0 else np.inf)
    union = np.logical_or(binary, mask).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(binary, mask).sum() / union)


def energy_ratio(cam, mask):
    total = cam.sum()
    area_frac = mask.mean()
    if total <= 0 or area_frac <= 0:
        return float("nan")
    return float((cam[mask].sum() / total) / area_frac)


def concentration(cam, top_frac=0.10):
    """Share of total heat held by the hottest `top_frac` of pixels."""
    total = cam.sum()
    if total <= 0:
        return float("nan")
    flat = np.sort(cam.ravel())[::-1]
    k = max(1, int(round(top_frac * flat.size)))
    return float(flat[:k].sum() / total)


EMPTY_EPS = 1e-8


def score_localisable(cams, masks, present, dilate_px=4, thresh_frac=0.5):
    """
    cams    (N, H, W) for one artifact
    masks   list of (H, W) bool or None
    present (N,) bool - whether that artifact is actually in the image
    """
    hits, ious, ratios, areas, n = [], [], [], [], 0
    for cam, mask, p in zip(cams, masks, present):
        if not p or mask is None or not mask.any():
            continue
        m = dilate(mask, dilate_px)
        hits.append(pointing_hit(cam, m))
        ious.append(iou(cam, m, thresh_frac))
        ratios.append(energy_ratio(cam, m))
        areas.append(float(m.mean()))
        n += 1
    if n == 0:
        return dict(n=0)
    area = float(np.mean(areas))
    pg = float(np.mean(hits))
    # CHANCE BASELINES.  A map whose peak falls in a uniformly random pixel hits
    # the mask at exactly the mask's area fraction, so pointing game and IoU are
    # uninterpretable without it: a mask covering 27% of the image scores 0.27
    # for free.  Lift = score / chance is the number to quote.
    return dict(n=n,
                mask_area_fraction=area,
                pointing_game=pg,
                pointing_chance=area,
                pointing_lift=pg / max(area, 1e-9),
                iou=float(np.nanmean(ious)),
                energy_ratio=float(np.nanmean(ratios)),
                energy_ratio_median=float(np.nanmedian(ratios)))


def score_concentration(cams, peaks, present, top_frac=0.10):
    """
    How focused is the heat, measured ONLY on non-empty maps where the artifact
    is actually present.

    Comparing present against absent does not work: for an absent artifact the
    map is usually exactly all-zero, and normalising that by its own maximum
    manufactures a spike, which scores as near-perfect "concentration".  What
    the absent case can report honestly is how often the map is empty --
    i.e. how often Grad-CAM correctly finds no evidence at all.
    """
    nonempty = peaks > EMPTY_EPS
    out = dict(top_frac=top_frac,
               empty_fraction_present=float(np.mean(~nonempty[present]))
               if present.sum() else float("nan"),
               empty_fraction_absent=float(np.mean(~nonempty[~present]))
               if (~present).sum() else float("nan"))
    sel = present & nonempty
    out["n_scored"] = int(sel.sum())
    if sel.sum():
        c = np.array([concentration(m, top_frac) for m in cams[sel]])
        out["concentration_present"] = float(np.nanmean(c))
    return out
