"""
The simulation pipeline: material map -> sinogram -> corruptions -> FBP image.

KEY OPTIMISATION (and a correctness win).  radon() is LINEAR, so

    radon(mu_E image) = sum_m  mu(m, E) * radon(mask_m)

We therefore forward-project one binary mask per material ONCE per phantom and
rebuild any energy's sinogram as a cheap linear combination.  All 6 energy bins
and all 6 configs of a phantom reuse the same cache, and the monochromatic and
polyenergetic paths are guaranteed to come from one consistent physical model.

PHYSICAL ORDERING of the corruption operators (this is not cosmetic):
  metal           acts on the OBJECT          -> changes which mu is used
  beam hardening  acts on BEAM FORMATION      -> changes how projections combine
  scatter         adds photons AT THE DETECTOR -> added to intensity
  ring            is a DETECTOR GAIN error    -> applied after scatter arrives
  Poisson noise   is DETECTION                -> last
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import radon, iradon

import config as C
import materials as M
from spectrum import spectrum_weights, water_calibration_gain

THETA = np.linspace(0.0, C.ANGLE_RANGE_DEG, C.N_ANGLES, endpoint=False)


# ----------------------------------------------------------------- projection cache
def build_projection_cache(phantom):
    """
    Forward-project one mask per material (metal site excluded) plus the metal
    site itself.  Returns {'mats': {mat_id: sinogram}, 'metal': sinogram,
    'host': mat_id}.  ~3-5 radon calls per phantom, reused by every config.
    """
    lab, metal = phantom["label_map"], phantom["metal_mask"]
    cache = {"mats": {}, "host": phantom["metal_host"]}
    for mat in np.unique(lab):
        if mat == M.AIR:                      # mu_air is ~1e-4 of the others: skip
            continue
        mask = (lab == mat) & ~metal
        if mask.any():
            cache["mats"][int(mat)] = radon(mask.astype(np.float32), theta=THETA, circle=True)
    cache["metal"] = radon(metal.astype(np.float32), theta=THETA, circle=True)
    return cache


def line_integrals(cache, energy_kev, metal_material=None, metal_scale=1.0):
    """Sinogram of line integrals at one energy. metal_material=None -> metal OFF."""
    p = None
    for mat, sino in cache["mats"].items():
        mu = float(M.mu_per_pixel(mat, energy_kev)[0])
        p = mu * sino if p is None else p + mu * sino
    if p is None:
        p = np.zeros_like(cache["metal"])
    if metal_material is None:
        mu_site = float(M.mu_per_pixel(cache["host"], energy_kev)[0])
    else:
        mu_site = float(M.mu_per_pixel(metal_material, energy_kev)[0]) * metal_scale
    return p + mu_site * cache["metal"]


# ----------------------------------------------------------------- corruptions
def form_projection(cache, labels, sev, phantom, rng):
    """
    Build the detector intensity leaving the object.
    Handles METAL (object-domain) and BEAM HARDENING (beam-formation domain).
    """
    metal_mat = phantom["metal_material"] if labels["metal"] else None
    metal_scale = 1.0
    if labels["metal"]:
        lo, hi = C.METAL_DENSITY_SCALE
        metal_scale = lo + sev["metal"] * (hi - lo)

    if labels["beam_hardening"]:
        w = spectrum_weights(sev["beam_hardening"])
        intensity = None
        for wi, e in zip(w, C.E_BINS_KEV):
            p_e = line_integrals(cache, e, metal_mat, metal_scale)
            term = wi * np.exp(-p_e)
            intensity = term if intensity is None else intensity + term
        p = -np.log(np.clip(intensity, 1e-12, None))
        p *= water_calibration_gain(sev["beam_hardening"])   # removes the brightness cue
    else:
        p = line_integrals(cache, C.E_EFF_KEV, metal_mat, metal_scale)

    return np.exp(-p), metal_scale


def apply_scatter(intensity, severity):
    """
    Convolution-kernel scatter (spec item 8, guidelines S2.4).

    The scatter SOURCE is (1 - I): scatter is generated where photons are
    actually removed from the primary beam, i.e. inside the object, and falls
    off in air.  Convolving that source with a broad kernel gives a smooth,
    low-frequency field that is nearly FLAT across the object while the primary
    I varies strongly with path length.

    That flatness is the whole point.  Adding a flat S to a varying I boosts
    the thick (small-I) central rays proportionally more than the thin edge
    rays, so -log(I + S) is depressed most at the centre -- cupping.  An earlier
    version blurred I itself, which made S track I, produced a near-uniform
    offset instead of cupping, and would have made the beam-hardening/scatter
    confusion pair (required experiment 3) artificially easy to separate.

    SPR is renormalised to mean what it says inside the object.
    """
    lo, hi = C.SCATTER_SPR
    spr = lo + severity * (hi - lo)
    source = np.clip(1.0 - intensity, 0.0, None)          # scatter is born in the object
    field = gaussian_filter(source, sigma=(C.SCATTER_SIGMA_DET, C.SCATTER_SIGMA_ANG),
                            mode="nearest")
    obj = intensity < 0.85                                 # rays that cross the object
    if obj.sum() < 32 or field[obj].mean() <= 0:
        return intensity, 0.0
    gain = spr * float(intensity[obj].mean()) / float(field[obj].mean())
    return intensity + gain * field, spr


def ring_gains(n_det, severity, rng, r_max):
    """
    Per-detector-channel gain error, constant across every view angle.

    Faulty channels are drawn so that the ring RADIUS they produce falls inside
    the sample (a detector channel at offset d from the centre reconstructs to
    a ring of radius d).  A fault at the detector edge makes a ring in empty
    air, which is not an artifact OF THE OBJECT and is invisible to any
    object-based detector -- in calibration that alone held ring detectability
    down to 75%.  Restricting the faults to the channels that actually image
    the sample is both physically reasonable and what CT QA cares about.
    """
    n_lo, n_hi = C.RING_N_CHANNELS
    g_lo, g_hi = C.RING_GAIN_PCT
    n_bad = int(round(n_lo + severity * (n_hi - n_lo)))
    amp = (g_lo + severity * (g_hi - g_lo)) / 100.0

    centre = (n_det - 1) / 2.0
    r_max = float(np.clip(r_max, 14.0, centre - 4.0))
    radii = rng.uniform(6.0, r_max, size=n_bad * 4)
    signs = rng.choice([-1.0, 1.0], size=radii.size)
    cand = np.round(centre + signs * radii).astype(int)
    cand = np.clip(cand, 2, n_det - 3)
    uniq = np.array(sorted(set(cand.tolist())))
    rng.shuffle(uniq)                     # otherwise [:n_bad] takes one side only
    idx = uniq[:n_bad]
    if idx.size == 0:
        idx = np.array([int(centre)])

    gains = np.ones(n_det)
    per_channel = amp * rng.uniform(0.6, 1.0, size=idx.size) * rng.choice([-1.0, 1.0], size=idx.size)
    gains[idx] = 1.0 + per_channel
    return gains, idx, amp, int(idx.size)


def detect(intensity, rng, i0=None):
    """Poisson photon detection. Starved rays (behind metal) become wildly noisy."""
    i0 = float(C.I0 if i0 is None else i0)
    counts = rng.poisson(np.clip(i0 * intensity, 0.0, None))
    counts = np.maximum(counts, 1.0)              # log of zero is the starvation mechanism
    return -np.log(counts / i0)


def reconstruct(sinogram):
    return iradon(sinogram, theta=THETA, circle=True, filter_name="ramp",
                  output_size=C.N_PIX).astype(np.float32)


# ----------------------------------------------------------------- one sample
def simulate(phantom, cache, labels, sev, rng, i0=None):
    """
    labels: {artifact_name: bool}       <- this IS the ground-truth vector
    sev:    {artifact_name: float}      <- normalised severity in [0,1]
    i0:     incident photon count for this sample (dose nuisance variable)
    Returns (image, params, masks)
    """
    i0 = float(C.I0 if i0 is None else i0)
    intensity, metal_scale = form_projection(cache, labels, sev, phantom, rng)
    params = {"metal_scale": float(metal_scale), "i0": i0}

    if labels["scatter"]:
        intensity, spr = apply_scatter(intensity, sev["scatter"])
        params["scatter_spr"] = float(spr)

    masks = {}
    if labels["ring"]:
        r_eff = float(np.sqrt(max(phantom["meta"]["body_area_px"], 1) / np.pi))
        gains, idx, amp, n_bad = ring_gains(intensity.shape[0], sev["ring"], rng,
                                            r_max=0.88 * r_eff)
        intensity = intensity * gains[:, None]
        params.update(ring_gain_amp=float(amp), ring_n_channels=int(n_bad),
                      ring_channels=sorted(int(i) for i in idx))
        masks["ring"] = ring_mask_from_channels(idx)

    if labels["beam_hardening"]:
        from spectrum import filtration_mm, mean_energy_kev
        params.update(bh_filtration_mm=float(filtration_mm(sev["beam_hardening"])),
                      bh_mean_energy_kev=float(mean_energy_kev(sev["beam_hardening"])))
    if labels["metal"]:
        masks["metal"] = phantom["metal_mask"]
        params.update(metal_material=M.NAMES[phantom["metal_material"]],
                      metal_area_px=int(phantom["metal_mask"].sum()))

    sinogram = detect(intensity, rng, i0=i0)
    image = reconstruct(sinogram)
    return image, params, masks


def ring_mask_from_channels(channels):
    """
    Free localisation mask.  A detector channel at offset d from the centre maps
    to a ring of radius d in the reconstruction (exact for parallel beam).
    """
    n = C.N_PIX
    yy, xx = np.mgrid[0:n, 0:n]
    c = n / 2.0 - 0.5
    r = np.hypot(yy - c, xx - c)
    mask = np.zeros((n, n), bool)
    for ch in np.atleast_1d(channels):
        radius = abs(float(ch) - c)
        mask |= np.abs(r - radius) <= 1.5
    return mask
