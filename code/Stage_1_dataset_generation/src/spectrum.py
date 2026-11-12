"""
Polyenergetic tube spectrum and the beam-hardening severity dial.  [D4]

Kramers' law bremsstrahlung from a tungsten anode, hardened by t mm of
aluminium filtration.  Severity is controlled by the FILTRATION THICKNESS:
a thin filter leaves a soft (low mean energy) beam, which hardens strongly as
it crosses the object and therefore produces strong cupping.

The returned weights are photon-number weights per simulation bin, normalised
to sum to 1, so sum_E w_E exp(-p_E) is a detector intensity in [0, 1].
"""
import numpy as np
import config as C
from materials import mu_water_per_pixel, _loglog_interp, MU_OVER_RHO, DENSITY, ALU


def _mu_al_per_mm(energies_kev):
    """Aluminium linear attenuation coefficient per mm, for the filter."""
    mu_rho = _loglog_interp(np.asarray(energies_kev, float), C.E_GRID_KEV, MU_OVER_RHO[ALU])
    return mu_rho * DENSITY[ALU] * 0.1       # cm^-1 -> mm^-1


def filtration_mm(severity):
    """Map normalised severity s in [0,1] -> Al filter thickness (mm). s=1 is softest."""
    s = float(np.clip(severity, 0.0, 1.0))
    return C.BH_FILT_MM_WEAK + s * (C.BH_FILT_MM_STRONG - C.BH_FILT_MM_WEAK)


def spectrum_weights(severity, energies_kev=None):
    """Photon-number weights per energy bin, normalised to sum to 1."""
    e = np.asarray(C.E_BINS_KEV if energies_kev is None else energies_kev, float)
    t_mm = filtration_mm(severity)
    # Kramers: photon number per unit energy ~ (kVp/E - 1)
    raw = np.clip(C.KVP / e - 1.0, 0.0, None)
    raw = raw * np.exp(-_mu_al_per_mm(e) * t_mm)       # filter hardening
    raw[e >= C.KVP] = 0.0
    total = raw.sum()
    if total <= 0:
        raise ValueError("degenerate spectrum")
    return raw / total


def mean_energy_kev(severity):
    w = spectrum_weights(severity)
    return float(np.dot(w, C.E_BINS_KEV))


def water_calibration_gain(severity):
    """
    First-order water linearisation gain.  [avoids a brightness shortcut]

    A polyenergetic scan reports systematically LOWER line integrals than a
    monochromatic scan at E_EFF, which would hand the classifier a trivial
    global-brightness cue for 'beam hardening present'.  Real scanners remove
    exactly this by water calibration.  We apply the same first-order fix:
    rescale so that a thin water path matches the monochromatic reference.
    What survives is the genuine NON-LINEAR cupping, which is the artifact.
    """
    w = spectrum_weights(severity)
    mu_w = mu_water_per_pixel(C.E_BINS_KEV)
    mu_bar = float(np.dot(w, mu_w))                       # thin-path limit
    mu_ref = float(mu_water_per_pixel(C.E_EFF_KEV)[0])
    return mu_ref / mu_bar
