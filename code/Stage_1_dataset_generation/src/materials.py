"""
Material attenuation table.  [D4]

The phantom is an INTEGER MATERIAL MAP, not a greyscale image.  Beam hardening
exists only because mu depends on energy differently for different materials,
so the generator must be able to ask "what is mu of material m at energy E".

mu/rho values are approximate NIST XCOM / ASTAR-grade tabulations (cm^2/g) on
config.E_GRID_KEV, log-log interpolated to the simulation bins.  They are
reference-range values, not exact XCOM output: good enough for the guidelines'
requirement to "keep a physically meaningful attenuation scale" (S9), and
documented as approximate in the README.
"""
import numpy as np
import config as C

AIR, PMMA, PE, ALU, TITANIUM, STEEL = 0, 1, 2, 3, 4, 5

NAMES = {AIR: "air", PMMA: "pmma", PE: "polyethylene", ALU: "aluminium",
         TITANIUM: "titanium", STEEL: "steel"}
DENSITY = {AIR: 0.001205, PMMA: 1.19, PE: 0.94, ALU: 2.699,
           TITANIUM: 4.54, STEEL: 7.874}

# mu/rho [cm^2/g] at E = 20, 30, 40, 50, 60, 80, 100, 120 keV
MU_OVER_RHO = {
    AIR:      [0.7779, 0.3538, 0.2485, 0.2080, 0.1875, 0.1662, 0.1541, 0.1458],
    PMMA:     [0.5789, 0.3173, 0.2397, 0.2088, 0.1927, 0.1751, 0.1641, 0.1564],
    PE:       [0.4342, 0.2777, 0.2251, 0.2009, 0.1866, 0.1710, 0.1606, 0.1533],
    ALU:      [3.4410, 1.1280, 0.5685, 0.3681, 0.2778, 0.2018, 0.1704, 0.1536],
    TITANIUM: [11.210, 3.6000, 1.7050, 0.9891, 0.6625, 0.3741, 0.2621, 0.2093],
    STEEL:    [25.680, 8.1760, 3.6290, 1.9580, 1.2050, 0.5952, 0.3717, 0.2778],
}
# water, used only for the beam-hardening water calibration
WATER_MU_OVER_RHO = [0.8096, 0.3756, 0.2683, 0.2269, 0.2059, 0.1837, 0.1707, 0.1614]
WATER_RHO = 1.0

# Base is a POLYMER HOUSING, with aluminium demoted to an inclusion material.
# Reason (measured, not assumed): a detector gain error produces the same
# ABSOLUTE ring amplitude whatever the object, so against an aluminium body
# (mu ~3x polymer) the same fault is ~3x less visible.  With aluminium bodies in
# the pool, ring detectability split cleanly by base material -- 5/5 polymer
# phantoms detectable at full severity versus 5/9 aluminium -- which would have
# made the ring severity label mean different things on different phantoms.
# Polymer housing + metal/air inclusions is also the canonical industrial-CT
# sample.  Shape, inclusion layout and insert size still carry the diversity.
BASE_MATERIALS = (PMMA, PE)
INCLUSION_MATERIALS = (AIR, PMMA, PE, ALU)
# Titanium is retained in the table but excluded from the insert pool: at 60 keV
# it needs a ~40 px radius to reach starving opacity, which would dominate the
# phantom.  Steel is the realistic industrial choice at this scale.
METAL_MATERIALS = (STEEL,)


def _loglog_interp(e_query, e_grid, values):
    """Interpolate mu/rho in log-log space (the physically correct regime away from edges)."""
    return np.exp(np.interp(np.log(e_query), np.log(e_grid), np.log(values)))


def mu_per_pixel(material, energies_kev):
    """Linear attenuation coefficient in units of 'per pixel' at each energy."""
    e = np.atleast_1d(np.asarray(energies_kev, float))
    mu_rho = _loglog_interp(e, C.E_GRID_KEV, MU_OVER_RHO[material])
    return mu_rho * DENSITY[material] * C.PX_CM


def mu_water_per_pixel(energies_kev):
    e = np.atleast_1d(np.asarray(energies_kev, float))
    mu_rho = _loglog_interp(e, C.E_GRID_KEV, WATER_MU_OVER_RHO)
    return mu_rho * WATER_RHO * C.PX_CM


def mu_table(energies_kev):
    """{material_id: array over energies} in per-pixel units."""
    return {m: mu_per_pixel(m, energies_kev) for m in NAMES}
