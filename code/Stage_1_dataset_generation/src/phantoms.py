"""
Procedural phantom generation.  [D4]

Output is a MATERIAL LABEL MAP (integers), plus a fixed candidate metal site.

Why the metal site is fixed per phantom: the projection cache in pipeline.py
forward-projects one mask per material ONCE per phantom and then builds any
energy's sinogram as a linear combination (radon is linear).  That makes all 6
configs and all 6 energy bins essentially free.  Keeping the metal GEOMETRY
fixed per phantom preserves the cache; metal severity is then dialled through
the metal's attenuation (density/material), and the metal AREA still varies
across the 400 phantoms, so the dataset retains insert-size diversity.
"""
import numpy as np
import materials as M
import config as C


def _radial_blob(shape, rng, radius, centre, n_harm, amp):
    """Organic closed outline: r(theta) = R * (1 + sum_k a_k cos(k theta + phi_k))."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    dy, dx = yy - centre[0], xx - centre[1]
    r = np.hypot(dy, dx)
    th = np.arctan2(dy, dx)
    boundary = np.full(shape, float(radius))
    for k in range(2, 2 + n_harm):
        a = rng.uniform(-amp, amp) / (k - 1)
        phi = rng.uniform(0, 2 * np.pi)
        boundary = boundary * (1.0 + a * np.cos(k * th + phi))
    return r < boundary


def _ellipse(shape, centre, radii, angle_rad):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    dy, dx = yy - centre[0], xx - centre[1]
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    u = (dx * ca + dy * sa) / radii[0]
    v = (-dx * sa + dy * ca) / radii[1]
    return (u * u + v * v) < 1.0


def make_phantom(phantom_id, seed):
    """
    Returns a dict:
      label_map   int array, material id per pixel (metal NOT yet inserted)
      metal_mask  bool array, the fixed candidate metal site
      metal_host  material id originally occupying the metal site
      meta        dict of generation parameters (goes into the manifest)
    """
    rng = np.random.default_rng(seed)
    n = C.N_PIX
    shape = (n, n)
    centre = (n / 2.0 - 0.5, n / 2.0 - 0.5)

    radius = rng.integers(*C.OBJ_RADIUS_PX)
    base = int(rng.choice(M.BASE_MATERIALS))
    body = _radial_blob(shape, rng, radius, centre, C.N_HARMONICS, C.HARMONIC_AMP)

    # keep everything strictly inside the reconstruction circle
    yy, xx = np.mgrid[0:n, 0:n]
    inscribed = np.hypot(yy - centre[0], xx - centre[1]) < (n / 2.0 - 2)
    body &= inscribed

    label_map = np.full(shape, M.AIR, dtype=np.uint8)
    label_map[body] = base

    # --- inclusions: aluminium, air voids and a second polymer, placed inside
    # the body.  Aluminium is one of the choices (it gives the polyenergetic
    # beam something to harden against) but is not forced: forcing a large
    # aluminium fraction made the cupping index track inclusion layout rather
    # than the artifact, and raised the object mean enough to suppress rings.
    incl_log = []

    def _place(mat, r_range, tries=60):
        for _attempt in range(tries):
            rr = int(rng.integers(*r_range))
            ang = rng.uniform(0, np.pi)
            ar = rng.uniform(0.55, 1.0)
            cy = rng.integers(rr + 2, n - rr - 2)
            cx = rng.integers(rr + 2, n - rr - 2)
            el = _ellipse(shape, (cy, cx), (rr, max(3, int(rr * ar))), ang)
            if el.sum() and np.all(body[el]):       # must sit fully inside the body
                label_map[el] = mat
                incl_log.append(dict(material=M.NAMES[mat], cy=int(cy), cx=int(cx),
                                     r=rr, aspect=round(float(ar), 3)))
                return True
        return False

    _place(M.ALU, C.ALU_RADIUS_PX)                  # at least one aluminium part
    choices = [m for m in M.INCLUSION_MATERIALS if m != base]
    for _ in range(int(rng.integers(*C.N_INCLUSIONS))):
        _place(int(rng.choice(choices)), C.INCL_RADIUS_PX)

    # --- fixed candidate metal site, sized by OPACITY so it can actually starve
    metal_mat = int(rng.choice(M.METAL_MATERIALS))
    mu_metal_px = float(M.mu_per_pixel(metal_mat, C.E_EFF_KEV)[0])
    tau_max = rng.uniform(*C.METAL_TAU_MAX)
    metal_r = int(round(min(tau_max / (2.0 * mu_metal_px), C.METAL_RADIUS_LIMIT_PX)))

    metal_mask = np.zeros(shape, bool)
    # shrink the insert if no single-material site of this size exists in the body
    for radius_try in range(metal_r, 3, -1):
        placed = False
        for _attempt in range(120):
            cy = rng.integers(radius_try + 3, n - radius_try - 3)
            cx = rng.integers(radius_try + 3, n - radius_try - 3)
            cand = _ellipse(shape, (cy, cx), (radius_try, radius_try), 0.0)
            # must sit inside the body AND inside a single material, so that the
            # metal-OFF image is bit-for-bit the clean phantom (label purity)
            if cand.sum() and np.all(body[cand]) and np.unique(label_map[cand]).size == 1:
                metal_mask, metal_r, placed = cand, radius_try, True
                break
        if placed:
            break
    if not metal_mask.any():                            # degenerate fallback: body centroid
        cy, cx = (np.argwhere(body).mean(axis=0)).astype(int)
        metal_r = 6
        metal_mask = _ellipse(shape, (cy, cx), (metal_r, metal_r), 0.0) & body
    tau_actual = mu_metal_px * 2.0 * metal_r

    host = int(np.bincount(label_map[metal_mask].ravel()).argmax())

    return dict(
        phantom_id=int(phantom_id),
        label_map=label_map,
        metal_mask=metal_mask,
        metal_host=host,
        metal_material=metal_mat,
        meta=dict(phantom_id=int(phantom_id), seed=int(seed),
                  base_material=M.NAMES[base], outer_radius_px=int(radius),
                  n_inclusions=len(incl_log), inclusions=incl_log,
                  alu_area_fraction=round(float((label_map == M.ALU).sum() / max(body.sum(), 1)), 4),
                  metal_material=M.NAMES[metal_mat], metal_radius_px=int(metal_r),
                  metal_area_px=int(metal_mask.sum()),
                  metal_tau_max=round(float(tau_actual), 3),
                  body_area_px=int(body.sum())),
    )
