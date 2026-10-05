"""Head geometry: from distance along the ray to device-frame points, and ring calibration of R0 / h.

Device frame: origin on the head rotation axis at crown level, z up along the rotation axis.  The beam of
row k starts at arm offset R0 from the axis, misses the axis sideways by h, sits H0 above the crown and is
tilted up by φ_k.  For a spot at distance z along the ray and head angle θ:

    r      = sqrt((R0 + z·cosφ)² + h²)
    α_wall = θ + atan(h / (R0 + z·cosφ))
    height = H0 + z·sinφ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .fitting import cylinder_distances, fit_cylinder, outlier_mask


@dataclass
class HeadGeometry:
    R0: float
    h: float
    H0: float

    def to_dict(self) -> dict:
        return {"R0": self.R0, "h": self.h, "H0": self.H0}


def wall_polar(z, theta, phi, g: HeadGeometry):
    """Return (r, α, height) of wall points in the device frame."""
    xh = g.R0 + np.asarray(z) * np.cos(phi)
    r = np.sqrt(xh**2 + g.h**2)
    alpha = np.asarray(theta) + np.arctan2(g.h, xh)
    height = g.H0 + np.asarray(z) * np.sin(phi)
    return r, alpha, height


def device_points(z, theta, phi, g: HeadGeometry) -> np.ndarray:
    r, a, hgt = wall_polar(z, theta, phi, g)
    return np.column_stack([r * np.cos(a), r * np.sin(a), hgt])


def arm_clearance(phi_lo: float, g: HeadGeometry, arm_height: float, arm_reach: float) -> float:
    """Height of the lowest ray above the arm top at the arm tip (must be > 0: no blind zones)."""
    return g.H0 + (arm_reach - g.R0) * np.tan(phi_lo) - arm_height


@dataclass
class RingCalibration:
    R0: float
    h: float
    D_known: float
    D_measured: float
    iterations: int
    rms: float
    offset_xy: tuple[float, float]
    n_outliers: int = 0


def calibrate_ring(z: np.ndarray, theta: np.ndarray, phi: np.ndarray, D_known: float, g0: HeadGeometry,
                   h_measured: float, sigma_h: float = 1.0, tol: float = 1e-5, max_iter: int = 20,
                   estimate_h: bool = True) -> RingCalibration:
    """Calibrate R0 (and h with a Gaussian prior) from a scan inside a ring of known diameter.

    R0 ← R0 + mean(D_known − D_meas)/2 until converged; h maximises the posterior
    −Σres²/σ² − (h − h_measured)²/σ_h², where σ is the point noise at h_measured.
    """
    g = HeadGeometry(g0.R0, h_measured, g0.H0)
    # glints / reflections: mark points outside max(5·1.4826·MAD, 0.1 mm) of the cylinder and exclude them
    keep = np.ones(np.size(z), bool)
    P = device_points(z, theta, phi, g)
    for _ in range(3):
        c = fit_cylinder(P[keep])
        d = cylinder_distances(P, c.point[0], c.point[1], c.direction[0] / c.direction[2],
                               c.direction[1] / c.direction[2], c.z_ref) - c.R
        keep = ~outlier_mask(d)
    z, theta, phi = np.asarray(z)[keep], np.asarray(theta)[keep], np.broadcast_to(phi, np.shape(keep))[keep]

    def fit(gg):
        return fit_cylinder(device_points(z, theta, phi, gg))

    if estimate_h:
        sigma2 = fit(g).rms ** 2

        def neg_post(hh):
            c = fit(HeadGeometry(g.R0, hh, g.H0))
            return c.n * c.rms**2 / sigma2 + (hh - h_measured) ** 2 / sigma_h**2

        res = minimize_scalar(neg_post, bounds=(h_measured - 5 * sigma_h, h_measured + 5 * sigma_h),
                              method="bounded", options={"xatol": 1e-3})
        g.h = float(res.x)

    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007
        c = fit(g)
        step = (D_known - 2 * c.R) / 2.0
        g.R0 += step
        if abs(step) < tol:
            break
    c = fit(g)
    return RingCalibration(g.R0, g.h, D_known, 2 * c.R, it, c.rms, (float(c.point[0]), float(c.point[1])),
                           int((~keep).sum()))
