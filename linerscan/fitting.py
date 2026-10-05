"""Geometric fits used for every height band.

* Kåsa algebraic circle fit (start value) and geometric Levenberg–Marquardt circle fit.
* Local cylinder fit (axis tilt + offset + radius) – removes device tilt and off-centre placement.
* Fourier radius profile ρ(α) about the fitted centre, and the diameter function D(α) = ρ(α) + ρ(α+π).
* Direct least-squares ellipse (Fitzgibbon, numerically stable form of Halíř & Flusser) as a cross-check.
* Robust outlier threshold  max(k·1.4826·MAD, floor).

All lengths in mm, angles in radians unless a name ends in ``_deg``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

MAD_SCALE = 1.4826


# --------------------------------------------------------------------------- robust statistics
def robust_threshold(residuals: np.ndarray, k: float = 5.0, floor: float = 0.1) -> float:
    """Outlier threshold max(k·1.4826·MAD, floor) on residuals (about their median)."""
    r = np.asarray(residuals, float)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    return float(max(k * MAD_SCALE * mad, floor))


def outlier_mask(residuals: np.ndarray, k: float = 5.0, floor: float = 0.1) -> np.ndarray:
    """True where a residual is an outlier: |r − median(r)| > max(k·1.4826·MAD, floor)."""
    r = np.asarray(residuals, float)
    thr = robust_threshold(r, k, floor)
    return np.abs(r - np.median(r)) > thr


# --------------------------------------------------------------------------- circles
@dataclass
class CircleFit:
    xc: float
    yc: float
    R: float
    rms: float
    n: int

    @property
    def D(self) -> float:
        return 2.0 * self.R


def kasa_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Kåsa algebraic fit: minimise Σ(x²+y²+Dx+Ey+F)². Returns (xc, yc, R)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    xs, ys = x - mx, y - my
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    rhs = -(xs**2 + ys**2)
    (D, E, F), *_ = np.linalg.lstsq(A, rhs, rcond=None)
    xc, yc = -D / 2.0, -E / 2.0
    R = float(np.sqrt(max(xc**2 + yc**2 - F, 0.0)))
    return float(xc + mx), float(yc + my), R


def fit_circle(x: np.ndarray, y: np.ndarray, init: tuple[float, float, float] | None = None) -> CircleFit:
    """Geometric circle fit (LM) started from the Kåsa solution."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    p0 = np.array(init if init is not None else kasa_circle(x, y), float)

    def res(p):
        return np.hypot(x - p[0], y - p[1]) - p[2]

    def jac(p):
        dx, dy = x - p[0], y - p[1]
        d = np.hypot(dx, dy)
        d[d == 0] = 1e-12
        return np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])

    sol = least_squares(res, p0, jac=jac, method="lm", xtol=1e-12, ftol=1e-12)
    r = sol.fun
    return CircleFit(float(sol.x[0]), float(sol.x[1]), float(sol.x[2]), float(np.sqrt(np.mean(r**2))), x.size)


# --------------------------------------------------------------------------- cylinder
@dataclass
class CylinderFit:
    point: np.ndarray  # a point on the axis at z = z_ref
    direction: np.ndarray  # unit axis direction (z-component > 0)
    R: float
    rms: float
    n: int
    z_ref: float

    @property
    def tilt_deg(self) -> tuple[float, float]:
        """Axis tilt about the device x and y axes, in degrees (tx = dx/dz, ty = dy/dz)."""
        d = self.direction
        return float(np.degrees(np.arctan2(d[0], d[2]))), float(np.degrees(np.arctan2(d[1], d[2])))

    def axis_xy_at(self, z: float) -> np.ndarray:
        d = self.direction
        t = (z - self.point[2]) / d[2]
        return self.point[:2] + t * d[:2]


def _axis_dir(tx: float, ty: float) -> np.ndarray:
    v = np.array([tx, ty, 1.0])
    return v / np.linalg.norm(v)


def cylinder_distances(P: np.ndarray, x0: float, y0: float, tx: float, ty: float, z_ref: float) -> np.ndarray:
    u = _axis_dir(tx, ty)
    w = P - np.array([x0, y0, z_ref])
    along = w @ u
    perp = w - along[:, None] * u[None, :]
    return np.linalg.norm(perp, axis=1)


def fit_cylinder(P: np.ndarray, init: tuple[float, float, float] | None = None) -> CylinderFit:
    """Fit a right circular cylinder to points P (N×3) whose axis is roughly along +z.

    Parameters: axis point (x0, y0) at z_ref = mean(z), slopes (tx, ty) = (dx/dz, dy/dz) and radius R.
    Start: Kåsa circle on the xy projection, zero tilt.  Solved with Levenberg–Marquardt.
    """
    P = np.asarray(P, float)
    z_ref = float(P[:, 2].mean())
    xc, yc, R = init if init is not None else kasa_circle(P[:, 0], P[:, 1])
    p0 = np.array([xc, yc, 0.0, 0.0, R])

    def res(p):
        return cylinder_distances(P, p[0], p[1], p[2], p[3], z_ref) - p[4]

    sol = least_squares(res, p0, method="lm", x_scale=np.array([1.0, 1.0, 1e-3, 1e-3, 1.0]),
                        xtol=1e-12, ftol=1e-12)
    x0, y0, tx, ty, Rf = sol.x
    rms = float(np.sqrt(np.mean(sol.fun**2)))
    return CylinderFit(np.array([x0, y0, z_ref]), _axis_dir(tx, ty), float(Rf), rms, P.shape[0], z_ref)


def rotation_to_z(u: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping unit vector u onto +z (Rodrigues, minimal rotation)."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s = np.linalg.norm(v)
    c = float(u @ z)
    if s < 1e-15:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def to_axis_frame(P: np.ndarray, cyl: CylinderFit) -> np.ndarray:
    """Express points in a frame whose z-axis is the fitted cylinder axis (origin on the axis at z_ref)."""
    Rm = rotation_to_z(cyl.direction)
    return (np.asarray(P, float) - cyl.point) @ Rm.T


# --------------------------------------------------------------------------- Fourier radius profile
@dataclass
class FourierFit:
    order: int
    coef: np.ndarray  # [c0, a1, b1, ..., aN, bN, (g)]
    cov: np.ndarray
    sigma: float  # residual standard deviation
    n: int
    taper: bool = False
    s_mid: float = 0.0

    def rho(self, alpha: np.ndarray) -> np.ndarray:
        """Radius at the band mid height."""
        A = fourier_design(np.atleast_1d(alpha), self.order)
        return A @ self.coef[: 2 * self.order + 1]

    def diameter_vector(self, alpha: float) -> np.ndarray:
        """Gradient g of D(α) = gᵀ·coef (odd harmonics cancel)."""
        g = np.zeros_like(self.coef)
        g[0] = 2.0
        for n in range(2, self.order + 1, 2):
            g[2 * n - 1] = 2.0 * np.cos(n * alpha)
            g[2 * n] = 2.0 * np.sin(n * alpha)
        return g

    def diameter(self, alpha: np.ndarray) -> np.ndarray:
        """D(α) = ρ(α) + ρ(α+π)."""
        alpha = np.atleast_1d(np.asarray(alpha, float))
        return np.array([self.diameter_vector(a) @ self.coef for a in alpha])

    def diameter_sigma(self, alpha: float) -> float:
        g = self.diameter_vector(alpha)
        return float(np.sqrt(max(g @ self.cov @ g, 0.0)))

    @property
    def mean_diameter(self) -> float:
        return float(2.0 * self.coef[0])


def fourier_design(alpha: np.ndarray, order: int) -> np.ndarray:
    alpha = np.asarray(alpha, float)
    cols = [np.ones_like(alpha)]
    for n in range(1, order + 1):
        cols += [np.cos(n * alpha), np.sin(n * alpha)]
    return np.column_stack(cols)


def fit_fourier(alpha: np.ndarray, rho: np.ndarray, order: int = 8, s: np.ndarray | None = None) -> FourierFit:
    """Least-squares Fourier series of ``order`` for ρ(α).

    If ``s`` (height of each point) is given a linear taper term g·(s − s_mid) is added, so that the
    harmonic coefficients describe the band mid-height even when the wall is tapered inside the band.
    """
    alpha = np.asarray(alpha, float)
    rho = np.asarray(rho, float)
    A = fourier_design(alpha, order)
    s_mid = 0.0
    if s is not None:
        s = np.asarray(s, float)
        s_mid = 0.5 * (float(s.min()) + float(s.max()))
        A = np.column_stack([A, s - s_mid])
    coef, *_ = np.linalg.lstsq(A, rho, rcond=None)
    r = rho - A @ coef
    dof = max(rho.size - A.shape[1], 1)
    sigma = float(np.sqrt(r @ r / dof))
    AtA = A.T @ A
    cov = np.linalg.pinv(AtA) * sigma**2
    return FourierFit(order, coef, cov, sigma, rho.size, s is not None, s_mid)


def fourier_residuals(fit: FourierFit, alpha: np.ndarray, rho: np.ndarray, s: np.ndarray | None = None):
    A = fourier_design(alpha, fit.order)
    if fit.taper:
        A = np.column_stack([A, np.asarray(s, float) - fit.s_mid])
    return np.asarray(rho, float) - A @ fit.coef


@dataclass
class DiameterStats:
    D_FA: float
    D_PS: float
    ovality: float  # |D_PS − D_FA|  (UMA definition)
    D_max: float
    D_max_angle_deg: float
    D_min: float
    D_min_angle_deg: float
    max_minus_min: float
    D_mean: float
    u_D_FA: float  # 1σ statistical
    u_D_PS: float
    u_ovality: float
    profile_deg: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    profile_D: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


def diameter_stats(fit: FourierFit, fa_angle_deg: float = 0.0, step_deg: float = 0.1) -> DiameterStats:
    """Diameter results from a Fourier fit whose angles are measured in the liner frame.

    D_FA is the diameter along the fore–aft line (α = fa_angle), D_PS along port–starboard (α = fa_angle+90°).
    """
    grid = np.arange(0.0, 180.0, step_deg)
    a = np.radians(grid)
    # vectorised D(α)
    Dg = 2.0 * fit.coef[0] * np.ones_like(a)
    for n in range(2, fit.order + 1, 2):
        Dg += 2.0 * (fit.coef[2 * n - 1] * np.cos(n * a) + fit.coef[2 * n] * np.sin(n * a))
    a_fa = np.radians(fa_angle_deg)
    a_ps = a_fa + np.pi / 2
    D_FA = float(fit.diameter(a_fa)[0])
    D_PS = float(fit.diameter(a_ps)[0])
    g = fit.diameter_vector(a_ps) - fit.diameter_vector(a_fa)
    u_ov = float(np.sqrt(max(g @ fit.cov @ g, 0.0)))
    imax, imin = int(np.argmax(Dg)), int(np.argmin(Dg))
    return DiameterStats(
        D_FA=D_FA, D_PS=D_PS, ovality=abs(D_PS - D_FA),
        D_max=float(Dg[imax]), D_max_angle_deg=float(grid[imax]),
        D_min=float(Dg[imin]), D_min_angle_deg=float(grid[imin]),
        max_minus_min=float(Dg[imax] - Dg[imin]), D_mean=fit.mean_diameter,
        u_D_FA=fit.diameter_sigma(a_fa), u_D_PS=fit.diameter_sigma(a_ps), u_ovality=u_ov,
        profile_deg=grid, profile_D=Dg,
    )


# --------------------------------------------------------------------------- ellipse
@dataclass
class EllipseFit:
    xc: float
    yc: float
    a: float  # semi-major
    b: float  # semi-minor
    angle_deg: float  # direction of the major axis, [0, 180)

    @property
    def D_major(self) -> float:
        return 2 * self.a

    @property
    def D_minor(self) -> float:
        return 2 * self.b


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> EllipseFit:
    """Direct least-squares ellipse fit (Fitzgibbon et al. 1999; stable form by Halíř & Flusser 1998)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    s = np.sqrt(np.mean((x - mx) ** 2 + (y - my) ** 2))
    xs, ys = (x - mx) / s, (y - my) / s
    D1 = np.column_stack([xs**2, xs * ys, ys**2])
    D2 = np.column_stack([xs, ys, np.ones_like(xs)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
    w, v = np.linalg.eig(M)
    v = np.real(v)
    cond = 4 * v[0] * v[2] - v[1] ** 2
    idx = np.where(cond > 0)[0]
    if idx.size == 0:
        raise ValueError("ellipse fit failed (no elliptical solution)")
    a1 = v[:, idx[0]]
    A, B, C = a1
    Dc, Ec, F = T @ a1
    den = B**2 - 4 * A * C
    x0 = (2 * C * Dc - B * Ec) / den
    y0 = (2 * A * Ec - B * Dc) / den
    F0 = A * x0**2 + B * x0 * y0 + C * y0**2 + Dc * x0 + Ec * y0 + F
    Mq = np.array([[A, B / 2], [B / 2, C]])
    lam, vec = np.linalg.eigh(Mq)
    ax = np.sqrt(-F0 / lam)
    i_major = int(np.argmax(ax))
    major_dir = vec[:, i_major]
    ang = float(np.degrees(np.arctan2(major_dir[1], major_dir[0])) % 180.0)
    return EllipseFit(float(x0 * s + mx), float(y0 * s + my), float(ax.max() * s), float(ax.min() * s), ang)
