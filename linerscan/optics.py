"""Laser-line triangulation: per-row models, calibration, model selection, thermal compensation.

Per image row the laser spot column ``u`` maps to the distance ``z`` along that row's laser ray:

* ``hyperbolic``  z = a/(u − u0) + b                                 (pinhole camera)
* ``distortion``  z = a/(ũ − u0) + b,  ũ = u + k·sc·((u − uc)/sc)³    (1-D radial lens distortion)
* ``cubic``       z = c0 + c1·s + c2·s² + c3·s³,  s = S/(u − u0)      (fallback)

Fitting: u0 by a 1-D search with the other parameters solved linearly (variable projection), then all
parameters refined by Levenberg–Marquardt.  Model choice by leave-one-distance-out RMS: a more complex
model must be ≥ 20 % *and* ≥ 0.001 mm better than the simpler one it replaces.

A ``surface`` variant (called the "plane model" in the brief) constrains the per-row parameters to
low-order polynomials across rows; it is chosen over independent rows by the same rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

KINDS = ("hyperbolic", "distortion", "cubic")
N_PARAMS = {"hyperbolic": 3, "distortion": 4, "cubic": 5}
CUBIC_S = 1000.0  # scale for s = S/(u − u0) so that s ~ 1


# --------------------------------------------------------------------------- resolution
def depth_resolution(z: float | np.ndarray, pixel_mm: float, focal_mm: float, baseline_mm: float,
                     subpixel_n: float = 10.0):
    """σ_z ≈ z²·p / (f·B·N)."""
    return np.asarray(z, float) ** 2 * pixel_mm / (focal_mm * baseline_mm * subpixel_n)


# --------------------------------------------------------------------------- model evaluation
def model_z(kind: str, p: np.ndarray, u: np.ndarray, uc: float = 0.0, sc: float = 1.0) -> np.ndarray:
    """Evaluate a row model. ``p`` may be (n_params,) or (n_points, n_params) for per-point rows."""
    u = np.asarray(u, float)
    p = np.asarray(p, float)
    P = np.moveaxis(p, -1, 0)
    if kind == "hyperbolic":
        a, b, u0 = P[0], P[1], P[2]
        return a / (u - u0) + b
    if kind == "distortion":
        a, b, u0, k = P[0], P[1], P[2], P[3]
        ut = u + k * sc * ((u - uc) / sc) ** 3
        return a / (ut - u0) + b
    if kind == "cubic":
        c0, c1, c2, c3, u0 = P[0], P[1], P[2], P[3], P[4]
        s = CUBIC_S / (u - u0)
        return c0 + s * (c1 + s * (c2 + s * c3))
    raise ValueError(kind)


def model_offset(kind: str, p: np.ndarray) -> np.ndarray:
    """The additive constant of a model (b, or c0 for the cubic)."""
    P = np.moveaxis(np.asarray(p, float), -1, 0)
    return P[0] if kind == "cubic" else P[1]


def _u0_index(kind: str) -> int:
    return 4 if kind == "cubic" else 2


# --------------------------------------------------------------------------- single-row fitting
def _linear_part(kind: str, u: np.ndarray, z: np.ndarray, u0: float):
    if kind == "cubic":
        s = CUBIC_S / (u - u0)
        A = np.column_stack([np.ones_like(s), s, s**2, s**3])
    else:
        A = np.column_stack([1.0 / (u - u0), np.ones_like(u)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    r = z - A @ coef
    return coef, float(r @ r)


def _varpro_u0(kind: str, u: np.ndarray, z: np.ndarray) -> float:
    """1-D search for u0 (outside the data range, on the side implied by the sign of dz/du)."""
    umin, umax = float(u.min()), float(u.max())
    span = max(umax - umin, 1e-6)
    decreasing = np.corrcoef(u, z)[0, 1] < 0  # a > 0  ->  u0 below the data

    def u0_of(t):
        return umin - np.exp(t) if decreasing else umax + np.exp(t)

    def cost(t):
        return _linear_part(kind, u, z, u0_of(t))[1]

    ts = np.linspace(np.log(1e-3 * span), np.log(1e3 * span), 121)
    costs = np.array([cost(t) for t in ts])
    i = int(np.argmin(costs))
    lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, ts.size - 1)]
    if hi > lo:
        res = minimize_scalar(cost, bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
        return u0_of(res.x)
    return u0_of(ts[i])


def _lm_refine(kind: str, u, z, p0, uc, sc):
    def res(p):
        return model_z(kind, p, u, uc, sc) - z

    scale = np.abs(np.asarray(p0, float)) * 1e-3 + 1e-6
    sol = least_squares(res, p0, method="lm", x_scale=scale, xtol=1e-14, ftol=1e-14, max_nfev=2000)
    return sol.x


def fit_row(kind: str, u: np.ndarray, z: np.ndarray, uc: float = 0.0, sc: float = 1.0,
            p0: np.ndarray | None = None) -> np.ndarray:
    """Fit one row model; ``p0`` (warm start) skips the variable-projection search."""
    u = np.asarray(u, float)
    z = np.asarray(z, float)
    if p0 is None:
        hyp_kind = "cubic" if kind == "cubic" else "hyperbolic"
        u0 = _varpro_u0(hyp_kind, u, z)
        coef, _ = _linear_part(hyp_kind, u, z, u0)
        if kind == "cubic":
            p0 = np.array([*coef, u0])
        else:
            p0 = np.array([coef[0], coef[1], u0])
            if kind == "hyperbolic":
                return _lm_refine(kind, u, z, p0, uc, sc)
            p0 = _lm_refine("hyperbolic", u, z, p0, uc, sc)
            p0 = np.array([*p0, 0.0])
    return _lm_refine(kind, u, z, np.asarray(p0, float), uc, sc)


def loo_errors_row(kind: str, u: np.ndarray, z: np.ndarray, p_full: np.ndarray, uc: float, sc: float):
    """Leave-one-distance-out prediction errors for one row (warm-started from the full fit)."""
    n = u.size
    err = np.empty(n)
    for j in range(n):
        m = np.arange(n) != j
        p = fit_row(kind, u[m], z[m], uc, sc, p0=p_full)
        err[j] = model_z(kind, p, u[j:j + 1], uc, sc)[0] - z[j]
    return err


def choose_simplest(candidates: list[tuple[str, float]], rel: float = 0.20, abs_mm: float = 0.001) -> str:
    """Candidates ordered simplest → most complex, as (name, LOO RMS).

    A more complex model replaces the current choice only if it is ≥ ``rel`` *and* ≥ ``abs_mm`` better.
    """
    best_name, best = candidates[0]
    for name, loo in candidates[1:]:
        if loo <= (1.0 - rel) * best and best - loo >= abs_mm:
            best_name, best = name, loo
    return best_name


# --------------------------------------------------------------------------- surface ("plane") model
def _poly_basis(w: np.ndarray, deg: int) -> np.ndarray:
    return np.vander(np.asarray(w, float), deg + 1, increasing=True)


def fit_surface(kind: str, U: np.ndarray, Z: np.ndarray, w: np.ndarray, deg: int, uc: float, sc: float,
                per_row_init: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Joint fit with every row parameter a polynomial of degree ``deg`` in normalised row coordinate w.

    U, Z: (n_dist, n_rows).  Returns polynomial coefficients C (n_params, deg+1).
    """
    V = _poly_basis(w, deg)  # (n_rows, deg+1)
    C0, *_ = np.linalg.lstsq(V, per_row_init, rcond=None)  # (deg+1, n_params)
    C0 = C0.T
    npar = C0.shape[0]
    if mask is None:
        mask = np.ones_like(U, bool)
    rows = np.broadcast_to(np.arange(U.shape[1])[None, :], U.shape)[mask]
    u = U[mask]
    z = Z[mask]

    def res(c):
        C = c.reshape(npar, deg + 1)
        P = V @ C.T  # (n_rows, n_params)
        return model_z(kind, P[rows], u, uc, sc) - z

    scale = np.abs(C0.ravel()) * 1e-3 + 1e-6
    sol = least_squares(res, C0.ravel(), method="lm", x_scale=scale, xtol=1e-14, ftol=1e-14, max_nfev=4000)
    return sol.x.reshape(npar, deg + 1)


def surface_params(C: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _poly_basis(w, C.shape[1] - 1) @ C.T


# --------------------------------------------------------------------------- thermal model
@dataclass
class ThermalModel:
    """Instrument drift vs temperature, applied to any row model.

    z_T = b + (1 + κ(T))·(z_cal(u − Δu0(T)) − b) + Δb(T),   polynomials in (T − T0).
    """
    T0: float = 20.0
    du0: list[float] = field(default_factory=lambda: [0.0])  # px, coefficients of (T−T0)^1, ^2, ...
    kappa: list[float] = field(default_factory=lambda: [0.0])  # relative scale
    db: list[float] = field(default_factory=lambda: [0.0])  # mm

    @staticmethod
    def _poly(c, dT):
        return sum(ci * dT ** (i + 1) for i, ci in enumerate(c))

    def terms(self, T):
        dT = np.asarray(T, float) - self.T0
        return self._poly(self.du0, dT), self._poly(self.kappa, dT), self._poly(self.db, dT)


# --------------------------------------------------------------------------- calibration container
@dataclass
class TriangulationCalibration:
    kind: str
    structure: str  # "per_row" | "surface"
    rows_v: np.ndarray  # sensor row index of every calibrated row
    phi: np.ndarray  # elevation of each row's laser ray (rad)
    params: np.ndarray  # (n_rows, n_params)
    uc: float
    sc: float
    loo_rms: dict = field(default_factory=dict)  # model name -> LOO RMS (mm)
    thermal: ThermalModel = field(default_factory=ThermalModel)

    def z(self, u: np.ndarray, row: np.ndarray, T: float | np.ndarray | None = None) -> np.ndarray:
        """Distance along the ray for spot column(s) u in calibrated row index(es) ``row``."""
        u = np.asarray(u, float)
        P = self.params[np.asarray(row)]
        if T is None:
            return model_z(self.kind, P, u, self.uc, self.sc)
        du0, kap, db = self.thermal.terms(T)
        b = model_offset(self.kind, P)
        z0 = model_z(self.kind, P, u - du0, self.uc, self.sc)
        return b + (1.0 + kap) * (z0 - b) + db

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "structure": self.structure, "rows_v": self.rows_v.tolist(),
            "phi": self.phi.tolist(), "params": self.params.tolist(), "uc": self.uc, "sc": self.sc,
            "loo_rms": self.loo_rms,
            "thermal": {"T0": self.thermal.T0, "du0": list(self.thermal.du0), "kappa": list(self.thermal.kappa),
                        "db": list(self.thermal.db)},
        }

    @classmethod
    def from_dict(cls, d: dict) -> TriangulationCalibration:
        th = d.get("thermal", {})
        return cls(d["kind"], d["structure"], np.array(d["rows_v"]), np.array(d["phi"]), np.array(d["params"]),
                   d["uc"], d["sc"], d.get("loo_rms", {}), ThermalModel(**th) if th else ThermalModel())


def calibrate_triangulation(U: np.ndarray, d_rail: np.ndarray, phi: np.ndarray, rows_v: np.ndarray,
                            uc: float, sc: float, surface_deg: int = 4, rel: float = 0.20,
                            abs_mm: float = 0.001, kinds: tuple[str, ...] = KINDS) -> TriangulationCalibration:
    """Calibrate all rows from flat-target captures at known rail distances.

    U: (n_dist, n_rows) spot columns (NaN = no detection); d_rail: (n_dist,) horizontal target distances;
    phi: (n_rows,) ray elevations.  The calibrated quantity is z = d/cos φ (distance along the ray), so that
    z·cos φ reproduces the horizontal distance exactly and the reconstruction is insensitive to φ errors.
    """
    U = np.asarray(U, float)
    n_dist, n_rows = U.shape
    if n_dist < 10:
        raise ValueError("need captures at >= 10 distances")
    Z = np.asarray(d_rail, float)[:, None] / np.cos(np.asarray(phi, float))[None, :]
    valid = np.isfinite(U)

    per_row: dict[str, np.ndarray] = {}
    loo: dict[str, float] = {}
    for kind in kinds:
        P = np.zeros((n_rows, N_PARAMS[kind]))
        errs = []
        for k in range(n_rows):
            m = valid[:, k]
            P[k] = fit_row(kind, U[m, k], Z[m, k], uc, sc)
            errs.append(loo_errors_row(kind, U[m, k], Z[m, k], P[k], uc, sc))
        per_row[kind] = P
        loo[kind] = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
    kind = choose_simplest([(k, loo[k]) for k in kinds], rel, abs_mm)

    structure, params = "per_row", per_row[kind]
    if kind != "cubic" and n_rows > surface_deg + 2:
        w = (rows_v - rows_v.mean()) / max(np.ptp(rows_v) / 2.0, 1.0)
        C = fit_surface(kind, U, Z, w, surface_deg, uc, sc, per_row[kind], valid)
        errs = []
        for j in range(n_dist):
            m = valid.copy()
            m[j] = False
            Cj = fit_surface(kind, U, Z, w, surface_deg, uc, sc, surface_params(C, w), m)
            Pj = surface_params(Cj, w)
            vk = valid[j]
            errs.append(model_z(kind, Pj[vk], U[j, vk], uc, sc) - Z[j, vk])
        loo["surface_" + kind] = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
        if choose_simplest([("surface", loo["surface_" + kind]), ("per_row", loo[kind])], rel, abs_mm) == "surface":
            structure, params = "surface", surface_params(C, w)
    return TriangulationCalibration(kind, structure, np.asarray(rows_v), np.asarray(phi, float), params,
                                    uc, sc, loo)


def calibrate_thermal(cal: TriangulationCalibration, captures: list[tuple[float, np.ndarray]],
                      d_rail: np.ndarray, T0: float = 20.0, deg: int = 2) -> ThermalModel:
    """Fit global drift terms (Δu0, κ, Δb) per chamber temperature, then polynomials in (T − T0).

    captures: list of (temperature, U (n_dist, n_rows)) taken on the rail inside a climatic chamber.
    """
    Z = np.asarray(d_rail, float)[:, None] / np.cos(cal.phi)[None, :]
    rows = np.broadcast_to(np.arange(Z.shape[1])[None, :], Z.shape)
    temps, terms = [], []
    for T, U in captures:
        m = np.isfinite(U)
        u, z, r = U[m], Z[m], rows[m]
        P = cal.params[r]
        b = model_offset(cal.kind, P)

        def res(q, u=u, z=z, P=P, b=b):
            z0 = model_z(cal.kind, P, u - q[0], cal.uc, cal.sc)
            return b + (1 + q[1]) * (z0 - b) + q[2] - z

        sol = least_squares(res, np.zeros(3), method="lm", x_scale=np.array([0.01, 1e-5, 1e-3]))
        temps.append(T)
        terms.append(sol.x)
    temps = np.asarray(temps) - T0
    terms = np.asarray(terms)
    deg = min(deg, len(temps) - 1)
    # least-squares polynomial through the origin (no drift at T0 by definition of the reference calibration)
    A = np.column_stack([temps ** (i + 1) for i in range(deg)])
    coefs = [np.linalg.lstsq(A, terms[:, j], rcond=None)[0].tolist() for j in range(3)]
    return ThermalModel(T0=T0, du0=coefs[0], kappa=coefs[1], db=coefs[2])
