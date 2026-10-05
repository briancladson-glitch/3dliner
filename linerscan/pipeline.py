"""Post-retrieval processing: raw scan → 3D points → per-band liner geometry.

1. u → z per row (calibrated model + thermal compensation at the logged camera temperature).
2. z, encoder θ → device-frame points (head geometry R0, h, H0).
3. Crown depth vs time from the upward ToF roof distance, anchored at BDC (start) and at the stop crank angle
   read from the flywheel.
4. Pass 1: bands by approximate depth, cylinder fits → global axis direction (device tilt).
5. Pass 2: depth corrected with the tilt; per band: cylinder fit (tilt + offset) → circle fit about the band
   axis (Kåsa → LM) → order-8 Fourier ρ(α) with taper term → D_FA, D_PS, ovality, max/min, ellipse check.
   Outliers beyond max(5·1.4826·MAD, 0.1 mm) are marked, excluded and counted — never silently dropped.
6. Yaw (FORE direction) from the operator's placement, refined with the scavenge-port pattern if seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import DeviceCalibration
from .config import EngineSpec, JobInfo, ProcessingConfig
from .device.safety import roof_distance
from .fitting import (
    CylinderFit,
    DiameterStats,
    EllipseFit,
    cylinder_distances,
    diameter_stats,
    fit_circle,
    fit_cylinder,
    fit_ellipse,
    fit_fourier,
    fourier_residuals,
    outlier_mask,
    rotation_to_z,
)
from .geometry import wall_polar
from .kinematics import crown_depth
from .rawscan import RawScan


# --------------------------------------------------------------------------- points
@dataclass
class Points:
    P: np.ndarray  # (N, 3) device frame: x, y, height above crown
    t: np.ndarray
    s_c: np.ndarray  # crown depth at capture time
    row: np.ndarray
    depth: np.ndarray  # depth below liner top (updated after tilt estimation)


def raw_to_points(raw: RawScan, cal: DeviceCalibration, crown_fn) -> Points:
    F, K = raw.u.shape
    T = np.interp(raw.frame_t, raw.temp_t, raw.temps[:, 0])
    m = np.isfinite(raw.u)
    fi, row = np.nonzero(m)
    u = raw.u[m]
    tri = cal.triangulation
    z = tri.z(u, row, T[fi])
    r, a, hgt = wall_polar(z, raw.frame_theta[fi], tri.phi[row], cal.geometry)
    P = np.column_stack([r * np.cos(a), r * np.sin(a), hgt])
    t = raw.frame_t[fi]
    s_c = crown_fn(t)
    return Points(P, t, s_c, row, s_c - hgt)


# --------------------------------------------------------------------------- height reference
@dataclass
class HeightModel:
    A: float
    B: float
    roof_bdc: float
    roof_stop: float
    s_bdc: float
    s_stop: float
    t_tof: np.ndarray = field(repr=False)
    roof_smooth: np.ndarray = field(repr=False)
    noise: float = 0.0

    def __call__(self, t):
        return self.A + self.B * np.interp(t, self.t_tof, self.roof_smooth)


def _local_linear(t, y, half_window):
    out = np.empty_like(y)
    c = np.cumsum
    # running sums for a centred window (uniform weights) – fast O(n)
    idx_lo = np.searchsorted(t, t - half_window)
    idx_hi = np.searchsorted(t, t + half_window, side="right")
    S0 = np.concatenate([[0], c(np.ones_like(t))])
    S1 = np.concatenate([[0], c(t)])
    S2 = np.concatenate([[0], c(t * t)])
    Sy = np.concatenate([[0], c(y)])
    Sty = np.concatenate([[0], c(t * y)])
    n = S0[idx_hi] - S0[idx_lo]
    st = S1[idx_hi] - S1[idx_lo]
    stt = S2[idx_hi] - S2[idx_lo]
    sy = Sy[idx_hi] - Sy[idx_lo]
    sty = Sty[idx_hi] - Sty[idx_lo]
    den = n * stt - st**2
    b = np.where(den > 1e-9, (n * sty - st * sy) / np.where(den > 1e-9, den, 1), 0.0)
    a = (sy - b * st) / n
    out = a + b * t
    return out


def height_model(raw: RawScan, engine: EngineSpec, job: JobInfo, smooth_s: float = 1.5) -> HeightModel:
    """Crown depth s_c(t) = A + B·roof(t); A, B from the BDC start and the stop crank angle read at the flywheel."""
    t = raw.tof_t
    roof = np.array([roof_distance(r) for r in raw.tof])
    ok = np.isfinite(roof)
    t, roof = t[ok], roof[ok]
    ev = {e[2].split()[0]: e[0] for e in raw.events if e[1] == "state"}
    t_scan = ev.get("SCAN", t[0])
    t_stop = ev.get("STOPPED", t[-1])
    pre = (t < t_scan - 8.0) & (t > t_scan - 60.0)
    if pre.sum() < 10:
        pre = t < t_scan - 8.0
    post = t > t_stop - 4.0
    roof0, roof1 = float(np.median(roof[pre])), float(np.median(roof[post]))
    if job.stop_angle_read_deg is None:
        raise ValueError("stop crank angle (read at the flywheel) is required for the height reference")
    s0 = float(crown_depth(180.0, engine))
    s1 = float(crown_depth(abs(job.stop_angle_read_deg), engine))
    B = (s0 - s1) / (roof0 - roof1)
    A = s0 - B * roof0
    sm = _local_linear(t, roof, smooth_s)
    sm[t <= t_scan - 8.0] = roof0
    sm[t >= t_stop - 4.0] = roof1
    noise = float(np.std(roof[post] - roof1))
    return HeightModel(A, B, roof0, roof1, s0, s1, t, sm, noise)


# --------------------------------------------------------------------------- bands
@dataclass
class BandResult:
    depth: float
    lo: float
    hi: float
    n_points: int
    n_outliers: int
    coverage_pct: float
    max_gap_deg: float
    port_zone: bool
    tilt_deg: tuple[float, float]
    centre_xy: tuple[float, float]
    circle_D: float
    circle_rms: float
    sigma_r: float
    stats: DiameterStats
    ellipse: EllipseFit | None
    ellipse_dev: float
    flags: list[str]
    U_k2: float = 0.0  # expanded uncertainty of a diameter (k=2), statistical + systematic
    coef: np.ndarray | None = field(default=None, repr=False)


def _robust_cylinder(P: np.ndarray, cfg: ProcessingConfig, rounds: int = 3) -> tuple[CylinderFit, np.ndarray]:
    keep = np.ones(len(P), bool)
    cyl = fit_cylinder(P)
    for _ in range(rounds):
        cyl = fit_cylinder(P[keep])
        d = cylinder_distances(P, cyl.point[0], cyl.point[1], cyl.direction[0] / cyl.direction[2],
                               cyl.direction[1] / cyl.direction[2], cyl.z_ref) - cyl.R
        new = ~outlier_mask(d, cfg.outlier_k, cfg.outlier_floor)
        if np.array_equal(new, keep):
            break
        keep = new
    return cyl, keep


def analyze_band(pts: Points, idx: np.ndarray, lo: float, hi: float, yaw_deg: float, cfg: ProcessingConfig,
                 engine: EngineSpec, u_sys: float = 0.0, axis_dir: np.ndarray | None = None,
                 collect: dict | None = None) -> BandResult | None:
    P = pts.P[idx]
    if len(P) < cfg.min_points_per_band:
        return None
    flags: list[str] = []
    cyl, keep = _robust_cylinder(P, cfg)
    if cfg.tilt_mode == "global" and axis_dir is not None:
        cyl = CylinderFit(cyl.point, axis_dir, cyl.R, cyl.rms, cyl.n, cyl.z_ref)
    Rm = rotation_to_z(cyl.direction)
    Q = (P - cyl.point) @ Rm.T
    s = pts.depth[idx]
    circ = fit_circle(Q[keep, 0], Q[keep, 1])
    x, y = Q[:, 0] - circ.xc, Q[:, 1] - circ.yc
    rho = np.hypot(x, y)
    alpha = np.arctan2(y, x) + np.radians(yaw_deg)
    # Fourier fit with iterative outlier marking (threshold from all residuals each round)
    inl = keep.copy()
    fit = fit_fourier(alpha[inl], rho[inl], cfg.fourier_order, s[inl] if cfg.fourier_taper else None)
    for _ in range(cfg.outlier_iterations):
        res = fourier_residuals(fit, alpha, rho, s if cfg.fourier_taper else None)
        new = ~outlier_mask(res, cfg.outlier_k, cfg.outlier_floor)
        if np.array_equal(new, inl):
            break
        inl = new
        fit = fit_fourier(alpha[inl], rho[inl], cfg.fourier_order, s[inl] if cfg.fourier_taper else None)
    st = diameter_stats(fit, fa_angle_deg=0.0)
    # coverage
    hist = np.bincount((np.degrees(alpha[inl]) % 360).astype(int) % 360, minlength=360) > 0
    cov = float(hist.mean() * 100)
    occ = np.nonzero(hist)[0]
    gaps = np.diff(np.concatenate([occ, [occ[0] + 360]])) - 1 if occ.size else np.array([360])
    max_gap = float(gaps.max())
    port = bool(hi > engine.port_top_depth and lo < engine.port_bottom_depth)
    if port:
        flags.append("scavenge-port zone: diameters across port bars only")
    if max_gap > cfg.max_angle_gap_deg:
        flags.append(f"angular gap {max_gap:.0f}° > {cfg.max_angle_gap_deg:.0f}°")
    ell, edev = None, float("nan")
    try:
        ell = fit_ellipse(rho[inl] * np.cos(alpha[inl]), rho[inl] * np.sin(alpha[inl]))
        edev = max(abs(ell.D_major - st.D_max), abs(ell.D_minor - st.D_min))
        if edev > cfg.ellipse_flag_mm:
            flags.append(f"shape not elliptical (ellipse vs Fourier {edev:.3f} mm)")
    except (ValueError, np.linalg.LinAlgError):
        flags.append("ellipse fit failed")
    n_out = int((~inl).sum())
    tx, ty = cyl.tilt_deg
    u_stat = max(st.u_D_FA, st.u_D_PS)
    if collect is not None:
        collect["alpha"].append(alpha[inl])
        collect["depth"].append(s[inl])
        collect["rho"].append(rho[inl])
        collect["out_alpha"].append(alpha[~inl])
        collect["out_depth"].append(s[~inl])
    ac = cyl.axis_xy_at(0.0)
    return BandResult(0.5 * (lo + hi), lo, hi, int(len(P)), n_out, cov, max_gap, port, (tx, ty),
                      (float(ac[0]), float(ac[1])), circ.D, circ.rms, fit.sigma, st, ell, edev, flags,
                      2 * float(np.hypot(u_stat, u_sys)), fit.coef)


# --------------------------------------------------------------------------- yaw from the port pattern
def estimate_yaw_from_ports(raw: RawScan, cal: DeviceCalibration, hm: HeightModel, engine: EngineSpec,
                            centre_xy: tuple[float, float], yaw_guess: float) -> tuple[float, float] | None:
    """Phase of the scavenge-port bars → yaw modulo the port pitch, resolved with the operator's estimate.

    Returns (yaw_deg, strength) or None.  Uses the fraction of rays that return a wall hit per azimuth bin
    (normalised by how often each azimuth was sampled), so the uneven sweep sampling does not bias the phase.
    """
    n = engine.n_ports
    tri, g = cal.triangulation, cal.geometry
    x_nom = engine.bore / 2 - g.R0
    s_c = hm(raw.frame_t)
    hgt = g.H0 + x_nom * np.tan(tri.phi)
    depth = s_c[:, None] - hgt[None, :]
    lo, hi = engine.port_top_depth + 30, engine.port_bottom_depth - 30
    sel = (depth > lo) & (depth < hi)
    if sel.sum() < 5000:
        return None
    th = np.broadcast_to(raw.frame_theta[:, None], depth.shape)[sel] + np.arctan2(g.h, g.R0 + x_nom)
    # azimuth about the liner centre of the wall point hit by a ray leaving the device axis at angle th
    C = -np.asarray(centre_xy)
    e = np.column_stack([np.cos(th), np.sin(th)])
    ce = e @ C
    R = engine.bore / 2
    tt = ce + np.sqrt(ce**2 - C @ C + R**2)
    Pw = tt[:, None] * e - C
    az = np.arctan2(Pw[:, 1], Pw[:, 0])
    hit = np.isfinite(raw.u)[sel].astype(float)
    bins = np.linspace(-np.pi, np.pi, 1441)
    tot = np.histogram(az, bins)[0]
    wal = np.histogram(az, bins, weights=hit)[0]
    m = tot > 3
    frac = wal[m] / tot[m]
    cen = 0.5 * (bins[1:] + bins[:-1])[m]
    z = np.sum((frac - frac.mean()) * np.exp(1j * n * cen))
    strength = float(np.abs(z) / m.sum())
    if strength < 0.05:
        return None
    bar_dev = np.angle(z) / n  # bar centre in device azimuth (mod pitch)
    pitch = 2 * np.pi / n
    bar_liner = np.radians(engine.port_phase_deg) + pitch / 2
    yaw_mod = bar_liner - bar_dev
    d = np.angle(np.exp(1j * n * (yaw_mod - np.radians(yaw_guess)))) / n
    return float(np.degrees(np.radians(yaw_guess) + d)), strength


# --------------------------------------------------------------------------- whole scan
@dataclass
class ScanResult:
    profile: list[BandResult]
    positions: list[BandResult]
    tilt_deg: tuple[float, float]
    yaw_deg: float
    yaw_source: str
    height: HeightModel
    n_points: int
    n_outliers: int
    depth_range: tuple[float, float]
    map_alpha: np.ndarray
    map_depth: np.ndarray
    map_dev: np.ndarray  # radius − nominal radius (mm), NaN = no data
    outlier_alpha: np.ndarray
    outlier_depth: np.ndarray
    temps: dict
    meta: dict


def process(raw: RawScan, cal: DeviceCalibration, engine: EngineSpec, job: JobInfo,
            cfg: ProcessingConfig | None = None, u_sys: float = 0.0, refine_yaw: bool = True) -> ScanResult:
    cfg = cfg or ProcessingConfig()
    hm = height_model(raw, engine, job)
    pts = raw_to_points(raw, cal, hm)
    order = np.argsort(pts.depth)
    for k in ("P", "t", "s_c", "row", "depth"):
        setattr(pts, k, getattr(pts, k)[order])
    dmin = max(float(np.quantile(pts.depth, 0.001)), 0.0)
    dmax = float(np.quantile(pts.depth, 0.999))

    def sel(lo, hi):
        i0, i1 = np.searchsorted(pts.depth, [lo, hi])
        return np.arange(i0, i1)

    # pass 1: global tilt from bands outside the port zone
    tilts, dirs = [], []
    for lo in np.arange(dmin + 20, min(dmax, engine.port_top_depth) - cfg.band_height, 3 * cfg.band_step):
        idx = sel(lo, lo + cfg.band_height)
        if idx.size >= cfg.min_points_per_band:
            cyl, _ = _robust_cylinder(pts.P[idx], cfg, rounds=2)
            dirs.append(cyl.direction)
            tilts.append(cyl.tilt_deg)
    u_axis = np.median(np.array(dirs), axis=0) if dirs else np.array([0, 0, 1.0])
    u_axis /= np.linalg.norm(u_axis)
    pts.depth = pts.s_c - pts.P @ u_axis
    order = np.argsort(pts.depth)
    for k in ("P", "t", "s_c", "row", "depth"):
        setattr(pts, k, getattr(pts, k)[order])
    tilt = (float(np.degrees(np.arctan2(u_axis[0], u_axis[2]))), float(np.degrees(np.arctan2(u_axis[1], u_axis[2]))))

    # yaw: operator estimate, refined with the scavenge-port pattern
    yaw, src = job.yaw_deg, "operator placement estimate"
    if refine_yaw and dmax > engine.port_top_depth + 60:
        mid = 0.5 * (engine.port_top_depth + min(engine.port_bottom_depth, dmax))
        idx = sel(mid - 15, mid + 15)
        if idx.size >= cfg.min_points_per_band:
            cyl, _ = _robust_cylinder(pts.P[idx], cfg, rounds=2)
            c0 = cyl.axis_xy_at(0.0)
            est = estimate_yaw_from_ports(raw, cal, hm, engine, (float(c0[0]), float(c0[1])), job.yaw_deg)
            if est is not None:
                yaw, src = est[0], f"scavenge-port pattern (strength {est[1]:.2f}), placement {job.yaw_deg:.1f}°"

    # pass 2: profile bands + measuring positions
    collect = {"alpha": [], "depth": [], "rho": [], "out_alpha": [], "out_depth": []}
    profile = []
    start = np.floor(dmin / cfg.band_step) * cfg.band_step
    for lo in np.arange(start, dmax, cfg.band_step):
        hi = lo + cfg.band_height
        if lo < dmin - cfg.band_height / 3 or hi > dmax + cfg.band_height / 3:
            continue
        b = analyze_band(pts, sel(lo, hi), lo, hi, yaw, cfg, engine, u_sys, u_axis, collect)
        if b is not None:
            profile.append(b)
    positions = []
    for d in engine.measuring_depths:
        lo, hi = d - cfg.band_height / 2, d + cfg.band_height / 2
        if lo < dmin - 5 or hi > dmax + 5:
            continue
        b = analyze_band(pts, sel(lo, hi), lo, hi, yaw, cfg, engine, u_sys, u_axis)
        if b is not None:
            positions.append(b)

    # unrolled deviation map (radius − nominal radius), as measured
    a = np.degrees(np.concatenate(collect["alpha"])) % 360 if collect["alpha"] else np.zeros(0)
    dd = np.concatenate(collect["depth"]) if collect["depth"] else np.zeros(0)
    rr = np.concatenate(collect["rho"]) - engine.bore / 2 if collect["rho"] else np.zeros(0)
    ab = np.arange(0, 360 + cfg.map_angle_step_deg, cfg.map_angle_step_deg)
    db = np.arange(start, dmax + cfg.map_depth_step, cfg.map_depth_step)
    ssum = np.histogram2d(dd, a, [db, ab], weights=rr)[0]
    cnt = np.histogram2d(dd, a, [db, ab])[0]
    dev = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
    temps = {"camera_mean": float(np.mean(raw.temps[:, 0])) if len(raw.temps) else float("nan"),
             "camera_scan": float(np.interp(np.median(raw.frame_t), raw.temp_t, raw.temps[:, 0])),
             "laser_max": float(np.max(raw.temps[:, 1])) if len(raw.temps) else float("nan")}
    n_out = sum(b.n_outliers for b in profile)
    return ScanResult(profile, positions, tilt, yaw, src, hm, int(len(pts.depth)), n_out, (dmin, dmax),
                      0.5 * (ab[1:] + ab[:-1]), 0.5 * (db[1:] + db[:-1]), dev,
                      np.degrees(np.concatenate(collect["out_alpha"])) % 360 if collect["out_alpha"] else np.zeros(0),
                      np.concatenate(collect["out_depth"]) if collect["out_depth"] else np.zeros(0), temps,
                      {"raw": raw.meta, "cal_id": cal.cal_id, "events": raw.events})
