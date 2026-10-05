import numpy as np
import pytest

from linerscan.config import PRESETS, DeviceSpec, EngineSpec
from linerscan.geometry import HeadGeometry, arm_clearance, calibrate_ring, device_points, wall_polar
from linerscan.kinematics import (
    crank_angle_for_displacement,
    overhead_clearance,
    piston_displacement,
    piston_speed,
    plan_scan,
    plan_stop_angle,
)
from linerscan.optics import (
    ThermalModel,
    TriangulationCalibration,
    calibrate_thermal,
    calibrate_triangulation,
    choose_simplest,
    depth_resolution,
    fit_row,
    model_z,
)
from linerscan.peak import detect_peak, render_row, subpixel_factor

# ----------------------------------------------------------------------------- triangulation


def test_resolution_formula():
    # z = 500 mm, p = 3 µm, f = 12 mm, B = 65 mm, N = 10  ->  0.0962 mm
    assert depth_resolution(500, 0.003, 12, 65, 10) == pytest.approx(500**2 * 0.003 / (12 * 65 * 10))
    assert depth_resolution(500, 0.003, 12, 65, 10) == pytest.approx(0.09615, abs=1e-5)


def _synthetic_row(a=260000.0, b=-1.5, u0=-150.0, k=0.0, uc=640, sc=640, n=15, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    z = np.linspace(310, 520, n)
    u_ideal = u0 + a / (z - b)
    # invert the distortion variant so that the forward model is exact
    u = u_ideal.copy()
    for _ in range(50):
        u = u_ideal - k * sc * ((u - uc) / sc) ** 3
    return u + rng.normal(0, noise, n), z


def test_hyperbolic_fit_exact():
    u, z = _synthetic_row()
    p = fit_row("hyperbolic", u, z)
    assert p == pytest.approx([260000.0, -1.5, -150.0], rel=1e-6)


def test_distortion_and_cubic_fit_exact():
    u, z = _synthetic_row(k=0.002)
    p = fit_row("distortion", u, z, 640, 640)
    assert p[3] == pytest.approx(0.002, rel=1e-3)
    assert np.abs(model_z("distortion", p, u, 640, 640) - z).max() < 1e-6
    pc = fit_row("cubic", u, z)
    assert np.abs(model_z("cubic", pc, u) - z).max() < 2e-3


def test_model_selection_rule():
    assert choose_simplest([("a", 0.010), ("b", 0.0085)]) == "a"  # 15 %: not enough
    assert choose_simplest([("a", 0.010), ("b", 0.0079)]) == "b"  # 21 % and 2.1 µm
    assert choose_simplest([("a", 0.0040), ("b", 0.0031)]) == "a"  # 22 % but only 0.9 µm
    assert choose_simplest([("a", 0.010), ("b", 0.0079), ("c", 0.007)]) == "b"
    assert choose_simplest([("a", 0.010), ("b", 0.0079), ("c", 0.005)]) == "c"


def _rig(k=0.0, noise=0.01, extra=0.0, seed=1, n_rows=12):
    """Simulated rail captures for one module: U (n_dist, n_rows)."""
    rng = np.random.default_rng(seed)
    p = PRESETS["L"]
    phi = np.radians(np.linspace(p.fan_lo_deg, p.fan_hi_deg, n_rows))
    rows_v = 400 + p.fx_px * np.tan(phi - phi.mean())
    d = np.linspace(305, 510, 15)
    Z = d[:, None] / np.cos(phi)[None, :]
    a = p.fx_px * p.baseline_mm / np.cos(phi - phi.mean())
    u0 = -150 + 2.0 * np.linspace(-1, 1, n_rows)
    ui = u0[None, :] + a[None, :] / (Z + 1.0)
    ui += extra * np.sin(Z / 30.0)  # an odd nonlinearity only the cubic can follow
    u = ui.copy()
    for _ in range(50):
        u = ui - k * 640 * ((u - 640) / 640) ** 3
    return u + rng.normal(0, noise, u.shape), d, phi, rows_v


def test_calibration_picks_hyperbolic_without_distortion():
    U, d, phi, rv = _rig(k=0.0)
    cal = calibrate_triangulation(U, d, phi, rv, 640, 640)
    assert cal.kind == "hyperbolic"
    assert cal.loo_rms["hyperbolic"] < 0.01


def test_calibration_picks_distortion_with_distortion():
    U, d, phi, rv = _rig(k=0.01)
    cal = calibrate_triangulation(U, d, phi, rv, 640, 640)
    assert cal.kind == "distortion"
    Z = d[:, None] / np.cos(phi)[None, :]
    rows = np.broadcast_to(np.arange(U.shape[1]), U.shape)
    assert np.abs(cal.z(U, rows) - Z).max() < 0.03


def test_calibration_falls_back_to_cubic():
    U, d, phi, rv = _rig(extra=0.3, noise=0.003)
    cal = calibrate_triangulation(U, d, phi, rv, 640, 640)
    assert cal.kind == "cubic"


def test_calibration_roundtrip_dict():
    U, d, phi, rv = _rig()
    cal = calibrate_triangulation(U, d, phi, rv, 640, 640)
    cal2 = TriangulationCalibration.from_dict(cal.to_dict())
    assert np.allclose(cal2.z(U[3], np.arange(U.shape[1])), cal.z(U[3], np.arange(U.shape[1])))


def test_thermal_calibration_recovers_drift():
    U, d, phi, rv = _rig(noise=0.002)
    cal = calibrate_triangulation(U, d, phi, rv, 640, 640)
    truth = ThermalModel(T0=20.0, du0=[0.015, 1e-4], kappa=[1.2e-5, 0.0], db=[0.002, 0.0])
    caps = []
    Z = d[:, None] / np.cos(phi)[None, :]
    rows = np.broadcast_to(np.arange(U.shape[1]), U.shape)
    for T in (20.0, 40.0, 60.0, 80.0):
        du0, kap, db = truth.terms(T)
        # forward: z = b + (1+κ)(z0(u−Δu0) − b) + Δb  ->  find u by Newton on the calibrated model
        b = cal.params[rows, 1]
        z0_target = (Z - db - b) / (1 + kap) + b
        u = U.copy()
        for _ in range(30):
            f = cal.z(u, rows) - z0_target
            df = (cal.z(u + 1e-3, rows) - cal.z(u, rows)) / 1e-3
            u = u - f / df
        caps.append((T, u + du0))
    th = calibrate_thermal(cal, caps, d)
    cal.thermal = th
    z80 = cal.z(caps[-1][1], rows, T=80.0)
    assert np.abs(z80 - Z).max() < 0.01
    # without compensation the error would be large
    assert np.abs(cal.z(caps[-1][1], rows) - Z).max() > 0.5


# ----------------------------------------------------------------------------- head geometry


def test_wall_polar_known_answer():
    g = HeadGeometry(R0=30.0, h=2.0, H0=13.0)
    z, phi = 400.0, np.radians(5.0)
    r, a, hgt = wall_polar(z, 0.0, phi, g)
    xh = 30 + 400 * np.cos(phi)
    assert r == pytest.approx(np.sqrt(xh**2 + 4))
    assert a == pytest.approx(np.arctan(2 / xh))
    assert hgt == pytest.approx(13 + 400 * np.sin(phi))


def test_chord_through_offset_axis_is_short():
    """A chord through a point 20 mm off centre, perpendicular to the offset, is e²/R = 1.6 mm short of the
    diameter on a 500 bore (each half-chord, i.e. a radius read through the device axis, is 0.8 mm short)."""
    R, e = 250.0, 20.0
    chord = 2 * np.sqrt(R**2 - e**2)
    assert 2 * R - chord == pytest.approx(e**2 / R, rel=0.01)
    assert R - chord / 2 == pytest.approx(0.8, abs=0.01)


def test_arms_below_line_of_sight():
    dev = DeviceSpec()
    g = HeadGeometry(dev.R0_nominal, dev.h_nominal, dev.optics_height)
    for p in PRESETS.values():
        assert arm_clearance(np.radians(p.fan_lo_deg), g, dev.arm_height, dev.arm_reach) > 2.0


def test_ring_calibration_recovers_R0_and_h():
    rng = np.random.default_rng(11)
    true = HeadGeometry(R0=31.2, h=2.6, H0=13.0)
    D = 880.0
    off = np.array([9.0, -6.0])
    theta = rng.uniform(-np.pi, np.pi, 30000)
    phi = np.radians(rng.uniform(0, 9, theta.size))
    # exact ray/circle intersection for the true geometry
    dir_ = np.column_stack([np.cos(theta), np.sin(theta)])
    nrm = np.column_stack([-np.sin(theta), np.cos(theta)])
    O = true.R0 * dir_ + true.h * nrm - off
    b = (O * dir_).sum(1)
    c = (O**2).sum(1) - (D / 2) ** 2
    x = -b + np.sqrt(b**2 - c)
    z = x / np.cos(phi) + rng.normal(0, 0.05, x.size)
    res = calibrate_ring(z, theta, phi, D, HeadGeometry(30.0, 2.0, 13.0), h_measured=2.0)
    assert res.R0 == pytest.approx(true.R0, abs=0.002)
    assert res.h == pytest.approx(true.h, abs=0.5)
    assert res.D_measured == pytest.approx(D, abs=1e-4)
    assert res.offset_xy == pytest.approx(tuple(off), abs=0.01)
    pts = device_points(z, theta, phi, HeadGeometry(res.R0, res.h, 13.0))
    assert np.hypot(pts[:, 0] - off[0], pts[:, 1] - off[1]).mean() == pytest.approx(D / 2, abs=0.002)


# ----------------------------------------------------------------------------- kinematics


def test_slider_crank_known_values():
    S, L = 2660.0, 2885.0
    assert piston_displacement(0.0, S, L) == pytest.approx(0.0, abs=1e-9)
    assert piston_displacement(180.0, S, L) == pytest.approx(S, abs=1e-9)
    r = S / 2
    assert piston_displacement(90.0, S, L) == pytest.approx(r + L - np.sqrt(L**2 - r**2))
    x = piston_displacement(37.0, S, L)
    assert crank_angle_for_displacement(x, S, L) == pytest.approx(37.0, abs=1e-9)
    # numerical derivative check of the speed
    rpm = 0.1
    w = rpm * 360 / 60  # deg/s
    th = 70.0
    num = (piston_displacement(th + 1e-4, S, L) - piston_displacement(th - 1e-4, S, L)) / 2e-4 * w
    assert piston_speed(th, S, L, rpm) == pytest.approx(num, rel=1e-6)


def test_stop_angle_leaves_margin():
    e, d = EngineSpec(), DeviceSpec()
    st = plan_stop_angle(e, d, margin=40.0)
    assert overhead_clearance(st, e, d) == pytest.approx(40.0, abs=1e-6)
    assert 0 < st < 90


def test_scan_plan_is_sane():
    e, d = EngineSpec(), DeviceSpec()
    plan = plan_scan(e, d, PRESETS["L"])
    assert plan.sweep_rate_dps > 0 and plan.angle_step_deg < 1.0
    assert plan.top_depth < 250  # strip reaches above the top-ring reversal zone
    assert plan.points_per_band_min > 5000
    assert plan.alarm_clearance < 40.0 < plan.warn_clearance + 40


# ----------------------------------------------------------------------------- peak detection


def test_peak_detection_subpixel():
    rng = np.random.default_rng(2)
    for c in (100.3, 640.77, 1100.5):
        u, amp = detect_peak(render_row(c, rng=rng))
        assert u == pytest.approx(c, abs=0.25)
        assert amp > 50
    u, amp = detect_peak(render_row(500, amplitude=0.0, rng=rng))
    assert np.isnan(u)


def test_subpixel_factor_supports_N_10():
    assert subpixel_factor(n_trials=300) > 8.0
