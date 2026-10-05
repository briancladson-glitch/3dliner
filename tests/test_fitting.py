import numpy as np
import pytest

from linerscan.fitting import (
    diameter_stats,
    fit_circle,
    fit_cylinder,
    fit_ellipse,
    fit_fourier,
    fourier_residuals,
    kasa_circle,
    outlier_mask,
    robust_threshold,
    rotation_to_z,
    to_axis_frame,
)


def test_kasa_exact_circle():
    a = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    x, y = 12.5 + 490 * np.cos(a), -7.25 + 490 * np.sin(a)
    xc, yc, R = kasa_circle(x, y)
    assert xc == pytest.approx(12.5, abs=1e-9)
    assert yc == pytest.approx(-7.25, abs=1e-9)
    assert R == pytest.approx(490, abs=1e-9)


def test_geometric_circle_noisy_partial_arc():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, np.pi, 4000)  # half circle: Kåsa is biased, LM is not
    r = 300 + rng.normal(0, 0.1, a.size)
    x, y = 5 + r * np.cos(a), 2 + r * np.sin(a)
    c = fit_circle(x, y)
    assert c.R == pytest.approx(300, abs=0.01)
    assert c.xc == pytest.approx(5, abs=0.01)
    assert c.rms == pytest.approx(0.1, rel=0.05)


def test_cylinder_recovers_tilt_offset_radius():
    rng = np.random.default_rng(4)
    n = 20000
    t = np.radians(1.0)  # 1 degree tilt about y (axis leans towards +x)
    u = np.array([np.sin(t), 0.0, np.cos(t)])
    a = rng.uniform(0, 2 * np.pi, n)
    s = rng.uniform(0, 30, n)
    # orthonormal frame around the axis
    e1 = np.array([np.cos(t), 0, -np.sin(t)])
    e2 = np.array([0, 1.0, 0])
    R = 490.0
    P = (np.array([20.0, -10.0, 0.0]) + s[:, None] * u + R * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2))
    P += rng.normal(0, 0.1, P.shape) * 0.577
    cyl = fit_cylinder(P)
    tx, ty = cyl.tilt_deg
    assert tx == pytest.approx(1.0, abs=0.01)
    assert ty == pytest.approx(0.0, abs=0.01)
    assert cyl.R == pytest.approx(R, abs=0.005)
    Q = to_axis_frame(P, cyl)
    assert np.hypot(Q[:, 0], Q[:, 1]).mean() == pytest.approx(R, abs=0.005)


def test_plane_cut_through_tilted_bore_looks_oval():
    """Error of a horizontal cut through a bore tilted by θ is ≈ D·θ²/2 (0.15 mm at 1° on 980 mm)."""
    D, th = 980.0, np.radians(1.0)
    assert D / np.cos(th) - D == pytest.approx(D * th**2 / 2, rel=1e-3)
    assert D * th**2 / 2 == pytest.approx(0.149, abs=0.001)


def test_rotation_to_z():
    u = np.array([0.1, -0.2, 0.97])
    u /= np.linalg.norm(u)
    Rm = rotation_to_z(u)
    assert np.allclose(Rm @ u, [0, 0, 1])
    assert np.allclose(Rm @ Rm.T, np.eye(3))


def test_fourier_diameters_known_shape():
    a = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    # radius with offset harmonic (1st), ovality (2nd), 3rd and 4th harmonics
    rho = 490 + 0.3 * np.cos(a) + 0.05 * np.cos(2 * (a - np.radians(30))) + 0.02 * np.sin(3 * a) \
        + 0.01 * np.cos(4 * a)
    f = fit_fourier(a, rho, order=8)
    st = diameter_stats(f, fa_angle_deg=0.0)
    D = lambda x: 980 + 0.1 * np.cos(2 * (x - np.radians(30))) + 0.02 * np.cos(4 * x)  # noqa: E731
    assert st.D_FA == pytest.approx(D(0.0), abs=1e-9)
    assert st.D_PS == pytest.approx(D(np.pi / 2), abs=1e-9)
    assert st.ovality == pytest.approx(abs(D(np.pi / 2) - D(0.0)), abs=1e-9)
    grid = np.radians(np.arange(0, 180, 0.01))
    assert st.D_max == pytest.approx(D(grid).max(), abs=1e-6)
    assert st.D_min == pytest.approx(D(grid).min(), abs=1e-6)
    assert st.max_minus_min == pytest.approx(st.D_max - st.D_min)
    assert st.D_mean == pytest.approx(980.0, abs=1e-9)


def test_fourier_noise_averaging_and_taper():
    rng = np.random.default_rng(5)
    n = 20000
    a = rng.uniform(0, 2 * np.pi, n)
    s = rng.uniform(-15, 15, n)
    sig = 0.1
    rho = 400 + 0.002 * s + 0.05 * np.cos(2 * a) + rng.normal(0, sig, n)
    f = fit_fourier(a, rho, order=8, s=s)
    st = diameter_stats(f)
    assert st.D_mean == pytest.approx(800.0, abs=5 * st.u_D_FA)
    # statistical uncertainty of D(α) is of order 6σ/√N or better
    assert st.u_D_FA < 6 * sig / np.sqrt(n)
    res = fourier_residuals(f, a, rho, s)
    assert np.std(res) == pytest.approx(sig, rel=0.05)


def test_ellipse_direct_fit():
    rng = np.random.default_rng(6)
    t = rng.uniform(0, 2 * np.pi, 5000)
    A, B, ang = 490.2, 489.9, np.radians(35)
    x0, y0 = 3.0, -4.0
    x = x0 + A * np.cos(t) * np.cos(ang) - B * np.sin(t) * np.sin(ang)
    y = y0 + A * np.cos(t) * np.sin(ang) + B * np.sin(t) * np.cos(ang)
    e = fit_ellipse(x + rng.normal(0, 0.01, t.size), y + rng.normal(0, 0.01, t.size))
    assert e.a == pytest.approx(A, abs=0.003)
    assert e.b == pytest.approx(B, abs=0.003)
    assert e.angle_deg == pytest.approx(35.0, abs=1.0)
    assert e.xc == pytest.approx(x0, abs=0.01)


def test_outlier_rule():
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, 1000)
    # tight data: the 0.1 mm floor governs
    assert robust_threshold(r) == pytest.approx(0.1)
    r[:5] = [0.5, -0.4, 0.2, 0.11, -0.3]
    m = outlier_mask(r)
    assert m[:5].all() and m.sum() == 5
    # noisy data: 5·1.4826·MAD governs
    r2 = rng.normal(0, 0.1, 100000)
    assert robust_threshold(r2) == pytest.approx(0.5, rel=0.03)
