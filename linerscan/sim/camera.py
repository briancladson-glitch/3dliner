"""Ground truth of one optical module + head, and the frame renderer (ray/bore intersection → spot column).

The renderer works on the per-row level: for every frame and evaluated row it intersects that row's laser
ray with the true surface, converts the distance to a sensor column with the true camera model (pinhole +
radial distortion + thermal drift) and adds detection noise, glints and drop-outs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import OpticalPreset
from ..optics import ThermalModel


@dataclass
class HeadTruth:
    R0: float = 31.2
    h: float = 2.6
    H0: float = 13.0
    runout: tuple[float, float] = (0.002, 0.001)  # mm, 1st and 2nd harmonic of bearing runout
    runout_phase: tuple[float, float] = (0.7, 2.1)
    enc_err_deg: tuple[float, float] = (0.008, 0.004)  # residual encoder harmonics after its self-calibration
    enc_phase: tuple[float, float] = (0.3, 1.9)
    enc_bits: int = 18
    enc_noise_deg: float = 0.002

    def R0_at(self, theta):
        return (self.R0 + self.runout[0] * np.cos(theta + self.runout_phase[0])
                + self.runout[1] * np.cos(2 * theta + self.runout_phase[1]))

    def encoder(self, theta_true, rng: np.random.Generator):
        th = np.asarray(theta_true, float)
        e = (np.radians(self.enc_err_deg[0]) * np.sin(th + self.enc_phase[0])
             + np.radians(self.enc_err_deg[1]) * np.sin(2 * th + self.enc_phase[1]))
        m = th + e + np.radians(self.enc_noise_deg) * rng.standard_normal(th.shape)
        q = 2 * np.pi / 2**self.enc_bits
        return np.round(m / q) * q


@dataclass
class CameraTruth:
    preset: OpticalPreset
    phi: np.ndarray  # true row elevations (rad)
    a: np.ndarray  # px·mm
    b: float  # mm
    u0: np.ndarray  # px
    k1: float
    uc: float
    yn: np.ndarray
    thermal: ThermalModel
    hysteresis_px: float = 0.0  # unmodelled drift, fixed per scan
    sigma_u: float = 0.08
    sigma_u_far: float = 0.05
    p_glint: float = 0.002
    p_drop: float = 0.01
    x_offset: float = 0.0  # rail-zero offset (absorbed by the R0 calibration)
    extra: dict = field(default_factory=dict)

    @classmethod
    def design(cls, preset: OpticalPreset, rng: np.random.Generator, k1: float = -0.08,
               thermal: ThermalModel | None = None) -> CameraTruth:
        phi = preset.phi + np.radians(0.02) * rng.standard_normal(preset.n_rows)
        dlt = phi - np.radians(preset.tilt_deg)
        fx = preset.fx_px
        a = fx * preset.baseline_mm / np.cos(dlt) * (1 + 2e-4 * rng.standard_normal())
        b = -1.2
        x_mid = 0.5 * (preset.x_min + preset.x_max)
        z_mid = x_mid / np.cos(phi)
        w = np.linspace(-1, 1, preset.n_rows)
        u0 = preset.sensor_cols / 2 - a / (z_mid - b) + 3.0 * w  # 3 px laser roll across the fan
        th = thermal or ThermalModel(T0=20.0, du0=[0.015, 8e-5], kappa=[1.1e-5], db=[0.002])
        return cls(preset, phi, a, b, u0, k1, preset.sensor_cols / 2, np.tan(dlt), th,
                   x_offset=float(rng.uniform(-2, 2)))

    def u_of_z(self, z, row, T):
        """Noise-free sensor column for distance z along the ray of row index ``row`` at temperature T."""
        z = np.asarray(z, float)
        row = np.asarray(row)
        du0, kap, db = self.thermal.terms(T)
        ui = self.u0[row] + du0 + self.hysteresis_px + self.a[row] * (1 + kap) / (z - self.b - db)
        x = (ui - self.uc) / self.preset.fx_px
        r2 = x**2 + self.yn[row] ** 2
        return self.uc + (ui - self.uc) * (1 + self.k1 * r2)

    def measure(self, z, row, T, rng: np.random.Generator):
        u = self.u_of_z(z, row, T)
        zr = np.asarray(z, float) / self.preset.x_max
        sig = self.sigma_u + self.sigma_u_far * zr**2
        u = u + sig * rng.standard_normal(u.shape)
        g = rng.random(u.shape) < self.p_glint
        u = np.where(g, u + rng.choice([-1, 1], u.shape) * rng.uniform(3, 40, u.shape), u)
        u = np.where(rng.random(u.shape) < self.p_drop, np.nan, u)
        u = np.where((u < 2) | (u > self.preset.sensor_cols - 3), np.nan, u)
        return u


@dataclass
class DevicePose:
    """Pose of the device on the crown: liner frame point = c(s_c) + R·p_device − s_c·ẑ."""
    yaw_deg: float = 92.7
    tilt_deg: tuple[float, float] = (0.35, -0.20)  # rotation about liner x and y
    offset: tuple[float, float] = (14.0, -9.0)  # device axis from liner axis at the start (mm)
    drift_per_m: tuple[float, float] = (0.05, -0.03)  # lateral drift per metre of piston travel
    wobble: float = 0.04  # mm, slow lateral wobble amplitude

    def R(self) -> np.ndarray:
        y = np.radians(self.yaw_deg)
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        ax, ay = np.radians(self.tilt_deg)
        Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        return Rx @ Ry @ Rz

    def centre(self, s_c, s_c0: float):
        s_c = np.asarray(s_c, float)
        trav = (s_c0 - s_c) / 1000.0
        cx = self.offset[0] + self.drift_per_m[0] * trav + self.wobble * np.sin(2 * np.pi * s_c / 1500.0)
        cy = self.offset[1] + self.drift_per_m[1] * trav + self.wobble * np.cos(2 * np.pi * s_c / 1100.0)
        return np.stack([cx, cy], axis=-1)


def render_frames(theta_true, s_c, centre_xy, R_dev, surface, cam: CameraTruth, head: HeadTruth, T,
                  rng: np.random.Generator, n_iter: int = 3, return_truth: bool = False):
    """Spot columns u (F, n_rows) for frames with head angle θ, crown depth s_c and device centre (F, 2)."""
    th = np.asarray(theta_true, float)[:, None]
    F, K = th.shape[0], cam.phi.size
    phi = cam.phi[None, :]
    R0 = head.R0_at(th)
    c, s = np.cos(th), np.sin(th)
    # origin and direction in the device frame
    Od = np.stack([np.broadcast_to(R0 * c - head.h * s, (F, K)), np.broadcast_to(R0 * s + head.h * c, (F, K)),
                   np.full((F, K), head.H0)], -1)
    dd = np.stack([np.cos(phi) * c, np.cos(phi) * s, np.broadcast_to(np.sin(phi), (F, K))], -1)
    dd = np.broadcast_to(dd, (F, K, 3))
    O = Od @ R_dev.T
    d = dd @ R_dev.T
    O = O + np.concatenate([np.asarray(centre_xy)[:, None, :].repeat(K, 1),
                            -np.asarray(s_c, float)[:, None, None].repeat(K, 1)], -1)
    A = d[..., 0] ** 2 + d[..., 1] ** 2
    B = O[..., 0] * d[..., 0] + O[..., 1] * d[..., 1]
    C0 = O[..., 0] ** 2 + O[..., 1] ** 2
    r = np.full((F, K), float(np.mean(surface.radius(0.0, 100.0))))
    for _ in range(n_iter):
        z = (-B + np.sqrt(np.maximum(B**2 - A * (C0 - r**2), 0))) / A
        P = O + z[..., None] * d
        alpha = np.arctan2(P[..., 1], P[..., 0])
        depth = -P[..., 2]
        r = surface.radius(alpha, depth)
    z = (-B + np.sqrt(np.maximum(B**2 - A * (C0 - r**2), 0))) / A
    P = O + z[..., None] * d
    alpha = np.arctan2(P[..., 1], P[..., 0])
    depth = -P[..., 2]
    ok = surface.has_wall(alpha, depth)
    rows = np.broadcast_to(np.arange(K)[None, :], (F, K))
    Tb = np.broadcast_to(np.asarray(T, float).reshape(-1, 1) if np.ndim(T) else T, (F, K))
    u = cam.measure(z, rows, Tb, rng)
    u = np.where(ok, u, np.nan)
    if return_truth:
        return u, z, alpha, depth
    return u
