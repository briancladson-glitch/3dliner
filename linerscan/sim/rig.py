"""Simulated factory calibration: rail captures (lab + climatic chamber) and the master-ring scan."""

from __future__ import annotations

import numpy as np

from ..calibration import DeviceCalibration
from ..config import DeviceSpec, OpticalPreset
from ..device.head import pass_duration, sweep_angle
from ..geometry import HeadGeometry, calibrate_ring
from ..optics import TriangulationCalibration, calibrate_thermal, calibrate_triangulation
from .camera import CameraTruth, DevicePose, HeadTruth, render_frames
from .liner import RingModel


def rail_capture(cam: CameraTruth, d_rail: np.ndarray, T: float, rng: np.random.Generator, n_avg: int = 50,
                 rail_sigma: float = 0.003) -> np.ndarray:
    """Flat matte target stepped along a rail with a 5 µm scale; spot columns averaged over n_avg frames."""
    x = d_rail + rail_sigma * rng.standard_normal(d_rail.size) + cam.x_offset
    z = x[:, None] / np.cos(cam.phi)[None, :]
    rows = np.broadcast_to(np.arange(cam.phi.size), z.shape)
    u = cam.u_of_z(z, rows, T)
    return u + (cam.sigma_u + cam.sigma_u_far) / np.sqrt(n_avg) * rng.standard_normal(u.shape)


def ring_scan(cam: CameraTruth, head: HeadTruth, ring: RingModel, T_device: float, rng: np.random.Generator,
              pose: DevicePose | None = None, frame_rate: float = 200.0, rate_dps: float = 110.0,
              n_passes: int = 2):
    """Scan inside a master ring standing on a flat plate.  Returns (u (F,K), encoder θ (F,))."""
    pose = pose or DevicePose(yaw_deg=0.0, tilt_deg=(0.15, -0.1), offset=(9.0, -6.0), drift_per_m=(0, 0), wobble=0)
    T_pass = pass_duration(rate_dps, 2000.0, 190.0)
    t = np.arange(0, n_passes * T_pass, 1 / frame_rate)
    th = np.radians(sweep_angle(t, rate_dps, 2000.0, 190.0))
    s_c = np.full(t.size, ring.height)
    centre = np.tile(np.array(pose.offset), (t.size, 1))
    u = render_frames(th, s_c, centre, pose.R(), ring, cam, head, T_device, rng)
    return u, head.encoder(th, rng)


def points_z(tri: TriangulationCalibration, u: np.ndarray, T: float | np.ndarray | None):
    """Distances along the rays for an (F, K) block of spot columns (NaN kept)."""
    F, K = u.shape
    rows = np.broadcast_to(np.arange(K), (F, K))
    if T is not None and np.ndim(T):
        T = np.broadcast_to(np.asarray(T).reshape(-1, 1), (F, K))
    return tri.z(u, rows, T)


def ring_points(tri, u, theta, T):
    z = points_z(tri, u, T)
    F, K = u.shape
    m = np.isfinite(z)
    th = np.broadcast_to(theta[:, None], (F, K))[m]
    ph = np.broadcast_to(tri.phi[None, :], (F, K))[m]
    return z[m], th, ph


def factory_calibration(preset: OpticalPreset, cam: CameraTruth, head: HeadTruth, rng: np.random.Generator,
                        dev: DeviceSpec | None = None, T_lab: float = 21.0,
                        chamber_temps: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0),
                        ring_D20: float | None = None, h_measured: float | None = None) -> DeviceCalibration:
    dev = dev or DeviceSpec()
    d_rail = np.linspace(preset.x_min - 10, preset.x_max + 10, 15)
    phi_hat = cam.phi + np.radians(0.005) * rng.standard_normal(cam.phi.size)
    U = rail_capture(cam, d_rail, T_lab, rng)
    tri = calibrate_triangulation(U, d_rail, phi_hat, preset.rows_v(), cam.uc, cam.uc)
    caps = [(T, rail_capture(cam, d_rail, T, rng)) for T in chamber_temps]
    tri.thermal = calibrate_thermal(tri, caps, d_rail, T0=T_lab)
    # master ring, certified at 20 °C, scanned in the lab
    ring_D20 = ring_D20 or round(0.5 * (preset.bore_min + preset.bore_max) / 10) * 10
    ring = RingModel(ring_D20, temperature_c=T_lab)
    u, th = ring_scan(cam, head, ring, T_lab, rng)
    z, th, ph = ring_points(tri, u, th, T_lab)
    h_meas = h_measured if h_measured is not None else head.h + 0.4 * rng.standard_normal()
    g0 = HeadGeometry(dev.R0_nominal, h_meas, dev.optics_height)
    rc = calibrate_ring(z, th, ph, ring.D_at_temp, g0, h_measured=h_meas)
    return DeviceCalibration(preset.name, tri, HeadGeometry(rc.R0, rc.h, dev.optics_height), ring_D20,
                             rc.D_measured, rc.rms, T_lab)
