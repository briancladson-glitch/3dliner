"""The simulated world the device lives in: turning gear + operator, device temperature, ToF view of the
cylinder cover, and the head motor.  Drivers in ``linerscan.device.sim_drivers`` read from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EngineSpec
from ..device.head import sweep_angle
from ..kinematics import crown_depth
from .camera import CameraTruth, DevicePose, HeadTruth, render_frames


@dataclass
class ThermalTruth:
    T_start: float = 36.0  # engine-room temperature of the device when it goes in
    T_liner: float = 62.0
    tau_s: float = 360.0
    laser_offset: float = 0.6
    battery_lag: float = 2.5
    sensor_noise: float = 0.02

    def device_T(self, t):
        t = np.maximum(np.asarray(t, float), 0.0)
        return self.T_liner - (self.T_liner - self.T_start) * np.exp(-t / self.tau_s)

    def sensors(self, t, rng):
        Tc = self.device_T(t)
        Tl = Tc + self.laser_offset
        Tb = self.T_liner - (self.T_liner - self.T_start) * np.exp(-max(t, 0) / (self.tau_s * self.battery_lag))
        return np.array([Tc, Tl, Tb]) + self.sensor_noise * rng.standard_normal(3)


@dataclass
class ToFTruth:
    """Two upward VL53L8CX-class 8×8 multizone sensors on the device top."""
    positions: tuple = ((60.0, 0.0, 19.0), (-60.0, 0.0, 19.0))
    fov_deg: float = 45.0
    bias: tuple[float, float] = (3.0, -2.0)
    scale: tuple[float, float] = (1.004, 0.996)
    noise_mm: float = 2.0
    noise_rel: float = 0.0025
    p_invalid: float = 0.02
    max_range: float = 4000.0
    valve_radius_frac: float = 0.18
    dome_rise: float = 25.0
    failed: tuple[bool, bool] = (False, False)

    def zone_dirs(self) -> np.ndarray:
        g = np.radians((np.arange(8) - 3.5) * self.fov_deg / 8)
        ax, ay = np.meshgrid(g, g, indexing="ij")
        d = np.stack([np.tan(ax), np.tan(ay), np.ones_like(ax)], -1).reshape(64, 3)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def true_ranges(self, s_c: float, centre_xy, R_dev, e: EngineSpec) -> np.ndarray:
        dirs = self.zone_dirs() @ R_dev.T
        out = np.empty((2, 64))
        Rb = e.bore / 2
        s_valve = e.crown_depth_tdc - e.overhead_clearance_tdc
        for i, p in enumerate(self.positions):
            O = R_dev @ np.array(p) + np.array([centre_xy[0], centre_xy[1], -s_c])
            best = np.full(64, np.inf)
            for s_plane, rmin, rmax in ((s_valve, 0.0, self.valve_radius_frac * e.bore),
                                        (s_valve - self.dome_rise, self.valve_radius_frac * e.bore, Rb)):
                t = (-s_plane - O[2]) / dirs[:, 2]
                P = O + t[:, None] * dirs
                rr = np.hypot(P[:, 0], P[:, 1])
                ok = (t > 0) & (rr >= rmin) & (rr < rmax)
                best = np.where(ok, np.minimum(best, t), best)
            # liner / cover bore wall
            A = dirs[:, 0] ** 2 + dirs[:, 1] ** 2
            B = O[0] * dirs[:, 0] + O[1] * dirs[:, 1]
            C = O[0] ** 2 + O[1] ** 2 - Rb**2
            tw = (-B + np.sqrt(np.maximum(B**2 - A * C, 0))) / np.maximum(A, 1e-12)
            best = np.minimum(best, tw)
            out[i] = best
        return out

    def read(self, s_c, centre_xy, R_dev, e: EngineSpec, rng) -> np.ndarray:
        r = self.true_ranges(s_c, centre_xy, R_dev, e)
        sc = np.array(self.scale)[:, None]
        bi = np.array(self.bias)[:, None]
        m = r * sc + bi + (self.noise_mm + self.noise_rel * r) * rng.standard_normal(r.shape)
        bad = (rng.random(r.shape) < self.p_invalid) | (r > self.max_range)
        bad[np.array(self.failed)] = True
        return np.where(bad, np.nan, m)


class EngineMotion:
    """Turning gear driven by an operator who stops at the planned angle (or misses it and reacts to the buzzer)."""

    def __init__(self, e: EngineSpec, rpm: float = 0.1, t_start: float = 60.0, stop_btdc: float = 15.0,
                 read_error_deg: float = 0.2, overshoot: bool = False, reaction_s: float = 4.0,
                 stop_time_s: float = 0.5, fluct: float = 0.04, period_s: float = 45.0, dt: float = 0.01):
        self.e, self.rpm, self.t_start, self.stop_btdc = e, rpm, t_start, stop_btdc
        self.read_error_deg, self.overshoot, self.reaction_s = read_error_deg, overshoot, reaction_s
        self.stop_time_s, self.fluct, self.period_s, self.dt = stop_time_s, fluct, period_s, dt
        self._chunks_t: list[np.ndarray] = [np.array([0.0])]
        self._chunks_a: list[np.ndarray] = [np.array([180.0])]
        self._full: tuple[np.ndarray, np.ndarray] | None = None
        self.t_cmd: float | None = None
        self.t_alarm_heard: float | None = None

    @property
    def t_hist(self) -> np.ndarray:
        return self._history()[0]

    @property
    def a_hist(self) -> np.ndarray:
        return self._history()[1]

    def _history(self):
        if self._full is None:
            self._full = (np.concatenate(self._chunks_t), np.concatenate(self._chunks_a))
            self._chunks_t, self._chunks_a = [self._full[0]], [self._full[1]]
        return self._full

    @property
    def angle(self) -> float:
        return float(self._chunks_a[-1][-1])

    def advance(self, t1: float, alarm_sounding: bool = False) -> None:
        t, a = float(self._chunks_t[-1][-1]), float(self._chunks_a[-1][-1])
        tl, al = [], []
        if alarm_sounding and self.t_alarm_heard is None:
            self.t_alarm_heard = t
        while t < t1 - 1e-12:
            dt = min(self.dt, t1 - t)
            if self.t_cmd is None:
                if not self.overshoot and t >= self.t_start and a <= self.stop_btdc + self.read_error_deg:
                    self.t_cmd = t
                if self.t_alarm_heard is not None and t >= self.t_alarm_heard + self.reaction_s:
                    self.t_cmd = t
            w = 0.0
            if t >= self.t_start:
                w = self.rpm * 6.0 * (1 + self.fluct * np.sin(2 * np.pi * (t - self.t_start) / self.period_s))
                if self.t_cmd is not None:
                    w *= max(0.0, 1 - (t - self.t_cmd) / self.stop_time_s)
            a -= w * dt
            t += dt
            tl.append(t)
            al.append(a)
        if tl:
            self._chunks_t.append(np.array(tl))
            self._chunks_a.append(np.array(al))
            self._full = None

    def angle_at(self, t):
        t = np.asarray(t, float)
        if len(self._chunks_t) >= 2 and t.size and np.min(t) >= self._chunks_t[-2][0]:
            return np.interp(t, np.concatenate(self._chunks_t[-2:]), np.concatenate(self._chunks_a[-2:]))
        th, ah = self._history()
        return np.interp(t, th, ah)

    def crown_depth_at(self, t):
        return crown_depth(np.abs(self.angle_at(t)), self.e)



@dataclass
class SweepCommand:
    t0: float
    rate_dps: float
    accel_dps2: float
    half_deg: float


@dataclass
class SimWorld:
    engine: EngineSpec
    surface: object  # LinerModel or RingModel
    cam: CameraTruth
    head: HeadTruth
    pose: DevicePose
    thermal: ThermalTruth
    tof: ToFTruth
    motion: EngineMotion
    seed: int = 0
    sweep: SweepCommand | None = None
    laser_on: bool = False
    camera_enabled: bool = True  # False: skip rendering (safety-only simulations)
    rng: np.random.Generator = field(init=False)
    truth_log: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.R_dev = self.pose.R()
        self.s_c0 = float(self.motion.crown_depth_at(0.0))

    def head_angle(self, t):
        if self.sweep is None:
            return np.zeros_like(np.asarray(t, float))
        s = self.sweep
        return np.radians(sweep_angle(np.asarray(t, float) - s.t0, s.rate_dps, s.accel_dps2, s.half_deg))

    def centre(self, t):
        return self.pose.centre(self.motion.crown_depth_at(t), self.s_c0)

    def frames(self, t: np.ndarray):
        """Render camera frames at times t: returns (u (F, K), encoder angle (F,))."""
        t = np.asarray(t, float)
        th = self.head_angle(t)
        if not self.laser_on or t.size == 0 or not self.camera_enabled:
            return np.full((t.size, self.cam.phi.size), np.nan), self.head.encoder(th, self.rng)
        s_c = self.motion.crown_depth_at(t)
        u = render_frames(th, s_c, self.centre(t), self.R_dev, self.surface, self.cam, self.head,
                          self.thermal.device_T(t), self.rng)
        return u, self.head.encoder(th, self.rng)

    def tof_read(self, t: float) -> np.ndarray:
        fail_at = getattr(self, "extra_fail_at", None)
        if fail_at is not None and t >= fail_at:
            self.tof.failed = (True, True)
        s_c = float(self.motion.crown_depth_at(t))
        return self.tof.read(s_c, self.centre(t), self.R_dev, self.engine, self.rng)

    def true_clearance(self, t: float, device_height: float = 20.0) -> float:
        from ..config import DeviceSpec
        from ..kinematics import overhead_clearance
        return float(overhead_clearance(abs(float(self.motion.angle_at(t))), self.engine,
                                        DeviceSpec(height=device_height)))

    def temps(self, t: float) -> np.ndarray:
        return self.thermal.sensors(t, self.rng)
