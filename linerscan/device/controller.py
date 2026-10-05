"""On-board scan controller (main processor firmware logic), written against the driver interfaces.

SOAK    laser off, head parked; wait for thermal equilibrium (dT/dt small) or the soak time-out.
ARMED   laser on (interlock: both ToF see a roof < 4 m), head sweeping ±190°, recording; wait for motion.
SCAN    piston rising (overhead clearance closing); keep recording; quick-look diameters over BLE.
STOPPED motion ended; post-roll recording, then laser off, head parked, file closed, summary sent.

The safety monitor is updated every tick from the ToF sensors and drives the buzzer directly.  Status goes
out over BLE every second, best effort.  Nothing safety-related waits for the radio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..fitting import kasa_circle
from ..rawscan import RawScan, RawScanWriter
from .drivers import Drivers
from .safety import SafetyConfig, SafetyMonitor


@dataclass
class ControllerConfig:
    frame_rate: float
    sweep_rate_dps: float
    sweep_accel_dps2: float = 2000.0
    sweep_half_deg: float = 190.0
    tick_s: float = 0.1
    soak_max_s: float = 900.0
    soak_min_s: float = 60.0
    soak_rate_K_per_min: float = 0.3
    motion_speed_mm_s: float = 1.5
    motion_confirm_s: float = 2.0
    stop_confirm_s: float = 6.0
    post_roll_s: float = 4.0
    max_run_s: float = 3600.0
    status_period_s: float = 1.0
    interlock_range_mm: float = 4000.0
    quicklook_period_s: float = 5.0
    fault_end_s: float = 30.0  # a persistent safety fault ends the recording (the operator has stopped)
    safety: SafetyConfig = field(default_factory=lambda: SafetyConfig(55.0, 35.0))


class ScanController:
    def __init__(self, drv: Drivers, cfg: ControllerConfig, meta: dict, n_rows: int, quicklook=None):
        self.drv, self.cfg = drv, cfg
        self.safety = SafetyMonitor(cfg.safety)
        self.rec = RawScanWriter(meta, n_rows)
        self.state = "SOAK"
        self.quicklook = quicklook  # callable(FrameBlock-like arrays) -> dict, optional
        self.status_log: list[dict] = []
        self._ql_buf: list = []

    def _set_state(self, t, s, text=""):
        self.state = s
        self.rec.event(t, "state", f"{s} {text}".strip())
        self.drv.radio.send({"t": round(t, 2), "type": "state", "state": s, "text": text})

    def run(self) -> RawScan:
        d, c = self.drv, self.cfg
        t = d.clock.now()
        t_temp_next = t
        t_status_next = t
        t_ql_next = t
        temp_hist: list[tuple[float, float]] = []
        t_soak0 = t
        t_motion = None
        t_still = None
        t_stopped = None
        self._set_state(t, "SOAK")
        while True:
            t = d.clock.now() + c.tick_s
            d.clock.wait_until(t)
            # ---------------------------------------------------------------- safety first
            ranges = [s.read() for s in d.tof]
            st = self.safety.update(t, ranges)
            d.buzzer.set(self.safety.buzzer_pattern)
            self.rec.tof(t, np.stack(ranges))
            if d.rangefinder is not None:
                self.rec.rangefinder(t, d.rangefinder.read())
            if self.safety.events and self.safety.events[-1][0] == t:
                ev = self.safety.events[-1]
                self.rec.event(t, "safety", f"{ev[1]} clearance={ev[2]:.0f}mm")
                d.radio.send({"t": round(t, 2), "type": "safety", "state": ev[1], "clearance_mm": round(ev[2])})
            # ---------------------------------------------------------------- housekeeping
            if t >= t_temp_next:
                T = d.temps.read()
                self.rec.temps(t, T)
                temp_hist.append((t, float(T[0])))
                t_temp_next = t + 1.0
            if self.state in ("ARMED", "SCAN", "STOPPED"):
                fb = d.camera.read_frames()
                self.rec.frames(fb.t, fb.theta, fb.u)
                if self.quicklook is not None and fb.t.size:
                    self._ql_buf.append(fb)
            # ---------------------------------------------------------------- state machine
            if self.state == "SOAK":
                rate = np.inf
                if len(temp_hist) > 60:
                    tt, TT = np.array(temp_hist[-60:]).T
                    rate = abs(np.polyfit(tt, TT, 1)[0]) * 60
                inside = all(np.nanmin(r) < c.interlock_range_mm if np.isfinite(r).any() else False for r in ranges)
                if (t - t_soak0 >= c.soak_min_s and (rate < c.soak_rate_K_per_min or t - t_soak0 >= c.soak_max_s)
                        and inside):
                    d.laser.set(True)
                    self.safety.armed = True
                    d.head.start_sweep(c.sweep_rate_dps, c.sweep_accel_dps2, c.sweep_half_deg)
                    d.camera.start(c.frame_rate)
                    why = f"dT/dt {rate:.2f} K/min" if np.isfinite(rate) else "soak time limit"
                    self._set_state(t, "ARMED", f"after {t - t_soak0:.0f} s soak ({why})")
            elif self.state == "ARMED":
                if self.safety.closing_speed > c.motion_speed_mm_s:
                    t_motion = t_motion or t
                    if t - t_motion >= c.motion_confirm_s:
                        self._set_state(t, "SCAN", "piston rising")
                else:
                    t_motion = None
            if self.state in ("ARMED", "SCAN") and self.safety.state == "FAULT":
                t_fault = self.safety.events[-1][0] if self.safety.events else t
                if t - t_fault >= c.fault_end_s:
                    t_stopped = t
                    self._set_state(t, "STOPPED", "ToF fault: recording ended")
            if self.state == "SCAN":
                if abs(self.safety.closing_speed) < c.motion_speed_mm_s / 2:
                    t_still = t_still or t
                    if t - t_still >= c.stop_confirm_s:
                        t_stopped = t
                        self._set_state(t, "STOPPED", f"clearance {self.safety.clearance:.0f}mm")
                else:
                    t_still = None
            elif self.state == "STOPPED":
                if t - t_stopped >= c.post_roll_s:
                    d.camera.stop()
                    d.laser.set(False)
                    d.head.park()
                    self._set_state(t, "DONE", f"{sum(x.size for x in self.rec.ft)} frames")
            # ---------------------------------------------------------------- BLE status / quick look
            if t >= t_status_next:
                msg = {"t": round(t, 1), "type": "status", "state": self.state, "safety": st,
                       "clearance_mm": None if not np.isfinite(self.safety.clearance) else round(self.safety.clearance),
                       "closing_mm_s": round(self.safety.closing_speed, 2),
                       "T_cam": round(temp_hist[-1][1], 2) if temp_hist else None,
                       "frames": int(sum(x.size for x in self.rec.ft))}
                self.status_log.append(msg)
                d.radio.send(msg)
                t_status_next = t + c.status_period_s
            if self.quicklook is not None and self.state == "SCAN" and t >= t_ql_next and self._ql_buf:
                ql = self.quicklook(self._ql_buf)
                self._ql_buf = []
                if ql:
                    ql.update({"t": round(t, 1), "type": "quicklook"})
                    d.radio.send(ql)
                    self.rec.event(t, "quicklook", f"D~{ql.get('D_mm', float('nan')):.2f}")
                t_ql_next = t + c.quicklook_period_s
            if self.state == "DONE" or t > c.max_run_s:
                break
        raw = self.rec.close()
        raw.meta["safety_events"] = [list(e) for e in self.safety.events]
        return raw


def make_quicklook(cal, geom, decimate: int = 8):
    """On-board quick look: Kåsa circle through a decimated slice of the last few seconds (device frame)."""
    from ..geometry import device_points

    def ql(blocks):
        t = np.concatenate([b.t for b in blocks])[::decimate]
        th = np.concatenate([b.theta for b in blocks])[::decimate]
        u = np.concatenate([b.u for b in blocks])[::decimate]
        if t.size < 20:
            return None
        k = u.shape[1] // 2
        uu = u[:, k]
        m = np.isfinite(uu)
        if m.sum() < 20:
            return None
        z = cal.z(uu[m], np.full(m.sum(), k))
        P = device_points(z, th[m], cal.phi[k], geom)
        _, _, R = kasa_circle(P[:, 0], P[:, 1])
        return {"D_mm": round(2 * R, 2), "n": int(m.sum())}

    return ql
