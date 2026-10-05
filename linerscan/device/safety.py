"""Secondary crush protection: two upward multizone ToF sensors + buzzer.

Runs on the safety MCU, independent of the main processor and of the radio.  The *primary* protection is
the planned stop crank angle (``linerscan.kinematics.plan_stop_angle``) given to the turning-gear operator.

States: NORMAL → WARN (clearance below the planned-stop margin: "you are at / past the planned stop")
→ ALARM (clearance below the alarm level, or time-to-contact too short) ; FAULT when both sensors have no
valid reading while armed (fail-safe: sounds like an alarm).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

PATTERN = {"NORMAL": "off", "WARN": "warn", "ALARM": "alarm", "FAULT": "fault"}


@dataclass
class SafetyConfig:
    warn_clearance: float
    alarm_clearance: float
    ttc_alarm_s: float = 8.0
    min_valid_zones: int = 8
    fault_timeout_s: float = 1.0
    sensor_to_top: float = 1.0  # device top above the ToF aperture
    trend_window_s: float = 5.0
    alarm_hold_s: float = 10.0
    warn_debounce_s: float = 0.5  # WARN must persist this long (ALARM and FAULT act immediately)
    footprint_mm: float = 200.0  # only zones that look at the space above the device count (not the far wall)
    fov_deg: float = 45.0


def zone_angles(fov_deg: float = 45.0) -> np.ndarray:
    """Angle of each of the 64 zones from the sensor normal (rad), row-major like the sensor output."""
    g = np.radians((np.arange(8) - 3.5) * fov_deg / 8)
    ax, ay = np.meshgrid(g, g, indexing="ij")
    return np.arctan(np.hypot(np.tan(ax), np.tan(ay))).ravel()


class SafetyMonitor:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.state = "NORMAL"
        self.armed = False
        self.hist: deque = deque()
        self.last_valid_t: float | None = None
        self.alarm_since: float | None = None
        self.clearance = np.inf
        self.closing_speed = 0.0
        self._warn_since: float | None = None
        self.sensor_ok = [True, True]
        self.events: list[tuple[float, str, float]] = []
        g = zone_angles(cfg.fov_deg)
        self._cos, self._sin = np.cos(g), np.sin(g)
        self._central = g <= g.min() + 1e-9  # the 4 central zones always look at the roof

    @staticmethod
    def _robust_min(r: np.ndarray, k: int = 2) -> float:
        v = np.sort(r[np.isfinite(r)])
        return float(v[min(k - 1, v.size - 1)])

    def update(self, t: float, ranges: list[np.ndarray]) -> str:
        c = self.cfg
        per, med = [], []
        for i, r in enumerate(ranges):
            valid = np.isfinite(r)
            vert = np.where(valid, r, np.inf) * self._cos
            foot = valid & ((np.where(valid, r, np.inf) * self._sin <= c.footprint_mm) | self._central)
            ok = valid.sum() >= c.min_valid_zones and foot.sum() >= 2
            self.sensor_ok[i] = bool(ok)
            if ok:
                per.append(self._robust_min(vert[foot]))
                cz = valid & self._central
                if cz.any():
                    med.append(float(np.median(vert[cz])))
        if per:
            self.last_valid_t = t
            self.clearance = min(per) - c.sensor_to_top
            if med:
                self.hist.append((t, float(np.mean(med))))
            while self.hist and self.hist[0][0] < t - c.trend_window_s:
                self.hist.popleft()
            if len(self.hist) >= 5:
                tt, cc = np.array(self.hist).T
                slope = np.polyfit(tt - tt.mean(), cc, 1)[0]
                self.closing_speed = float(-slope)
        new = "NORMAL"
        if not per:
            if self.armed and (self.last_valid_t is None or t - self.last_valid_t > c.fault_timeout_s):
                new = "FAULT"
            else:
                new = self.state
        else:
            ttc = self.clearance / self.closing_speed if self.closing_speed > 0.5 else np.inf
            if self.clearance < c.alarm_clearance or ttc < c.ttc_alarm_s:
                new = "ALARM"
            elif self.clearance < c.warn_clearance:
                self._warn_since = self._warn_since if self._warn_since is not None else t
                new = "WARN" if (t - self._warn_since >= c.warn_debounce_s or self.state == "WARN") else self.state
            if self.clearance >= c.warn_clearance:
                self._warn_since = None
        # an alarm latches for a minimum time and while still inside the warn zone
        if self.state == "ALARM" and new != "ALARM" and new != "FAULT":
            if t - (self.alarm_since or t) < c.alarm_hold_s or self.clearance < c.warn_clearance:
                new = "ALARM"
        if new == "ALARM" and self.state != "ALARM":
            self.alarm_since = t
        if new != self.state:
            self.events.append((t, new, float(self.clearance)))
        self.state = new
        return new

    @property
    def buzzer_pattern(self) -> str:
        return PATTERN[self.state]


def roof_distance(ranges: np.ndarray, fov_deg: float = 45.0, k: int = 2) -> float:
    """Vertical distance to the lowest overhead feature seen by the central zones (height reference).

    Only the 4 central zones of each sensor are used (outer zones reach the liner wall on small bores), and the
    k-th smallest of them is taken: the lowest feature (exhaust-valve face) stays the reference even when some
    zone footprints move onto the higher cover dome as the range changes.  The noise bias of an order statistic
    grows linearly with range and is absorbed by the two-anchor scale.  ``ranges``: (2, 64).
    """
    g = zone_angles(fov_deg)
    central = g <= g.min() + 1e-9
    r = np.asarray(ranges, float)[:, central] * np.cos(g[central])
    v = np.sort(r[np.isfinite(r)])
    return float(v[min(k - 1, v.size - 1)]) if v.size >= 2 else float("nan")
