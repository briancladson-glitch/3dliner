"""Head sweep profile: back-and-forth ±half_deg with trapezoidal velocity (constant rate, finite accel)."""

from __future__ import annotations

import numpy as np


def sweep_angle(t, rate_dps: float, accel_dps2: float, half_deg: float):
    """Commanded head angle (deg) at time t after the sweep start; starts at −half moving positive."""
    t = np.maximum(np.asarray(t, float), 0.0)
    ta = rate_dps / accel_dps2
    Th = 2 * half_deg / rate_dps + ta  # duration of one pass
    k = np.floor(t / Th)
    tau = t - k * Th
    p = np.where(tau < ta, -half_deg + 0.5 * accel_dps2 * tau**2,
                 np.where(tau < Th - ta, -half_deg + 0.5 * accel_dps2 * ta**2 + rate_dps * (tau - ta),
                          half_deg - 0.5 * accel_dps2 * (Th - tau) ** 2))
    return np.where(k % 2 == 0, p, -p)


def pass_duration(rate_dps: float, accel_dps2: float, half_deg: float) -> float:
    return 2 * half_deg / rate_dps + rate_dps / accel_dps2
