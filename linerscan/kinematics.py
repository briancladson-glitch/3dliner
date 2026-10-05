"""Engine kinematics, the planned stop crank angle (primary crush safety) and scan planning.

Crank angle convention: degrees *before* TDC (BTDC).  BDC = 180°, TDC = 0°.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .config import DeviceSpec, EngineSpec, OpticalPreset


def piston_displacement(theta_btdc_deg, stroke: float, conrod: float):
    """Distance of the piston below its TDC position (slider-crank)."""
    th = np.radians(np.asarray(theta_btdc_deg, float))
    r = stroke / 2.0
    return r * (1 - np.cos(th)) + conrod - np.sqrt(conrod**2 - (r * np.sin(th)) ** 2)


def crank_angle_for_displacement(x: float, stroke: float, conrod: float) -> float:
    """Inverse of piston_displacement on [0°, 180°]."""
    if not 0 <= x <= stroke:
        raise ValueError("displacement outside stroke")
    if x == 0:
        return 0.0
    if x == stroke:
        return 180.0
    return float(brentq(lambda t: piston_displacement(t, stroke, conrod) - x, 0.0, 180.0, xtol=1e-12))


def piston_speed(theta_btdc_deg, stroke: float, conrod: float, crank_rpm: float):
    """|dx/dt| in mm/s for a crank turning at crank_rpm."""
    th = np.radians(np.asarray(theta_btdc_deg, float))
    r = stroke / 2.0
    w = crank_rpm * 2 * np.pi / 60.0
    lam = r / conrod
    return np.abs(w * r * np.sin(th) * (1 + lam * np.cos(th) / np.sqrt(1 - (lam * np.sin(th)) ** 2)))


def crown_depth(theta_btdc_deg, e: EngineSpec):
    """Depth of the crown below the liner top."""
    return e.crown_depth_tdc + piston_displacement(theta_btdc_deg, e.stroke, e.conrod)


def overhead_clearance(theta_btdc_deg, e: EngineSpec, d: DeviceSpec):
    """Free space between the top of the device and the lowest overhead feature."""
    return e.overhead_clearance_tdc + piston_displacement(theta_btdc_deg, e.stroke, e.conrod) - d.height


def plan_stop_angle(e: EngineSpec, d: DeviceSpec, margin: float = 40.0, min_angle_deg: float = 5.0) -> float:
    """Planned stop crank angle (BTDC) leaving ``margin`` mm between device top and the lowest overhead feature.

    Never closer to TDC than ``min_angle_deg``.
    """
    need = margin + d.height - e.overhead_clearance_tdc  # required displacement below TDC
    if need <= 0:
        return float(min_angle_deg)
    return max(crank_angle_for_displacement(need, e.stroke, e.conrod), float(min_angle_deg))


@dataclass
class ScanPlan:
    preset: str
    stop_angle_btdc: float
    turning_gear_rpm: float
    duration_s: float
    max_piston_speed: float
    strip_length_min: float
    sweep_rate_dps: float
    frame_rate: float
    angle_step_deg: float
    top_depth: float  # shallowest depth reached by the strip at the stop
    bottom_depth: float  # deepest depth seen at BDC
    points_per_band_min: float
    alarm_clearance: float
    warn_clearance: float


def plan_scan(e: EngineSpec, d: DeviceSpec, p: OpticalPreset, turning_gear_rpm: float = 0.1,
              margin: float = 40.0, frame_rate: float = 200.0, coverage_margin: float = 0.75,
              band: float = 30.0, reaction_s: float = 5.0) -> ScanPlan:
    """Size the sweep so every height is swept over the full ±190° while it is inside the laser strip."""
    stop = plan_stop_angle(e, d, margin)
    th = np.linspace(stop, 180.0, 2001)
    vmax = float(piston_speed(th, e.stroke, e.conrod, turning_gear_rpm).max())
    x_nom = e.bore / 2.0 - d.R0_nominal
    x_min = max(x_nom - 25.0, p.x_min)  # worst case: device 25 mm off centre
    strip = x_min * (np.tan(np.radians(p.fan_hi_deg)) - np.tan(np.radians(p.fan_lo_deg)))
    t_in_strip = strip / vmax
    sweep_rate = 2 * d.sweep_half_deg / (coverage_margin * t_in_strip)
    duration = (180.0 - stop) / (turning_gear_rpm * 6.0)
    strip_top = x_nom * np.tan(np.radians(p.fan_hi_deg)) + d.optics_height
    top = float(crown_depth(stop, e) - strip_top)
    bottom = float(crown_depth(180.0, e) - d.optics_height)
    n_rows_per_mm = p.n_rows / strip
    pts = frame_rate * n_rows_per_mm * strip / vmax * band
    # alarm when the remaining clearance would be used up within the operator's reaction time at the
    # speed the piston has near the planned stop (speed falls towards TDC, so this is conservative)
    v_stop = float(piston_speed(stop, e.stroke, e.conrod, turning_gear_rpm))
    alarm = max(10.0, v_stop * reaction_s * 2.0)
    warn = alarm + 0.5 * margin
    return ScanPlan(p.name, float(stop), turning_gear_rpm, float(duration), vmax, float(strip),
                    float(sweep_rate), frame_rate, float(sweep_rate / frame_rate), top, bottom, float(pts),
                    float(alarm), float(warn))
