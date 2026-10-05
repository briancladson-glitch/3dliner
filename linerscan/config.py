"""Configuration objects: engine, optical presets, device, processing and report settings.

Every default here that is a judgement call is listed in docs/DECISIONS.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import numpy as np


@dataclass
class EngineSpec:
    """Geometry of one engine type (all mm, angles in degrees)."""
    name: str = "2-stroke crosshead, 980 bore (simulated)"
    bore: float = 980.0
    stroke: float = 2660.0
    conrod: float = 2885.0
    crown_depth_tdc: float = 15.0  # crown (periphery) below liner top at TDC
    overhead_clearance_tdc: float = 12.0  # crown to lowest cover / exhaust-valve feature at TDC
    port_top_depth: float = 2400.0  # top edge of the scavenge ports below liner top
    port_bottom_depth: float = 2700.0
    n_ports: int = 24
    port_width_deg: float = 11.0
    port_phase_deg: float = 7.5  # centre of first port, measured from FORE
    liner_cte: float = 11.0e-6  # grey cast iron, 1/K
    measuring_depths: list[float] = field(default_factory=lambda: [
        40, 80, 120, 160, 200, 250, 300, 400, 550, 700, 900, 1150, 1400, 1700, 2000, 2300])

    @property
    def crank_radius(self) -> float:
        return self.stroke / 2.0


@dataclass
class OpticalPreset:
    """One of the three factory-calibrated optical modules (camera + lens + 405 nm line laser on an Invar bench)."""
    name: str
    bore_min: float
    bore_max: float
    focal_mm: float
    baseline_mm: float
    fan_lo_deg: float
    fan_hi_deg: float
    x_min: float  # horizontal working range of the laser spot from the beam origin
    x_max: float
    pixel_mm: float = 0.003
    sensor_cols: int = 1280
    sensor_rows: int = 800
    n_rows: int = 48  # (binned) rows evaluated per frame
    subpixel_n: float = 10.0

    @property
    def phi(self) -> np.ndarray:
        """Elevation of each evaluated row's laser ray (rad), lowest first."""
        return np.radians(np.linspace(self.fan_lo_deg, self.fan_hi_deg, self.n_rows))

    @property
    def tilt_deg(self) -> float:
        return 0.5 * (self.fan_lo_deg + self.fan_hi_deg)

    @property
    def fx_px(self) -> float:
        return self.focal_mm / self.pixel_mm

    def rows_v(self) -> np.ndarray:
        d = self.phi - np.radians(self.tilt_deg)
        return self.sensor_rows / 2.0 + self.fx_px * np.tan(d)

    def sigma_z(self, z):
        from .optics import depth_resolution
        return depth_resolution(z, self.pixel_mm, self.focal_mm, self.baseline_mm, self.subpixel_n)

    def covers(self, bore: float) -> bool:
        return self.bore_min <= bore <= self.bore_max


PRESETS: dict[str, OpticalPreset] = {
    "S": OpticalPreset("S", 400.0, 560.0, focal_mm=6.0, baseline_mm=50.0, fan_lo_deg=-0.5, fan_hi_deg=14.0,
                       x_min=130.0, x_max=290.0),
    "M": OpticalPreset("M", 560.0, 760.0, focal_mm=8.0, baseline_mm=60.0, fan_lo_deg=-0.5, fan_hi_deg=12.0,
                       x_min=215.0, x_max=390.0),
    "L": OpticalPreset("L", 760.0, 980.0, focal_mm=12.0, baseline_mm=65.0, fan_lo_deg=-0.5, fan_hi_deg=9.0,
                       x_min=315.0, x_max=500.0),
}


def preset_for_bore(bore: float) -> OpticalPreset:
    for p in PRESETS.values():
        if p.covers(bore):
            return p
    raise ValueError(f"bore {bore} mm outside 400-980 mm")


@dataclass
class DeviceSpec:
    """Fixed mechanical facts of the scanner (mm)."""
    length: float = 350.0
    width: float = 100.0
    height: float = 20.0
    arm_height: float = 8.0
    arm_reach: float = 175.0  # from the rotation axis
    optics_height: float = 13.0  # H0: laser origin above the crown
    R0_nominal: float = 30.0  # laser origin offset from the rotation axis
    h_nominal: float = 2.0  # sideways miss of the beam from the axis
    sweep_half_deg: float = 190.0
    tof_height: float = 19.0  # ToF sensor aperture above the crown
    serial: str = "LS-0001"
    firmware: str = "0.1.0-sim"


@dataclass
class ProcessingConfig:
    band_height: float = 30.0
    band_step: float = 30.0
    fourier_order: int = 8
    fourier_taper: bool = True
    outlier_k: float = 5.0
    outlier_floor: float = 0.1
    outlier_iterations: int = 5
    min_points_per_band: int = 2000
    max_angle_gap_deg: float = 30.0
    tilt_mode: str = "per_band"  # per_band (shape-aware local fit) | per_band_circular | global
    ellipse_flag_mm: float = 0.05  # flag if ellipse and Fourier max/min disagree by more
    gradient_flag_mm: float = 0.10  # flag if the mean diameter changes more than this across a band
    map_angle_step_deg: float = 2.0
    map_depth_step: float = 10.0


@dataclass
class ReportLimits:
    """Placeholder limits – MUST be replaced with the engine maker's values (see docs/DECISIONS.md)."""
    wear_limit_mm: float | None = None  # on diameter; None -> derived from wear_limit_pct
    wear_limit_pct: float = 0.8  # of nominal bore – placeholder only
    ovality_limit_mm: float = 0.5  # placeholder only
    limits_are_placeholders: bool = True
    reference_temperature_c: float | None = None  # None -> report as measured (open question for UMA)

    def wear_limit(self, bore: float) -> float:
        return self.wear_limit_mm if self.wear_limit_mm is not None else bore * self.wear_limit_pct / 100.0


@dataclass
class JobInfo:
    report_no: str = "LS-SIM-0001"
    vessel: str = "MV Simulation (no real vessel)"
    imo: str = "—"
    engine_maker_type: str = "2-stroke crosshead, 980 bore (simulated)"
    engine_serial: str = "—"
    cylinder: str = "3"
    running_hours: str = "—"
    hours_since_overhaul: str = "—"
    location: str = "Dubai (simulation)"
    date: str = "2026-10-05"
    surveyor: str = "UMA Marine — (name)"
    client: str = "—"
    yaw_deg: float = 90.0  # device x-axis azimuth from FORE (operator placement estimate)
    stop_angle_read_deg: float | None = None  # crank angle BTDC read from the flywheel at stop
    liner_temperature_c: float | None = None


def _to_jsonable(o):
    if is_dataclass(o):
        return {f.name: _to_jsonable(getattr(o, f.name)) for f in fields(o)}
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(_to_jsonable(obj), indent=2))


def dataclass_from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


__all__ = ["EngineSpec", "OpticalPreset", "PRESETS", "preset_for_bore", "DeviceSpec", "ProcessingConfig",
           "ReportLimits", "JobInfo", "save_json", "dataclass_from_dict", "asdict"]
