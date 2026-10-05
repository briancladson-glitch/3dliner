"""Reproducible end-to-end scenarios: an engine, a worn liner, a device with its true imperfections, the
factory calibration and one scan stroke through the on-board controller."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..calibration import DeviceCalibration
from ..config import PRESETS, DeviceSpec, EngineSpec, JobInfo, OpticalPreset
from ..device.controller import ControllerConfig, ScanController, make_quicklook
from ..device.safety import SafetyConfig
from ..device.sim_drivers import make_sim_drivers
from ..kinematics import ScanPlan, plan_scan
from ..rawscan import RawScan
from .camera import CameraTruth, DevicePose, HeadTruth
from .liner import LinerModel
from .rig import factory_calibration
from .world import EngineMotion, SimWorld, ThermalTruth, ToFTruth

_BASE_DEPTHS = [40, 80, 120, 160, 200, 250, 300, 400, 550, 700, 900, 1150, 1400, 1700, 2000, 2300]


def example_engine(preset: str) -> EngineSpec:
    """One representative engine per optical preset (simulated, not a specific maker's data)."""
    if preset == "S":
        e = EngineSpec(name="2-stroke crosshead, 450 bore (simulated)", bore=450.0, stroke=1900.0, conrod=2100.0,
                       port_top_depth=1700.0, port_bottom_depth=1930.0, n_ports=16, port_width_deg=15.0,
                       port_phase_deg=11.25)
    elif preset == "M":
        e = EngineSpec(name="2-stroke crosshead, 650 bore (simulated)", bore=650.0, stroke=2600.0, conrod=2800.0,
                       port_top_depth=2330.0, port_bottom_depth=2630.0, n_ports=20, port_width_deg=12.5,
                       port_phase_deg=9.0)
    else:
        e = EngineSpec()
    k = e.stroke / 2660.0
    e.measuring_depths = sorted({round(d * k / 5) * 5 for d in _BASE_DEPTHS})
    return e


@dataclass
class Scenario:
    preset: OpticalPreset
    engine: EngineSpec
    device: DeviceSpec
    cam: CameraTruth
    head: HeadTruth
    liner: LinerModel
    pose: DevicePose
    thermal: ThermalTruth
    tof: ToFTruth
    seed: int
    rpm: float = 0.1
    overshoot: bool = False
    stop_read_error_deg: float = 0.2
    yaw_estimate_deg: float = 90.0
    extra: dict = field(default_factory=dict)


def build_scenario(preset: str = "L", seed: int = 0, **kw) -> Scenario:
    rng = np.random.default_rng(seed)
    p = PRESETS[preset]
    e = kw.pop("engine", None) or example_engine(preset)
    dev = DeviceSpec()
    cam = CameraTruth.design(p, rng)
    sc_hyst = float(0.01 * rng.standard_normal())  # unmodelled drift between calibration and field use
    head = HeadTruth(R0=dev.R0_nominal + 1.2 + 0.2 * rng.standard_normal(), h=2.6, H0=dev.optics_height)
    liner = LinerModel(e, top_ring_tdc_depth=e.crown_depth_tdc + 0.055 * e.stroke,
                       wear_peak_D=0.0019 * e.bore, wear_decay=0.2 * e.stroke)
    pose = DevicePose(yaw_deg=92.7, tilt_deg=(0.35, -0.20), offset=(14.0, -9.0))
    sc = Scenario(p, e, dev, cam, head, liner, pose, ThermalTruth(), ToFTruth(), seed)
    sc.extra["hysteresis_px"] = sc_hyst
    for k, v in kw.items():
        setattr(sc, k, v)
    return sc


def calibrate(sc: Scenario) -> DeviceCalibration:
    return factory_calibration(sc.preset, sc.cam, sc.head, np.random.default_rng(sc.seed + 1000), sc.device)


def run_scan(sc: Scenario, cal: DeviceCalibration, plan: ScanPlan | None = None, soak_max_s: float = 900.0,
             radio_loss: float = 0.2, t_start_margin: float = 15.0, render: bool = True,
             tof_fail_at: float | None = None):
    """One stroke BDC → stop.  Returns (raw, job, world, controller, plan)."""
    plan = plan or plan_scan(sc.engine, sc.device, sc.preset, turning_gear_rpm=sc.rpm)
    sc.cam.hysteresis_px = sc.extra.get("hysteresis_px", 0.0)
    rng = np.random.default_rng(sc.seed + 2000)
    read_err = sc.stop_read_error_deg * float(rng.standard_normal())
    # the operator starts the turning gear after the soak (procedure: on the base-station "ARMED" + 15 s)
    t_start = soak_max_s + t_start_margin
    motion = EngineMotion(sc.engine, rpm=sc.rpm, t_start=t_start, stop_btdc=plan.stop_angle_btdc,
                          read_error_deg=0.0, overshoot=sc.overshoot)
    world = SimWorld(sc.engine, sc.liner, sc.cam, sc.head, sc.pose, sc.thermal, sc.tof, motion, seed=sc.seed + 3000,
                     camera_enabled=render)
    drv = make_sim_drivers(world, radio_loss=radio_loss)
    if tof_fail_at is not None:
        world.extra_fail_at = tof_fail_at
    cc = ControllerConfig(frame_rate=plan.frame_rate, sweep_rate_dps=plan.sweep_rate_dps, soak_max_s=soak_max_s,
                          soak_min_s=min(60.0, soak_max_s),
                          safety=SafetyConfig(plan.warn_clearance, plan.alarm_clearance))
    meta = {"device": sc.device.serial, "firmware": sc.device.firmware, "preset": sc.preset.name,
            "cal_id": cal.cal_id, "plan": plan.__dict__, "engine": sc.engine.name}
    ctl = ScanController(drv, cc, meta, sc.preset.n_rows, quicklook=make_quicklook(cal.triangulation, cal.geometry))
    raw = ctl.run()
    raw.meta["radio"] = {"sent": drv.radio.sent, "received": len(drv.radio.received)}
    stop_true = float(np.min(motion.a_hist))
    job = JobInfo(engine_maker_type=sc.engine.name, yaw_deg=sc.yaw_estimate_deg,
                  stop_angle_read_deg=round(stop_true + read_err, 1),
                  liner_temperature_c=sc.thermal.T_liner)
    return raw, job, world, ctl, plan


def quick_scenario(preset: str = "L", seed: int = 0, **kw) -> Scenario:
    """Variant with a short soak for tests (device already near liner temperature)."""
    sc = build_scenario(preset, seed, **kw)
    sc.thermal = replace(sc.thermal, T_start=sc.thermal.T_liner - 3.0)
    return sc


__all__ = ["Scenario", "build_scenario", "calibrate", "run_scan", "example_engine", "quick_scenario", "RawScan"]
