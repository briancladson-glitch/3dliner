"""Command line interface.

    linerscan plan      --bore 980 --stroke 2660 --conrod 2885 [--rpm 0.1]
    linerscan simulate  --preset L --seed 0 --out out/scan_L      (factory calibration + one stroke)
    linerscan process   out/scan_L/raw.npz --cal out/scan_L/cal.json --job out/scan_L/job.json --pdf report.pdf
    linerscan example   [--out examples]                           (regenerates the example report)
    linerscan validate                                             (simulation campaign → docs/VALIDATION.md)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import DeviceCalibration
from .config import PRESETS, DeviceSpec, EngineSpec, JobInfo, ProcessingConfig, ReportLimits, dataclass_from_dict
from .config import save_json as _save_json
from .kinematics import plan_scan, preset_for_bore_safe
from .pipeline import process
from .rawscan import RawScan


def cmd_plan(a) -> None:
    e = EngineSpec(bore=a.bore, stroke=a.stroke, conrod=a.conrod, crown_depth_tdc=a.crown_depth_tdc,
                   overhead_clearance_tdc=a.overhead_clearance_tdc)
    p = preset_for_bore_safe(a.bore)
    plan = plan_scan(e, DeviceSpec(), p, turning_gear_rpm=a.rpm, margin=a.margin)
    for k, v in plan.__dict__.items():
        print(f"{k:22s} {v:.3f}" if isinstance(v, float) else f"{k:22s} {v}")


def cmd_simulate(a) -> None:
    from .sim.scenario import build_scenario, calibrate, quick_scenario, run_scan

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = calibrate(build_scenario(a.preset, a.seed))
    sc = quick_scenario(a.preset, a.seed) if a.quick else build_scenario(a.preset, a.seed)
    raw, job, world, ctl, plan = run_scan(sc, cal, soak_max_s=60.0 if a.quick else 900.0)
    cal.save(out / "cal.json")
    raw.save(out / "raw.npz")
    _save_json(job, out / "job.json")
    _save_json(sc.engine, out / "engine.json")
    (out / "ble_log.jsonl").write_text("\n".join(json.dumps(m) for m in ctl.drv.radio.received))
    print(f"wrote {out}: {raw.n_frames} frames; BLE {len(ctl.drv.radio.received)}/{ctl.drv.radio.sent} packets")


def cmd_process(a) -> None:
    raw = RawScan.load(a.raw)
    cal = DeviceCalibration.load(a.cal)
    job = dataclass_from_dict(JobInfo, json.loads(Path(a.job).read_text()))
    eng = dataclass_from_dict(EngineSpec, json.loads(Path(a.engine).read_text())) if a.engine else EngineSpec()
    res = process(raw, cal, eng, job, ProcessingConfig(), u_sys=a.u_sys)
    res.meta["u_sys"] = a.u_sys
    for b in res.positions:
        s = b.stats
        print(f"{b.depth:6.0f}  FA {s.D_FA:8.3f}  PS {s.D_PS:8.3f}  ov {s.ovality:.3f}  "
              f"max {s.D_max:8.3f}@{s.D_max_angle_deg:3.0f}  min {s.D_min:8.3f}@{s.D_min_angle_deg:3.0f}")
    if a.pdf:
        from .report import build_report

        build_report(res, cal, eng, job, ReportLimits(), a.pdf, simulated=not a.real)
        print("report:", a.pdf)


def cmd_example(a) -> None:
    from .example import make_example

    print(make_example(Path(a.out), seed=a.seed))


def cmd_validate(a) -> None:
    from .sim.campaign import main

    main()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="linerscan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="planned stop angle, sweep rate and alarm levels for an engine")
    p.add_argument("--bore", type=float, required=True)
    p.add_argument("--stroke", type=float, required=True)
    p.add_argument("--conrod", type=float, required=True)
    p.add_argument("--crown-depth-tdc", type=float, default=15.0)
    p.add_argument("--overhead-clearance-tdc", type=float, default=12.0)
    p.add_argument("--rpm", type=float, default=0.1, help="turning-gear crank speed")
    p.add_argument("--margin", type=float, default=60.0, help="clearance left above the device at the stop (mm)")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("simulate", help="simulate factory calibration and one scan stroke")
    p.add_argument("--preset", choices=list(PRESETS), default="L")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/sim")
    p.add_argument("--quick", action="store_true", help="short thermal soak")
    p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser("process", help="process a retrieved scan")
    p.add_argument("raw")
    p.add_argument("--cal", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--engine")
    p.add_argument("--pdf")
    p.add_argument("--u-sys", type=float, default=0.008, help="systematic standard uncertainty (mm)")
    p.add_argument("--real", action="store_true", help="omit the SIMULATED watermark")
    p.set_defaults(fn=cmd_process)
    p = sub.add_parser("example", help="regenerate the example report")
    p.add_argument("--out", default="examples")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_example)
    p = sub.add_parser("validate", help="run the simulation validation campaign")
    p.set_defaults(fn=cmd_validate)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
