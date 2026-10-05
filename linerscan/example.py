"""Builds the example (simulated) survey report shipped in ``examples/``."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import JobInfo, ReportLimits, save_json
from .pipeline import process
from .report import build_report
from .sim.rig import ring_check
from .sim.scenario import build_scenario, calibrate, run_scan

U_SYS = 0.008  # mm, systematic standard uncertainty from the simulation campaign (docs/VALIDATION.md)


def make_example(out: Path, seed: int = 0, preset: str = "L") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    sc = build_scenario(preset, seed)
    cal = calibrate(sc)
    rng = np.random.default_rng(seed + 77)
    cal.ring_checks.append(ring_check(cal, sc.cam, sc.head, 34.0, rng, "before job"))
    raw, job, world, ctl, plan = run_scan(sc, cal, soak_max_s=900.0)
    cal.ring_checks.append(ring_check(cal, sc.cam, sc.head, 41.0, rng, "after job"))
    job = JobInfo(report_no="LS-2026-0001-SIM", vessel="MV Example (simulated vessel)", imo="—",
                  engine_maker_type=sc.engine.name, engine_serial="—", cylinder="3", running_hours="38 450 (example)",
                  hours_since_overhaul="11 200 (example)", location="Dubai anchorage (simulated)", date="2026-10-05",
                  surveyor="UMA Marine — name to be entered", client="— (example)", yaw_deg=job.yaw_deg,
                  stop_angle_read_deg=job.stop_angle_read_deg, liner_temperature_c=job.liner_temperature_c)
    res = process(raw, cal, sc.engine, job, u_sys=U_SYS)
    res.meta["u_sys"] = U_SYS
    pdf = build_report(res, cal, sc.engine, job, ReportLimits(), out / "LinerScan_example_report.pdf",
                       plan=plan, simulated=True, workdir=out / "figures")
    with open(out / "example_bands.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth_mm", "D_FA", "D_PS", "ovality", "D_max", "D_max_deg", "D_min", "D_min_deg", "max_minus_min",
                    "D_mean", "U_k2", "points", "outliers", "coverage_pct", "flags"])
        for b in res.profile:
            s = b.stats
            w.writerow([f"{b.depth:.0f}", f"{s.D_FA:.3f}", f"{s.D_PS:.3f}", f"{s.ovality:.3f}", f"{s.D_max:.3f}",
                        f"{s.D_max_angle_deg:.1f}", f"{s.D_min:.3f}", f"{s.D_min_angle_deg:.1f}",
                        f"{s.max_minus_min:.3f}", f"{s.D_mean:.3f}", f"{b.U_k2:.4f}", b.n_points, b.n_outliers,
                        f"{b.coverage_pct:.0f}", " | ".join(b.flags)])
    save_json(job, out / "example_job.json")
    (out / "example_ble_status.jsonl").write_text(
        "\n".join(__import__("json").dumps(m) for m in ctl.drv.radio.received[:400]))
    return pdf
