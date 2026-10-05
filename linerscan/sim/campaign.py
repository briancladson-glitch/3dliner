"""Validation campaign: simulate, process and compare with ground truth for several cases.

    python -m linerscan.sim.campaign            # writes docs/VALIDATION.md and out/validation.json
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..calibration import DeviceCalibration
from ..config import ProcessingConfig
from ..optics import ThermalModel
from ..pipeline import process
from .scenario import build_scenario, calibrate, quick_scenario, run_scan
from .validate import QUANTITIES, validate

_CAL_CACHE: dict[tuple[str, int], DeviceCalibration] = {}


def _cal(preset: str, seed: int) -> DeviceCalibration:
    key = (preset, seed)
    if key not in _CAL_CACHE:
        _CAL_CACHE[key] = calibrate(build_scenario(preset, seed))
    return _CAL_CACHE[key]


def run_case(preset: str = "L", seed: int = 0, liner_temp: float = 62.0, thermal_comp: bool = True,
             cfg: ProcessingConfig | None = None, soak_s: float = 60.0) -> dict:
    t0 = time.time()
    cal = _cal(preset, seed)
    sc = quick_scenario(preset, seed)
    sc.thermal = replace(sc.thermal, T_liner=liner_temp, T_start=liner_temp - 3.0)
    sc.liner.temperature_c = liner_temp
    raw, job, world, ctl, plan = run_scan(sc, cal, soak_max_s=soak_s)
    pcal = cal
    if not thermal_comp:
        pcal = DeviceCalibration.from_dict(cal.to_dict())
        pcal.triangulation.thermal = ThermalModel(T0=cal.T_cal)
    res = process(raw, pcal, sc.engine, job, cfg)
    v = validate(res, sc.liner, sc.pose.yaw_deg, sc.pose.tilt_deg)
    ev = {e[2].split()[0]: e[0] for e in raw.events if e[1] == "state"}
    tt = np.linspace(ev["SCAN"], ev["STOPPED"], 400)
    h_err = res.height(tt) - world.motion.crown_depth_at(tt)
    return {
        "case": f"{preset} bore {sc.engine.bore:.0f}, liner {liner_temp:.0f} °C, seed {seed}"
                + ("" if thermal_comp else ", NO thermal compensation"),
        "preset": preset, "bore": sc.engine.bore, "seed": seed, "liner_temp": liner_temp,
        "thermal_comp": thermal_comp, "device_temp_scan": res.temps["camera_scan"],
        "summary": {k: v[k] for k in ("profile", "profile_ports", "profile_steep", "positions")},
        "yaw_error_deg": v["yaw_error_deg"],
        "tilt_true": list(sc.pose.tilt_deg), "tilt_est": list(res.tilt_deg),
        "height_err_max": float(np.max(np.abs(h_err))), "height_err_rms": float(np.sqrt(np.mean(h_err**2))),
        "n_points": res.n_points, "n_outliers": res.n_outliers, "n_bands": len(res.profile),
        "frames": raw.n_frames, "calib_model": f"{cal.triangulation.kind}/{cal.triangulation.structure}",
        "R0": cal.geometry.R0, "h": cal.geometry.h, "seconds": time.time() - t0,
    }


DEFAULT_CASES = [
    {"preset": "S", "seed": 0}, {"preset": "M", "seed": 0}, {"preset": "L", "seed": 0},
    {"preset": "L", "seed": 1}, {"preset": "L", "seed": 0, "liner_temp": 80.0},
    {"preset": "L", "seed": 0, "liner_temp": 80.0, "thermal_comp": False},
]


def to_markdown(results: list[dict]) -> str:
    L = ["# Simulation validation", "",
         "Measured − ground truth for every 30 mm profile band (port-zone and steep-gradient bands listed "
         "separately).  All values in mm.  Truth = order-8 Fourier projection of the true band-mean profile "
         "(what an ideal instrument would report).  Regenerate with `python -m linerscan.sim.campaign`.", ""]
    L += ["| case | bands | max\\|err\\| D_FA | D_PS | ovality | D_max | D_min | RMS D_FA | RMS D_PS | "
          "yaw err ° | height err max | model |", "|" + "---|" * 12]
    for r in results:
        p = r["summary"]["profile"]
        L.append(f"| {r['case']} | {p['D_FA']['n']} | {p['D_FA']['max_abs']:.3f} | {p['D_PS']['max_abs']:.3f} | "
                 f"{p['ovality']['max_abs']:.3f} | {p['D_max']['max_abs']:.3f} | {p['D_min']['max_abs']:.3f} | "
                 f"{p['D_FA']['rms']:.4f} | {p['D_PS']['rms']:.4f} | {r['yaw_error_deg']:+.2f} | "
                 f"{r['height_err_max']:.1f} | {r['calib_model']} |")
    L += ["", "## Special bands", "", "| case | port-zone bands: max\\|err\\| D_FA / D_PS | "
          "steep-gradient bands: n, max\\|err\\| D_mean |", "|---|---|---|"]
    for r in results:
        pp, st = r["summary"]["profile_ports"], r["summary"]["profile_steep"]
        L.append(f"| {r['case']} | {pp['D_FA']['max_abs']:.3f} / {pp['D_PS']['max_abs']:.3f} | "
                 f"{st['D_mean']['n']}, {st['D_mean']['max_abs']:.3f} |")
    L += ["", "## Standard measuring positions (VAK-style table)", "",
          "| case | positions | max\\|err\\| D_FA | D_PS | ovality | D_max | D_min |", "|---|---|---|---|---|---|---|"]
    for r in results:
        p = r["summary"]["positions"]
        L.append(f"| {r['case']} | {p['D_FA']['n']} | " + " | ".join(
            f"{p[q]['max_abs']:.3f}" for q in ("D_FA", "D_PS", "ovality", "D_max", "D_min")) + " |")
    L += ["", "Points, outliers and run time:", ""]
    for r in results:
        L.append(f"- {r['case']}: {r['frames']} frames, {r['n_points']:,} points, {r['n_outliers']:,} marked "
                 f"outliers, device {r['device_temp_scan']:.1f} °C, tilt true {r['tilt_true']} est "
                 f"[{r['tilt_est'][0]:.3f}, {r['tilt_est'][1]:.3f}], {r['seconds']:.0f} s")
    return "\n".join(L) + "\n"


def main(cases=None, out_dir: str = "out", md_path: str = "docs/VALIDATION.md") -> list[dict]:
    cases = cases or DEFAULT_CASES
    Path(out_dir).mkdir(exist_ok=True)
    results = []
    for c in cases:
        r = run_case(**c)
        print(r["case"], {q: round(r["summary"]["profile"][q]["max_abs"], 4) for q in QUANTITIES}, flush=True)
        results.append(r)
    Path(out_dir, "validation.json").write_text(json.dumps(results, indent=1))
    Path(md_path).write_text(to_markdown(results))
    return results


if __name__ == "__main__":
    main()
