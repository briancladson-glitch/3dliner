"""End-to-end simulation tests: factory calibration → one stroke through the controller → processing,
compared with the simulator's ground truth.  Spec: ±0.05 mm on diameter."""

import numpy as np
import pytest

from linerscan.calibration import DeviceCalibration
from linerscan.config import ReportLimits
from linerscan.optics import ThermalModel
from linerscan.pipeline import process
from linerscan.sim.rig import ring_check
from linerscan.sim.scenario import build_scenario, quick_scenario, run_scan
from linerscan.sim.validate import QUANTITIES, validate

SPEC = 0.05


@pytest.fixture(scope="module")
def run_L(cal_L):
    sc = quick_scenario("L", seed=0)
    raw, job, world, ctl, plan = run_scan(sc, cal_L, soak_max_s=60)
    res = process(raw, cal_L, sc.engine, job, u_sys=0.008)
    return sc, raw, job, world, res


def test_L_diameters_within_spec(run_L):
    sc, raw, job, world, res = run_L
    v = validate(res, sc.liner, sc.pose.yaw_deg, sc.pose.tilt_deg)
    for key in ("profile", "positions"):
        for q in QUANTITIES:
            assert v[key][q]["max_abs"] < SPEC, (key, q, v[key][q])
    assert v["profile"]["D_FA"]["n"] > 70
    assert abs(v["yaw_error_deg"]) < 0.5  # recovered from the scavenge-port pattern (placement was 2.7° off)
    assert "scavenge-port" in res.yaw_source


def test_L_geometry_and_height(run_L):
    sc, raw, job, world, res = run_L
    assert res.tilt_deg[0] == pytest.approx(sc.pose.tilt_deg[0], abs=0.03)
    assert res.tilt_deg[1] == pytest.approx(sc.pose.tilt_deg[1], abs=0.03)
    ev = {e[2].split()[0]: e[0] for e in raw.events if e[1] == "state"}
    tt = np.linspace(ev["SCAN"], ev["STOPPED"], 200)
    assert np.max(np.abs(res.height(tt) - world.motion.crown_depth_at(tt))) < 3.0
    assert res.height.top_edge_frames > 0  # stop anchored on the liner-top edge


def test_L_outliers_marked_not_dropped(run_L):
    sc, raw, job, world, res = run_L
    # glints (0.2 % of points in the simulator) are marked, counted and kept in the result
    assert 0.001 < res.n_outliers / res.n_points < 0.004
    assert res.outlier_alpha.size == res.n_outliers
    assert res.n_outliers == sum(b.n_outliers for b in res.profile)
    # the 0.25 mm scuff at 301.5°, 600–640 mm is inside the noise-based threshold, so it stays as data and
    # shows on the wall map as a local bump
    ia = np.argmin(np.abs(res.map_alpha - 301.0))
    rows = (res.map_depth > 605) & (res.map_depth < 635)
    bump = np.nanmean(res.map_dev[rows, ia]) - np.nanmean(res.map_dev[rows][:, [ia - 6, ia + 6]])
    assert bump > 0.1


def test_report_pdf(run_L, cal_L, tmp_path):
    from linerscan.report import build_report

    sc, raw, job, world, res = run_L
    pdf = build_report(res, cal_L, sc.engine, job, ReportLimits(), tmp_path / "r.pdf")
    data = pdf.read_bytes()
    assert data[:4] == b"%PDF" and data.count(b"/Type /Page") >= 6


def test_S_small_bore_within_spec(cal_S):
    sc = quick_scenario("S", seed=0)
    raw, job, world, ctl, plan = run_scan(sc, cal_S, soak_max_s=60)
    res = process(raw, cal_S, sc.engine, job)
    v = validate(res, sc.liner, sc.pose.yaw_deg, sc.pose.tilt_deg)
    for q in QUANTITIES:
        assert v["profile"][q]["max_abs"] < SPEC, (q, v["profile"][q])
    assert res.depth_range[1] < sc.engine.crown_depth_tdc + sc.engine.stroke  # nothing below BDC


def test_thermal_compensation_holds_at_80C(cal_L):
    sc = build_scenario("L", seed=0)
    rng = np.random.default_rng(5)
    comp = ring_check(cal_L, sc.cam, sc.head, 80.0, rng, "80C")
    raw_cal = DeviceCalibration.from_dict(cal_L.to_dict())
    raw_cal.triangulation.thermal = ThermalModel(T0=cal_L.T_cal)
    uncomp = ring_check(raw_cal, sc.cam, sc.head, 80.0, np.random.default_rng(5), "80C")
    assert abs(comp["error"]) < 0.02
    assert abs(uncomp["error"]) > 0.2  # without compensation the instrument is far out
