import numpy as np
import pytest

from linerscan.device.head import pass_duration, sweep_angle
from linerscan.device.safety import SafetyConfig, SafetyMonitor
from linerscan.rawscan import RawScan
from linerscan.sim.scenario import quick_scenario, run_scan


def test_sweep_profile_continuous_and_bounded():
    t = np.arange(0, 20, 1e-3)
    a = sweep_angle(t, 110.0, 2000.0, 190.0)
    assert a.min() == pytest.approx(-190.0, abs=1e-3) and a.max() == pytest.approx(190.0, abs=1e-3)
    v = np.diff(a) / 1e-3
    assert np.abs(v).max() == pytest.approx(110.0, rel=1e-3)
    assert np.abs(np.diff(v)).max() / 1e-3 <= 2000.0 * 1.01  # acceleration limit, no jumps
    T = pass_duration(110.0, 2000.0, 190.0)
    assert sweep_angle(T, 110.0, 2000.0, 190.0) == pytest.approx(190.0)
    # every pass covers 380°, i.e. a full circle plus 20° overlap
    assert np.ptp(sweep_angle(np.linspace(0, T, 1000), 110.0, 2000.0, 190.0)) == pytest.approx(380.0)


def _ranges(clear, n=64, noise=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return [np.full(n, clear + 1.0) + noise * rng.standard_normal(n) for _ in range(2)]


def test_safety_states_and_latching():
    m = SafetyMonitor(SafetyConfig(warn_clearance=50.0, alarm_clearance=35.0))
    m.armed = True
    t = 0.0
    for c in np.linspace(200, 60, 1000):  # approaching slowly (1.4 mm/s)
        t += 0.1
        assert m.update(t, _ranges(c)) == "NORMAL"
    for c in np.linspace(60, 45, 100):
        t += 0.1
        m.update(t, _ranges(c))
    assert m.state == "WARN"
    t += 0.1
    assert m.update(t, _ranges(34.0)) == "ALARM"
    t += 0.1
    assert m.update(t, _ranges(80.0)) == "ALARM"  # latched for the hold time
    for _ in range(120):
        t += 0.1
        m.update(t, _ranges(80.0))
    assert m.state == "NORMAL"


def test_safety_time_to_contact_alarm():
    m = SafetyMonitor(SafetyConfig(warn_clearance=50.0, alarm_clearance=20.0, ttc_alarm_s=8.0))
    m.armed = True
    t, c = 0.0, 300.0
    while c > 60:
        t += 0.1
        c -= 4.0  # 40 mm/s: fast turning gear
        m.update(t, _ranges(c))
        if m.state == "ALARM":
            break
    assert m.state == "ALARM" and c > 60  # alarm long before the clearance threshold


def test_safety_fault_is_fail_safe():
    m = SafetyMonitor(SafetyConfig(50.0, 35.0))
    m.armed = True
    m.update(0.1, _ranges(500))
    nan = [np.full(64, np.nan)] * 2
    m.update(0.6, nan)
    assert m.state == "NORMAL"  # within the timeout
    m.update(1.5, nan)
    assert m.state == "FAULT" and m.buzzer_pattern == "fault"


def test_wall_zones_ignored_at_long_range():
    """Outer zones see the liner wall ~1.4 m away while the roof is 2.6 m up: clearance must be the roof."""
    from linerscan.device.safety import zone_angles

    g = zone_angles()
    m = SafetyMonitor(SafetyConfig(50.0, 35.0))
    r = np.where(g < np.radians(5), 2600.0 / np.cos(g), 450.0 / np.sin(g))
    m.update(0.1, [r, r])
    assert m.clearance == pytest.approx(2600.0 - 1.0, abs=1.0)


@pytest.fixture(scope="module")
def fake_cal(cal_L):
    return cal_L


def test_overshoot_alarm_prevents_crush(cal_L):
    """The operator misses the planned stop; the ToF alarm makes him stop with clearance left. Radio is dead."""
    sc = quick_scenario("L", seed=3, overshoot=True)
    raw, job, world, ctl, plan = run_scan(sc, cal_L, soak_max_s=30, render=False, radio_loss=1.0)
    states = [e[1] for e in raw.meta["safety_events"]]
    assert "ALARM" in states
    t_end = world.motion.t_hist[-1]
    min_clear = min(world.true_clearance(t) for t in np.arange(0, t_end, 0.1))
    assert min_clear > 5.0
    assert ctl.drv.radio.received == []  # nothing got through, safety still worked


def test_normal_stop_has_no_alarm(cal_L):
    sc = quick_scenario("L", seed=4)
    raw, job, world, ctl, plan = run_scan(sc, cal_L, soak_max_s=30, render=False)
    assert "ALARM" not in [e[1] for e in raw.meta["safety_events"]]
    assert [e[2].split()[0] for e in raw.events if e[1] == "state"] == ["SOAK", "ARMED", "SCAN", "STOPPED", "DONE"]
    t_end = world.motion.t_hist[-1]
    assert min(world.true_clearance(t) for t in np.arange(0, t_end, 0.5)) > 55.0


def test_tof_failure_triggers_fault_and_stop(cal_L):
    sc = quick_scenario("L", seed=5)
    raw, job, world, ctl, plan = run_scan(sc, cal_L, soak_max_s=30, render=False, tof_fail_at=150.0)
    assert "FAULT" in [e[1] for e in raw.meta["safety_events"]]
    assert world.motion.t_cmd is not None and world.motion.t_cmd < 160.0


def test_rawscan_roundtrip(tmp_path, cal_L):
    sc = quick_scenario("L", seed=6)
    raw, *_ = run_scan(sc, cal_L, soak_max_s=30, render=False)
    p = tmp_path / "scan.npz"
    raw.save(p)
    r2 = RawScan.load(p)
    assert r2.n_frames == raw.n_frames and r2.meta["preset"] == "L"
    assert np.allclose(r2.tof, raw.tof, equal_nan=True)
