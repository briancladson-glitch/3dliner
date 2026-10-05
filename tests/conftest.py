"""Shared fixtures.  Factory calibrations are slow (~20 s each), so they are cached per test session."""

import pytest

from linerscan.calibration import DeviceCalibration
from linerscan.sim.scenario import build_scenario, calibrate

_CACHE: dict[str, DeviceCalibration] = {}


def get_cal(preset: str) -> DeviceCalibration:
    if preset not in _CACHE:
        _CACHE[preset] = calibrate(build_scenario(preset, seed=0))
    return _CACHE[preset]


@pytest.fixture(scope="session")
def cal_L():
    return get_cal("L")
