"""Device calibration bundle: triangulation (+ thermal) per optical module, head geometry, ring records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import HeadGeometry
from .optics import TriangulationCalibration


@dataclass
class DeviceCalibration:
    preset: str
    triangulation: TriangulationCalibration
    geometry: HeadGeometry
    ring_D20: float
    ring_D_measured: float
    ring_rms: float
    T_cal: float
    cal_id: str = "CAL-SIM-0001"
    date: str = "2026-10-01"
    ring_checks: list = field(default_factory=list)  # [{"when", "T", "D_known", "D_meas", "error"}]

    def to_dict(self) -> dict:
        return {"preset": self.preset, "triangulation": self.triangulation.to_dict(),
                "geometry": self.geometry.to_dict(), "ring_D20": self.ring_D20,
                "ring_D_measured": self.ring_D_measured, "ring_rms": self.ring_rms, "T_cal": self.T_cal,
                "cal_id": self.cal_id, "date": self.date, "ring_checks": self.ring_checks}

    @classmethod
    def from_dict(cls, d: dict) -> DeviceCalibration:
        return cls(d["preset"], TriangulationCalibration.from_dict(d["triangulation"]), HeadGeometry(**d["geometry"]),
                   d["ring_D20"], d["ring_D_measured"], d["ring_rms"], d["T_cal"], d.get("cal_id", ""),
                   d.get("date", ""), d.get("ring_checks", []))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> DeviceCalibration:
        return cls.from_dict(json.loads(Path(path).read_text()))
