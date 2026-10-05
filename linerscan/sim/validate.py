"""Compare a processed scan with the simulator's ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pipeline import BandResult, ScanResult
from .liner import LinerModel

QUANTITIES = ("D_FA", "D_PS", "ovality", "D_max", "D_min", "D_mean")


@dataclass
class BandError:
    depth: float
    port_zone: bool
    err: dict  # quantity -> measured − truth (mm)
    truth: dict
    steep: bool = False


def band_errors(bands: list[BandResult], liner: LinerModel) -> list[BandError]:
    out = []
    for b in bands:
        tr = liner.band_truth(b.lo, b.hi)
        e, t = {}, {}
        for q in QUANTITIES:
            t[q] = getattr(tr, q)
            e[q] = getattr(b.stats, q) - t[q]
        out.append(BandError(b.depth, b.port_zone, e, t, b.steep))
    return out


def summarize(errs: list[BandError], include_special: bool = False) -> dict:
    """Error statistics; port-zone and steep-gradient bands are excluded unless include_special."""
    sel = [e for e in errs if include_special or not (e.port_zone or e.steep)]
    res = {}
    for q in QUANTITIES:
        v = np.array([e.err[q] for e in sel])
        res[q] = {"max_abs": float(np.max(np.abs(v))) if v.size else float("nan"),
                  "rms": float(np.sqrt(np.mean(v**2))) if v.size else float("nan"),
                  "mean": float(np.mean(v)) if v.size else float("nan"), "n": int(v.size)}
    return res


def validate(result: ScanResult, liner: LinerModel, true_yaw: float, true_tilt: tuple[float, float]) -> dict:
    prof = band_errors(result.profile, liner)
    pos = band_errors(result.positions, liner)
    return {
        "profile": summarize(prof), "profile_ports": summarize([e for e in prof if e.port_zone], True),
        "profile_steep": summarize([e for e in prof if e.steep], True),
        "positions": summarize(pos), "yaw_error_deg": result.yaw_deg - true_yaw,
        "n_bands": len(result.profile), "n_points": result.n_points, "n_outliers": result.n_outliers,
        "band_errors": prof, "position_errors": pos,
    }
