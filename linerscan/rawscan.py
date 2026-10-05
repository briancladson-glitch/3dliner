"""On-board raw scan record (what the device stores; processed after retrieval)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RawScan:
    meta: dict
    frame_t: np.ndarray  # (F,) s
    frame_theta: np.ndarray  # (F,) rad, head encoder
    u: np.ndarray  # (F, K) px, NaN = no line
    tof_t: np.ndarray  # (M,)
    tof: np.ndarray  # (M, 2, 64) mm
    temp_t: np.ndarray  # (N,)
    temps: np.ndarray  # (N, 3) camera, laser, battery °C
    events: list = field(default_factory=list)  # (t, kind, text)

    @property
    def n_frames(self) -> int:
        return int(self.frame_t.size)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, frame_t=self.frame_t, frame_theta=self.frame_theta, u=self.u.astype(np.float32),
                            tof_t=self.tof_t, tof=self.tof.astype(np.float32), temp_t=self.temp_t, temps=self.temps,
                            meta=np.array(json.dumps({"meta": self.meta, "events": self.events})))

    @classmethod
    def load(cls, path: str | Path) -> RawScan:
        d = np.load(path, allow_pickle=False)
        m = json.loads(str(d["meta"]))
        return cls(m["meta"], d["frame_t"], d["frame_theta"], d["u"].astype(float), d["tof_t"], d["tof"].astype(float),
                   d["temp_t"], d["temps"], m["events"])


class RawScanWriter:
    """Append-only recorder used by the controller (the real device streams the same records to eMMC)."""

    def __init__(self, meta: dict, n_rows: int):
        self.meta, self.n_rows = meta, n_rows
        self.ft, self.fth, self.fu = [], [], []
        self.tt, self.tr = [], []
        self.mt, self.mv = [], []
        self.events: list = []

    def frames(self, t, theta, u):
        if len(t):
            self.ft.append(np.asarray(t))
            self.fth.append(np.asarray(theta))
            self.fu.append(np.asarray(u, np.float32))

    def tof(self, t, ranges):
        self.tt.append(t)
        self.tr.append(np.asarray(ranges))

    def temps(self, t, v):
        self.mt.append(t)
        self.mv.append(np.asarray(v))

    def event(self, t, kind, text):
        self.events.append((float(t), kind, text))

    def close(self) -> RawScan:
        cat = (lambda a, shape: np.concatenate(a) if a else np.zeros(shape))  # noqa: E731
        return RawScan(self.meta, cat(self.ft, (0,)), cat(self.fth, (0,)), cat(self.fu, (0, self.n_rows)),
                       np.array(self.tt), np.array(self.tr).reshape(-1, 2, 64), np.array(self.mt),
                       np.array(self.mv).reshape(-1, 3), self.events)
