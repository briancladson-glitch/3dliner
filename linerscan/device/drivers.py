"""Hardware abstraction layer.  The scan controller only talks to these interfaces; the simulator implements
them in ``sim_drivers`` and real drivers (camera SoM, encoder, VL53L8CX, buzzer, BLE) implement them later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameBlock:
    t: np.ndarray  # (F,) capture time, s (hardware timestamp at mid-exposure)
    theta: np.ndarray  # (F,) head encoder angle latched at the camera trigger, rad
    u: np.ndarray  # (F, n_rows) sub-pixel laser column per evaluated row, NaN = no line


class Clock(ABC):
    @abstractmethod
    def now(self) -> float: ...

    @abstractmethod
    def wait_until(self, t: float) -> None: ...


class Camera(ABC):
    """Global-shutter camera + on-board peak detection: delivers per-row sub-pixel laser columns."""

    @abstractmethod
    def start(self, frame_rate: float) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read_frames(self) -> FrameBlock: ...


class Laser(ABC):
    @abstractmethod
    def set(self, on: bool) -> None: ...


class HeadMotor(ABC):
    @abstractmethod
    def start_sweep(self, rate_dps: float, accel_dps2: float, half_deg: float) -> None: ...

    @abstractmethod
    def park(self) -> None: ...


class RangeSensor(ABC):
    """One upward multizone ToF sensor (VL53L8CX, 8×8 zones)."""

    @abstractmethod
    def read(self) -> np.ndarray:  # (64,) mm, NaN = invalid zone
        ...


class RangeFinder(ABC):
    """Upward narrow-beam laser distance module used as the height reference (not for safety)."""

    @abstractmethod
    def read(self) -> float:  # mm, NaN = no valid reading
        ...


class TempSensors(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:  # [camera, laser, battery] °C
        ...


class Buzzer(ABC):
    @abstractmethod
    def set(self, pattern: str) -> None:  # off | warn | alarm | fault
        ...


class Radio(ABC):
    @abstractmethod
    def send(self, msg: dict) -> bool:  # best effort; False if not delivered
        ...


@dataclass
class Drivers:
    clock: Clock
    camera: Camera
    laser: Laser
    head: HeadMotor
    tof: tuple[RangeSensor, RangeSensor]
    temps: TempSensors
    buzzer: Buzzer
    radio: Radio
    rangefinder: RangeFinder | None = None
