"""Driver implementations backed by the simulator (``linerscan.sim.world.SimWorld``)."""

from __future__ import annotations

import numpy as np

from ..sim.world import SimWorld, SweepCommand
from .drivers import (
    Buzzer,
    Camera,
    Clock,
    Drivers,
    FrameBlock,
    HeadMotor,
    Laser,
    Radio,
    RangeFinder,
    RangeSensor,
    TempSensors,
)


class SimClock(Clock):
    def __init__(self, world: SimWorld, buzzer: SimBuzzer):
        self.world, self.buzzer, self.t = world, buzzer, 0.0

    def now(self) -> float:
        return self.t

    def wait_until(self, t: float) -> None:
        self.world.motion.advance(t, alarm_sounding=self.buzzer.pattern in ("alarm", "fault"))
        self.t = t


class SimCamera(Camera):
    def __init__(self, world: SimWorld, clock: SimClock):
        self.world, self.clock = world, clock
        self.fps = 0.0
        self.t_next = None

    def start(self, frame_rate: float) -> None:
        self.fps = frame_rate
        self.t_next = self.clock.now()

    def stop(self) -> None:
        self.t_next = None

    def read_frames(self) -> FrameBlock:
        k = self.world.cam.phi.size
        if self.t_next is None:
            return FrameBlock(np.zeros(0), np.zeros(0), np.zeros((0, k)))
        t = np.arange(self.t_next, self.clock.now(), 1.0 / self.fps)
        if t.size:
            self.t_next = t[-1] + 1.0 / self.fps
        u, th = self.world.frames(t)
        return FrameBlock(t, th, u.astype(np.float32))


class SimLaser(Laser):
    def __init__(self, world: SimWorld):
        self.world = world

    def set(self, on: bool) -> None:
        self.world.laser_on = on


class SimHead(HeadMotor):
    def __init__(self, world: SimWorld, clock: SimClock):
        self.world, self.clock = world, clock

    def start_sweep(self, rate_dps, accel_dps2, half_deg) -> None:
        self.world.sweep = SweepCommand(self.clock.now(), rate_dps, accel_dps2, half_deg)

    def park(self) -> None:
        self.world.sweep = None


class SimToF(RangeSensor):
    def __init__(self, world: SimWorld, clock: SimClock, index: int):
        self.world, self.clock, self.index = world, clock, index
        self._cache = (None, None)

    def read(self) -> np.ndarray:
        t = self.clock.now()
        if self._cache[0] != t:
            self._cache = (t, self.world.tof_read(t))
        return self._cache[1][self.index]


class SimRangeFinder(RangeFinder):
    def __init__(self, world: SimWorld, clock: SimClock):
        self.world, self.clock = world, clock

    def read(self) -> float:
        return self.world.rangefinder_read(self.clock.now())


class SimTemps(TempSensors):
    def __init__(self, world: SimWorld, clock: SimClock):
        self.world, self.clock = world, clock

    def read(self) -> np.ndarray:
        return self.world.temps(self.clock.now())


class SimBuzzer(Buzzer):
    def __init__(self):
        self.pattern = "off"
        self.log: list[tuple[float, str]] = []

    def set(self, pattern: str) -> None:
        self.pattern = pattern


class SimRadio(Radio):
    """BLE link to the base station; drops packets at random (safety must not depend on it)."""

    def __init__(self, loss: float = 0.2, seed: int = 0):
        self.loss = loss
        self.rng = np.random.default_rng(seed)
        self.received: list[dict] = []
        self.sent = 0

    def send(self, msg: dict) -> bool:
        self.sent += 1
        if self.rng.random() < self.loss:
            return False
        self.received.append(msg)
        return True


def make_sim_drivers(world: SimWorld, radio_loss: float = 0.2) -> Drivers:
    buzzer = SimBuzzer()
    clock = SimClock(world, buzzer)
    return Drivers(clock=clock, camera=SimCamera(world, clock), laser=SimLaser(world), head=SimHead(world, clock),
                   tof=(SimToF(world, clock, 0), SimToF(world, clock, 1)), temps=SimTemps(world, clock),
                   buzzer=buzzer, radio=SimRadio(radio_loss, seed=world.seed + 7),
                   rangefinder=SimRangeFinder(world, clock) if world.rangefinder is not None else None)
