"""Ground-truth bore surfaces for the simulator: a worn liner with scavenge ports, and a master ring.

Liner frame: x → FORE, y → PORT, z up, origin on the liner axis at the liner top (cover joint face).
Azimuth α is measured from FORE towards PORT (counter-clockwise seen from above).  ``s`` is the depth below
the liner top (mm, positive down).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EngineSpec
from ..fitting import diameter_stats, fit_fourier


@dataclass
class Scratch:
    alpha_deg: float
    width_deg: float
    depth_mm: float  # radial, positive = material removed
    s0: float
    s1: float


@dataclass
class LinerModel:
    engine: EngineSpec
    temperature_c: float = 62.0
    top_ring_tdc_depth: float = 160.0  # wear peak (top-ring reversal at TDC)
    wear_peak_D: float = 1.85  # mean diameter increase at the peak (mm)
    wear_base_frac: float = 0.12
    wear_decay: float = 520.0
    ridge_width: float = 6.0
    ovality_peak: float = 0.32  # D_max − D_min at the peak (mm)
    ovality_base: float = 0.06
    ovality_angle_deg: float = 28.0  # direction of the largest diameter (from FORE)
    clover_peak: float = 0.03  # radial 4th harmonic amplitude at the peak
    harmonics: dict = field(default_factory=lambda: {3: (0.012, 40.0), 5: (0.004, 10.0), 6: (0.003, 70.0),
                                                     7: (0.002, 5.0), 8: (0.002, 33.0)})
    scratches: list[Scratch] = field(default_factory=lambda: [
        Scratch(200.0, 0.8, 0.05, 300.0, 1400.0),
        Scratch(301.5, 3.0, 0.25, 600.0, 640.0),
    ])

    @property
    def R_hot(self) -> float:
        return 0.5 * self.engine.bore * (1 + self.engine.liner_cte * (self.temperature_c - 20.0))

    def _shape(self, s):
        """Normalised wear shape: 1 at the reversal point, ridge above, exponential decay below."""
        s = np.asarray(s, float)
        f = self.wear_base_frac
        below = f + (1 - f) * np.exp(-(s - self.top_ring_tdc_depth) / self.wear_decay)
        above = np.exp(-(self.top_ring_tdc_depth - s) / self.ridge_width)
        return np.where(s >= self.top_ring_tdc_depth, below, above)

    def mean_wear(self, s):
        return self.wear_peak_D * self._shape(s)

    def radius(self, alpha, s):
        """True wall radius (hot) at azimuth α (rad) and depth s (mm)."""
        alpha = np.asarray(alpha, float)
        s = np.asarray(s, float)
        g = self._shape(s)
        r = self.R_hot + 0.5 * self.wear_peak_D * g
        a2 = 0.25 * (self.ovality_base + (self.ovality_peak - self.ovality_base) * g)
        beta = np.radians(self.ovality_angle_deg) + np.radians(8.0) * s / self.engine.stroke
        r = r + a2 * np.cos(2 * (alpha - beta))
        r = r + self.clover_peak * g * np.cos(4 * (alpha - np.radians(12.0)))
        for n, (amp, ph) in self.harmonics.items():
            r = r + amp * np.cos(n * (alpha - np.radians(ph)))
        for sc in self.scratches:
            da = np.angle(np.exp(1j * (alpha - np.radians(sc.alpha_deg))))
            inside = (np.abs(da) < np.radians(sc.width_deg) / 2) & (s >= sc.s0) & (s <= sc.s1)
            r = r + np.where(inside, sc.depth_mm, 0.0)
        return r

    def has_wall(self, alpha, s):
        """False inside a scavenge port, above the liner top or below the port bottom."""
        e = self.engine
        alpha = np.asarray(alpha, float)
        s = np.asarray(s, float)
        pitch = 2 * np.pi / e.n_ports
        rel = np.mod(alpha - np.radians(e.port_phase_deg) + pitch / 2, pitch) - pitch / 2
        in_port = ((s >= e.port_top_depth) & (s <= e.port_bottom_depth)
                   & (np.abs(rel) < np.radians(e.port_width_deg) / 2))
        return (~in_port) & (s >= 0.0) & (s <= e.port_bottom_depth)

    def band_truth(self, s0: float, s1: float, order: int = 8, n_alpha: int = 3600, n_s: int = 31):
        """Ideal-instrument result for a band: order-N Fourier projection of the band-mean profile."""
        a = np.linspace(0, 2 * np.pi, n_alpha, endpoint=False)
        ss = np.linspace(s0, s1, n_s)
        rho = self.radius(a[None, :], ss[:, None]).mean(axis=0)
        return diameter_stats(fit_fourier(a, rho, order), fa_angle_deg=0.0)


@dataclass
class RingModel:
    """Master ring of known diameter (certified at 20 °C), wall between depths 0 and ``height``."""
    D20: float
    height: float = 60.0
    temperature_c: float = 21.5
    cte: float = 11.5e-6  # steel

    @property
    def D_at_temp(self) -> float:
        return self.D20 * (1 + self.cte * (self.temperature_c - 20.0))

    def radius(self, alpha, s):
        return np.full(np.broadcast(np.asarray(alpha), np.asarray(s)).shape, 0.5 * self.D_at_temp)

    def has_wall(self, alpha, s):
        s = np.asarray(s, float)
        return np.broadcast_to((s >= 0) & (s <= self.height), np.broadcast(np.asarray(alpha), s).shape)
