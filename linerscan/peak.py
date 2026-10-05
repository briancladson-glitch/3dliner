"""Sub-pixel laser-line detection in one image row (runs on board in the real device).

Method: background subtraction, the strongest pixel above a threshold, then a weighted Gaussian fit
(parabola through ln I) over a small window.  Returns NaN when no valid line is present.
"""

from __future__ import annotations

import numpy as np


def detect_peak(row: np.ndarray, threshold: float = 20.0, half_window: int = 3) -> tuple[float, float]:
    """Return (sub-pixel column, peak amplitude) of the laser line in one row, or (nan, 0).

    Gaussian fit = weighted least-squares parabola through ln(intensity) over ±half_window pixels
    (weights I², the inverse variance of ln I), centred on the strongest pixel.
    """
    row = np.asarray(row, float)
    s = row - np.median(row)
    i = int(np.argmax(s))
    amp = float(s[i])
    if amp < threshold or i < half_window or i > row.size - 1 - half_window:
        return float("nan"), 0.0
    x = np.arange(-half_window, half_window + 1, dtype=float)
    y = s[i - half_window:i + half_window + 1]
    m = y > 0.05 * amp
    if m.sum() >= 3:
        w = y[m]
        A = np.column_stack([np.ones(m.sum()), x[m], x[m] ** 2]) * w[:, None]
        c, *_ = np.linalg.lstsq(A, np.log(y[m]) * w, rcond=None)
        if c[2] < 0:
            return i - c[1] / (2 * c[2]), amp
    w = np.clip(y, 0, None)  # fallback: centre of gravity
    return i + float((x * w).sum() / w.sum()), amp


def render_row(centre: float, n: int = 1280, sigma_px: float = 1.6, amplitude: float = 180.0,
               background: float = 8.0, read_noise: float = 2.0, speckle: float = 0.03,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic (binned) camera row: Gaussian laser line, speckle, shot and read noise (8-bit-like DN).

    Speckle default 3 % per pixel corresponds to ~12 % raw speckle contrast after binning 16 sensor rows.
    """
    rng = rng or np.random.default_rng(0)
    x = np.arange(n)
    prof = amplitude * np.exp(-0.5 * ((x - centre) / sigma_px) ** 2)
    prof *= 1 + speckle * rng.standard_normal(n)
    sig = background + np.clip(prof, 0, None)
    sig = rng.poisson(sig).astype(float) + read_noise * rng.standard_normal(n)
    return np.clip(sig, 0, 255)


def subpixel_factor(n_trials: int = 400, seed: int = 1, **kw) -> float:
    """Empirical N in σ_u ≈ 1/N px for the synthetic line model (used to justify N ≈ 10)."""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_trials):
        c = rng.uniform(100, 1100)
        u, _ = detect_peak(render_row(c, rng=rng, **kw))
        errs.append(u - c)
    return 1.0 / float(np.std(errs))
