"""Magnitude specification masks (passband / stopband limits)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Band:
    """|H| in dB must lie in [lo_db, hi_db] for f in [f_lo, f_hi] (Hz).

    ``None`` means unbounded on that side.
    """

    f_lo: float
    f_hi: float
    lo_db: float | None = None
    hi_db: float | None = None

    def __post_init__(self):
        if self.f_hi < self.f_lo:
            raise ValueError("f_hi must be >= f_lo")


@dataclass(frozen=True)
class Mask:
    bands: tuple[Band, ...] = ()

    def excess_db(self, f: np.ndarray, mag_db: np.ndarray) -> np.ndarray:
        """Per-frequency violation in dB (0 where the spec is met)."""
        f = np.asarray(f)
        mag_db = np.asarray(mag_db)
        out = np.zeros_like(mag_db, dtype=float)
        for b in self.bands:
            sel = (f >= b.f_lo) & (f <= b.f_hi)
            if b.hi_db is not None:
                out[sel] = np.maximum(out[sel], mag_db[sel] - b.hi_db)
            if b.lo_db is not None:
                out[sel] = np.maximum(out[sel], b.lo_db - mag_db[sel])
        return out

    def worst_violation_db(self, f, mag_db) -> float:
        e = self.excess_db(f, mag_db)
        return float(e.max()) if e.size else 0.0

    def grid(self, n_per_band: int = 128) -> np.ndarray:
        """Frequencies that sample every band (for optimization)."""
        pts = [np.linspace(b.f_lo, b.f_hi, n_per_band) for b in self.bands]
        return np.unique(np.concatenate(pts)) if pts else np.array([])


def lowpass_mask(fs: float, f_pass: float, f_stop: float,
                 ripple_db: float = 1.0, atten_db: float = 40.0) -> Mask:
    return Mask((
        Band(0.0, f_pass, -ripple_db, 0.0),
        Band(f_stop, fs / 2, None, -atten_db),
    ))


def highpass_mask(fs: float, f_stop: float, f_pass: float,
                  ripple_db: float = 1.0, atten_db: float = 40.0) -> Mask:
    return Mask((
        Band(0.0, f_stop, None, -atten_db),
        Band(f_pass, fs / 2, -ripple_db, 0.0),
    ))


def bandpass_mask(fs: float, f_stop1: float, f_pass1: float, f_pass2: float,
                  f_stop2: float, ripple_db: float = 1.0,
                  atten_db: float = 40.0) -> Mask:
    return Mask((
        Band(0.0, f_stop1, None, -atten_db),
        Band(f_pass1, f_pass2, -ripple_db, 0.0),
        Band(f_stop2, fs / 2, None, -atten_db),
    ))
