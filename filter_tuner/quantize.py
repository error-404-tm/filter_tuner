"""Fixed-point realization of a second-order-section cascade.

Covers coefficient quantization (and where the quantized poles land),
per-section gain analysis / scaling for overflow headroom, and a
bit-accurate integer model of the direct-form-I biquad that the generated
Verilog implements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class QFormat:
    """Signed two's-complement word with ``frac_bits`` fractional bits."""

    word_bits: int = 16
    frac_bits: int = 14

    def __post_init__(self):
        if not 2 <= self.word_bits <= 64:
            raise ValueError("word_bits must be in 2..64")
        if not 0 <= self.frac_bits < self.word_bits:
            raise ValueError("frac_bits must be in 0..word_bits-1")

    @property
    def int_bits(self) -> int:
        return self.word_bits - self.frac_bits  # includes sign bit

    @property
    def lsb(self) -> float:
        return 2.0 ** -self.frac_bits

    @property
    def min_int(self) -> int:
        return -(1 << (self.word_bits - 1))

    @property
    def max_int(self) -> int:
        return (1 << (self.word_bits - 1)) - 1

    @property
    def range(self) -> tuple[float, float]:
        return self.min_int * self.lsb, self.max_int * self.lsb

    def label(self) -> str:
        return f"Q{self.int_bits}.{self.frac_bits} ({self.word_bits}-bit)"


def to_int(x, fmt: QFormat) -> tuple[np.ndarray, bool]:
    """Round to nearest and saturate. Returns (int array, any_saturated)."""
    q = np.round(np.asarray(x, dtype=float) / fmt.lsb)
    sat = bool(np.any(q > fmt.max_int) or np.any(q < fmt.min_int))
    q = np.clip(q, fmt.min_int, fmt.max_int).astype(np.int64)
    return q, sat


def required_int_bits(sos: np.ndarray) -> int:
    """Integer bits (incl. sign) needed so no coefficient saturates."""
    m = float(np.max(np.abs(sos[:, [0, 1, 2, 4, 5]])))
    return max(1, int(np.floor(np.log2(m))) + 2) if m > 0 else 1


# --------------------------------------------------------------------------
# section scaling
# --------------------------------------------------------------------------

def cumulative_gains(sos: np.ndarray, norm: str = "linf", n: int = 2048) -> np.ndarray:
    """Norm of the transfer function from the input to each section's output."""
    out = []
    h_cum = np.ones(n, dtype=complex)
    for sec in sos:
        _, h = signal.sosfreqz(sec[None, :], worN=n)
        h_cum = h_cum * h
        if norm == "linf":
            out.append(float(np.max(np.abs(h_cum))))
        elif norm == "l2":
            out.append(float(np.sqrt(np.mean(np.abs(h_cum) ** 2))))
        else:
            raise ValueError("norm must be 'linf' or 'l2'")
    return np.array(out)


@dataclass(frozen=True, eq=False)
class Scaling:
    sos: np.ndarray            # scaled sections
    output_gain: float         # multiply the cascade output by this to restore H
    section_gains: np.ndarray  # cumulative norms after scaling (~1)


def scale_sections(sos: np.ndarray, norm: str = "linf") -> Scaling:
    """Scale numerators so the cumulative gain to every section output is 1."""
    sos = np.array(sos, dtype=float)
    g = cumulative_gains(sos, norm)
    prev = 1.0
    for i, gi in enumerate(g):
        if gi > 0:
            sos[i, :3] *= prev / gi
            prev = gi
    return Scaling(sos, float(g[-1]) if len(g) else 1.0, cumulative_gains(sos, norm))


def headroom_bits(sos: np.ndarray) -> np.ndarray:
    """Extra integer bits each section output needs for a full-scale input
    (L-inf bound: no overflow for sinusoids; worst case uses the L1 norm)."""
    g = cumulative_gains(sos, "linf")
    return np.maximum(0, np.ceil(np.log2(np.maximum(g, 1e-30)))).astype(int)


# --------------------------------------------------------------------------
# coefficient quantization
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class QuantizedSOS:
    fmt: QFormat
    ideal: np.ndarray        # float sections
    ints: np.ndarray         # (n, 6) int coefficients (a0 column = 1.0 quantized)
    sos: np.ndarray          # float reconstruction of the integer coefficients
    zeros: np.ndarray
    poles: np.ndarray
    saturated: bool
    stable: bool
    max_pole_radius: float
    max_coef_error: float


def quantize_sos(sos: np.ndarray, fmt: QFormat) -> QuantizedSOS:
    sos = np.asarray(sos, dtype=float)
    cols = [0, 1, 2, 4, 5]
    ints = np.zeros(sos.shape, dtype=np.int64)
    ints[:, cols], sat = to_int(sos[:, cols], fmt)
    # a0 (column 3) is implicit, normalized to 1, in the hardware
    q = ints.astype(float) * fmt.lsb
    q[:, 3] = 1.0
    zeros, poles = [], []
    for sec in q:
        b, a = sec[:3], sec[3:]
        zeros.extend(_roots_trim(b))
        poles.extend(_roots_trim(a))
    poles = np.array(poles, dtype=complex)
    rmax = float(np.max(np.abs(poles))) if len(poles) else 0.0
    return QuantizedSOS(
        fmt=fmt, ideal=sos, ints=ints, sos=q,
        zeros=np.array(zeros, dtype=complex), poles=poles,
        saturated=sat, stable=rmax < 1.0, max_pole_radius=rmax,
        max_coef_error=float(np.max(np.abs(q - sos)[:, [0, 1, 2, 4, 5]])),
    )


def _roots_trim(c):
    """Roots of a quadratic in z^-1 written as c0 + c1 z^-1 + c2 z^-2."""
    c = np.asarray(c, dtype=float)
    if c[0] == 0:
        return []  # degenerate (zero leading coef): not representable as roots here
    r = np.roots(c)
    # a trailing zero coefficient means a root at z = 0 in this form
    return list(r) + [0.0] * (2 - len(r))


# --------------------------------------------------------------------------
# bit-accurate DF-I model (matches export/verilog.py)
# --------------------------------------------------------------------------

def _round_shift(acc: int, shift: int) -> int:
    if shift == 0:
        return acc
    return (acc + (1 << (shift - 1))) >> shift


def _sat(v: int, bits: int) -> int:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return lo if v < lo else hi if v > hi else v


def simulate_fixed(ints: np.ndarray, coef_frac_bits: int, x_int, data_bits: int) -> np.ndarray:
    """Run integer samples through the integer DF-I cascade.

    Per section: acc = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2 (full precision),
    y = saturate(round_half_up(acc >> coef_frac_bits), data_bits).
    """
    x = [int(v) for v in np.asarray(x_int).ravel()]
    for sec in np.asarray(ints, dtype=np.int64):
        b0, b1, b2, _, a1, a2 = (int(c) for c in sec)
        x1 = x2 = y1 = y2 = 0
        y = []
        for xn in x:
            acc = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            yn = _sat(_round_shift(acc, coef_frac_bits), data_bits)
            x2, x1 = x1, xn
            y2, y1 = y1, yn
            y.append(yn)
        x = y
    return np.array(x, dtype=np.int64)


def float_to_data(x, data_bits: int) -> np.ndarray:
    """Map floats in [-1, 1) to Q1.(data_bits-1) integers."""
    fmt = QFormat(data_bits, data_bits - 1)
    return to_int(x, fmt)[0]


def data_to_float(x_int, data_bits: int) -> np.ndarray:
    return np.asarray(x_int, dtype=float) / (1 << (data_bits - 1))
