"""Frequency/time analysis computed directly from zeros, poles and gain.

The filter is never expanded into a single high-order polynomial for
analysis: responses come from the factored form and second-order sections,
which stays accurate for high orders and poles near the unit circle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .model import FilterState, GainSpec


@dataclass(frozen=True, eq=False)
class ZPK:
    zeros: np.ndarray
    poles: np.ndarray
    k: float
    # poles added at the origin to make an improper (more zeros than poles)
    # H(z) causal. They only change phase / time responses by a pure delay.
    padded_poles: int = 0


@dataclass(frozen=True, eq=False)
class Response:
    f: np.ndarray            # frequency axis, in units of fs (0 .. fs/2)
    h: np.ndarray            # complex response
    mag_db: np.ndarray
    phase: np.ndarray        # unwrapped, rad
    group_delay: np.ndarray  # samples (nan where a root sits on the circle)
    impulse: np.ndarray
    step: np.ndarray
    sos: np.ndarray
    zpk: ZPK
    stable: bool
    max_pole_radius: float


def eval_zpk(zeros, poles, k, z) -> np.ndarray:
    """Evaluate k * prod(z - zi) / prod(z - pi) at arbitrary complex points."""
    z = np.asarray(z, dtype=complex)
    num = np.ones_like(z)
    den = np.ones_like(z)
    for zi in zeros:
        num = num * (z - zi)
    for pi in poles:
        den = den * (z - pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        return k * num / den


def resolve_gain(zeros, poles, gain: GainSpec, fs: float) -> float:
    """Gain k that satisfies the gain specification."""
    if gain.mode == "manual":
        return float(gain.value)
    if gain.mode == "peak":
        _, h = signal.freqz_zpk(zeros, poles, 1.0, worN=4096)
        ref = float(np.nanmax(np.abs(h)))
    else:
        f = {"dc": 0.0, "nyquist": fs / 2, "freq": gain.freq}[gain.mode]
        ref = float(abs(eval_zpk(zeros, poles, 1.0, np.exp(2j * np.pi * f / fs))))
    if not np.isfinite(ref) or ref < 1e-300:
        # reference point sits on a zero or pole: fall back to unit gain
        return 1.0
    return 1.0 / ref


def state_zpk(state: FilterState) -> ZPK:
    z, p = state.zeros, state.poles
    pad = max(0, len(z) - len(p))
    if pad:
        p = np.concatenate([p, np.zeros(pad, dtype=complex)])
    k = resolve_gain(z, p, state.gain, state.fs)
    return ZPK(z, p, k, pad)


def group_delay_zpk(zeros, poles, w) -> np.ndarray:
    """Group delay in samples from root locations (exact, no polynomial)."""
    e = np.exp(1j * np.asarray(w))
    tau = np.zeros(e.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in zeros:
            tau -= np.real(e / (e - c))
        for c in poles:
            tau += np.real(e / (e - c))
    tau[~np.isfinite(tau)] = np.nan
    return tau


def zpk_to_sos(zpk: ZPK) -> np.ndarray:
    if len(zpk.zeros) == 0 and len(zpk.poles) == 0:
        return np.array([[zpk.k, 0, 0, 1, 0, 0]], dtype=float)
    return signal.zpk2sos(zpk.zeros, zpk.poles, zpk.k)


def analyze(state: FilterState, n_freq: int = 2048, n_time: int = 256) -> Response:
    zpk = state_zpk(state)
    fs = state.fs
    grid = np.linspace(0.0, fs / 2, n_freq)  # includes Nyquist
    f, h = signal.freqz_zpk(zpk.zeros, zpk.poles, zpk.k, worN=grid, fs=fs)
    w = 2 * np.pi * f / fs

    with np.errstate(divide="ignore"):
        mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    phase = np.unwrap(np.angle(h))
    gd = group_delay_zpk(zpk.zeros, zpk.poles, w)

    sos = zpk_to_sos(zpk)
    x = np.zeros(n_time)
    x[0] = 1.0
    impulse = signal.sosfilt(sos, x)
    step = np.cumsum(impulse)

    rmax = float(np.max(np.abs(zpk.poles))) if len(zpk.poles) else 0.0
    return Response(
        f=f, h=h, mag_db=mag_db, phase=phase, group_delay=gd,
        impulse=impulse, step=step, sos=sos, zpk=zpk,
        stable=bool(rmax < 1.0), max_pole_radius=rmax,
    )


def log_magnitude_grid(zpk: ZPK, extent: float = 1.5, n: int = 200) -> np.ndarray:
    """20 log10 |H(z)| over a square patch of the z-plane, indexed [x, y]."""
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    H = eval_zpk(zpk.zeros, zpk.poles, zpk.k, X + 1j * Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 20 * np.log10(np.abs(H))
    return np.nan_to_num(g, nan=0.0, posinf=200.0, neginf=-200.0)
