"""Seed designs from classical IIR and FIR methods, as editable states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .model import POLE, FilterState, GainSpec, Root, roots_from_zpk

IIR_FAMILIES = {
    "Butterworth": "butter",
    "Chebyshev I": "cheby1",
    "Chebyshev II": "cheby2",
    "Elliptic": "ellip",
    "Bessel": "bessel",
}
BAND_TYPES = ("lowpass", "highpass", "bandpass", "bandstop")
FIR_METHODS = ("firwin", "remez")


@dataclass(frozen=True)
class IIRSpec:
    family: str = "Butterworth"
    btype: str = "lowpass"
    order: int = 4
    cutoff: tuple[float, ...] = (0.1,)   # Hz (one value, or two for band types)
    ripple_db: float = 1.0               # Chebyshev I / elliptic passband ripple
    atten_db: float = 40.0               # Chebyshev II / elliptic stopband


@dataclass(frozen=True)
class FIRSpec:
    method: str = "firwin"
    btype: str = "lowpass"
    numtaps: int = 31
    cutoff: tuple[float, ...] = (0.1,)
    window: str = "hamming"
    transition: float = 0.05             # remez transition width (Hz)


def _cutoff(spec_cutoff, btype):
    c = tuple(float(x) for x in spec_cutoff)
    needs_two = btype in ("bandpass", "bandstop")
    if needs_two and len(c) != 2:
        raise ValueError(f"{btype} needs two cutoff frequencies")
    if not needs_two:
        c = c[:1]
    return c if needs_two else c[0]


def iir_seed(spec: IIRSpec, fs: float = 1.0) -> FilterState:
    ftype = IIR_FAMILIES[spec.family]
    kw = {}
    if ftype in ("cheby1", "ellip"):
        kw["rp"] = spec.ripple_db
    if ftype in ("cheby2", "ellip"):
        kw["rs"] = spec.atten_db
    z, p, k = signal.iirfilter(
        spec.order, _cutoff(spec.cutoff, spec.btype), btype=spec.btype,
        ftype=ftype, output="zpk", fs=fs, **kw,
    )
    return FilterState(roots_from_zpk(z, p), GainSpec("manual", float(k)), fs)


def fir_taps(spec: FIRSpec, fs: float = 1.0) -> np.ndarray:
    cut = _cutoff(spec.cutoff, spec.btype)
    n = spec.numtaps
    if spec.btype in ("highpass", "bandstop") and n % 2 == 0:
        n += 1  # type I needed for a passband at Nyquist
    if spec.method == "firwin":
        pass_zero = spec.btype in ("lowpass", "bandstop")
        return signal.firwin(n, cut, window=spec.window, pass_zero=pass_zero, fs=fs)
    if spec.method == "remez":
        t = spec.transition
        nyq = fs / 2
        edges, desired = _remez_bands(spec.btype, np.atleast_1d(cut), t, nyq)
        return signal.remez(n, edges, desired, fs=fs)
    raise ValueError(f"unknown FIR method {spec.method}")


def _remez_bands(btype, cut, t, nyq):
    def clip(x):
        return float(np.clip(x, 0.0, nyq))

    if btype == "lowpass":
        return [0, clip(cut[0] - t / 2), clip(cut[0] + t / 2), nyq], [1, 0]
    if btype == "highpass":
        return [0, clip(cut[0] - t / 2), clip(cut[0] + t / 2), nyq], [0, 1]
    lo, hi = cut
    e = [0, clip(lo - t / 2), clip(lo + t / 2), clip(hi - t / 2), clip(hi + t / 2), nyq]
    return e, ([0, 1, 0] if btype == "bandpass" else [1, 0, 1])


def fir_seed(spec: FIRSpec, fs: float = 1.0) -> FilterState:
    b = fir_taps(spec, fs)
    return state_from_fir(b, fs)


def state_from_fir(b: np.ndarray, fs: float = 1.0) -> FilterState:
    b = np.trim_zeros(np.asarray(b, dtype=float), "f")
    zeros = np.roots(b) if len(b) > 1 else np.array([])
    # np.roots output is only approximately conjugate symmetric; symmetrize
    zeros = _symmetrize(zeros)
    roots = list(roots_from_zpk(zeros, []))
    n_poles = len(b) - 1
    roots += [Root(POLE, 0j, paired=True)] * (n_poles // 2)
    if n_poles % 2:
        roots.append(Root(POLE, 0j, paired=False))
    return FilterState(tuple(roots), GainSpec("manual", float(b[0])), fs)


def _symmetrize(r: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    r = np.asarray(r, dtype=complex)
    out = []
    used = np.zeros(len(r), bool)
    for i, x in enumerate(r):
        if used[i]:
            continue
        used[i] = True
        if abs(x.imag) <= tol * max(1, abs(x)):
            out.append(complex(x.real, 0))
            continue
        cand = [j for j in range(len(r)) if not used[j]]
        j = min(cand, key=lambda j: abs(r[j] - np.conj(x)))
        used[j] = True
        m = (x + np.conj(r[j])) / 2
        out += [m, np.conj(m)]
    return np.array(out)
