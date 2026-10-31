"""Refine a hand-placed design so its magnitude response meets a mask.

The user's placement is the starting point; the optimizer nudges every
root (radius/angle for pairs, position for real roots) and optionally the
gain to minimize squared mask violation, keeping poles inside the circle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.signal import freqz_zpk

from .dsp import resolve_gain
from .mask import Mask
from .model import POLE, FilterState, GainSpec, Root


@dataclass(frozen=True, eq=False)
class FitResult:
    state: FilterState
    cost_before: float
    cost_after: float
    worst_before_db: float
    worst_after_db: float
    iterations: int
    message: str


@dataclass(frozen=True)
class FitOptions:
    pole_rmax: float = 0.995
    zero_rmax: float = 1.5
    fit_gain: bool = True
    max_iter: int = 300
    n_per_band: int = 96
    stability_margin_weight: float = 0.0  # >0 pushes poles away from the circle


class _Codec:
    """Maps a FilterState to/from the optimizer's parameter vector."""

    def __init__(self, state: FilterState, opts: FitOptions):
        self.state = state
        self.opts = opts
        x0, bounds = [], []
        for r in state.roots:
            rmax = opts.pole_rmax if r.kind == POLE else opts.zero_rmax
            if r.paired:
                x0 += [min(r.radius, rmax), r.angle]
                bounds += [(0.0, rmax), (0.0, np.pi)]
            else:
                x0.append(float(np.clip(r.value.real, -rmax, rmax)))
                bounds.append((-rmax, rmax))
        if opts.fit_gain:
            x0.append(0.0)  # gain offset in dB relative to the starting gain
            bounds.append((-120.0, 120.0))
        self.x0 = np.array(x0)
        self.bounds = bounds
        z0, p0 = state.zeros, state.poles
        self.k0 = resolve_gain(z0, _pad(z0, p0), state.gain, state.fs)

    def roots(self, x):
        out, i = [], 0
        for r in self.state.roots:
            if r.paired:
                out.append(Root.polar(r.kind, x[i], x[i + 1], True))
                i += 2
            else:
                out.append(Root(r.kind, complex(x[i], 0), False))
                i += 1
        return out

    def zpk(self, x):
        roots = self.roots(x)
        z = np.array([c for r in roots if r.kind != POLE for c in r.points()], complex)
        p = np.array([c for r in roots if r.kind == POLE for c in r.points()], complex)
        p = _pad(z, p)
        if self.opts.fit_gain:
            k = self.k0 * 10 ** (x[-1] / 20)
        else:
            k = resolve_gain(z, p, self.state.gain, self.state.fs)
        return z, p, k

    def to_state(self, x) -> FilterState:
        roots = tuple(self.roots(x))
        gain = self.state.gain
        if self.opts.fit_gain:
            gain = GainSpec("manual", float(self.k0 * 10 ** (x[-1] / 20)))
        return FilterState(roots, gain, self.state.fs)


def _pad(z, p):
    pad = max(0, len(z) - len(p))
    return np.concatenate([p, np.zeros(pad, complex)]) if pad else p


def mask_cost(state: FilterState, mask: Mask, n_per_band: int = 96) -> tuple[float, float]:
    """(mean squared violation, worst violation) for a state."""
    codec = _Codec(state, FitOptions(fit_gain=False, n_per_band=n_per_band))
    f = mask.grid(n_per_band)
    z, p, k = codec.zpk(codec.x0)
    return _cost(z, p, k, f, mask, state.fs)


def _cost(z, p, k, f, mask, fs):
    _, h = freqz_zpk(z, p, k, worN=f, fs=fs)
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    e = mask.excess_db(f, mag)
    return float(np.mean(e ** 2)), float(e.max() if e.size else 0.0)


def fit_to_mask(state: FilterState, mask: Mask, opts: FitOptions | None = None) -> FitResult:
    opts = opts or FitOptions()
    if not mask.bands:
        raise ValueError("mask has no bands")
    if not state.roots:
        raise ValueError("nothing to optimize: add zeros/poles first")
    codec = _Codec(state, opts)
    f = mask.grid(opts.n_per_band)

    def objective(x):
        z, p, k = codec.zpk(x)
        c, _ = _cost(z, p, k, f, mask, state.fs)
        if opts.stability_margin_weight and len(p):
            c += opts.stability_margin_weight * float(np.sum(np.abs(p) ** 8))
        return c

    before = _cost(*codec.zpk(codec.x0), f, mask, state.fs)
    res = optimize.minimize(
        objective, codec.x0, method="L-BFGS-B", bounds=codec.bounds,
        options={"maxiter": opts.max_iter},
    )
    after = _cost(*codec.zpk(res.x), f, mask, state.fs)
    return FitResult(
        state=codec.to_state(res.x),
        cost_before=before[0], cost_after=after[0],
        worst_before_db=before[1], worst_after_db=after[1],
        iterations=int(res.nit), message=str(res.message),
    )
