"""Editable filter state: roots, gain specification and sample rate.

Everything here is immutable and GUI-free, so states can be compared,
stored in undo history and serialized without copying concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

ZERO = "zero"
POLE = "pole"
KINDS = (ZERO, POLE)


@dataclass(frozen=True)
class Root:
    """A zero or pole.

    A paired root stands for a conjugate pair; ``value`` is kept in the upper
    half-plane. An unpaired root is a single real root (imaginary part 0).
    """

    kind: str
    value: complex
    paired: bool = True

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        v = complex(self.value)
        v = complex(v.real, abs(v.imag)) if self.paired else complex(v.real, 0.0)
        object.__setattr__(self, "value", v)

    @classmethod
    def polar(cls, kind: str, radius: float, angle: float, paired: bool = True) -> "Root":
        return cls(kind, radius * np.exp(1j * angle), paired)

    @property
    def radius(self) -> float:
        return abs(self.value)

    @property
    def angle(self) -> float:
        return float(np.angle(self.value))

    @property
    def order(self) -> int:
        return 2 if self.paired else 1

    def points(self) -> list[complex]:
        """Every root location this entry contributes."""
        if self.paired:
            return [self.value, self.value.conjugate()]
        return [self.value]

    def moved_to(self, z: complex) -> "Root":
        return replace(self, value=complex(z))


@dataclass(frozen=True)
class GainSpec:
    """How the overall gain k is chosen.

    mode:
      manual   k = ``value``
      dc       |H| = 1 at f = 0
      nyquist  |H| = 1 at f = fs/2
      peak     max |H| = 1
      freq     |H| = 1 at ``freq`` (Hz, in the same units as fs)
    """

    mode: str = "manual"
    value: float = 1.0
    freq: float = 0.25

    MODES = ("manual", "dc", "nyquist", "peak", "freq")

    def __post_init__(self):
        if self.mode not in self.MODES:
            raise ValueError(f"gain mode must be one of {self.MODES}")


@dataclass(frozen=True)
class FilterState:
    roots: tuple[Root, ...] = ()
    gain: GainSpec = field(default_factory=GainSpec)
    fs: float = 1.0

    # --- queries -------------------------------------------------------
    def of_kind(self, kind: str) -> list[Root]:
        return [r for r in self.roots if r.kind == kind]

    def points(self, kind: str) -> np.ndarray:
        pts: list[complex] = []
        for r in self.of_kind(kind):
            pts.extend(r.points())
        return np.array(pts, dtype=complex)

    @property
    def zeros(self) -> np.ndarray:
        return self.points(ZERO)

    @property
    def poles(self) -> np.ndarray:
        return self.points(POLE)

    # --- edits (return new states) ---------------------------------------
    def with_root(self, index: int, root: Root) -> "FilterState":
        roots = list(self.roots)
        roots[index] = root
        return replace(self, roots=tuple(roots))

    def adding(self, root: Root) -> "FilterState":
        return replace(self, roots=self.roots + (root,))

    def removing(self, index: int) -> "FilterState":
        roots = list(self.roots)
        del roots[index]
        return replace(self, roots=tuple(roots))

    def with_gain(self, gain: GainSpec) -> "FilterState":
        return replace(self, gain=gain)

    def with_fs(self, fs: float) -> "FilterState":
        if fs <= 0:
            raise ValueError("fs must be positive")
        return replace(self, fs=float(fs))


def default_state() -> FilterState:
    """A resonator with a notch: one zero pair on the circle, one pole pair inside."""
    return FilterState(
        roots=(
            Root.polar(ZERO, 1.0, np.pi / 3),
            Root.polar(POLE, 0.85, np.pi / 3),
        ),
        gain=GainSpec("dc"),
    )


def roots_from_zpk(z, p, tol: float = 1e-9) -> tuple[Root, ...]:
    """Group raw zero/pole arrays into Root entries (conjugate pairs + real singles)."""
    out: list[Root] = []
    for kind, arr in ((ZERO, np.atleast_1d(z)), (POLE, np.atleast_1d(p))):
        arr = np.asarray(arr, dtype=complex)
        scale = max(1.0, float(np.max(np.abs(arr)))) if arr.size else 1.0
        t = tol * scale
        upper = arr[arr.imag > t]
        lower = list(arr[arr.imag < -t])
        real = arr[np.abs(arr.imag) <= t]
        for u in upper:
            # consume the matching conjugate so mismatched input is caught
            if lower:
                j = int(np.argmin([abs(c - np.conj(u)) for c in lower]))
                if abs(lower[j] - np.conj(u)) > 1e-6 * scale:
                    raise ValueError("roots are not conjugate-symmetric")
                lower.pop(j)
            else:
                raise ValueError("roots are not conjugate-symmetric")
            out.append(Root(kind, complex(u), paired=True))
        if lower:
            raise ValueError("roots are not conjugate-symmetric")
        for x in real:
            out.append(Root(kind, complex(x.real, 0.0), paired=False))
    return tuple(out)
