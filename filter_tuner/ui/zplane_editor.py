"""Editing operations on the z-plane, separated from drawing and Qt events.

The z-plane widget translates mouse events into calls on this class;
everything here works in z-plane coordinates on the Session.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Root
from ..session import Session


@dataclass
class _Drag:
    index: int
    conjugate: bool        # grabbed the lower-half-plane twin of a pair
    last: complex          # last mouse position (z-plane)
    value: complex         # unsnapped root position being tracked


class ZPlaneEditor:
    SNAP_TOL = 0.025
    FINE_SCALE = 0.1

    def __init__(self, session: Session):
        self.s = session
        self._drag: _Drag | None = None

    # --- selection / add / delete --------------------------------------------
    def select(self, index: int | None) -> None:
        self.s.selection.set(index)

    def add_at(self, z: complex, paired: bool) -> None:
        z = self._snap(complex(z), paired)
        root = Root(self.s.add_kind.value, z, paired)
        self.s.filter.set(self.s.filter.value.adding(root))
        self.s.selection.set(len(self.s.filter.value.roots) - 1)

    def delete(self, index: int | None = None) -> None:
        index = self.s.selection.value if index is None else index
        roots = self.s.filter.value.roots
        if index is None or not 0 <= index < len(roots):
            return
        self.s.filter.set(self.s.filter.value.removing(index))
        n = len(self.s.filter.value.roots)
        self.s.selection.set(min(index, n - 1) if n else None)

    def cycle_selection(self, step: int) -> None:
        n = len(self.s.filter.value.roots)
        if not n:
            return
        cur = self.s.selection.value
        self.s.selection.set(0 if cur is None else (cur + step) % n)

    # --- dragging ----------------------------------------------------------------
    def begin_drag(self, index: int, conjugate: bool, at: complex) -> None:
        root = self.s.filter.value.roots[index]
        self._drag = _Drag(index, conjugate, complex(at), root.value)
        self.s.selection.set(index)
        self.s.filter.begin_gesture()

    def drag_to(self, at: complex, fine: bool = False) -> None:
        d = self._drag
        if d is None:
            return
        at = complex(at)
        delta = (at - d.last) * (self.FINE_SCALE if fine else 1.0)
        d.last = at
        if d.conjugate:
            delta = delta.conjugate()
        d.value += delta
        state = self.s.filter.value
        root = state.roots[d.index]
        moved = root.moved_to(self._snap(d.value, root.paired))
        self.s.filter.set(state.with_root(d.index, moved))

    def end_drag(self) -> None:
        if self._drag is not None:
            self._drag = None
            self.s.filter.end_gesture()

    @property
    def dragging(self) -> bool:
        return self._drag is not None

    # --- snapping ------------------------------------------------------------------
    def _snap(self, z: complex, paired: bool) -> complex:
        v = self.s.view.value
        if not paired:
            z = complex(z.real, 0.0)
        r = abs(z)
        if v.snap_unit_circle and abs(r - 1.0) < self.SNAP_TOL and r > 0:
            z = z / r
        if paired and v.snap_real_axis and abs(z.imag) < self.SNAP_TOL:
            z = complex(z.real, 0.0)
        return z
