"""List of zeros/poles with precise numeric editing of the selected one."""

from __future__ import annotations

import numpy as np
from pyqtgraph.Qt import QtWidgets

from ...model import POLE, ZERO, Root
from ...session import Session
from ..widgets import blocked, dspin


def describe(i: int, r: Root) -> str:
    sym = "○" if r.kind == ZERO else "×"
    if r.paired:
        return f"{sym} {i + 1:>2}  {r.kind} pair   r={r.radius:.5f}  θ={r.angle / np.pi:.5f}π"
    return f"{sym} {i + 1:>2}  {r.kind} real   x={r.value.real:+.5f}"


class RootEditor(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        lay = QtWidgets.QVBoxLayout(self)

        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self._on_row)
        lay.addWidget(self.list, stretch=1)

        buttons = QtWidgets.QGridLayout()
        for col, (label, kind, paired) in enumerate([
            ("+ zero pair", ZERO, True), ("+ pole pair", POLE, True),
        ]):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind, p=paired: self._add(k, p))
            buttons.addWidget(b, 0, col)
        for col, (label, kind, paired) in enumerate([
            ("+ real zero", ZERO, False), ("+ real pole", POLE, False),
        ]):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind, p=paired: self._add(k, p))
            buttons.addWidget(b, 1, col)
        self.toggle_btn = QtWidgets.QPushButton("Pair ↔ real")
        self.toggle_btn.setToolTip("Turn a conjugate pair into a single real root or back")
        self.toggle_btn.clicked.connect(self._toggle_paired)
        self.del_btn = QtWidgets.QPushButton("Delete")
        self.del_btn.clicked.connect(self._delete)
        buttons.addWidget(self.toggle_btn, 2, 0)
        buttons.addWidget(self.del_btn, 2, 1)
        lay.addLayout(buttons)

        form = QtWidgets.QFormLayout()
        self.radius = dspin(0.0, 3.0, 1.0, decimals=6)
        self.angle = dspin(0.0, 1.0, 0.0, decimals=6, suffix=" π")
        self.dist = dspin(-1.0, 1.0, 0.0, decimals=7)
        self.dist.setToolTip("1 − r: fine control of distance to the unit circle")
        self.real = dspin(-3.0, 3.0, 0.0, decimals=6)
        self.imag = dspin(0.0, 3.0, 0.0, decimals=6)
        form.addRow("radius r", self.radius)
        form.addRow("1 − r", self.dist)
        form.addRow("angle θ", self.angle)
        form.addRow("real", self.real)
        form.addRow("imag", self.imag)
        lay.addLayout(form)

        self.radius.valueChanged.connect(lambda v: self._edit_polar(r=v))
        self.dist.valueChanged.connect(lambda v: self._edit_polar(r=1.0 - v))
        self.angle.valueChanged.connect(lambda v: self._edit_polar(theta=v * np.pi))
        self.real.valueChanged.connect(lambda v: self._edit_cart(re=v))
        self.imag.valueChanged.connect(lambda v: self._edit_cart(im=v))
        self._fields = (self.radius, self.dist, self.angle, self.real, self.imag)

        session.filter.subscribe(lambda _: self._refresh())
        session.selection.subscribe(lambda _: self._refresh())
        self._refresh()

    # --- view sync ---------------------------------------------------------------------
    def _refresh(self):
        state = self.s.filter.value
        sel = self.s.selection.value
        with blocked(self.list, *self._fields):
            if self.list.count() != len(state.roots):
                self.list.clear()
                self.list.addItems([describe(i, r) for i, r in enumerate(state.roots)])
            else:
                for i, r in enumerate(state.roots):
                    self.list.item(i).setText(describe(i, r))
            self.list.setCurrentRow(-1 if sel is None else sel)
            root = state.roots[sel] if sel is not None else None
            for w in self._fields + (self.toggle_btn, self.del_btn):
                w.setEnabled(root is not None)
            if root is None:
                return
            self.radius.setValue(root.radius)
            self.dist.setValue(1.0 - root.radius)
            self.angle.setValue(root.angle / np.pi)
            self.real.setValue(root.value.real)
            self.imag.setValue(root.value.imag)
            for w in (self.angle, self.imag, self.radius, self.dist):
                w.setEnabled(root.paired)

    def _on_row(self, row: int):
        self.s.selection.set(row if row >= 0 else None)

    # --- edits ---------------------------------------------------------------------------
    def _selected(self):
        sel = self.s.selection.value
        if sel is None:
            return None, None
        return sel, self.s.filter.value.roots[sel]

    def _edit_polar(self, r=None, theta=None):
        i, root = self._selected()
        if root is None:
            return
        r = root.radius if r is None else max(0.0, r)
        theta = root.angle if theta is None else theta
        new = Root.polar(root.kind, r, theta, root.paired)
        self.s.filter.set(self.s.filter.value.with_root(i, new), coalesce_key=("polar", i))

    def _edit_cart(self, re=None, im=None):
        i, root = self._selected()
        if root is None:
            return
        re = root.value.real if re is None else re
        im = root.value.imag if im is None else im
        new = root.moved_to(complex(re, im))
        self.s.filter.set(self.s.filter.value.with_root(i, new), coalesce_key=("cart", i))

    def _add(self, kind: str, paired: bool):
        n = len(self.s.filter.value.roots)
        # spread new roots out so they never land on top of an existing one
        angle = np.pi * ((0.15 + 0.17 * n) % 1.0)
        z = 0.7 * np.exp(1j * angle) if paired else complex(0.5 - 0.1 * (n % 10), 0)
        self.s.filter.set(self.s.filter.value.adding(Root(kind, z, paired)))
        self.s.selection.set(n)

    def _toggle_paired(self):
        i, root = self._selected()
        if root is None:
            return
        z = complex(root.value.real, 0) if root.paired else complex(root.value.real, 0.1)
        new = Root(root.kind, z, not root.paired)
        self.s.filter.set(self.s.filter.value.with_root(i, new))

    def _delete(self):
        i, _ = self._selected()
        if i is None:
            return
        self.s.filter.set(self.s.filter.value.removing(i))
        n = len(self.s.filter.value.roots)
        self.s.selection.set(min(i, n - 1) if n else None)
