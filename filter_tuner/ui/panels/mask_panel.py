"""Edit the magnitude spec mask and run the optimizer against it."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from ... import mask as masks
from ...mask import Band, Mask
from ...optimize import FitOptions, fit_to_mask
from ...session import Session
from ..widgets import blocked, dspin, error_box, ispin

COLUMNS = ("f from", "f to", "min dB", "max dB")


def _num(text: str):
    text = text.strip()
    return None if text in ("", "-", "none", "None") else float(text)


class MaskTable(QtWidgets.QTableWidget):
    """Bands as rows; blank min/max means unbounded."""

    def __init__(self, session: Session, parent=None):
        super().__init__(0, len(COLUMNS), parent)
        self.s = session
        self.setHorizontalHeaderLabels(COLUMNS)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.cellChanged.connect(lambda *_: self._commit())
        session.mask.subscribe(self._load, call_now=True)

    def _load(self, mask: Mask):
        with blocked(self):
            self.setRowCount(len(mask.bands))
            for i, b in enumerate(mask.bands):
                for j, v in enumerate((b.f_lo, b.f_hi, b.lo_db, b.hi_db)):
                    self.setItem(i, j, QtWidgets.QTableWidgetItem("" if v is None else f"{v:g}"))

    def _commit(self):
        bands = []
        try:
            for i in range(self.rowCount()):
                vals = [_num(self.item(i, j).text() if self.item(i, j) else "")
                        for j in range(len(COLUMNS))]
                if vals[0] is None or vals[1] is None:
                    raise ValueError(f"row {i + 1}: both frequencies are required")
                bands.append(Band(*vals))
        except ValueError as e:
            self.s.status.set(f"Mask not updated: {e}")
            return
        self.s.mask.set(Mask(tuple(bands)))

    def add_band(self):
        fs = self.s.filter.value.fs
        self.s.mask.set(Mask(self.s.mask.value.bands + (Band(0.0, fs / 4, None, 0.0),)))

    def remove_selected(self):
        rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True)
        bands = list(self.s.mask.value.bands)
        for r in rows:
            del bands[r]
        self.s.mask.set(Mask(tuple(bands)))


class MaskPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        lay = QtWidgets.QVBoxLayout(self)

        tmpl = QtWidgets.QHBoxLayout()
        for name in ("Lowpass", "Highpass", "Bandpass"):
            b = QtWidgets.QPushButton(name)
            b.setToolTip("Replace the mask with a template you can then edit")
            b.clicked.connect(lambda _=False, n=name: self._template(n))
            tmpl.addWidget(b)
        lay.addLayout(tmpl)

        self.table = MaskTable(session)
        lay.addWidget(self.table, stretch=1)
        row = QtWidgets.QHBoxLayout()
        add, rem, clear = (QtWidgets.QPushButton(t) for t in ("Add band", "Remove", "Clear"))
        add.clicked.connect(self.table.add_band)
        rem.clicked.connect(self.table.remove_selected)
        clear.clicked.connect(lambda: session.mask.set(Mask()))
        for b in (add, rem, clear):
            row.addWidget(b)
        lay.addLayout(row)

        box = QtWidgets.QGroupBox("Optimizer")
        form = QtWidgets.QFormLayout(box)
        self.fit_gain = QtWidgets.QCheckBox("Optimize gain too")
        self.fit_gain.setChecked(True)
        self.rmax = dspin(0.5, 0.99999, 0.995, decimals=5)
        self.iters = ispin(10, 5000, 300)
        form.addRow(self.fit_gain)
        form.addRow("Max pole radius", self.rmax)
        form.addRow("Max iterations", self.iters)
        go = QtWidgets.QPushButton("Fit current design to mask")
        go.clicked.connect(self._fit)
        form.addRow(go)
        self.result = QtWidgets.QLabel()
        self.result.setWordWrap(True)
        form.addRow(self.result)
        lay.addWidget(box)

        session.analysis.subscribe(lambda _: self._show_violation())
        session.mask.subscribe(lambda _: self._show_violation())
        self._show_violation()

    def _template(self, name):
        fs = self.s.filter.value.fs
        m = {
            "Lowpass": lambda: masks.lowpass_mask(fs, 0.1 * fs, 0.15 * fs),
            "Highpass": lambda: masks.highpass_mask(fs, 0.25 * fs, 0.3 * fs),
            "Bandpass": lambda: masks.bandpass_mask(fs, 0.1 * fs, 0.15 * fs, 0.25 * fs, 0.3 * fs),
        }[name]()
        self.s.mask.set(m)

    def _show_violation(self):
        m = self.s.mask.value
        if not m.bands:
            self.result.setText("No mask. Add bands or pick a template.")
            return
        r = self.s.analysis.value
        worst = m.worst_violation_db(r.f, r.mag_db)
        self.result.setText("Spec met ✓" if worst <= 1e-6 else f"Worst violation: {worst:.2f} dB")

    def _fit(self):
        opts = FitOptions(pole_rmax=self.rmax.value(), fit_gain=self.fit_gain.isChecked(),
                          max_iter=self.iters.value())
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            res = fit_to_mask(self.s.filter.value, self.s.mask.value, opts)
        except Exception as e:
            return error_box(self, "Optimization failed", e)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.s.filter.set(res.state)
        self.s.status.set(
            f"Fit: worst violation {res.worst_before_db:.2f} → {res.worst_after_db:.2f} dB "
            f"in {res.iterations} iterations (Ctrl+Z to undo)"
        )
