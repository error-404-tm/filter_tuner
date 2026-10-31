"""Seed the editor from a classical IIR or FIR design."""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets

from ... import design
from ...session import Session
from ..widgets import combo, dspin, error_box, ispin


class _BandInputs:
    """Band type + one or two cutoff spin boxes, shared by the IIR/FIR forms."""

    def __init__(self, form: QtWidgets.QFormLayout):
        self.btype = combo(design.BAND_TYPES)
        self.c1 = dspin(0.0, 1e12, 0.1, decimals=6)
        self.c2 = dspin(0.0, 1e12, 0.2, decimals=6)
        form.addRow("Type", self.btype)
        form.addRow("Cutoff 1", self.c1)
        form.addRow("Cutoff 2", self.c2)
        self.btype.currentTextChanged.connect(self._sync)
        self._sync()

    def _sync(self):
        self.c2.setEnabled(self.btype.currentText() in ("bandpass", "bandstop"))

    def cutoffs(self) -> tuple[float, ...]:
        if self.c2.isEnabled():
            return (self.c1.value(), self.c2.value())
        return (self.c1.value(),)

    def set_fs(self, fs: float):
        for w in (self.c1, self.c2):
            w.setMaximum(fs / 2 * 0.9999)


class DesignPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        lay = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Replaces the current zeros/poles with a classical design that you can "
            "then hand-tune. Frequencies are in the units of fs. Undo restores the old one."
        )
        note.setWordWrap(True)
        lay.addWidget(note)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._iir_tab(), "IIR")
        tabs.addTab(self._fir_tab(), "FIR")
        lay.addWidget(tabs)
        lay.addStretch(1)
        session.filter.subscribe(lambda st: self._set_fs(st.fs), call_now=True)

    def _iir_tab(self):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        self.family = combo(design.IIR_FAMILIES)
        form.addRow("Family", self.family)
        self.iir_band = _BandInputs(form)
        self.order = ispin(1, 40, 4)
        self.ripple = dspin(0.001, 20, 1.0, decimals=3, suffix=" dB")
        self.atten = dspin(1, 200, 40, decimals=1, suffix=" dB")
        form.addRow("Order", self.order)
        form.addRow("Passband ripple", self.ripple)
        form.addRow("Stopband atten.", self.atten)
        go = QtWidgets.QPushButton("Use this design")
        go.clicked.connect(self._make_iir)
        form.addRow(go)
        return w

    def _fir_tab(self):
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        self.method = combo(design.FIR_METHODS)
        form.addRow("Method", self.method)
        self.fir_band = _BandInputs(form)
        self.taps = ispin(2, 255, 31)
        self.window = combo(["hamming", "hann", "blackman", "kaiser,8", "boxcar"])
        self.window.setEditable(True)
        self.transition = dspin(1e-6, 1e12, 0.05, decimals=6)
        form.addRow("Taps", self.taps)
        form.addRow("Window (firwin)", self.window)
        form.addRow("Transition (remez)", self.transition)
        go = QtWidgets.QPushButton("Use this design")
        go.clicked.connect(self._make_fir)
        form.addRow(go)
        return w

    def _set_fs(self, fs):
        self.iir_band.set_fs(fs)
        self.fir_band.set_fs(fs)

    def _make_iir(self):
        spec = design.IIRSpec(
            family=self.family.currentText(), btype=self.iir_band.btype.currentText(),
            order=self.order.value(), cutoff=self.iir_band.cutoffs(),
            ripple_db=self.ripple.value(), atten_db=self.atten.value(),
        )
        try:
            st = design.iir_seed(spec, self.s.filter.value.fs)
        except Exception as e:
            return error_box(self, "IIR design failed", e)
        self.s.replace_filter(st, f"{spec.family} {spec.btype}, order {spec.order}")

    def _make_fir(self):
        win = self.window.currentText()
        if "," in win:
            name, arg = win.split(",", 1)
            win = (name.strip(), float(arg))
        spec = design.FIRSpec(
            method=self.method.currentText(), btype=self.fir_band.btype.currentText(),
            numtaps=self.taps.value(), cutoff=self.fir_band.cutoffs(),
            window=win, transition=self.transition.value(),
        )
        try:
            st = design.fir_seed(spec, self.s.filter.value.fs)
        except Exception as e:
            return error_box(self, "FIR design failed", e)
        self.s.replace_filter(st, f"FIR {spec.method} {spec.btype}, {spec.numtaps} taps")
