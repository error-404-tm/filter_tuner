"""Fixed-point settings and a report on what quantization does to the filter."""

from __future__ import annotations

import numpy as np
from pyqtgraph.Qt import QtWidgets

from ...project_io import QuantSettings
from ...quantize import QFormat
from ...session import FixedPointResult, Session
from .. import theme
from ..widgets import blocked, combo, ispin, mono_text

SCALINGS = {"none": "None", "linf": "L∞ (no sinusoid overflow)", "l2": "L2 (noise power)"}


def report(fx: FixedPointResult, data_bits: int) -> str:
    q = fx.quantized
    lines = [
        f"Coefficients: {q.fmt.label()}, range [{q.fmt.range[0]:g}, {q.fmt.range[1]:g}]",
        f"Integer bits needed by largest coefficient: {fx.required_int_bits}",
        f"Max coefficient error: {q.max_coef_error:.3g}  (LSB {q.fmt.lsb:.3g})",
        f"Quantized max |pole|: {q.max_pole_radius:.6f}",
    ]
    if fx.settings.scaling != "none":
        lines.append(f"Sections scaled ({fx.settings.scaling}); output gain {fx.output_gain:.6g}")
    lines += ["", "Section  headroom  b0 b1 b2 | a1 a2 (integers)"]
    for i, (sec, hb) in enumerate(zip(q.ints, fx.headroom_bits)):
        b0, b1, b2, _, a1, a2 = (int(v) for v in sec)
        lines.append(f"  {i + 1:>3}    +{hb} bit   {b0} {b1} {b2} | {a1} {a2}")
    lines += ["", f"Data path: Q1.{data_bits - 1}. Headroom = extra integer bits a section",
              "output needs for a full-scale sinusoid input (scale to bring it to 0)."]
    return "\n".join(lines)


class QuantPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.word = ispin(2, 32, 16)
        self.frac = ispin(1, 31, 14)
        self.data = ispin(4, 32, 16)
        self.scaling = combo(SCALINGS.values())
        self.overlay = QtWidgets.QCheckBox("Show quantized roots and response")
        form.addRow("Coefficient bits", self.word)
        form.addRow("Fractional bits", self.frac)
        form.addRow("Data bits", self.data)
        form.addRow("Section scaling", self.scaling)
        form.addRow(self.overlay)
        lay.addLayout(form)

        self.status = QtWidgets.QLabel()
        lay.addWidget(self.status)
        self.text = mono_text()
        lay.addWidget(self.text, stretch=1)

        for w in (self.word, self.frac, self.data):
            w.valueChanged.connect(lambda _: self._commit())
        self.scaling.currentIndexChanged.connect(lambda _: self._commit())
        self.overlay.toggled.connect(lambda _: self._commit())

        session.quant_settings.subscribe(self._load, call_now=True)
        session.fixed.subscribe(lambda _: self._show(), call_now=True)

    def _load(self, qs: QuantSettings):
        with blocked(self.word, self.frac, self.data, self.scaling, self.overlay):
            self.word.setValue(qs.coef.word_bits)
            self.frac.setMaximum(qs.coef.word_bits - 1)
            self.frac.setValue(qs.coef.frac_bits)
            self.data.setValue(qs.data_bits)
            self.scaling.setCurrentIndex(list(SCALINGS).index(qs.scaling))
            self.overlay.setChecked(qs.show_overlay)

    def _commit(self):
        word = self.word.value()
        self.frac.setMaximum(word - 1)
        qs = QuantSettings(
            QFormat(word, min(self.frac.value(), word - 1)), self.data.value(),
            list(SCALINGS)[self.scaling.currentIndex()], self.overlay.isChecked(),
        )
        self.s.quant_settings.set(qs)

    def _show(self):
        fx = self.s.fixed.value
        if fx is None:
            return
        q = fx.quantized
        ideal = self.s.analysis.value.stable
        if q.saturated:
            msg, style = "Coefficients SATURATE: use fewer fractional bits", theme.BAD_STYLE
        elif not q.stable:
            msg, style = "Quantized filter is UNSTABLE", theme.BAD_STYLE
        elif ideal and np.any(fx.headroom_bits > 0) and fx.settings.scaling == "none":
            msg, style = "OK, but sections need headroom (try scaling)", theme.GOOD_STYLE
        else:
            msg, style = "Quantized filter OK", theme.GOOD_STYLE
        self.status.setText(msg)
        self.status.setStyleSheet(style)
        self.text.setPlainText(report(fx, fx.settings.data_bits))
