"""Sample rate, gain normalization and a stability/summary readout."""

from __future__ import annotations

import numpy as np
from pyqtgraph.Qt import QtWidgets

from ...model import GainSpec
from ...session import Session
from .. import theme
from ..widgets import blocked, combo, dspin

GAIN_LABELS = {
    "manual": "Manual k",
    "dc": "0 dB at DC",
    "nyquist": "0 dB at Nyquist",
    "peak": "0 dB at peak",
    "freq": "0 dB at frequency",
}


class FilterSettingsPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        lay = QtWidgets.QVBoxLayout(self)

        self.stability = QtWidgets.QLabel()
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        lay.addWidget(self.stability)
        lay.addWidget(self.summary)

        form = QtWidgets.QFormLayout()
        self.fs = dspin(1e-6, 1e12, 1.0, decimals=4)
        self.fs.setToolTip("Sample rate. 1.0 means normalized (cycles/sample).")
        self.mode = combo(GAIN_LABELS.values())
        self.k = dspin(-1e9, 1e9, 1.0, decimals=8)
        self.freq = dspin(0.0, 1e12, 0.25, decimals=6)
        form.addRow("Sample rate fs", self.fs)
        form.addRow("Gain", self.mode)
        form.addRow("k", self.k)
        form.addRow("ref. frequency", self.freq)
        lay.addLayout(form)
        lay.addStretch(1)

        self.fs.valueChanged.connect(self._set_fs)
        self.mode.currentIndexChanged.connect(lambda _: self._set_gain())
        self.k.valueChanged.connect(lambda _: self._set_gain())
        self.freq.valueChanged.connect(lambda _: self._set_gain())

        session.filter.subscribe(lambda _: self._refresh())
        session.analysis.subscribe(lambda _: self._refresh_summary())
        self._refresh()

    def _mode_key(self) -> str:
        return list(GAIN_LABELS)[self.mode.currentIndex()]

    def _refresh(self):
        st = self.s.filter.value
        with blocked(self.fs, self.mode, self.k, self.freq):
            self.fs.setValue(st.fs)
            self.mode.setCurrentIndex(list(GAIN_LABELS).index(st.gain.mode))
            self.k.setValue(self.s.analysis.value.zpk.k if st.gain.mode != "manual"
                            else st.gain.value)
            self.freq.setMaximum(st.fs / 2)
            self.freq.setValue(st.gain.freq)
            self.k.setEnabled(st.gain.mode == "manual")
            self.freq.setEnabled(st.gain.mode == "freq")
        self._refresh_summary()

    def _refresh_summary(self):
        r = self.s.analysis.value
        if r.stable:
            self.stability.setText(f"Stable  (max |p| = {r.max_pole_radius:.6f})")
            self.stability.setStyleSheet(theme.GOOD_STYLE)
        else:
            self.stability.setText(f"UNSTABLE  (max |p| = {r.max_pole_radius:.6f})")
            self.stability.setStyleSheet(theme.BAD_STYLE)
        peak = float(np.max(r.mag_db))
        self.summary.setText(
            f"{len(r.zpk.zeros)} zeros, {len(r.zpk.poles)} poles, "
            f"{len(r.sos)} biquad section(s)\n"
            f"k = {r.zpk.k:.6g}   peak |H| = {peak:.2f} dB at "
            f"{r.f[int(np.argmax(r.mag_db))]:.4g}"
        )
        if self.s.filter.value.gain.mode != "manual":
            with blocked(self.k):
                self.k.setValue(r.zpk.k)

    def _set_fs(self, fs: float):
        st = self.s.filter.value
        # keep the reference frequency at the same normalized position
        g = st.gain
        g = GainSpec(g.mode, g.value, g.freq * fs / st.fs)
        self.s.filter.set(st.with_fs(fs).with_gain(g), coalesce_key="fs")

    def _set_gain(self):
        st = self.s.filter.value
        mode = self._mode_key()
        value = self.k.value() if mode == "manual" else st.gain.value
        if mode == "manual" and st.gain.mode != "manual":
            value = self.s.analysis.value.zpk.k  # keep the current gain
        gain = GainSpec(mode, value, self.freq.value())
        self.s.filter.set(st.with_gain(gain), coalesce_key="gain")
