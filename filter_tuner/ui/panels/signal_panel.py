"""Run a recorded signal (WAV/CSV/NPY) through the filter and compare spectra."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ... import signal_io
from ...session import Session
from .. import theme
from ..widgets import error_box

MAX_PLOT_POINTS = 20000


class SpectrogramView(pg.PlotWidget):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("inferno"))
        self.addItem(self.img)
        self.setLabel("bottom", "time", units="s")
        self.setLabel("left", "frequency")

    def show_spec(self, spec: signal_io.Spectrogram, levels):
        self.img.setImage(spec.db, levels=levels, autoLevels=False)
        t0, t1 = (spec.t[0], spec.t[-1]) if len(spec.t) > 1 else (0.0, 1.0)
        self.img.setRect(QtCore.QRectF(t0, spec.f[0], max(t1 - t0, 1e-12), spec.f[-1] - spec.f[0]))
        self.autoRange()


class SignalPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        self.loaded: signal_io.LoadedSignal | None = None
        self.y: np.ndarray | None = None

        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        load = QtWidgets.QPushButton("Load signal…")
        load.clicked.connect(self._load)
        self.save_btn = QtWidgets.QPushButton("Save filtered…")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        row.addWidget(load)
        row.addWidget(self.save_btn)
        lay.addLayout(row)

        opts = QtWidgets.QHBoxLayout()
        self.zero_phase = QtWidgets.QCheckBox("Zero-phase (filtfilt)")
        self.use_quant = QtWidgets.QCheckBox("Quantized coefficients")
        self.use_fs = QtWidgets.QCheckBox("Use file's sample rate")
        self.use_fs.setChecked(True)
        for w in (self.zero_phase, self.use_quant, self.use_fs):
            opts.addWidget(w)
            w.toggled.connect(lambda _: self._run())
        lay.addLayout(opts)

        self.info = QtWidgets.QLabel("No signal loaded (WAV, CSV/TXT with value in last column, or NPY).")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

        self.wave = pg.PlotWidget(title="Waveform")
        self.wave.addLegend()
        self.wave_in = self.wave.plot(pen=pg.mkPen((150, 150, 150)), name="input")
        self.wave_out = self.wave.plot(pen=pg.mkPen(theme.STEP_COLOR), name="filtered")
        self.spec_in = SpectrogramView("Input spectrogram")
        self.spec_out = SpectrogramView("Filtered spectrogram")
        self.spec_out.setXLink(self.spec_in)
        self.spec_out.setYLink(self.spec_in)
        for w in (self.wave, self.spec_in, self.spec_out):
            lay.addWidget(w, stretch=1)

        session.analysis.subscribe(lambda _: self._run())
        session.fixed.subscribe(lambda _: self.use_quant.isChecked() and self._run())

    # --- actions --------------------------------------------------------------------
    def _load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load signal", "", "Signals (*.wav *.csv *.txt *.npy);;All files (*)")
        if not path:
            return
        try:
            self.loaded = signal_io.load_signal(path)
        except Exception as e:
            return error_box(self, "Could not load signal", e)
        self._run()

    def _save(self):
        if self.y is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save filtered signal", "filtered.wav", "WAV (*.wav);;CSV (*.csv);;NPY (*.npy)")
        if path:
            signal_io.save_signal(path, self.y, self._fs())
            self.s.status.set(f"Wrote {path}")

    def _fs(self) -> float:
        if self.use_fs.isChecked() and self.loaded and self.loaded.fs:
            return self.loaded.fs
        return self.s.filter.value.fs

    def _run(self):
        if self.loaded is None or not self.isVisible():
            return
        x = self.loaded.x
        fs = self._fs()
        sos = self.s.analysis.value.sos
        if self.use_quant.isChecked() and self.s.fixed.value is not None:
            fx = self.s.fixed.value
            sos = fx.quantized.sos.copy()
            sos[0, :3] *= fx.output_gain
        y = signal_io.apply_sos(sos, x, self.zero_phase.isChecked())
        self.y = y
        self.save_btn.setEnabled(True)

        if fs != self.s.filter.value.fs:
            note = (f"  Note: file fs = {fs:g} but filter fs = {self.s.filter.value.fs:g};"
                    " the filter acts at its normalized frequencies.")
        else:
            note = ""
        self.info.setText(f"{self.loaded.source}: {len(x)} samples @ {fs:g}.{note}")

        step = max(1, len(x) // MAX_PLOT_POINTS)
        t = np.arange(0, len(x), step) / fs
        self.wave_in.setData(t, x[::step])
        self.wave_out.setData(t, y[::step])
        si, so = signal_io.spectrogram(x, fs), signal_io.spectrogram(y, fs)
        top = float(np.max(si.db))
        levels = (top - 100, top)
        self.spec_in.show_spec(si, levels)
        self.spec_out.show_spec(so, levels)

    def showEvent(self, ev):
        super().showEvent(ev)
        self._run()
