"""Magnitude / phase / group-delay / time-response plots.

Each plot is its own small class that knows how to draw one view of a
``Response``; ``ResponsePlots`` just lays them out and feeds them.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..dsp import Response
from ..mask import Mask
from ..session import Session, ViewOptions
from . import theme


def _freq_label(fs: float) -> str:
    return "Frequency (cycles/sample)" if fs == 1.0 else "Frequency (Hz)"


def _xs(f: np.ndarray, view: ViewOptions):
    """Frequency axis data; drop f=0 on a log axis."""
    return (f > 0) if view.log_freq else slice(None)


class _FreqPlot:
    title = ""
    ylabel = ("", None)

    def __init__(self, plot: pg.PlotItem):
        self.plot = plot
        plot.setTitle(self.title)
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("left", *self.ylabel)
        plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.curve = plot.plot(pen=pg.mkPen(theme.CURVE_COLOR, width=2))

    def set_view(self, view: ViewOptions, fs: float):
        self.plot.setLogMode(x=view.log_freq, y=False)
        self.plot.setLabel("bottom", _freq_label(fs))


class MagnitudePlot(_FreqPlot):
    title = "Magnitude"
    ylabel = ("|H|", "dB")

    def __init__(self, plot):
        super().__init__(plot)
        self.quant_curve = plot.plot(pen=pg.mkPen(theme.QUANT_COLOR, width=1, style=theme.DASH))
        self.mask_curve = plot.plot(pen=pg.mkPen(theme.MASK_COLOR, width=2), connect="finite")
        self.violation = plot.plot(pen=None, symbol="o", symbolSize=3,
                                   symbolBrush=theme.MASK_COLOR, symbolPen=None)

    def update(self, r: Response, view: ViewOptions, mask: Mask, quant=None):
        sel = _xs(r.f, view)
        self.curve.setData(r.f[sel], r.mag_db[sel])
        if quant is not None:
            qs = _xs(quant.response_f, view)
            self.quant_curve.setData(quant.response_f[qs], quant.response_db[qs])
        else:
            self.quant_curve.setData([], [])
        self._draw_mask(r, view, mask)
        lo = max(float(np.min(r.mag_db[sel])), float(np.max(r.mag_db)) - 120)
        hi = float(np.max(r.mag_db))
        self.plot.setYRange(lo - 3, hi + 3, padding=0)

    def _draw_mask(self, r, view, mask: Mask):
        xs, ys = [], []
        for b in mask.bands:
            f_lo = max(b.f_lo, r.f[1]) if view.log_freq else b.f_lo
            for lim in (b.hi_db, b.lo_db):
                if lim is not None:
                    xs += [f_lo, b.f_hi, np.nan]
                    ys += [lim, lim, np.nan]
        self.mask_curve.setData(np.array(xs, float), np.array(ys, float))
        if mask.bands:
            bad = mask.excess_db(r.f, r.mag_db) > 1e-6
            bad &= r.f > 0 if view.log_freq else True
            self.violation.setData(r.f[bad], r.mag_db[bad])
        else:
            self.violation.setData([], [])


class PhasePlot(_FreqPlot):
    title = "Phase (unwrapped)"
    ylabel = ("phase", "rad")

    def update(self, r: Response, view: ViewOptions):
        sel = _xs(r.f, view)
        self.curve.setData(r.f[sel], r.phase[sel])


class GroupDelayPlot(_FreqPlot):
    title = "Group delay"
    ylabel = ("τ", "samples")

    def update(self, r: Response, view: ViewOptions):
        sel = _xs(r.f, view)
        gd = r.group_delay.copy()
        # keep spikes from roots on/near the circle from flattening the plot
        finite = gd[np.isfinite(gd)]
        if finite.size:
            lim = max(10.0, 4 * float(np.percentile(np.abs(finite), 95)))
            gd[np.abs(gd) > lim] = np.nan
        self.curve.setData(r.f[sel], gd[sel], connect="finite")


class TimePlot:
    def __init__(self, plot: pg.PlotItem):
        self.plot = plot
        plot.setTitle("Impulse / step response")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("bottom", "n")
        plot.addLegend(offset=(-10, 10))
        self.impulse = plot.plot(pen=pg.mkPen(theme.CURVE_COLOR, width=1), symbol="o",
                                 symbolSize=4, symbolBrush=theme.CURVE_COLOR, name="h[n]")
        self.step = plot.plot(pen=pg.mkPen(theme.STEP_COLOR, width=2), name="step")

    def update(self, r: Response):
        n = np.arange(len(r.impulse))
        self.impulse.setData(n, r.impulse)
        self.step.setData(n, r.step)


class ResponsePlots(pg.GraphicsLayoutWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent=parent)
        self.session = session
        self.mag = MagnitudePlot(self.addPlot(row=0, col=0))
        self.phase = PhasePlot(self.addPlot(row=0, col=1))
        self.gd = GroupDelayPlot(self.addPlot(row=1, col=0))
        self.time = TimePlot(self.addPlot(row=1, col=1))
        self.phase.plot.setXLink(self.mag.plot)
        self.gd.plot.setXLink(self.mag.plot)

        for obs in (session.analysis, session.view, session.mask, session.fixed):
            obs.subscribe(lambda _: self.refresh())
        self.refresh()

    def refresh(self):
        s = self.session
        r, view = s.analysis.value, s.view.value
        fs = s.filter.value.fs
        for p in (self.mag, self.phase, self.gd):
            p.set_view(view, fs)
        quant = s.fixed.value if s.quant_settings.value.show_overlay else None
        self.mag.update(r, view, s.mask.value, quant)
        self.phase.update(r, view)
        self.gd.update(r, view)
        self.time.update(r)
