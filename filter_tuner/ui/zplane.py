"""Z-plane view: draws roots, the optional |H(z)| landscape and the quantized
overlay, and forwards mouse gestures to a ZPlaneEditor.

Mouse:
  click empty space      add a conjugate pair of the current kind (Z/P)
  shift-click            add a single real root
  click a root           select it
  drag a root            move it (the conjugate follows); hold Shift for fine
  right-click a root     delete it
  drag empty space       pan; wheel zooms
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from ..dsp import log_magnitude_grid
from ..model import POLE, ZERO
from ..session import Session
from . import theme
from .zplane_editor import ZPlaneEditor

LEFT = QtCore.Qt.MouseButton.LeftButton
RIGHT = QtCore.Qt.MouseButton.RightButton
SHIFT = QtCore.Qt.KeyboardModifier.ShiftModifier

HEATMAP_EXTENT = 1.6
HIT_PIXELS = 12


class _ZPlaneViewBox(pg.ViewBox):
    def __init__(self, editor: ZPlaneEditor, hit_test):
        super().__init__(lockAspect=True)
        self._editor = editor
        self._hit = hit_test

    def _z(self, scene_pos) -> complex:
        p = self.mapSceneToView(scene_pos)
        return complex(p.x(), p.y())

    def mouseClickEvent(self, ev):
        hit = self._hit(ev.scenePos())
        if ev.button() == LEFT:
            ev.accept()
            if hit is not None:
                self._editor.select(hit[0])
            else:
                paired = not (ev.modifiers() & SHIFT)
                self._editor.add_at(self._z(ev.scenePos()), paired)
        elif ev.button() == RIGHT and hit is not None:
            ev.accept()
            self._editor.delete(hit[0])
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != LEFT:
            return super().mouseDragEvent(ev, axis)
        if ev.isStart():
            hit = self._hit(ev.buttonDownScenePos())
            if hit is None:
                return super().mouseDragEvent(ev, axis)
            self._editor.begin_drag(hit[0], hit[1], self._z(ev.buttonDownScenePos()))
        if not self._editor.dragging:
            return super().mouseDragEvent(ev, axis)
        ev.accept()
        self._editor.drag_to(self._z(ev.scenePos()), fine=bool(ev.modifiers() & SHIFT))
        if ev.isFinish():
            self._editor.end_drag()


class ZPlaneWidget(pg.PlotWidget):
    def __init__(self, session: Session, editor: ZPlaneEditor, parent=None):
        self.session = session
        self.editor = editor
        vb = _ZPlaneViewBox(editor, self._hit_test)
        super().__init__(parent=parent, viewBox=vb, title="Z-plane")
        self.vb = vb
        self._build_items()
        self.setXRange(-1.5, 1.5)
        self.setYRange(-1.5, 1.5)

        session.filter.subscribe(lambda _: self._draw_roots())
        session.selection.subscribe(lambda _: self._draw_roots())
        session.analysis.subscribe(lambda _: self._draw_heatmap())
        session.view.subscribe(lambda _: self._draw_heatmap())
        session.fixed.subscribe(lambda _: self._draw_quantized())
        session.quant_settings.subscribe(lambda _: self._draw_quantized())
        self._draw_roots()
        self._draw_heatmap()
        self._draw_quantized()

    # --- items ----------------------------------------------------------------------
    def _build_items(self):
        p = self.getPlotItem()
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setLabel("bottom", "Re z")
        p.setLabel("left", "Im z")

        self.heatmap = pg.ImageItem()
        self.heatmap.setZValue(-10)
        self.heatmap.setColorMap(pg.colormap.get("viridis"))
        p.addItem(self.heatmap)

        t = np.linspace(0, 2 * np.pi, 512)
        p.plot(np.cos(t), np.sin(t), pen=pg.mkPen((200, 200, 200), style=theme.DASH))

        self.zero_item = pg.ScatterPlotItem(
            symbol="o", size=14, brush=None, pen=pg.mkPen(theme.ZERO_COLOR, width=2))
        self.pole_item = pg.ScatterPlotItem(
            symbol="x", size=16, pen=pg.mkPen(theme.POLE_COLOR, width=2),
            brush=pg.mkBrush(theme.POLE_COLOR))
        self.quant_item = pg.ScatterPlotItem(
            symbol="+", size=9, pen=pg.mkPen(theme.QUANT_COLOR, width=1),
            brush=pg.mkBrush(theme.QUANT_COLOR))
        self.select_item = pg.ScatterPlotItem(
            symbol="s", size=26, brush=None, pen=pg.mkPen(theme.SELECT_COLOR, width=2))
        self.order_text: list[pg.TextItem] = []
        for item in (self.zero_item, self.pole_item, self.quant_item, self.select_item):
            p.addItem(item)

    # --- drawing ----------------------------------------------------------------------
    def _draw_roots(self):
        state = self.session.filter.value
        for kind, item in ((ZERO, self.zero_item), (POLE, self.pole_item)):
            pts = state.points(kind)
            item.setData(pts.real, pts.imag)
        self._draw_multiplicity(state)
        sel = self.session.selection.value
        if sel is not None and sel < len(state.roots):
            pts = np.array(state.roots[sel].points())
            self.select_item.setData(pts.real, pts.imag)
        else:
            self.select_item.setData([], [])

    def _draw_multiplicity(self, state):
        """Label coincident roots with their multiplicity (e.g. FIR poles at 0)."""
        for t in self.order_text:
            self.getPlotItem().removeItem(t)
        self.order_text = []
        for kind in (ZERO, POLE):
            pts = np.round(state.points(kind), 6)
            if not len(pts):
                continue
            uniq, counts = np.unique(pts, return_counts=True)
            for u, c in zip(uniq, counts):
                if c > 1:
                    color = theme.ZERO_COLOR if kind == ZERO else theme.POLE_COLOR
                    t = pg.TextItem(str(c), color=color, anchor=(-0.3, 1.1))
                    t.setPos(u.real, u.imag)
                    self.getPlotItem().addItem(t)
                    self.order_text.append(t)

    def _draw_heatmap(self):
        if not self.session.view.value.show_heatmap:
            self.heatmap.setVisible(False)
            return
        grid = log_magnitude_grid(self.session.analysis.value.zpk, HEATMAP_EXTENT, 180)
        lo, hi = np.percentile(grid, [2, 98])
        levels = (max(lo, -100.0), min(max(hi, lo + 1), 80.0))
        self.heatmap.setImage(grid, levels=levels, autoLevels=False)
        e = HEATMAP_EXTENT
        self.heatmap.setRect(QtCore.QRectF(-e, -e, 2 * e, 2 * e))  # after setImage
        self.heatmap.setVisible(True)

    def _draw_quantized(self):
        fx = self.session.fixed.value
        if fx is None or not self.session.quant_settings.value.show_overlay:
            self.quant_item.setData([], [])
            return
        pts = np.concatenate([fx.quantized.poles, fx.quantized.zeros])
        self.quant_item.setData(pts.real, pts.imag)

    # --- hit testing ------------------------------------------------------------------
    def _hit_test(self, scene_pos):
        """(root index, grabbed_conjugate) of the root nearest the cursor, or None."""
        best, best_d = None, HIT_PIXELS
        for i, root in enumerate(self.session.filter.value.roots):
            for j, z in enumerate(root.points()):
                sp = self.vb.mapViewToScene(QtCore.QPointF(z.real, z.imag))
                d = np.hypot(sp.x() - scene_pos.x(), sp.y() - scene_pos.y())
                if d < best_d:
                    best, best_d = (i, j == 1), d
        return best
