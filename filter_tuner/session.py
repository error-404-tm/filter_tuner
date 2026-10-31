"""Application state, wired together but independent of Qt.

``Session`` owns the editable values (filter with undo, selection, mask,
fixed-point settings) and the derived results (analysis, fixed-point
realization) that are recomputed whenever their inputs change. UI widgets
subscribe to these observables; none of them talk to each other directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfreqz

from . import dsp, quantize
from .mask import Mask
from .model import POLE, FilterState, default_state
from .observable import Observable, UndoableObservable
from .project_io import Project, QuantSettings


@dataclass(frozen=True, eq=False)
class FixedPointResult:
    settings: QuantSettings
    sos: np.ndarray                  # sections that get quantized (maybe scaled)
    output_gain: float
    quantized: quantize.QuantizedSOS
    response_f: np.ndarray
    response_db: np.ndarray
    headroom_bits: np.ndarray
    required_int_bits: int


@dataclass(frozen=True)
class ViewOptions:
    log_freq: bool = False
    show_heatmap: bool = False
    snap_unit_circle: bool = False
    snap_real_axis: bool = True


class Session:
    def __init__(self, state: FilterState | None = None):
        self.filter = UndoableObservable(state or default_state())
        self.selection: Observable[int | None] = Observable(0 if self.filter.value.roots else None)
        self.add_kind: Observable[str] = Observable(POLE)
        self.mask: Observable[Mask] = Observable(Mask())
        self.quant_settings: Observable[QuantSettings] = Observable(QuantSettings())
        self.view: Observable[ViewOptions] = Observable(ViewOptions())
        self.analysis: Observable[dsp.Response] = Observable(dsp.analyze(self.filter.value))
        self.fixed: Observable[FixedPointResult | None] = Observable(None)
        self.status: Observable[str] = Observable("")

        self.filter.subscribe(self._on_filter)
        self.quant_settings.subscribe(lambda _: self._recompute_fixed())
        self._recompute_fixed()

    # --- derived values ------------------------------------------------------
    def _on_filter(self, state: FilterState) -> None:
        sel = self.selection.value
        if sel is not None and sel >= len(state.roots):
            self.selection.set(len(state.roots) - 1 if state.roots else None)
        self.analysis.set(dsp.analyze(state))
        self._recompute_fixed()

    def _recompute_fixed(self) -> None:
        resp = self.analysis.value
        qs = self.quant_settings.value
        sos, gain = resp.sos, 1.0
        if qs.scaling != "none":
            sc = quantize.scale_sections(sos, qs.scaling)
            sos, gain = sc.sos, sc.output_gain
        q = quantize.quantize_sos(sos, qs.coef)
        f, h = sosfreqz(q.sos, worN=1024, fs=self.filter.value.fs)
        db = 20 * np.log10(np.maximum(np.abs(h) * gain, 1e-15))
        self.fixed.set(FixedPointResult(
            settings=qs, sos=sos, output_gain=gain, quantized=q,
            response_f=f, response_db=db,
            headroom_bits=quantize.headroom_bits(sos),
            required_int_bits=quantize.required_int_bits(resp.sos),
        ))

    # --- project -------------------------------------------------------------
    def project(self) -> Project:
        return Project(self.filter.value, self.mask.value, self.quant_settings.value)

    def load_project(self, p: Project) -> None:
        self.mask.set(p.mask)
        self.quant_settings.set(p.quant)
        self.filter.reset(p.state)
        self.selection.set(0 if p.state.roots else None)

    def replace_filter(self, state: FilterState, note: str = "") -> None:
        """An undoable wholesale replacement (seed design, optimizer result)."""
        self.filter.set(state)
        self.selection.set(0 if state.roots else None)
        if note:
            self.status.set(note)
