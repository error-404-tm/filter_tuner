"""Small widget helpers shared by the panels."""

from __future__ import annotations

from contextlib import contextmanager

from pyqtgraph.Qt import QtGui, QtWidgets


def dspin(lo: float, hi: float, value: float, decimals: int = 4,
          step: float | None = None, suffix: str = "") -> QtWidgets.QDoubleSpinBox:
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    if step is None:
        w.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
    else:
        w.setSingleStep(step)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    return w


def ispin(lo: int, hi: int, value: int) -> QtWidgets.QSpinBox:
    w = QtWidgets.QSpinBox()
    w.setRange(lo, hi)
    w.setValue(value)
    w.setKeyboardTracking(False)
    return w


def combo(items, current=None) -> QtWidgets.QComboBox:
    w = QtWidgets.QComboBox()
    w.addItems(list(items))
    if current is not None:
        w.setCurrentText(current)
    return w


def mono_text(read_only: bool = True) -> QtWidgets.QPlainTextEdit:
    w = QtWidgets.QPlainTextEdit()
    w.setReadOnly(read_only)
    f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    w.setFont(f)
    w.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
    return w


@contextmanager
def blocked(*widgets):
    """Update widgets programmatically without firing their change signals."""
    old = [w.blockSignals(True) for w in widgets]
    try:
        yield
    finally:
        for w, o in zip(widgets, old):
            w.blockSignals(o)


def error_box(parent, title: str, exc: Exception) -> None:
    QtWidgets.QMessageBox.warning(parent, title, f"{type(exc).__name__}: {exc}")
