"""Menu/toolbar actions and keyboard shortcuts."""

from __future__ import annotations

from dataclasses import replace

from pyqtgraph.Qt import QtGui, QtWidgets

from .. import project_io
from ..model import POLE, ZERO, FilterState, default_state
from ..session import Session
from .widgets import error_box
from .zplane_editor import ZPlaneEditor

FILE_FILTER = "Filter projects (*.json);;All files (*)"


class FileActions:
    """New / open / save, tracking the current file path."""

    def __init__(self, window: QtWidgets.QMainWindow, session: Session):
        self.w = window
        self.s = session
        self.path: str | None = None

    def new(self, blank: bool = False):
        self.s.load_project(project_io.Project(FilterState() if blank else default_state()))
        self.path = None
        self._title()

    def open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.w, "Open project", "", FILE_FILTER)
        if not path:
            return
        try:
            self.s.load_project(project_io.load(path))
        except Exception as e:
            return error_box(self.w, "Could not open project", e)
        self.path = path
        self._title()

    def save(self, save_as: bool = False):
        path = self.path
        if save_as or not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.w, "Save project", path or "filter.json", FILE_FILTER)
            if not path:
                return
        try:
            project_io.save(path, self.s.project())
        except Exception as e:
            return error_box(self.w, "Could not save project", e)
        self.path = path
        self.s.status.set(f"Saved {path}")
        self._title()

    def _title(self):
        name = self.path.rsplit("/", 1)[-1] if self.path else "untitled"
        self.w.setWindowTitle(f"filter_tuner — {name}")


def _action(parent, text, slot, shortcut=None, checkable=False, checked=False, tip=None):
    a = QtGui.QAction(text, parent)
    if shortcut:
        a.setShortcuts([QtGui.QKeySequence(s) for s in
                        (shortcut if isinstance(shortcut, (list, tuple)) else [shortcut])])
    if checkable:
        a.setCheckable(True)
        a.setChecked(checked)
        a.toggled.connect(slot)
    else:
        a.triggered.connect(lambda _=False: slot())
    if tip:
        a.setStatusTip(tip)
        a.setToolTip(tip)
    return a


def install(window: QtWidgets.QMainWindow, session: Session, editor: ZPlaneEditor,
            files: FileActions) -> QtWidgets.QToolBar:
    """Build menus + toolbar on ``window``; returns the toolbar."""
    mb = window.menuBar()
    tb = window.addToolBar("Main")
    tb.setObjectName("main_toolbar")

    # File
    m = mb.addMenu("&File")
    m.addAction(_action(window, "New (example)", files.new, "Ctrl+N"))
    m.addAction(_action(window, "New (empty)", lambda: files.new(blank=True)))
    m.addAction(_action(window, "Open…", files.open, "Ctrl+O"))
    m.addAction(_action(window, "Save", files.save, "Ctrl+S"))
    m.addAction(_action(window, "Save as…", lambda: files.save(True), "Ctrl+Shift+S"))
    m.addSeparator()
    m.addAction(_action(window, "Quit", window.close, "Ctrl+Q"))

    # Edit
    m = mb.addMenu("&Edit")
    undo = _action(window, "Undo", session.filter.undo, "Ctrl+Z")
    redo = _action(window, "Redo", session.filter.redo, ["Ctrl+Shift+Z", "Ctrl+Y"])

    def sync_history(_=None):
        undo.setEnabled(session.filter.can_undo)
        redo.setEnabled(session.filter.can_redo)

    session.filter.history_changed.subscribe(sync_history, call_now=True)
    m.addAction(undo)
    m.addAction(redo)
    m.addSeparator()
    m.addAction(_action(window, "Delete selected root", editor.delete, ["Delete", "Backspace"]))
    m.addAction(_action(window, "Next root", lambda: editor.cycle_selection(1), "S"))
    m.addAction(_action(window, "Previous root", lambda: editor.cycle_selection(-1), "A"))
    m.addAction(_action(window, "Deselect", lambda: editor.select(None), "Esc"))

    # Add mode (what clicking on empty z-plane space creates)
    group = QtGui.QActionGroup(window)
    add_zero = _action(window, "Click adds zeros", lambda on: on and session.add_kind.set(ZERO),
                       "Z", checkable=True, checked=session.add_kind.value == ZERO,
                       tip="Clicking empty z-plane space adds a zero pair (Shift: real zero)")
    add_pole = _action(window, "Click adds poles", lambda on: on and session.add_kind.set(POLE),
                       "P", checkable=True, checked=session.add_kind.value == POLE,
                       tip="Clicking empty z-plane space adds a pole pair (Shift: real pole)")
    for a in (add_zero, add_pole):
        group.addAction(a)
    session.add_kind.subscribe(
        lambda k: (add_zero.setChecked(k == ZERO), add_pole.setChecked(k == POLE)))

    # View
    m = mb.addMenu("&View")

    def view_toggle(text, field, shortcut=None, tip=None):
        def set_flag(on):
            session.view.set(replace(session.view.value, **{field: bool(on)}))
        return _action(window, text, set_flag, shortcut, checkable=True,
                       checked=getattr(session.view.value, field), tip=tip)

    toggles = [
        view_toggle("Log frequency", "log_freq", "L"),
        view_toggle("|H(z)| landscape", "show_heatmap", "H",
                    tip="Show 20·log10|H(z)| over the z-plane"),
        view_toggle("Snap to unit circle", "snap_unit_circle", "C"),
        view_toggle("Snap to real axis", "snap_real_axis", "R"),
    ]
    for a in toggles:
        m.addAction(a)

    tb.addAction(undo)
    tb.addAction(redo)
    tb.addSeparator()
    tb.addAction(add_zero)
    tb.addAction(add_pole)
    tb.addSeparator()
    for a in toggles:
        tb.addAction(a)
    mode = mb.addMenu("&Mode")
    mode.addAction(add_zero)
    mode.addAction(add_pole)
    return tb
