"""Main window: lays out the z-plane, response plots and dockable panels.

The window owns no filter logic. It creates the Session-bound widgets,
arranges them, and installs actions.
"""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from ..session import Session
from . import actions, panels
from .response_plots import ResponsePlots
from .zplane import ZPlaneWidget
from .zplane_editor import ZPlaneEditor

RIGHT = QtCore.Qt.DockWidgetArea.RightDockWidgetArea

# (dock title, panel class), first group and second group are tabbed separately
PANEL_GROUPS = (
    (("Roots", panels.RootEditor), ("Filter", panels.FilterSettingsPanel),
     ("Design", panels.DesignPanel), ("Mask", panels.MaskPanel)),
    (("Fixed-point", panels.QuantPanel), ("Export", panels.ExportPanel),
     ("Math", panels.MathPanel), ("Signal", panels.SignalPanel)),
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, session: Session | None = None):
        super().__init__()
        self.session = session or Session()
        self.resize(1600, 950)
        self.editor = ZPlaneEditor(self.session)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.zplane = ZPlaneWidget(self.session, self.editor)
        self.plots = ResponsePlots(self.session)
        split.addWidget(self.zplane)
        split.addWidget(self.plots)
        split.setSizes([600, 900])
        self.setCentralWidget(split)

        self.docks: dict[str, QtWidgets.QDockWidget] = {}
        groups = [[self._dock(title, cls(self.session)) for title, cls in group]
                  for group in PANEL_GROUPS]
        # tabify only after every dock exists (avoids stray tab bars in Qt)
        for first, *rest in groups:
            for dock in rest:
                self.tabifyDockWidget(first, dock)
        for first, *_ in groups:
            first.raise_()

        self.files = actions.FileActions(self, self.session)
        actions.install(self, self.session, self.editor, self.files)
        self.files._title()

        self.session.status.subscribe(lambda msg: self.statusBar().showMessage(msg, 8000))
        self.statusBar().showMessage(
            "Click: add pair · Shift-click: add real root · drag: move (Shift = fine) · "
            "right-click: delete · Z/P: zeros/poles · Ctrl+Z: undo"
        )

    def _dock(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(f"dock_{title}")
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        dock.setWidget(scroll)
        self.addDockWidget(RIGHT, dock)
        self.docks[title] = dock
        return dock
