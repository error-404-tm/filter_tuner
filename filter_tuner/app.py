"""Entry point: ``python -m filter_tuner [project.json]``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from . import project_io
    from .session import Session
    from .ui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    pg.setConfigOptions(antialias=True)
    session = Session()
    window = MainWindow(session)
    if len(argv) > 1:
        session.load_project(project_io.load(argv[1]))
        window.files.path = argv[1]
        window.files._title()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
