"""Live transfer-function text: factored H(z), polynomials, SOS, LaTeX."""

from __future__ import annotations

import numpy as np
from pyqtgraph.Qt import QtWidgets

from ...formatting import latex, summary
from ...session import Session
from ..widgets import mono_text


class MathPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        lay = QtWidgets.QVBoxLayout(self)
        self.text = mono_text()
        lay.addWidget(self.text)
        copy = QtWidgets.QPushButton("Copy LaTeX")
        copy.clicked.connect(self._copy_latex)
        lay.addWidget(copy)
        session.analysis.subscribe(lambda _: self._refresh(), call_now=True)

    def _refresh(self):
        r = self.s.analysis.value
        sos = np.array2string(r.sos, precision=6, suppress_small=True, max_line_width=120)
        bar = self.text.verticalScrollBar().value()
        self.text.setPlainText(
            summary(r.zpk) + "\n\nSecond-order sections [b0 b1 b2 a0 a1 a2]:\n" + sos
        )
        self.text.verticalScrollBar().setValue(bar)

    def _copy_latex(self):
        QtWidgets.QApplication.clipboard().setText(latex(self.s.analysis.value.zpk))
        self.s.status.set("LaTeX copied")
