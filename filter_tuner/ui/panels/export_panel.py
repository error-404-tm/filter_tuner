"""Generate CMSIS-DSP C and Verilog from the current fixed-point realization."""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets

from ...export import cmsis, verilog
from ...session import Session
from ..widgets import combo, error_box, mono_text


class ExportPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        lay = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit("filt")
        self.target = combo(list(cmsis.FORMATS) + ["Verilog cascade"])
        form.addRow("Name", self.name)
        form.addRow("Target", self.target)
        lay.addLayout(form)

        note = QtWidgets.QLabel(
            "CMSIS exports use the float sections (scaled if scaling is on). "
            "Verilog uses the integer coefficients from the Fixed-point tab and "
            "includes a self-checking testbench with bit-exact expected outputs."
        )
        note.setWordWrap(True)
        lay.addWidget(note)

        row = QtWidgets.QHBoxLayout()
        preview = QtWidgets.QPushButton("Preview")
        copy = QtWidgets.QPushButton("Copy")
        save = QtWidgets.QPushButton("Save…")
        preview.clicked.connect(self._preview)
        copy.clicked.connect(self._copy)
        save.clicked.connect(self._save)
        for b in (preview, copy, save):
            row.addWidget(b)
        lay.addLayout(row)
        self.text = mono_text()
        lay.addWidget(self.text, stretch=1)

    def _ident(self) -> str:
        n = "".join(c if c.isalnum() or c == "_" else "_" for c in self.name.text().strip())
        return n if n and not n[0].isdigit() else "filt"

    def _is_verilog(self) -> bool:
        return self.target.currentText().startswith("Verilog")

    def _verilog_bundle(self):
        fx = self.s.fixed.value
        return verilog.bundle(fx.quantized.ints, fx.quantized.fmt, fx.settings.data_bits,
                              name=f"{self._ident()}_cascade", output_gain=fx.output_gain)

    def _render(self) -> str:
        if self._is_verilog():
            b = self._verilog_bundle()
            return "\n".join(t for n, t in b.files.items() if n.endswith(".v") and not n.startswith("tb_"))
        fx = self.s.fixed.value
        text = cmsis.FORMATS[self.target.currentText()](fx.sos, self._ident())
        if abs(fx.output_gain - 1.0) > 1e-12:
            text = (f"/* Sections are scaled: multiply the output by {fx.output_gain:.9g} */\n"
                    + text)
        return text

    def _preview(self):
        try:
            self.text.setPlainText(self._render())
        except Exception as e:
            error_box(self, "Export failed", e)

    def _copy(self):
        self._preview()
        QtWidgets.QApplication.clipboard().setText(self.text.toPlainText())
        self.s.status.set("Copied to clipboard")

    def _save(self):
        try:
            if self._is_verilog():
                d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for Verilog files")
                if not d:
                    return
                paths = self._verilog_bundle().write(d)
                self.s.status.set(f"Wrote {len(paths)} files to {d}")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save C source", f"{self._ident()}_coeffs.c", "C source (*.c *.h)")
            if path:
                with open(path, "w") as f:
                    f.write(self._render())
                self.s.status.set(f"Wrote {path}")
        except Exception as e:
            error_box(self, "Export failed", e)
