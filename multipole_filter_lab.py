import sys
import numpy as np
from scipy import signal
import sympy as sp

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


def compute_filter(zeros, poles, gain=1.0, fs=1.0):
    b, a = signal.zpk2tf(zeros, poles, gain)
    w, h = signal.freqz(b, a, worN=2048, fs=fs)

    impulse = np.zeros(128)
    impulse[0] = 1.0
    impulse_response = signal.lfilter(b, a, impulse)

    stable = np.all(np.abs(poles) < 1)

    return b, a, w, h, impulse_response, stable


def symbolic_transfer_function_text(zeros, poles, gain=1.0):
    z = sp.symbols("z")

    numerator = sp.Float(gain)
    denominator = sp.Float(1.0)

    for zero in zeros:
        numerator *= z - sp.N(complex(zero), 5)

    for pole in poles:
        denominator *= z - sp.N(complex(pole), 5)

    H = sp.simplify(numerator / denominator)

    expanded_num = sp.expand(numerator)
    expanded_den = sp.expand(denominator)

    return (
        "H(z):\n"
        f"{H}\n\n"
        "Expanded numerator:\n"
        f"{expanded_num}\n\n"
        "Expanded denominator:\n"
        f"{expanded_den}\n\n"
        "LaTeX:\n"
        f"{sp.latex(H)}"
    )


class FilterLabWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Phase 1.5 Filter Lab")
        self.resize(1400, 900)

        self.gain = 1.0

        self.math_dirty = True

        # Initial filter parameters
        self.max_pairs = 6

        self.active_zero_pairs = 1
        self.active_pole_pairs = 1

        self.selected_zero_pair = 0
        self.selected_pole_pair = 0

        self.zero_pairs = [
            {"radius": 1.0, "angle": np.pi / 3}
            for _ in range(self.max_pairs)
        ]

        self.pole_pairs = [
            {"radius": 0.85, "angle": np.pi / 3}
            for _ in range(self.max_pairs)
        ]

        self._build_ui()
        self.update_filter()
        self.refresh_math_pane()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)

        # ------------------------
        # Left side: plots
        # ------------------------
        self.graphics = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graphics, stretch=4)

        pg.setConfigOptions(antialias=True)

        self.zplot = self.graphics.addPlot(row=0, col=0, title="Z-plane")
        self.zplot.setAspectLocked(True)
        self.zplot.showGrid(x=True, y=True)
        self.zplot.setLabel("bottom", "Real")
        self.zplot.setLabel("left", "Imag")
        self.zplot.setXRange(-1.5, 1.5)
        self.zplot.setYRange(-1.5, 1.5)

        theta = np.linspace(0, 2 * np.pi, 512)
        unit_x = np.cos(theta)
        unit_y = np.sin(theta)

        try:
            dash_style = QtCore.Qt.PenStyle.DashLine
        except AttributeError:
            dash_style = QtCore.Qt.DashLine

        self.zplot.plot(unit_x, unit_y, pen=pg.mkPen(style=dash_style))

        self.zero_points = self.zplot.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolSize=14,
            symbolBrush=None,
            symbolPen=pg.mkPen(width=2),
        )

        self.pole_points = self.zplot.plot(
            [],
            [],
            pen=None,
            symbol="x",
            symbolSize=18,
            symbolPen=pg.mkPen(width=3),
        )

        self.selected_zero_points = self.zplot.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolSize=22,
            symbolBrush=None,
            symbolPen=pg.mkPen(width=4),
        )

        self.selected_pole_points = self.zplot.plot(
            [],
            [],
            pen=None,
            symbol="x",
            symbolSize=26,
            symbolPen=pg.mkPen(width=5),
        )

        self.mag_plot = self.graphics.addPlot(row=0, col=1, title="Magnitude Response")
        self.mag_curve = self.mag_plot.plot([])
        self.mag_plot.showGrid(x=True, y=True)
        self.mag_plot.setLabel("bottom", "Normalized Frequency")
        self.mag_plot.setLabel("left", "Magnitude", units="dB")

        self.phase_plot = self.graphics.addPlot(row=1, col=0, title="Phase Response")
        self.phase_curve = self.phase_plot.plot([])
        self.phase_plot.showGrid(x=True, y=True)
        self.phase_plot.setLabel("bottom", "Normalized Frequency")
        self.phase_plot.setLabel("left", "Phase", units="rad")

        self.impulse_plot = self.graphics.addPlot(row=1, col=1, title="Impulse Response")
        self.impulse_curve = self.impulse_plot.plot([])
        self.impulse_plot.showGrid(x=True, y=True)
        self.impulse_plot.setLabel("bottom", "n")
        self.impulse_plot.setLabel("left", "h[n]")

        self.selected_zero_pair = 0
        self.selected_pole_pair = 0
        self.selected_kind = "zero"

        # ------------------------
        # Right side: controls and math
        # ------------------------
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, stretch=1)

        self.status_label = QtWidgets.QLabel("Stable: unknown")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        control_layout.addWidget(self.status_label)

        self.mode_label = QtWidgets.QLabel("")
        self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        control_layout.addWidget(self.mode_label)

        self.coeff_label = QtWidgets.QLabel("")
        self.coeff_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        control_layout.addWidget(self.coeff_label)

        zero_radius_box, self.zero_radius_slider, self.zero_radius_label = self._make_slider_group(
            title="Zero radius",
            minimum=0,
            maximum=150,
            initial=int(self.zero_pairs[self.selected_zero_pair]["radius"] * 100),
            callback=self.on_zero_radius_changed,
            scale=100,
        )
        control_layout.addWidget(zero_radius_box)

        zero_angle_box, self.zero_angle_slider, self.zero_angle_label = self._make_slider_group(
            title="Zero angle / π",
            minimum=0,
            maximum=100,
            initial=int((self.zero_pairs[self.selected_zero_pair]["angle"] / np.pi) * 100),
            callback=self.on_zero_angle_changed,
            scale=100,
        )
        control_layout.addWidget(zero_angle_box)

        pole_radius_box, self.pole_radius_slider, self.pole_radius_label = self._make_slider_group(
            title="Pole radius",
            minimum=0,
            maximum=120,
            initial=int(self.pole_pairs[self.selected_pole_pair]["radius"] * 100),
            callback=self.on_pole_radius_changed,
            scale=100,
        )
        control_layout.addWidget(pole_radius_box)

        pole_angle_box, self.pole_angle_slider, self.pole_angle_label = self._make_slider_group(
            title="Pole angle / π",
            minimum=0,
            maximum=100,
            initial=int((self.pole_pairs[self.selected_pole_pair]["angle"] / np.pi) * 100),
            callback=self.on_pole_angle_changed,
            scale=100,
        )
        control_layout.addWidget(pole_angle_box)

        self.refresh_math_button = QtWidgets.QPushButton("Refresh math pane (M)")
        self.refresh_math_button.clicked.connect(self.refresh_math_pane)
        control_layout.addWidget(self.refresh_math_button)

        self.math_box = QtWidgets.QTextEdit()
        self.math_box.setReadOnly(True)
        self.math_box.setMinimumHeight(250)
        control_layout.addWidget(QtWidgets.QLabel("Symbolic transfer function"))
        control_layout.addWidget(self.math_box, stretch=1)

    def _make_slider_group(self, title, minimum, maximum, initial, callback, scale):
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)

        label = QtWidgets.QLabel("")
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(initial)

        def on_change(value):
            actual = value / scale
            label.setText(f"{actual:.3f}")
            callback(actual)

        slider.valueChanged.connect(on_change)
        label.setText(f"{initial / scale:.3f}")

        layout.addWidget(label)
        layout.addWidget(slider)

        return box, slider, label

    def current_zeros_poles(self):
        zeros = []
        poles = []

        for pair in self.zero_pairs[:self.active_zero_pairs]:
            r = pair["radius"]
            angle = pair["angle"]

            z = r * np.exp(1j * angle)
            zeros.append(z)
            zeros.append(np.conj(z))

        for pair in self.pole_pairs[:self.active_pole_pairs]:
            r = pair["radius"]
            angle = pair["angle"]

            p = r * np.exp(1j * angle)
            poles.append(p)
            poles.append(np.conj(p))

        return np.array(zeros), np.array(poles)

    def update_filter(self):
        zeros, poles = self.current_zeros_poles()

        b, a, w, h, impulse_response, stable = compute_filter(
            zeros=zeros,
            poles=poles,
            gain=self.gain,
        )

        # Update z-plane points
        self.zero_points.setData(np.real(zeros), np.imag(zeros))
        self.pole_points.setData(np.real(poles), np.imag(poles))

        selected_zeros, selected_poles = self.selected_pair_points()

        if self.selected_kind == "zero":
            self.selected_zero_points.setData(np.real(selected_zeros), np.imag(selected_zeros))
            self.selected_pole_points.setData([], [])
        else:
            self.selected_zero_points.setData([], [])
            self.selected_pole_points.setData(np.real(selected_poles), np.imag(selected_poles))

        # Update magnitude response
        magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
        self.mag_curve.setData(w, magnitude_db)

        # Update phase response
        phase = np.unwrap(np.angle(h))
        self.phase_curve.setData(w, phase)

        # Update impulse response
        n = np.arange(len(impulse_response))
        self.impulse_curve.setData(n, impulse_response)

        # Update status
        if stable:
            self.status_label.setText("Stable: yes")
            self.status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: green;"
            )
        else:
            self.status_label.setText("Stable: NO — pole outside unit circle")
            self.status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: red;"
            )

        b_text = np.array2string(b, precision=4, suppress_small=True)
        a_text = np.array2string(a, precision=4, suppress_small=True)

        self.coeff_label.setText(
            f"b coefficients:\n{b_text}\n\n"
            f"a coefficients:\n{a_text}"
        )

        pair_status = (
            f"EDITING: {self.selected_kind.upper()}\n"
            f"Z: edit zeros | P: edit poles\n"
            f"S/A: cycle selected pair | +/-: add/remove active type | M: refresh math\n\n"
            f"Selected zero pair: {self.selected_zero_pair + 1}/{self.active_zero_pairs}\n"
            f"Selected pole pair: {self.selected_pole_pair + 1}/{self.active_pole_pairs}\n\n"
        )

        self.math_dirty = True

        if not self.math_box.toPlainText().endswith("Press M or click Refresh math pane."):
            self.math_box.setPlainText(
                self.current_pair_status_text()
                + "\nMath pane is stale. Press M or click Refresh math pane."
            )

        if self.selected_kind == "zero":
            self.mode_label.setText(
                f"Editing ZEROS | pair {self.selected_zero_pair + 1}/{self.active_zero_pairs} | "
                "P: poles | S: next zero pair | A: previous | +/-: add/remove zero pair"
            )
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        else:
            self.mode_label.setText(
                f"Editing POLES | pair {self.selected_pole_pair + 1}/{self.active_pole_pairs} | "
                "Z: zeros | S: next pole pair | A: previous | +/-: add/remove pole pair"
            )
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")

    def on_zero_radius_changed(self, value):
        self.zero_pairs[self.selected_zero_pair]["radius"] = value
        self.update_filter()

    def on_zero_angle_changed(self, value):
        self.zero_pairs[self.selected_zero_pair]["angle"] = value * np.pi
        self.update_filter()

    def on_pole_radius_changed(self, value):
        self.pole_pairs[self.selected_pole_pair]["radius"] = value
        self.update_filter()

    def on_pole_angle_changed(self, value):
        self.pole_pairs[self.selected_pole_pair]["angle"] = value * np.pi
        self.update_filter()

    def keyPressEvent(self, event):
        key = event.key()

        if key == self.qt_key("Key_Z"):
            self.selected_kind = "zero"
            self.load_selected_pair_into_sliders()
            self.update_filter()
            return

        if key == self.qt_key("Key_P"):
            self.selected_kind = "pole"
            self.load_selected_pair_into_sliders()
            self.update_filter()
            return


        if key in (self.qt_key("Key_Plus"), self.qt_key("Key_Equal")):
            self.add_pair()
            return

        if key in (self.qt_key("Key_Minus"), self.qt_key("Key_Underscore")):
            self.remove_pair()
            return
        
        if key == self.qt_key("Key_M"):
            self.refresh_math_pane()
            return
        
        if key == self.qt_key("Key_S"):
            self.select_next_pair()
            return

        if key == self.qt_key("Key_A"):
            self.select_previous_pair()
            return
        
        super().keyPressEvent(event)

    def qt_key(self, name):
        if hasattr(QtCore.Qt, "Key"):
            return getattr(QtCore.Qt.Key, name)
        return getattr(QtCore.Qt, name)

    
    def load_selected_pair_into_sliders(self):
        zr = self.zero_pairs[self.selected_zero_pair]["radius"]
        za = self.zero_pairs[self.selected_zero_pair]["angle"] / np.pi

        pr = self.pole_pairs[self.selected_pole_pair]["radius"]
        pa = self.pole_pairs[self.selected_pole_pair]["angle"] / np.pi

        items = [
            (self.zero_radius_slider, self.zero_radius_label, int(zr * 100), zr),
            (self.zero_angle_slider, self.zero_angle_label, int(za * 100), za),
            (self.pole_radius_slider, self.pole_radius_label, int(pr * 100), pr),
            (self.pole_angle_slider, self.pole_angle_label, int(pa * 100), pa),
        ]

        for slider, label, slider_value, actual_value in items:
            slider.blockSignals(True)
            slider.setValue(slider_value)
            slider.blockSignals(False)
            label.setText(f"{actual_value:.3f}")

    def selected_pair_points(self):
        if self.active_zero_pairs > 0:
            zp = self.zero_pairs[self.selected_zero_pair]
            z = zp["radius"] * np.exp(1j * zp["angle"])
            selected_zeros = np.array([z, np.conj(z)])
        else:
            selected_zeros = np.array([])

        if self.active_pole_pairs > 0:
            pp = self.pole_pairs[self.selected_pole_pair]
            p = pp["radius"] * np.exp(1j * pp["angle"])
            selected_poles = np.array([p, np.conj(p)])
        else:
            selected_poles = np.array([])

        return selected_zeros, selected_poles
    
    def clamp_selected_indices(self):
        self.selected_zero_pair = min(self.selected_zero_pair, self.active_zero_pairs - 1)
        self.selected_pole_pair = min(self.selected_pole_pair, self.active_pole_pairs - 1)

        self.selected_zero_pair = max(self.selected_zero_pair, 0)
        self.selected_pole_pair = max(self.selected_pole_pair, 0)


    def select_next_pair(self):
        if self.selected_kind == "zero":
            self.selected_zero_pair = (self.selected_zero_pair + 1) % self.active_zero_pairs
        else:
            self.selected_pole_pair = (self.selected_pole_pair + 1) % self.active_pole_pairs

        self.load_selected_pair_into_sliders()
        self.update_filter()


    def select_previous_pair(self):
        if self.selected_kind == "zero":
            self.selected_zero_pair = (self.selected_zero_pair - 1) % self.active_zero_pairs
        else:
            self.selected_pole_pair = (self.selected_pole_pair - 1) % self.active_pole_pairs

        self.load_selected_pair_into_sliders()
        self.update_filter()


    def add_pair(self):
        if self.selected_kind == "zero":
            if self.active_zero_pairs < self.max_pairs:
                self.active_zero_pairs += 1
                self.selected_zero_pair = self.active_zero_pairs - 1
        else:
            if self.active_pole_pairs < self.max_pairs:
                self.active_pole_pairs += 1
                self.selected_pole_pair = self.active_pole_pairs - 1

        self.load_selected_pair_into_sliders()
        self.update_filter()


    def remove_pair(self):
        if self.selected_kind == "zero":
            if self.active_zero_pairs > 1:
                self.active_zero_pairs -= 1
                self.clamp_selected_indices()
        else:
            if self.active_pole_pairs > 1:
                self.active_pole_pairs -= 1
                self.clamp_selected_indices()

        self.load_selected_pair_into_sliders()
        self.update_filter()

    def refresh_math_pane(self):
        zeros, poles = self.current_zeros_poles()

        self.math_box.setPlainText(
            self.current_pair_status_text()
            + symbolic_transfer_function_text(zeros, poles, self.gain)
        )

        self.math_dirty = False

    def current_pair_status_text(self):
        return (
            f"EDITING: {self.selected_kind.upper()}\n"
            f"Z: edit zeros | P: edit poles\n"
            f"S/A: cycle selected pair | +/-: add/remove active type | M: refresh math\n\n"
            f"Selected zero pair: {self.selected_zero_pair + 1}/{self.active_zero_pairs}\n"
            f"Selected pole pair: {self.selected_pole_pair + 1}/{self.active_pole_pairs}\n\n"
        )
    

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FilterLabWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()