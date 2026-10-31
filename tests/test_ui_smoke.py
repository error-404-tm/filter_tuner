"""Headless UI checks: editing gestures go through the Session correctly."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pyqtgraph.Qt").QtWidgets

from filter_tuner.model import POLE, ZERO  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    from filter_tuner.ui.main_window import MainWindow
    w = MainWindow()
    w.show()
    app.processEvents()
    yield w
    w.close()


def test_add_drag_delete_undo(window):
    s, ed = window.session, window.editor
    n0 = len(s.filter.value.roots)
    s.add_kind.set(ZERO)
    ed.add_at(0.2 + 0.6j, paired=True)
    assert len(s.filter.value.roots) == n0 + 1
    i = s.selection.value
    # drag the lower conjugate downward: the stored (upper) root moves up
    ed.begin_drag(i, conjugate=True, at=0.2 - 0.6j)
    for k in range(1, 11):
        ed.drag_to(0.2 - (0.6 + 0.01 * k) * 1j)
    ed.end_drag()
    assert np.isclose(s.filter.value.roots[i].value, 0.2 + 0.7j)
    s.filter.undo()  # whole drag is one step
    assert np.isclose(s.filter.value.roots[i].value, 0.2 + 0.6j)
    ed.delete()
    assert len(s.filter.value.roots) == n0


def test_fine_drag_and_snap(window):
    from dataclasses import replace
    s, ed = window.session, window.editor
    s.view.set(replace(s.view.value, snap_unit_circle=True))
    s.add_kind.set(POLE)
    ed.add_at(0.0 + 0.5j, paired=True)
    i = s.selection.value
    ed.begin_drag(i, False, 0.5j)
    ed.drag_to(0.5j + 1.0j, fine=True)  # fine: moves 0.1
    ed.end_drag()
    assert np.isclose(s.filter.value.roots[i].value, 0.6j)
    ed.begin_drag(i, False, 0.6j)
    ed.drag_to(0.6j + 0.39j)
    ed.end_drag()
    assert np.isclose(abs(s.filter.value.roots[i].value), 1.0)


def test_real_root_stays_real(window):
    s, ed = window.session, window.editor
    ed.add_at(0.3 + 0.4j, paired=False)
    i = s.selection.value
    assert s.filter.value.roots[i].value == 0.3
    ed.begin_drag(i, False, 0.3)
    ed.drag_to(0.5 + 0.5j)
    ed.end_drag()
    assert s.filter.value.roots[i].value == 0.5


def test_all_panels_survive_design_changes(window, app):
    from dataclasses import replace

    from filter_tuner import design, mask
    s = window.session
    for spec in (design.IIRSpec("Elliptic", "bandstop", 5, (0.1, 0.2)),):
        s.replace_filter(design.iir_seed(spec))
    s.replace_filter(design.fir_seed(design.FIRSpec("remez", "highpass", 40, (0.3,))))
    s.mask.set(mask.highpass_mask(1.0, 0.2, 0.3))
    s.view.set(replace(s.view.value, show_heatmap=True, log_freq=True))
    for dock in window.docks.values():
        dock.raise_()
        app.processEvents()
    s.filter.reset(s.filter.value.with_fs(48000.0))
    app.processEvents()
    assert s.analysis.value.f[-1] == pytest.approx(24000.0)


def test_signal_panel_filters_a_file(window, app, tmp_path):
    from filter_tuner import signal_io
    from filter_tuner.ui.panels import SignalPanel

    t = np.arange(4000) / 8000.0
    x = np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 3000 * t)
    path = tmp_path / "trace.csv"
    np.savetxt(path, np.column_stack([t, x]), delimiter=",")
    panel = window.findChild(SignalPanel)
    window.docks["Signal"].raise_()
    app.processEvents()
    panel.loaded = signal_io.load_signal(path)
    assert len(panel.loaded.x) == 4000
    panel.use_quant.setChecked(True)
    panel._run()
    assert panel.y is not None and panel.y.shape == x.shape
