import numpy as np
import pytest
from scipy import signal

from filter_tuner import design, dsp, formatting, mask, optimize, project_io
from filter_tuner.model import POLE, ZERO, FilterState, GainSpec, Root, default_state, roots_from_zpk
from filter_tuner.observable import UndoableObservable


def test_root_normalizes_to_upper_half_and_real():
    assert Root(ZERO, 0.3 - 0.4j).value == 0.3 + 0.4j
    assert Root(POLE, 0.3 - 0.4j, paired=False).value == 0.3 + 0j


def test_roots_from_zpk_round_trip():
    z, p, k = signal.ellip(5, 1, 40, 0.3, output="zpk")
    roots = roots_from_zpk(z, p)
    st = FilterState(roots, GainSpec("manual", k))
    assert sorted(np.round(st.zeros, 9), key=lambda c: (c.real, c.imag)) == \
        sorted(np.round(z, 9), key=lambda c: (c.real, c.imag))
    assert len(st.poles) == 5


def test_roots_from_zpk_rejects_asymmetric():
    with pytest.raises(ValueError):
        roots_from_zpk([0.5 + 0.5j], [])


@pytest.mark.parametrize("family", list(design.IIR_FAMILIES))
def test_iir_seed_matches_scipy(family):
    spec = design.IIRSpec(family, "lowpass", 5, (0.1,), 1.0, 40.0)
    st = design.iir_seed(spec, fs=1.0)
    r = dsp.analyze(st)
    kw = {"rp": 1.0} if family in ("Chebyshev I", "Elliptic") else {}
    kw.update({"rs": 40.0} if family in ("Chebyshev II", "Elliptic") else {})
    b, a = signal.iirfilter(5, 0.1, btype="lowpass", ftype=design.IIR_FAMILIES[family],
                            fs=1.0, **kw)
    _, h = signal.freqz(b, a, worN=r.f, fs=1.0)
    assert np.allclose(np.abs(r.h), np.abs(h), atol=1e-6)
    assert r.stable


def test_high_order_near_circle_stays_accurate():
    # order 24 elliptic would be wrecked by expanding to (b, a)
    st = design.iir_seed(design.IIRSpec("Elliptic", "bandpass", 12, (0.2, 0.21), 0.5, 80))
    r = dsp.analyze(st)
    assert r.stable
    sos = signal.ellip(12, 0.5, 80, [0.4, 0.42], btype="bandpass", output="sos")
    _, h = signal.sosfreqz(sos, worN=r.f, fs=1.0)
    assert np.allclose(np.abs(r.h), np.abs(h), atol=1e-6)


def test_gain_modes():
    base = default_state()
    for mode, f in (("dc", 0.0), ("nyquist", 0.5), ("freq", 0.1)):
        st = base.with_gain(GainSpec(mode, freq=0.1))
        zpk = dsp.state_zpk(st)
        h = dsp.eval_zpk(zpk.zeros, zpk.poles, zpk.k, np.exp(2j * np.pi * f))
        assert abs(abs(h) - 1) < 1e-9
    r = dsp.analyze(base.with_gain(GainSpec("peak")))
    assert abs(np.max(r.mag_db)) < 0.05


def test_group_delay_matches_scipy():
    st = design.iir_seed(design.IIRSpec("Butterworth", "lowpass", 4, (0.15,)))
    r = dsp.analyze(st)
    b, a = signal.sos2tf(r.sos)
    # stay away from the 4th-order zero at Nyquist, where scipy's
    # polynomial-based estimate loses precision
    sel = (r.f > 0) & (r.f < 0.45)
    _, gd = signal.group_delay((b, a), w=r.f[sel], fs=1.0)
    assert np.allclose(r.group_delay[sel], gd, atol=1e-6)


def test_more_zeros_than_poles_is_made_causal():
    st = FilterState((Root(ZERO, 0.5 + 0.5j), Root(ZERO, -0.9, paired=False)))
    r = dsp.analyze(st)
    assert r.zpk.padded_poles == 3
    # an FIR's impulse response is its taps
    taps = np.real(np.poly(st.zeros))
    assert np.allclose(r.impulse[:4], taps)


def test_fir_seed_reproduces_taps():
    spec = design.FIRSpec("firwin", "lowpass", 31, (0.12,))
    taps = design.fir_taps(spec)
    r = dsp.analyze(design.fir_seed(spec))
    assert np.allclose(r.impulse[:31], taps, atol=1e-9)
    assert np.allclose(r.impulse[31:], 0, atol=1e-12)


def test_remez_seed_runs():
    st = design.fir_seed(design.FIRSpec("remez", "bandpass", 41, (0.15, 0.3), transition=0.05))
    r = dsp.analyze(st)
    i = np.argmin(abs(r.f - 0.22))
    assert r.mag_db[i] > -1


def test_mask_and_optimizer_improves():
    m = mask.lowpass_mask(1.0, 0.1, 0.15, 1.0, 50.0)
    start = FilterState((
        Root.polar(ZERO, 1, 2.0), Root.polar(ZERO, 1, 2.6),
        Root.polar(POLE, 0.8, 0.3), Root.polar(POLE, 0.9, 0.55)), GainSpec("dc"))
    res = optimize.fit_to_mask(start, m)
    assert res.worst_after_db < res.worst_before_db
    assert np.all(np.abs(res.state.poles) <= 0.995 + 1e-12)


def test_optimizer_without_gain_keeps_gain_mode():
    m = mask.lowpass_mask(1.0, 0.1, 0.2, 2.0, 30.0)
    st = design.iir_seed(design.IIRSpec("Butterworth", "lowpass", 3, (0.12,)))
    st = st.with_gain(GainSpec("dc"))
    res = optimize.fit_to_mask(st, m, optimize.FitOptions(fit_gain=False, max_iter=50))
    assert res.state.gain.mode == "dc"


def test_formatting_runs_without_sympy():
    text = formatting.summary(dsp.state_zpk(default_state()))
    assert "H(z)" in text and "z⁻²" in text


def test_project_round_trip(tmp_path):
    p = project_io.Project(
        design.iir_seed(design.IIRSpec("Chebyshev I", "highpass", 3, (0.2,))),
        mask.highpass_mask(1.0, 0.1, 0.2),
    )
    path = tmp_path / "f.json"
    project_io.save(path, p)
    q = project_io.load(path)
    assert q.state == p.state and q.mask == p.mask and q.quant == p.quant


def test_undo_gesture_and_coalesce():
    o = UndoableObservable(0)
    o.begin_gesture()
    for v in range(1, 20):
        o.set(v)
    o.end_gesture()
    o.set(100, coalesce_key="k")
    o.set(101, coalesce_key="k")
    o.undo()
    assert o.value == 19
    o.undo()
    assert o.value == 0
    o.redo()
    assert o.value == 19
