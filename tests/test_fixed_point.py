import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from filter_tuner import design, dsp, quantize
from filter_tuner.export import cmsis, verilog

ELLIP = design.IIRSpec("Elliptic", "lowpass", 6, (0.1,), 1.0, 50.0)


def sos():
    return dsp.analyze(design.iir_seed(ELLIP)).sos


def test_quantization_converges_to_ideal():
    s = sos()
    errs = [quantize.quantize_sos(s, quantize.QFormat(w, w - 2)).max_coef_error
            for w in (10, 14, 18, 24)]
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_coarse_quantization_can_destabilize():
    # poles very close to z=1: 8-bit coefficients cannot hold them inside
    st = design.iir_seed(design.IIRSpec("Butterworth", "lowpass", 4, (0.002,)))
    q = quantize.quantize_sos(dsp.analyze(st).sos, quantize.QFormat(8, 6))
    assert not q.stable or q.max_pole_radius > 0.999


def test_scaling_normalizes_section_gain():
    sc = quantize.scale_sections(sos(), "linf")
    assert np.allclose(sc.section_gains, 1.0, atol=1e-6)
    _, h0 = signal.sosfreqz(sos(), worN=512)
    _, h1 = signal.sosfreqz(sc.sos, worN=512)
    assert np.allclose(h0, h1 * sc.output_gain, atol=1e-9)


def test_fixed_model_tracks_float():
    s = quantize.scale_sections(sos(), "linf").sos
    q = quantize.quantize_sos(s, quantize.QFormat(24, 21))
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.5, 0.5, 2000)
    yi = quantize.simulate_fixed(q.ints, 21, quantize.float_to_data(x, 24), 24)
    y = signal.sosfilt(q.sos, x)
    assert np.max(np.abs(quantize.data_to_float(yi, 24) - y)) < 1e-4


def test_cmsis_sign_convention():
    s = sos()
    text = cmsis.f32_df2t(s, "lp")
    body = text.split("lp_coeffs[5 * 3] = {", 1)[1].split("};", 1)[0]
    body = body.split("*/", 1)[1]  # drop the column comment
    vals = [float(v.strip().rstrip("f")) for v in body.split(",")]
    assert len(vals) == 15
    assert np.isclose(vals[3], -s[0, 4]) and np.isclose(vals[4], -s[0, 5])
    assert "arm_biquad_cascade_df1_init_q31" in cmsis.q31_df1(s)
    assert cmsis.post_shift(np.array([[1.5, 0, 0, 1, -1.9, 0.9]])) == 1


def test_verilog_bundle_contents(tmp_path):
    q = quantize.quantize_sos(sos(), quantize.QFormat(18, 15))
    b = verilog.bundle(q.ints, q.fmt, 16)
    paths = b.write(tmp_path)
    names = {p.name for p in paths}
    assert {"biquad_df1.v", "filter_cascade.v", "tb_filter_cascade.v",
            "coeffs.mem", "stimulus.mem", "expected.mem"} <= names
    assert len((tmp_path / "coeffs.mem").read_text().splitlines()) == 1 + len(q.ints)


HARNESS = r"""
#include "cascade.cc"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
static std::vector<uint32_t> rd(const char *f) {
  std::ifstream i(f); std::string l; std::vector<uint32_t> v;
  while (std::getline(i, l)) if (!l.empty() && l[0] != '/') v.push_back(std::stoul(l, nullptr, 16));
  return v;
}
int main() {
  cxxrtl_design::p_filter__cascade top;
  auto st = rd("stimulus.mem"), ex = rd("expected.mem");
  auto tick = [&] { top.p_clk.set<bool>(false); top.step(); top.p_clk.set<bool>(true); top.step(); };
  top.p_rst.set<bool>(true); tick(); tick(); top.p_rst.set<bool>(false);
  size_t k = 0, err = 0;
  for (size_t i = 0; i < st.size() + 32; i++) {
    top.p_in__valid.set<bool>(i < st.size());
    top.p_x.set<uint32_t>(i < st.size() ? st[i] : 0);
    tick();
    if (top.p_out__valid.get<bool>() && k < ex.size()) { err += top.p_y.get<uint32_t>() != ex[k]; k++; }
  }
  printf("%zu %zu\n", k, err);
  return (err || k != ex.size()) ? 1 : 0;
}
"""


def _cxxrtl_include():
    try:
        import yowasp_yosys
    except ImportError:
        return None
    inc = Path(yowasp_yosys.__file__).parent / "share/include/backends/cxxrtl/runtime"
    return inc if inc.exists() else None


@pytest.mark.skipif(not (shutil.which("yowasp-yosys") and shutil.which("g++") and _cxxrtl_include()),
                    reason="needs yowasp-yosys and g++")
@pytest.mark.parametrize("scaling", ["none", "linf"])
def test_verilog_is_bit_exact_with_python_model(tmp_path, scaling):
    s = sos()
    gain = 1.0
    if scaling != "none":
        sc = quantize.scale_sections(s, scaling)
        s, gain = sc.sos, sc.output_gain
    q = quantize.quantize_sos(s, quantize.QFormat(18, 15))
    verilog.bundle(q.ints, q.fmt, 16, output_gain=gain).write(tmp_path)
    expected = (tmp_path / "expected.mem").read_text().split()
    assert len(set(expected)) > 50  # the vectors exercise the filter
    subprocess.run(["yowasp-yosys", "-q", "-p",
                    "read_verilog biquad_df1.v filter_cascade.v; hierarchy -top filter_cascade; "
                    "proc; flatten; opt; write_cxxrtl cascade.cc"],
                   cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "harness.cc").write_text(HARNESS)
    subprocess.run(["g++", "-O1", "-std=c++14", f"-I{_cxxrtl_include()}", "harness.cc", "-o", "sim"],
                   cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run(["./sim"], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout
