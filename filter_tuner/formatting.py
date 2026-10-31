"""Human-readable and LaTeX renderings of the transfer function (no sympy)."""

from __future__ import annotations

import numpy as np

from .dsp import ZPK


def fmt_complex(c: complex, digits: int = 4) -> str:
    c = complex(c)
    re, im = round(c.real, digits), round(c.imag, digits)
    if im == 0:
        return f"{re:g}"
    if re == 0:
        return f"{im:g}j"
    sign = "+" if im > 0 else "-"
    return f"({re:g} {sign} {abs(im):g}j)"


def _factor(c: complex, var: str = "z") -> str:
    c = complex(c)
    if abs(c) < 1e-12:
        return var
    s = fmt_complex(c)
    if c.imag == 0 and c.real < 0:
        return f"({var} + {abs(c.real):g})"
    return f"({var} - {s})"


def factored_text(zpk: ZPK) -> str:
    num = "·".join(_factor(z) for z in zpk.zeros) or "1"
    den = "·".join(_factor(p) for p in zpk.poles) or "1"
    return f"H(z) = {zpk.k:.6g} · {num}\n       / {den}"


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def poly_text(coeffs: np.ndarray) -> str:
    """Polynomial in z^-1, e.g. ``1 - 0.85 z⁻¹ + 0.72 z⁻²``."""
    terms = []
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-14:
            continue
        mag = f"{abs(c):.6g}"
        power = "" if i == 0 else " z" + f"-{i}".translate(_SUP)
        sign = "-" if c < 0 else "+"
        terms.append((sign, f"{mag}{power}"))
    if not terms:
        return "0"
    first_sign, first = terms[0]
    out = ("-" if first_sign == "-" else "") + first
    for sign, t in terms[1:]:
        out += f" {sign} {t}"
    return out


def expanded_coeffs(zpk: ZPK) -> tuple[np.ndarray, np.ndarray]:
    """b, a in powers of z^-1 (display only: loses precision at high order)."""
    b = np.real(zpk.k * np.poly(zpk.zeros)) if len(zpk.zeros) else np.array([zpk.k])
    a = np.real(np.poly(zpk.poles)) if len(zpk.poles) else np.array([1.0])
    n = max(len(b), len(a))
    # align so both are polynomials in z^-1 with the same leading power
    b = np.concatenate([np.zeros(n - len(b)), b])
    a = np.concatenate([a, np.zeros(n - len(a))]) if len(a) < n else a
    return b, a


def latex(zpk: ZPK) -> str:
    def lat_factor(c):
        c = complex(c)
        if abs(c) < 1e-12:
            return "z"
        if c.imag == 0:
            return f"(z {'+' if c.real < 0 else '-'} {abs(c.real):.4g})"
        sign = "+" if c.imag > 0 else "-"
        return f"(z - ({c.real:.4g} {sign} {abs(c.imag):.4g}j))"

    num = " ".join(lat_factor(z) for z in zpk.zeros) or "1"
    den = " ".join(lat_factor(p) for p in zpk.poles) or "1"
    return rf"H(z) = {zpk.k:.6g}\,\frac{{{num}}}{{{den}}}"


def summary(zpk: ZPK) -> str:
    b, a = expanded_coeffs(zpk)
    lines = [
        factored_text(zpk),
        "",
        f"Order: {len(zpk.zeros)} zeros, {len(zpk.poles)} poles",
    ]
    if zpk.padded_poles:
        lines.append(
            f"({zpk.padded_poles} pole(s) at z=0 added so the filter is causal)"
        )
    lines += [
        "",
        "Numerator  B(z) = " + poly_text(b),
        "Denominator A(z) = " + poly_text(a),
    ]
    if max(len(b), len(a)) > 21:
        lines.append("(expanded coefficients are imprecise at this order; use SOS)")
    lines += ["", "LaTeX:", latex(zpk)]
    return "\n".join(lines)
