"""JSON save/load for a whole project (filter, mask, fixed-point settings)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .mask import Band, Mask
from .model import FilterState, GainSpec, Root
from .quantize import QFormat

VERSION = 1


@dataclass(frozen=True)
class QuantSettings:
    coef: QFormat = QFormat(16, 14)
    data_bits: int = 16
    scaling: str = "none"        # "none" | "linf" | "l2"
    show_overlay: bool = True


@dataclass(frozen=True)
class Project:
    state: FilterState
    mask: Mask = Mask()
    quant: QuantSettings = QuantSettings()


def to_dict(p: Project) -> dict:
    s = p.state
    return {
        "version": VERSION,
        "fs": s.fs,
        "gain": {"mode": s.gain.mode, "value": s.gain.value, "freq": s.gain.freq},
        "roots": [
            {"kind": r.kind, "re": r.value.real, "im": r.value.imag, "paired": r.paired}
            for r in s.roots
        ],
        "mask": [
            {"f_lo": b.f_lo, "f_hi": b.f_hi, "lo_db": b.lo_db, "hi_db": b.hi_db}
            for b in p.mask.bands
        ],
        "quant": {
            "word_bits": p.quant.coef.word_bits,
            "frac_bits": p.quant.coef.frac_bits,
            "data_bits": p.quant.data_bits,
            "scaling": p.quant.scaling,
            "show_overlay": p.quant.show_overlay,
        },
    }


def from_dict(d: dict) -> Project:
    if d.get("version", 1) > VERSION:
        raise ValueError(f"project file version {d['version']} is newer than this app")
    g = d.get("gain", {})
    state = FilterState(
        roots=tuple(
            Root(r["kind"], complex(r["re"], r["im"]), bool(r.get("paired", True)))
            for r in d.get("roots", [])
        ),
        gain=GainSpec(g.get("mode", "manual"), g.get("value", 1.0), g.get("freq", 0.25)),
        fs=float(d.get("fs", 1.0)),
    )
    mask = Mask(tuple(Band(**b) for b in d.get("mask", [])))
    q = d.get("quant", {})
    quant = QuantSettings(
        QFormat(q.get("word_bits", 16), q.get("frac_bits", 14)),
        q.get("data_bits", 16), q.get("scaling", "none"), q.get("show_overlay", True),
    )
    return Project(state, mask, quant)


def save(path: str | Path, project: Project) -> None:
    Path(path).write_text(json.dumps(to_dict(project), indent=2))


def load(path: str | Path) -> Project:
    return from_dict(json.loads(Path(path).read_text()))
