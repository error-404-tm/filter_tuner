"""Loading test signals, filtering them, and computing spectrograms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


@dataclass(frozen=True, eq=False)
class LoadedSignal:
    x: np.ndarray        # float64, mono
    fs: float | None     # sample rate if the file carries one
    source: str


def load_signal(path: str | Path, column: int = -1) -> LoadedSignal:
    """WAV (any integer/float format; first channel) or CSV/TXT/NPY.

    For text files with several columns the last column is the signal
    (so ``time, value`` traces work as-is) unless ``column`` is given.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".wav":
        fs, data = wavfile.read(path)
        data = np.asarray(data)
        if data.ndim > 1:
            data = data[:, 0]
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            data = (data.astype(np.float64) - (info.max + info.min + 1) / 2) / (info.max + 1)
        return LoadedSignal(data.astype(np.float64), float(fs), str(path))
    if ext == ".npy":
        data = np.load(path)
        return LoadedSignal(np.asarray(data, dtype=np.float64).ravel(), None, str(path))
    data = np.genfromtxt(path, delimiter="," if ext == ".csv" else None,
                         invalid_raise=False)
    data = data[~np.all(np.isnan(np.atleast_2d(data.T).T), axis=1)] if data.ndim > 1 else data
    if data.ndim > 1:
        data = data[:, column]
    data = data[np.isfinite(data)]
    return LoadedSignal(data.astype(np.float64), None, str(path))


def apply_sos(sos: np.ndarray, x: np.ndarray, zero_phase: bool = False) -> np.ndarray:
    if zero_phase:
        return signal.sosfiltfilt(sos, x)
    return signal.sosfilt(sos, x)


@dataclass(frozen=True, eq=False)
class Spectrogram:
    f: np.ndarray
    t: np.ndarray
    db: np.ndarray  # [time, freq]


def spectrogram(x: np.ndarray, fs: float, nperseg: int = 256) -> Spectrogram:
    nperseg = int(min(nperseg, max(16, len(x))))
    f, t, S = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    db = 10 * np.log10(np.maximum(S, 1e-20))
    return Spectrogram(f, t, db.T)


def save_signal(path: str | Path, y: np.ndarray, fs: float) -> None:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        peak = float(np.max(np.abs(y))) or 1.0
        wavfile.write(path, int(round(fs)), (y / max(peak, 1.0)).astype(np.float32))
    elif path.suffix.lower() == ".npy":
        np.save(path, y)
    else:
        np.savetxt(path, y, delimiter=",")
