"""Spectral, loudness, dynamics and stereo features used to compare renders to the reference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pyloudnorm
from scipy.signal import stft, welch

from ..audio import ANALYSIS_SR

# Third-octave centres 31.5 Hz .. 16 kHz (28 bands).
BAND_CENTERS = np.array([1000.0 * 2.0 ** (n / 3.0) for n in range(-15, 13)])
BAND_EDGES = np.concatenate([BAND_CENTERS / 2 ** (1 / 6), [BAND_CENTERS[-1] * 2 ** (1 / 6)]])
SILENCE_DB = -120.0


@dataclass
class Features:
    ltas_db: list[float]        # band power, dB (absolute, follows signal level)
    side_mid_db: list[float]    # per-band side/mid power ratio, dB (stereo width)
    lr_db: list[float]          # per-band left/right power ratio, dB (pan)
    lufs: float                 # integrated loudness
    crest_db: float             # peak-to-RMS
    dyn_range_db: float         # spread of short-term RMS (95th - 10th percentile)
    flatness: float             # mean spectral flatness 200 Hz-8 kHz (distortion density proxy)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Features":
        return cls(**d)

    @property
    def ltas(self) -> np.ndarray:
        return np.asarray(self.ltas_db)

    def tone_curve(self) -> np.ndarray:
        """LTAS with the overall level removed, i.e. spectral shape only."""
        return normalize_curve(self.ltas)


def normalize_curve(curve_db: np.ndarray, ref_band=(250.0, 4000.0)) -> np.ndarray:
    """Subtract the mean level over a midrange reference window."""
    curve_db = np.asarray(curve_db, dtype=float)
    mask = (BAND_CENTERS >= ref_band[0]) & (BAND_CENTERS <= ref_band[1])
    return curve_db - curve_db[mask].mean()


def band_powers(x: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Third-octave band power (linear) of a mono signal from its Welch PSD."""
    nperseg = min(8192, max(256, len(x)))
    f, psd = welch(x, fs=sr, nperseg=nperseg, scaling="spectrum")
    out = np.empty(len(BAND_CENTERS))
    for i in range(len(BAND_CENTERS)):
        m = (f >= BAND_EDGES[i]) & (f < BAND_EDGES[i + 1])
        out[i] = psd[m].sum() if m.any() else np.interp(BAND_CENTERS[i], f, psd)
    return out


def _db(p: np.ndarray | float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(p, 10 ** (SILENCE_DB / 10)))


def lufs(data: np.ndarray, sr: int = ANALYSIS_SR) -> float:
    if len(data) < int(0.4 * sr) or not np.any(data):
        return -70.0
    meter = pyloudnorm.Meter(sr)
    val = meter.integrated_loudness(np.asarray(data, dtype=np.float64))
    return float(val) if np.isfinite(val) else -70.0


def _short_term_rms_db(x: np.ndarray, sr: int, win_s: float = 0.4) -> np.ndarray:
    n = int(win_s * sr)
    if len(x) < n:
        return np.array([float(_db(np.mean(x**2) + 1e-20))])
    frames = len(x) // n
    rms = (x[: frames * n].reshape(frames, n) ** 2).mean(axis=1)
    return _db(rms)


def spectral_flatness(x: np.ndarray, sr: int = ANALYSIS_SR, lo: float = 200.0, hi: float = 8000.0) -> float:
    f, _, z = stft(x, fs=sr, nperseg=2048)
    mag = np.abs(z)
    band = (f >= lo) & (f <= hi)
    mag = mag[band] + 1e-12
    energy = (mag**2).sum(axis=0)
    active = energy > energy.max() * 1e-3 if energy.max() > 0 else np.zeros_like(energy, bool)
    if not active.any():
        return 0.0
    geo = np.exp(np.log(mag[:, active]).mean(axis=0))
    arith = mag[:, active].mean(axis=0)
    return float((geo / arith).mean())


def analyze(data: np.ndarray, sr: int = ANALYSIS_SR) -> Features:
    """Compute all features for a stereo (n, 2) or mono signal."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    left, right = data[:, 0], data[:, 1]
    mid, side = (left + right) / 2, (left - right) / 2
    p_mid, p_side = band_powers(mid, sr), band_powers(side, sr)
    p_l, p_r = band_powers(left, sr), band_powers(right, sr)
    ltas = _db(band_powers(mid, sr) + band_powers(side, sr))
    st = _short_term_rms_db(mid, sr)
    active = st[st > st.max() - 40] if len(st) else st
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(mid**2 + side**2))) if data.size else 0.0
    return Features(
        ltas_db=[round(float(v), 3) for v in ltas],
        side_mid_db=[round(float(v), 3) for v in _db(p_side) - _db(p_mid)],
        lr_db=[round(float(v), 3) for v in _db(p_l) - _db(p_r)],
        lufs=round(lufs(data, sr), 3),
        crest_db=round(float(_db(peak**2) - _db(rms**2)), 3) if rms > 0 else 0.0,
        dyn_range_db=round(float(np.percentile(active, 95) - np.percentile(active, 10)), 3)
        if len(active)
        else 0.0,
        flatness=round(spectral_flatness(mid, sr), 5),
    )
