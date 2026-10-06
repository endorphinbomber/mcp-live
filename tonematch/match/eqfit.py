"""Fit a residual tone difference to EQ Eight bands (low shelf, bells, high shelf)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import freqz

from ..analysis.features import BAND_CENTERS, BAND_EDGES

FS = 44100.0
# Five evaluation points per third-octave band, power-averaged = band smoothing.
_EVAL = np.array([np.geomspace(BAND_EDGES[i], BAND_EDGES[i + 1], 5) for i in range(len(BAND_CENTERS))])


@dataclass
class EqBand:
    kind: str        # "Low Shelf" | "Bell" | "High Shelf" (EQ Eight enum labels)
    freq: float
    gain_db: float
    q: float

    def to_dict(self) -> dict:
        return asdict(self)


def _biquad(kind: str, f0: float, gain_db: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """RBJ audio-EQ-cookbook coefficients."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / FS
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "Bell":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        den = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    else:
        sa = 2 * np.sqrt(a) * alpha
        if kind == "Low Shelf":
            b = [a * ((a + 1) - (a - 1) * cw + sa), 2 * a * ((a - 1) - (a + 1) * cw), a * ((a + 1) - (a - 1) * cw - sa)]
            den = [(a + 1) + (a - 1) * cw + sa, -2 * ((a - 1) + (a + 1) * cw), (a + 1) + (a - 1) * cw - sa]
        else:  # High Shelf
            b = [a * ((a + 1) + (a - 1) * cw + sa), -2 * a * ((a - 1) + (a + 1) * cw), a * ((a + 1) + (a - 1) * cw - sa)]
            den = [(a + 1) - (a - 1) * cw + sa, 2 * ((a - 1) - (a + 1) * cw), (a + 1) - (a - 1) * cw - sa]
    return np.asarray(b), np.asarray(den)


def band_response_db(bands: list[EqBand]) -> np.ndarray:
    """Third-octave-smoothed magnitude response (dB) of a set of bands."""
    pts = _EVAL.ravel()
    total = np.zeros_like(pts)
    for band in bands:
        b, a = _biquad(band.kind, band.freq, band.gain_db, band.q)
        _, h = freqz(b, a, worN=2 * np.pi * pts / FS)
        total += 20 * np.log10(np.abs(h) + 1e-12)
    power = (10 ** (total / 10)).reshape(_EVAL.shape).mean(axis=1)
    return 10 * np.log10(power)


def _layout(n: int) -> list[str]:
    if n < 3:
        return ["Bell"] * n
    return ["Low Shelf", *["Bell"] * (n - 2), "High Shelf"]


def fit_eq(target_db: np.ndarray, weights: np.ndarray, n_bands: int = 6, max_gain_db: float = 6.0,
           regularize: float = 0.08) -> list[EqBand]:
    """Least-squares fit of `n_bands` EQ Eight bands to a target correction curve (dB per
    third-octave band). Gains are bounded and lightly penalized so the EQ only does
    what clearly helps; bands that end up near 0 dB are dropped."""
    target = np.clip(np.asarray(target_db, float), -2 * max_gain_db, 2 * max_gain_db)
    target = np.convolve(np.pad(target, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    kinds = _layout(n_bands)
    f_init = np.geomspace(90.0, 8000.0, n_bands)
    x0, lo, hi = [], [], []
    for kind, f in zip(kinds, f_init):
        q0, qlo, qhi = (0.71, 0.5, 1.2) if kind != "Bell" else (1.0, 0.4, 4.0)
        x0 += [np.log(f), 0.0, q0]
        lo += [np.log(30.0), -max_gain_db, qlo]
        hi += [np.log(16000.0), max_gain_db, qhi]
    sw = np.sqrt(np.asarray(weights, float))

    def unpack(x):
        return [EqBand(k, float(np.exp(x[3 * i])), float(x[3 * i + 1]), float(x[3 * i + 2]))
                for i, k in enumerate(kinds)]

    def residual(x):
        model = band_response_db(unpack(x))
        gains = x[1::3]
        return np.concatenate([sw * (model - target), regularize * gains])

    res = least_squares(residual, np.asarray(x0), bounds=(lo, hi), x_scale=[1.0, 3.0, 0.5] * n_bands)
    return merge_close(unpack(res.x), max_gain_db)


def merge_close(bands: list[EqBand], max_gain_db: float, min_gain_db: float = 0.3) -> list[EqBand]:
    """Fold bells that converged onto (nearly) the same frequency into one band and drop
    bands too small to matter, so no EQ Eight band is wasted."""
    out: list[EqBand] = []
    for b in sorted(bands, key=lambda b: (b.kind, b.freq)):
        prev = out[-1] if out else None
        if (prev is not None and prev.kind == b.kind == "Bell"
                and abs(np.log2(b.freq / prev.freq)) < 1 / 6 and abs(np.log2(b.q / prev.q)) < 1):
            prev.gain_db = float(np.clip(prev.gain_db + b.gain_db, -max_gain_db, max_gain_db))
            continue
        out.append(EqBand(b.kind, b.freq, b.gain_db, b.q))
    return sorted([b for b in out if abs(b.gain_db) >= min_gain_db], key=lambda b: b.freq)
