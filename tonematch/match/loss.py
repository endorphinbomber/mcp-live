"""Distance between a render and its reference stem."""

from __future__ import annotations

import numpy as np

from ..analysis.features import BAND_CENTERS, Features


def _ramp(lo_hz: float, hi_hz: float, floor: float = 0.15) -> np.ndarray:
    """Weight 1 inside [lo, hi], fading (per octave) to `floor` outside."""
    octs = np.log2(BAND_CENTERS)
    w = np.ones_like(octs)
    w = np.where(octs < np.log2(lo_hz), np.maximum(floor, 1 - (np.log2(lo_hz) - octs)), w)
    w = np.where(octs > np.log2(hi_hz), np.maximum(floor, 1 - (octs - np.log2(hi_hz))), w)
    return w


# Where each source carries information that survives stem separation.
ROLE_WEIGHTS = {
    "guitar": _ramp(90.0, 7000.0),
    "bass": _ramp(40.0, 3000.0),
    "drums": _ramp(40.0, 12000.0),
    "mix": _ramp(35.0, 14000.0),
}


def weights_for(role: str) -> np.ndarray:
    return ROLE_WEIGHTS.get(role, ROLE_WEIGHTS["mix"])


def spectral_distance(render: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> float:
    """Weighted RMS difference (dB) between two level-normalized tone curves."""
    diff = np.clip(np.asarray(render) - np.asarray(ref), -30, 30)
    return float(np.sqrt(np.sum(weights * diff**2) / np.sum(weights)))


def tone_loss(render: Features, ref: Features, role: str) -> dict[str, float]:
    w = weights_for(role)
    spectral = spectral_distance(render.tone_curve(), ref.tone_curve(), w)
    flat = 40.0 * abs(render.flatness - ref.flatness)
    dyn = 0.3 * abs(render.crest_db - ref.crest_db) + 0.3 * abs(render.dyn_range_db - ref.dyn_range_db)
    silent = 50.0 if render.lufs <= -60 else 0.0
    return {"spectral": spectral, "flatness": flat, "dynamics": dyn, "silence": silent,
            "total": spectral + flat + dyn + silent}
