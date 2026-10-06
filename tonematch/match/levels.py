"""Closed-form gain and pan estimates."""

from __future__ import annotations

import math

import numpy as np

from ..analysis.features import Features
from .loss import weights_for

MAX_STEP_DB = 24.0


def gain_delta_db(render_lufs: float, target_lufs: float) -> float:
    if render_lufs <= -60:
        return 0.0  # silent render: gain would not help, something else is wrong
    return float(np.clip(target_lufs - render_lufs, -MAX_STEP_DB, MAX_STEP_DB))


def share_db(n_tracks: int) -> float:
    """Level of each of n uncorrelated tracks that sum to one stem."""
    return -10.0 * math.log10(max(1, n_tracks))


def weighted_band_mean(values: list[float], role: str) -> float:
    w = weights_for(role)
    return float(np.sum(w * np.asarray(values)) / np.sum(w))


def double_pan(ref: Features, role: str) -> float:
    """Pan amount p (0..1, applied as -p/+p) for a pair of uncorrelated doubles that
    reproduces the reference's side/mid ratio.

    With Live's balance-style pan, track A at -p and B at +p give
    L = a + (1-p)b, R = (1-p)a + b, hence S/M power = p^2 / (2-p)^2, so
    p = 2r / (1 + r) with r = sqrt(S/M).
    """
    sm_db = weighted_band_mean(ref.side_mid_db, role)
    r = math.sqrt(10 ** (sm_db / 10))
    return float(np.clip(2 * r / (1 + r), 0.0, 1.0))


def single_pan(ref: Features, role: str) -> float:
    """Pan for a single source from the reference's L/R balance (dB -> balance law)."""
    lr_db = weighted_band_mean(ref.lr_db, role)
    g = 10 ** (-abs(lr_db) / 20)            # quieter side gain relative to louder side
    p = 1.0 - g
    return float(np.clip(-p if lr_db > 0 else p, -1.0, 1.0)) if abs(lr_db) > 0.5 else 0.0
