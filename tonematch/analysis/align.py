"""Align the MIDI rendition with the reference recording and pick comparison regions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, stft

from ..audio import ANALYSIS_SR, mono
from ..midi import Song

HOP = 512
FRAME_RATE = ANALYSIS_SR / HOP


@dataclass
class Alignment:
    offset_s: float   # reference time of MIDI time 0
    scale: float      # reference seconds per MIDI second
    score: float      # normalized cross-correlation peak (0..1)

    def ref_time(self, midi_seconds: float) -> float:
        return self.offset_s + self.scale * midi_seconds

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Region:
    start_beat: float
    end_beat: float
    ref_start_s: float
    ref_end_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def onset_envelope(data: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Half-wave rectified log-spectral flux, normalized."""
    _, _, z = stft(mono(data), fs=sr, nperseg=2048, noverlap=2048 - HOP, boundary=None, padded=False)
    logmag = np.log1p(100 * np.abs(z))
    flux = np.maximum(np.diff(logmag, axis=1), 0).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    flux -= gaussian_filter1d(flux, 20)
    flux = np.maximum(flux, 0)
    return flux / (flux.max() + 1e-12)


def midi_envelope(onsets_s: np.ndarray, weights: np.ndarray, n_frames: int) -> np.ndarray:
    env = np.zeros(n_frames)
    idx = np.round(onsets_s * FRAME_RATE).astype(int)
    ok = (idx >= 0) & (idx < n_frames)
    np.add.at(env, idx[ok], weights[ok])
    env = gaussian_filter1d(env, 1.5)
    return env / (env.max() + 1e-12)


def _song_onsets(song: Song, prefer_role_part=None) -> tuple[np.ndarray, np.ndarray]:
    parts = [prefer_role_part] if prefer_role_part is not None else song.parts
    starts, weights = [], []
    for p in parts:
        for n in p.notes:
            starts.append(song.beats_to_seconds(n.start))
            weights.append(n.velocity / 127.0)
    return np.asarray(starts), np.asarray(weights)


def align(song: Song, reference: np.ndarray, drums_part=None, max_offset_s: float = 60.0,
          scales: np.ndarray | None = None, ref_env: np.ndarray | None = None) -> Alignment:
    """Find offset (and a small linear tempo correction) that best lines up MIDI onsets
    with the reference's onset envelope. Drums are the most reliable anchor."""
    if scales is None:
        scales = np.arange(0.97, 1.0301, 0.0025)
    env = onset_envelope(reference) if ref_env is None else ref_env
    starts, weights = _song_onsets(song, drums_part)
    if len(starts) == 0:
        raise ValueError("MIDI has no notes to align")
    max_lag = int(max_offset_s * FRAME_RATE)
    best = Alignment(0.0, 1.0, -1.0)
    env_c = env - env.mean()
    for scale in scales:
        n_frames = int((starts.max() * scale) * FRAME_RATE) + 10
        m = midi_envelope(starts * scale, weights, n_frames)
        m_c = m - m.mean()
        # Correlate for lags in [-max_lag, max_lag]: reference frame = midi frame + lag.
        corr = correlate(np.pad(env_c, (len(m_c), len(m_c))), m_c, mode="valid", method="fft")
        lags = np.arange(len(corr)) - len(m_c)
        keep = np.abs(lags) <= max_lag
        corr, lags = corr[keep], lags[keep]
        i = int(np.argmax(corr))
        denom = np.linalg.norm(m_c) * np.linalg.norm(env_c) + 1e-12
        score = float(corr[i] / denom)
        if score > best.score:
            best = Alignment(offset_s=float(lags[i] / FRAME_RATE), scale=float(scale), score=score)
    return best


def pick_regions(song: Song, alignment: Alignment, ref_duration_s: float, bars: int = 8,
                 count: int = 2) -> list[Region]:
    """Choose the densest non-overlapping windows where every part plays."""
    bpb = song.beats_per_bar()
    length = bars * bpb
    total = song.end_beat
    candidates = []
    start = 0.0
    while start + length <= total + 1e-9:
        end = start + length
        active = [sum(1 for n in p.notes if start <= n.start < end) for p in song.parts]
        ref_a, ref_b = alignment.ref_time(song.beats_to_seconds(start)), alignment.ref_time(
            song.beats_to_seconds(end))
        if ref_a >= 0 and ref_b <= ref_duration_s:
            coverage = sum(1 for a in active if a > 0) / max(1, len(active))
            candidates.append((coverage, sum(active), start, end, ref_a, ref_b))
        start += bpb
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    regions: list[Region] = []
    for _, _, a, b, ra, rb in candidates:
        if all(b <= r.start_beat or a >= r.end_beat for r in regions):
            regions.append(Region(a, b, ra, rb))
        if len(regions) == count:
            break
    if not regions:
        raise ValueError("No region of the MIDI maps inside the reference; check the alignment")
    return sorted(regions, key=lambda r: r.start_beat)
