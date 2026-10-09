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
    bpm: float = 0.0          # the song's average tempo over the region (0 = not stored, older analysis)
    why: str = ""             # how the region was chosen

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


BANDS_HZ = ((20.0, 250.0), (250.0, 2000.0), (2000.0, 16000.0))
FEATURES = ("palm-muted", "busier", "higher notes", "louder", "more low end", "more mids", "brighter")


def _band_frames(reference: np.ndarray, sr: int = ANALYSIS_SR, n: int = 4096) -> np.ndarray:
    """Per-frame band energies of the reference: (frames, len(BANDS_HZ))."""
    x = mono(reference)
    frames = len(x) // n
    if frames == 0:
        return np.zeros((0, len(BANDS_HZ)))
    spec = np.abs(np.fft.rfft(x[: frames * n].reshape(frames, n) * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return np.stack([spec[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in BANDS_HZ], axis=1)


def _describe(song: Song, a: float, b: float, bands: np.ndarray | None, ra: float, rb: float,
              n: int = 4096) -> np.ndarray:
    """What a window sounds/plays like: palm-mute share, note density, pitch, loudness, band balance."""
    pitched = [x for p in song.parts if not p.is_percussion for x in p.notes if a <= x.start < b]
    every = sum(1 for p in song.parts for x in p.notes if a <= x.start < b)
    pm = float(np.mean([bool(x.palm_mute) for x in pitched])) if pitched else 0.0
    pitch = float(np.mean([x.pitch for x in pitched])) if pitched else 0.0
    out = [pm, every / max(b - a, 1e-9), pitch]
    if bands is not None and len(bands):
        lo, hi = int(ra * ANALYSIS_SR / n), max(int(rb * ANALYSIS_SR / n), int(ra * ANALYSIS_SR / n) + 1)
        e = bands[lo:hi].mean(axis=0) + 1e-12
        total = e.sum()
        out += [10 * np.log10(total), *(10 * np.log10(e / total))]
    else:
        out += [0.0, 0.0, 0.0, 0.0]
    return np.asarray(out)


def _contrast_words(d: np.ndarray) -> str:
    words = [(FEATURES[i] if v > 0 else {"palm-muted": "more open", "busier": "sparser", "higher notes": "lower notes",
                                         "louder": "quieter", "more low end": "less low end",
                                         "more mids": "fewer mids", "brighter": "darker"}[FEATURES[i]], abs(v))
             for i, v in enumerate(d) if abs(v) >= 0.5]
    words.sort(key=lambda w: -w[1])
    return ", ".join(w for w, _ in words[:3]) or "a different part of the song"


def region_bpm(song: Song, start_beat: float, end_beat: float) -> float:
    """The song's average tempo over [start, end] (follows tempo changes)."""
    secs = song.beats_to_seconds(end_beat) - song.beats_to_seconds(start_beat)
    return (end_beat - start_beat) * 60.0 / secs if secs > 0 else song.bpm


def pick_regions(song: Song, alignment: Alignment, ref_duration_s: float, bars: int = 8,
                 count: int = 2, reference: np.ndarray | None = None,
                 gap_bars: float | None = None) -> list[Region]:
    """First the busiest window where every part plays; then, at least `gap_bars` (default: one
    region length) away from the others, the window that sounds most different from them
    (palm mutes vs open, register, density, loudness and band balance of the reference).
    Returned in that order, so `tone_regions = 1` uses the busiest one."""
    bpb = song.beats_per_bar()
    length = bars * bpb
    gap = (bars if gap_bars is None else gap_bars) * bpb
    bands = _band_frames(reference) if reference is not None else None
    cands = []
    start = 0.0
    while start + length <= song.end_beat + 1e-9:
        end = start + length
        active = [sum(1 for n in p.notes if start <= n.start < end) for p in song.parts]
        ra = alignment.ref_time(song.beats_to_seconds(start))
        rb = alignment.ref_time(song.beats_to_seconds(end))
        if ra >= 0 and rb <= ref_duration_s:
            coverage = sum(1 for a in active if a > 0) / max(1, len(active))
            cands.append({"a": start, "b": end, "ra": ra, "rb": rb, "coverage": coverage,
                          "notes": sum(active), "desc": _describe(song, start, end, bands, ra, rb)})
        start += bpb
    if not cands:
        raise ValueError("No region of the MIDI maps inside the reference; check the alignment")
    best_cov = max(c["coverage"] for c in cands)
    full = [c for c in cands if c["coverage"] == best_cov]
    most = max(c["notes"] for c in full)
    pool = [c for c in full if c["notes"] >= 0.4 * most]       # skip near-empty passages
    desc = np.stack([c["desc"] for c in cands])
    scale = desc.std(axis=0)
    scale[scale < 1e-9] = np.inf                                 # a feature that never changes counts 0
    for c in cands:
        c["z"] = (c["desc"] - desc.mean(axis=0)) / scale

    first = max(pool, key=lambda c: (c["notes"], -c["a"]))
    chosen = [dict(first, why="busiest section")]

    def apart(c, g):
        return all(c["b"] + g <= x["a"] or c["a"] >= x["b"] + g for x in chosen)

    while len(chosen) < count:
        for group, g in ((pool, gap), (pool, 0.0), (cands, 0.0)):
            options = [c for c in group if apart(c, g)]
            if options:
                break
        else:
            break
        nxt = max(options, key=lambda c: (min(float(np.linalg.norm(c["z"] - x["z"])) for x in chosen), c["notes"]))
        ref_z = np.mean([x["z"] for x in chosen], axis=0)
        chosen.append(dict(nxt, why="contrast: " + _contrast_words(nxt["z"] - ref_z)))
    return [Region(c["a"], c["b"], c["ra"], c["rb"], bpm=round(region_bpm(song, c["a"], c["b"]), 3), why=c["why"])
            for c in chosen]
