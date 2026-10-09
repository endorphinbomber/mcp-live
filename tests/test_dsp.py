import numpy as np
import pytest

from tonematch.analysis.align import Alignment, align, onset_envelope, pick_regions
from tonematch.analysis.features import BAND_CENTERS, analyze, normalize_curve
from tonematch.audio import ANALYSIS_SR
from tonematch.match.eqfit import EqBand, band_response_db, fit_eq
from tonematch.match.levels import double_pan, gain_delta_db, share_db, single_pan
from tonematch.match.loss import tone_loss
from tonematch.midi import Note, Part, Song, humanize

SR = ANALYSIS_SR


def noise(seconds=3.0, seed=0):
    return np.random.default_rng(seed).standard_normal(int(SR * seconds)) * 0.1


def test_lufs_tracks_gain():
    x = np.stack([noise(), noise(seed=1)], axis=1)
    a, b = analyze(x), analyze(x * 10 ** (-6 / 20))
    assert a.lufs - b.lufs == pytest.approx(6.0, abs=0.05)
    assert gain_delta_db(b.lufs, a.lufs) == pytest.approx(6.0, abs=0.05)


def test_tone_curve_is_level_independent():
    x = np.stack([noise()] * 2, axis=1)
    np.testing.assert_allclose(analyze(x).tone_curve(), analyze(x * 0.3).tone_curve(), atol=1e-2)


def test_eq_fit_recovers_known_curve():
    truth = [EqBand("Low Shelf", 120, 3.0, 0.71), EqBand("Bell", 800, -4.0, 1.5), EqBand("Bell", 3000, 2.5, 1.0)]
    target = band_response_db(truth)
    fitted = fit_eq(target, np.ones(len(BAND_CENTERS)), n_bands=6)
    assert np.abs(band_response_db(fitted) - target).max() < 1.2
    assert all(abs(b.gain_db) <= 6.0 for b in fitted)


def test_eq_fit_identity_is_flat():
    assert fit_eq(np.zeros(len(BAND_CENTERS)), np.ones(len(BAND_CENTERS))) == []


def test_tone_loss_prefers_matching_spectrum():
    from scipy.signal import butter, sosfilt
    base = noise(seed=3)
    dark = sosfilt(butter(2, 1500, "lowpass", fs=SR, output="sos"), base)
    ref = analyze(np.stack([base] * 2, axis=1))
    same = analyze(np.stack([noise(seed=4)] * 2, axis=1))
    other = analyze(np.stack([dark] * 2, axis=1))
    assert tone_loss(same, ref, "guitar")["total"] < tone_loss(other, ref, "guitar")["total"]


def test_double_pan_hard_and_mono():
    l, r = noise(seed=5), noise(seed=6)
    hard = analyze(np.stack([l, r], axis=1))
    mono = analyze(np.stack([l + r, l + r], axis=1))
    assert double_pan(hard, "guitar") > 0.9
    assert double_pan(mono, "guitar") < 0.05


def test_single_pan_direction():
    x = noise(seed=7)
    left_heavy = analyze(np.stack([x, x * 0.5], axis=1))
    assert single_pan(left_heavy, "bass") == pytest.approx(-0.5, abs=0.05)
    assert share_db(2) == pytest.approx(-3.01, abs=0.01)


def _click_song(bpm=120.0, beats=64):
    part = Part("Drums", 9, [Note(36 if i % 2 == 0 else 38, float(i), 0.25, 100) for i in range(beats)])
    part.notes += [Note(42, i + 0.5, 0.1, 60) for i in range(0, beats, 3)]
    return Song([part], [(0.0, bpm)])


def test_alignment_recovers_offset():
    song = _click_song()
    offset = 1.37
    audio = np.zeros(int(SR * (offset + 40)))
    rng = np.random.default_rng(0)
    for n in song.parts[0].notes:
        i = int((offset + song.beats_to_seconds(n.start)) * SR)
        audio[i:i + 2000] += rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300) * n.velocity / 127
    al = align(song, np.stack([audio] * 2, axis=1), scales=np.array([0.99, 1.0, 1.01]))
    assert al.offset_s == pytest.approx(offset, abs=0.03)
    assert al.scale == 1.0
    regions = pick_regions(song, al, len(audio) / SR, bars=4, count=2)
    assert len(regions) == 2
    a, b = sorted(regions, key=lambda r: r.start_beat)
    assert a.end_beat <= b.start_beat


def test_humanize_is_bounded_and_deterministic():
    notes = [Note(40, float(i), 0.5, 100) for i in range(32)]
    a = humanize(notes, 120, timing_ms=12, velocity=8, seed=1)
    b = humanize(notes, 120, timing_ms=12, velocity=8, seed=1)
    assert [n.start for n in a] == [n.start for n in b]
    max_shift_beats = 3 * 12 * 120 / 60000
    assert all(abs(x.start - y.start) <= max_shift_beats for x, y in zip(a, notes))
    assert any(x.start != y.start for x, y in zip(a, notes))
    assert all(1 <= n.velocity <= 127 for n in a)


def test_tempo_map_seconds():
    song = Song([], [(0.0, 120.0), (8.0, 60.0)])
    assert song.beats_to_seconds(8.0) == pytest.approx(4.0)
    assert song.beats_to_seconds(10.0) == pytest.approx(6.0)


def test_normalize_curve_zero_midband():
    c = normalize_curve(np.linspace(-10, 10, len(BAND_CENTERS)))
    mask = (BAND_CENTERS >= 250) & (BAND_CENTERS <= 4000)
    assert c[mask].mean() == pytest.approx(0.0, abs=1e-9)
