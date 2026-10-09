"""Region choice (busiest, then most different, spread out) and per-region tempo in Live."""

import math

import numpy as np
import pytest

from tests.fake_live import SR, FakeLiveServer
from tests.test_end_to_end import make_reference, write_midi
from tonematch.analysis.align import Alignment, Region, pick_regions, region_bpm
from tonematch.config import load_config
from tonematch.live.client import LiveClient, LiveError
from tonematch.live.session import LEAD_SECONDS, TakeFailed
from tonematch.midi import Note, Part, Song
from tonematch.pipeline import Project


def verse_chorus_song(bars=32):
    """Bars 0-15: palm-muted low chugs (verse); bars 16-31: open high chords (chorus)."""
    gtr, drums = Part("Guitar", 1), Part("Drums", 9, is_percussion=True)
    for beat in range(bars * 4):
        verse = beat < bars * 2
        for k in range(2 if verse else 1):
            gtr.notes.append(Note(40 if verse else 64, beat + k / 2, 0.4, 100, palm_mute=verse))
        drums.notes.append(Note(36 if beat % 2 == 0 else 38, beat, 0.1, 100))
    return Song([gtr, drums], [(0.0, 120.0), (bars * 2.0, 150.0)])


def test_second_region_contrasts_and_keeps_a_gap():
    song = verse_chorus_song()
    regions = pick_regions(song, Alignment(0.0, 1.0, 1.0), 1e6, bars=4, count=2)
    first, second = regions
    assert first.why == "busiest section" and first.end_beat <= 64          # busier, palm-muted verse
    assert second.start_beat >= 64 and second.why.startswith("contrast:")   # the open chorus
    assert "more open" in second.why
    assert abs(second.start_beat - first.end_beat) >= 16 or abs(first.start_beat - second.end_beat) >= 16


def test_gap_is_relaxed_when_the_song_is_short():
    song = verse_chorus_song(bars=9)             # no room for a one-region gap
    regions = pick_regions(song, Alignment(0.0, 1.0, 1.0), 1e6, bars=4, count=2)
    assert len(regions) == 2


def test_region_bpm_follows_tempo_changes():
    song = verse_chorus_song()
    assert region_bpm(song, 0, 16) == pytest.approx(120.0)
    assert region_bpm(song, 64, 80) == pytest.approx(150.0)
    assert 120.0 < region_bpm(song, 56, 72) < 150.0


def test_each_region_is_recorded_at_its_tempo(tmp_path, monkeypatch):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
    fake = server.live
    write_midi(tmp_path / "song.mid")
    (tmp_path / "tonematch.toml").write_text("""
[project]
reference = "ref.wav"
midi = "song.mid"
[analysis]
separator = "none"
stems = { drums = "stems/drums.wav", bass = "stems/bass.wav", guitar = "stems/guitar.wav" }
region_bars = 2
regions = 2
offset_s = 0.0
scale = 1.0
""")
    try:
        p = Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=lambda m: None)
        p.build()
        make_reference(fake, tmp_path)
        p.analyze()
        regs = p.state["analysis"]["regions"]
        regs[1]["bpm"] = 200.0                         # pretend the song speeds up in region 2
        seen = []
        real = fake.handle

        def handle(cmd, params):
            if cmd == "continue_playing":
                seen.append(fake.tempo)
            return real(cmd, params)

        fake.handle = handle
        fake.record_starts.clear()
        p.match(["levels"])
        assert set(seen) == {regs[0]["bpm"], 200.0}
        # every take switches recording on just before its own region (Live ignores the
        # playhead while stopped, so this only works by jumping while playing)
        expected = set()
        for r in regs:
            at = r["start_beat"] - (1.0 if r["start_beat"] >= 1 else 0.0)
            expected.add(max(0.0, at - max(1.0, math.ceil(LEAD_SECONDS * r["bpm"] / 60.0))))
        assert set(fake.record_starts) == expected
        assert regs[1]["start_beat"] > 0
        assert fake.tempo == p.song.bpm                # set back to the song tempo afterwards
    finally:
        server.close()


REGION = Region(start_beat=40.0, end_beat=48.0, ref_start_s=0.0, ref_end_s=2.0, bpm=240.0)


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
    write_midi(tmp_path / "song.mid")
    (tmp_path / "tonematch.toml").write_text('[project]\nmidi = "song.mid"\n')
    p = Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=lambda m: None)
    p.build()
    yield p, server.live
    server.close()


def test_take_is_trimmed_to_start_one_beat_before_the_region(proj):
    p, fake = proj
    fake.tempo = 240.0
    takes = p.live.record(["Bass"], REGION.start_beat, REGION.end_beat)
    (start,) = fake.record_starts
    assert start == 39.0 - 4.0                                        # 1 s lead-in at 240 BPM
    lead_s = 4.0 * 60 / 240
    assert len(takes["Bass"]) == pytest.approx((fake.seconds - lead_s) * SR, abs=2)


def test_slightly_late_recording_is_padded_too_late_is_retaken(proj):
    p, fake = proj
    fake.tempo = 240.0
    fake.record_delay_beats = 4.5                                     # on 0.5 beat after the pre-roll began
    takes = p.live.record(["Bass"], REGION.start_beat, REGION.end_beat)
    pad = int(round(0.5 * 60 / 240 * SR))
    assert np.all(takes["Bass"][:pad] == 0) and len(takes["Bass"]) == fake.seconds * SR + pad
    fake.record_delay_beats = 6.0                                     # missed the start of the region
    with pytest.raises(TakeFailed, match="beats late"):
        p.live.record(["Bass"], REGION.start_beat, REGION.end_beat)


def test_playhead_that_will_not_move_is_explained(proj):
    p, fake = proj
    fake.jump_fails = True
    with pytest.raises(LiveError, match="didn't move to beat"):
        p.live.record(["Bass"], REGION.start_beat, REGION.end_beat)
    assert not fake.playing
