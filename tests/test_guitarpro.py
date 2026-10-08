import pytest

from tests.gp_fixtures import write_gp5, write_gp7
from tonematch.gpfile import Bar, play_order
from tonematch.midi import (Note, Part, Song, classify_parts, humanize, load_song, mark_palm_mutes,
                            palm_mute_velocities, parse_bars)


def test_gp5_notes_repeats_ties_and_palm_mutes(tmp_path):
    song = load_song(write_gp5(tmp_path / "s.gp5"))
    assert song.end_beat == 24.0                      # bars 1 2 3 1 2 4
    assert song.tempos == [(0.0, 120.0)]
    roles = classify_parts(song)
    assert {r: p.name for r, p in roles.items()} == {"drums": "Drums", "bass": "Bass", "guitar": "Rhythm Guitar"}
    gtr = roles["guitar"].notes
    chugs = [n for n in gtr if n.palm_mute]
    assert len(chugs) == 16 and {n.pitch for n in chugs} == {36}          # 8 per pass, drop C low string
    assert [n.start for n in chugs[:8]] == [i * 0.5 for i in range(8)]
    tied = [n for n in gtr if n.start == 20.0]
    assert len(tied) == 1 and tied[0].duration == 4.0 and tied[0].pitch == 43   # half + tied half merged
    assert {n.pitch for n in roles["drums"].notes} == {36}


def test_gp7_notes_tempo_and_dead_notes(tmp_path):
    song = load_song(write_gp7(tmp_path / "s.gp"))
    assert song.tempos == [(0.0, 100.0), (24.0, 150.0)]
    (part,) = song.parts
    assert part.strings == [36, 43, 48, 53, 57, 62]
    assert sum(n.palm_mute for n in part.notes) == 24       # 8 chugs x 3 passes
    assert [n.duration for n in part.notes if n.start == 4.0] == [4.0, 2.0]   # tie extends the low note only
    assert any("dead" in line for line in song.notes_info)
    assert classify_parts(song)["guitar"] is part


def test_gpx_gives_clear_error(tmp_path):
    f = tmp_path / "old.gpx"
    f.write_bytes(b"BCFZ")
    with pytest.raises(ValueError, match="save as .gp"):
        load_song(f)


def test_bass_detected_from_tuning():
    bass = Part("Track 2", 1, [Note(28, 0, 1, 90)], strings=[28, 33, 38, 43])
    gtr = Part("Track 1", 0, [Note(40, 0, 1, 90)] * 5, strings=[36, 43, 48, 53, 57, 62])
    roles = classify_parts(Song([gtr, bass], [(0.0, 120.0)]))
    assert roles["bass"] is bass and roles["guitar"] is gtr


def _bars(*specs):
    return [Bar(**s) for s in specs]


def test_play_order_endings_and_ds_al_coda():
    info = []
    # 0 |: 1 [1. 2 :|] [2. 3] 4 (D.S. al Coda, segno at 1) ... 5 (To Coda) 6 ... 7 (Coda)
    bars = _bars({"repeat_open": True}, {"target": "segno"},
                 {"endings": {1}, "repeat_jumps": 1}, {"endings": {2}},
                 {"jump": "dasegnoalcoda"}, {"jump": "dacoda"}, {}, {"target": "coda"})
    order = play_order(bars, info)
    # First pass with the repeat, then D.S. back to the segno: only the last ending is played,
    # the D.S. isn't taken twice, and "To Coda" jumps to the coda.
    assert order[:7] == [0, 1, 2, 0, 1, 3, 4]
    assert order[7:] == [1, 3, 4, 5, 7]


def test_play_order_multibar_first_ending():
    bars = _bars({"repeat_open": True}, {"endings": {1}}, {"repeat_jumps": 1}, {"endings": {2}}, {})
    assert play_order(bars, []) == [0, 1, 2, 0, 3, 4]


def test_play_order_dc_al_fine():
    bars = _bars({}, {"target": "fine"}, {"jump": "dacapoalfine"})
    assert play_order(bars, []) == [0, 1, 2, 0, 1]


# ------------------------------------------------------------------ palm mutes
def chugs_and_chords():
    notes = [Note(36, i * 0.5, 0.5, 90) for i in range(8)]            # bar 1: chugs
    notes += [Note(36, 4.0, 4.0, 110), Note(43, 4.0, 4.0, 110)]      # bar 2: sustained power chord
    notes += [Note(60, 8.0 + i * 0.5, 0.5, 90) for i in range(8)]    # bar 3: high riff
    return notes


def test_heuristic_marks_low_chug_runs_only():
    marked, mode = mark_palm_mutes(chugs_and_chords(), {"mode": "auto"}, 4.0)
    assert mode == "heuristic"
    assert [n.palm_mute for n in marked] == [True] * 8 + [False] * 10


def test_file_flags_win_in_auto():
    notes = [Note(40, 0, 1, 90, True), Note(40, 1, 1, 90, False)]
    marked, mode = mark_palm_mutes(notes, {}, 4.0)
    assert mode == "file" and [n.palm_mute for n in marked] == [True, False]


def test_velocity_and_cc_modes():
    notes = [Note(40, 0, 1, 25), Note(40, 1, 1, 100)]
    marked, mode = mark_palm_mutes(notes, {"mode": "auto"}, 4.0)
    assert mode == "velocity" and [n.palm_mute for n in marked] == [True, False]
    ccs = [(0.0, 64, 127), (1.0, 64, 0)]
    marked, mode = mark_palm_mutes([Note(40, 0, 1, 90), Note(40, 1, 1, 90)], {"mode": "cc", "cc": 64}, 4.0, ccs)
    assert [n.palm_mute for n in marked] == [True, False]


def test_bar_overrides():
    notes = chugs_and_chords()
    marked, _ = mark_palm_mutes(notes, {"mode": "heuristic", "open_bars": "1", "pm_bars": "3"}, 4.0)
    assert [n.palm_mute for n in marked] == [False] * 8 + [False] * 2 + [True] * 8
    assert parse_bars("9-16, 33") == [(9, 16), (33, 33)]
    with pytest.raises(ValueError):
        parse_bars("nine")


def test_humanized_velocities_never_cross_threshold():
    opts = {"mode": "heuristic", "velocity": 20, "threshold": 40, "open_min": 41}
    notes = [Note(36, i * 0.5, 0.5, 42) for i in range(16)] + [Note(50, 8 + i, 2.0, 41) for i in range(16)]
    marked, _ = mark_palm_mutes(notes, opts, 4.0)
    for seed in range(20):
        out = palm_mute_velocities(humanize(marked, 120, 12, 8, seed), opts)
        assert all(n.velocity == 20 for n in out if n.palm_mute)
        assert all(n.velocity >= 41 for n in out if not n.palm_mute)
    with pytest.raises(ValueError):
        palm_mute_velocities(marked, {"velocity": 45})


def test_transpose_shifts_and_clamps():
    from tonematch.config import TrackSpec
    from tonematch.pipeline import prepare_notes
    part = Part("Bass", 0, [Note(28, 0, 1, 90), Note(120, 1, 1, 90)])
    song = Song([part], [(0.0, 120.0)])
    t = TrackSpec(name="Bass", role="bass", stem="bass", instrument="kontakt", chain=[], transpose=12)
    notes, summary = prepare_notes(song, part, t)
    assert [n.pitch for n in notes] == [40, 127]
    assert "transposed +12" in summary
    assert [n.pitch for n in part.notes] == [28, 120]          # source part untouched
