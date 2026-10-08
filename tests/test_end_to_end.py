"""Full closed loop against the fake Live: build -> analyze -> discover -> match -> export."""

import copy

import mido
import numpy as np
import pytest
import soundfile as sf

from tests.fake_live import SR, FakeLiveServer
from tonematch.config import load_config
from tonematch.live.client import LiveClient
from tonematch.match.loss import tone_loss
from tonematch.pipeline import Project


def write_midi(path, bpm=240, bars=16):
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    mid.tracks.append(meta)
    for name, ch, pitches in (("Drums", 9, [36, 38]), ("Bass", 0, [28, 31]), ("Guitar", 1, [40, 43, 47])):
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name))
        for i in range(bars * 4):
            tr.append(mido.Message("note_on", channel=ch, note=pitches[i % len(pitches)], velocity=100, time=0))
            tr.append(mido.Message("note_off", channel=ch, note=pitches[i % len(pitches)], velocity=0, time=480))
        mid.tracks.append(tr)
    mid.save(path)


def set_by_name(fake, track, device, param, value):
    t = next(t for t in fake.tracks if t.name == track)
    _, ps = next(d for d in t.devices if d[0] == device)
    next(p for p in ps if p.name == param).value = value


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
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
[match]
tone_trials = 16
tone_regions = 1
""")
    cfg = load_config(tmp_path / "tonematch.toml")
    proj = Project(cfg, LiveClient(port=server.port), echo=lambda s: None)
    yield proj, server.live, tmp_path
    server.close()


def make_reference(fake, tmp_path):
    """Dial in a 'true' sound on the fake, render stems + mix, then restore defaults."""
    defaults = {t.name: copy.deepcopy(t.devices) for t in fake.tracks}
    master_defaults = copy.deepcopy(fake.master)
    names = {t.name for t in fake.tracks}
    for g, amp, treble, bass in (("Gtr L", 2, 0.8, 0.3), ("Gtr R", 2, 0.75, 0.35)):
        if g not in names:
            continue
        set_by_name(fake, g, "Archetype Gojira", "Amp Type", amp)
        set_by_name(fake, g, "Archetype Gojira", "Hot Treble", treble)
        set_by_name(fake, g, "Archetype Gojira", "Hot Bass", bass)
        set_by_name(fake, g, "Archetype Gojira", "Hot Gain", 0.7)
    if "Bass" in names:
        set_by_name(fake, "Bass", "Pedal", "Treble", 0.2)
    set_by_name(fake, "Drums", "Utility", "Gain", 0.55)
    fake.seconds = 20.0
    stems = tmp_path / "stems"
    stems.mkdir()
    out = {t.name: fake.track_out(t) for t in fake.tracks}
    drums = sum(a for n, a in out.items() if n.startswith("Drums"))
    sf.write(stems / "drums.wav", drums.astype(np.float32), SR, subtype="FLOAT")
    if "Bass" in out:
        sf.write(stems / "bass.wav", out["Bass"].astype(np.float32), SR, subtype="FLOAT")
    if "Gtr L" in out:
        sf.write(stems / "guitar.wav", (out["Gtr L"] + out["Gtr R"]).astype(np.float32), SR, subtype="FLOAT")
    sf.write(tmp_path / "ref.wav", fake.master_out().astype(np.float32), SR, subtype="FLOAT")
    fake.seconds = 4.0
    for t in fake.tracks:
        t.devices = defaults[t.name]
    fake.master = master_defaults


def test_closed_loop(project):
    proj, fake, tmp_path = project
    todo = proj.build()
    assert [t.name for t in fake.tracks] == ["Drums", "Bass", "Gtr L", "Gtr R"]
    assert [d[0] for d in fake.tracks[2].devices] == ["Hellrazer", "Archetype Gojira", "EQ Eight", "Utility"]
    assert [d[0] for d in fake.master] == ["EQ Eight", "Glue Compressor", "Limiter"]
    assert fake.tracks[2].arrangement and fake.tempo == 240
    assert any("Kontakt" in s for s in todo)

    make_reference(fake, tmp_path)
    proj.analyze()
    report = proj.discover()
    gl = report["Gtr L/archetype_gojira"]
    assert gl["resolved"]["amp"] == "Amp Type" and "Hot" in gl["choices"]["amp"]

    def guitar_loss():
        feats = proj.render([proj.cfg.track("Gtr L")], [0])["Gtr L"][0]
        return tone_loss(feats, proj.ref_features(0, "guitar"), "guitar")["total"]

    before = guitar_loss()
    proj.match()
    after = guitar_loss()
    assert after < before * 0.6, (before, after)

    # Every bounce track was cleaned up.
    assert all(not t.name.startswith("tm-bounce") for t in fake.tracks)
    # Doubles end up panned wide on their configured sides (reference was hard L/R).
    ap = proj.applied()
    assert ap["pan"]["Gtr L"] < -0.8 and ap["pan"]["Gtr R"] > 0.8
    # Levels converge on the per-stem targets.
    last = proj.state["last_render"]
    target = proj.ref_features(0, "guitar").lufs - 3.01 - 6.0
    assert np.mean([f["lufs"] for f in last["Gtr L"]]) == pytest.approx(target, abs=1.5)

    from tonematch.export.presets import export_presets
    from tonematch.export.report import write_report
    out = export_presets(proj)
    assert (out / "presets" / "Gtr L" / "archetype_gojira.json").exists()
    assert "Utility gain" in (out / "SAVE_PRESETS.md").read_text()
    assert write_report(proj).read_text().startswith("<!doctype html>")

    # Rollback: reapplying the stored settings onto a reset device restores them.
    set_by_name(fake, "Gtr L", "Archetype Gojira", "Amp Type", 0)
    proj.reapply()
    stored = ap["params"]["Gtr L"]["archetype_gojira"]["0"]
    t = next(t for t in fake.tracks if t.name == "Gtr L")
    assert t.devices[1][1][0].value == stored


def test_kontakt_multi_outs(tmp_path, monkeypatch):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
    write_midi(tmp_path / "song.mid")
    (tmp_path / "tonematch.toml").write_text("""
[project]
reference = "ref.wav"
midi = "song.mid"
[analysis]
separator = "none"
stems = { drums = "stems/drums.wav", bass = "stems/bass.wav", guitar = "stems/guitar.wav" }
region_bars = 2
regions = 1
offset_s = 0.0
scale = 1.0
[match]
tone_trials = 3
[[tracks]]
name = "Drums"
role = "drums"
stem = "drums"
instrument = "kontakt"
chain = ["EQ Eight", "Utility"]
outputs = [{ name = "Kick", channel = "Post" }, { name = "Room", channel = "Post" }]
""")
    raw = load_config(tmp_path / "tonematch.toml")
    raw.tracks = [t for t in raw.tracks if t.name == "Drums"]   # drums only for this test
    proj = Project(raw, LiveClient(port=server.port), echo=lambda s: None)
    fake = server.live
    proj.build()
    assert [t.name for t in fake.tracks] == ["Drums", "Drums Kick", "Drums Room"]
    assert fake.tracks[1].input_type == "Drums" and fake.tracks[1].monitoring == 0
    make_reference(fake, tmp_path)
    proj.analyze()
    proj.match(["levels", "tone"])
    gains = proj.applied()["gain_db"]
    assert {"Drums", "Drums Kick", "Drums Room"} <= set(gains)
    assert len([e for e in proj.state["log"] if e["stage"] == "tone"]) == 3
    server.close()


def test_build_from_guitar_pro_sets_palm_mute_velocities(tmp_path, monkeypatch):
    from tests.gp_fixtures import write_gp5
    server = FakeLiveServer()
    write_gp5(tmp_path / "song.gp5")
    (tmp_path / "tonematch.toml").write_text('[project]\nscore = "song.gp5"\n')
    proj = Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=lambda s: None)
    proj.build()
    clips = {t.name: t.session_clip["notes"] for t in server.live.tracks}
    for name in ("Gtr L", "Gtr R"):
        vels = [n["velocity"] for n in clips[name]]
        assert vels.count(20) == 16                         # the 16 palm-muted chugs from the tab
        assert all(v >= 64 for v in vels if v != 20)        # open notes can't be muted by humanize
    assert all(n["velocity"] > 40 for n in clips["Bass"])   # palm mutes only on the guitar tracks
    assert {n["pitch"] for n in clips["Bass"]} == {28 + 12}   # default config: bass up one octave
    assert min(n["pitch"] for n in clips["Gtr L"]) == 36      # guitars untransposed
    assert server.live.tempo == 120
    server.close()
