"""tone_regions: every trial records each region and is scored on the average."""

from tests.fake_live import FakeLiveServer
from tests.test_end_to_end import make_reference, write_midi
from tonematch.config import load_config
from tonematch.live.client import LiveClient
from tonematch.pipeline import Project

TOML = """
[project]
reference = "ref.wav"
midi = "song.mid"
[analysis]
separator = "none"
stems = {{ drums = "stems/drums.wav", bass = "stems/bass.wav", guitar = "stems/guitar.wav" }}
region_bars = 2
regions = {regions}
offset_s = 0.0
scale = 1.0
[match]
tone_trials = 3
tone_regions = 2
"""


def project(tmp_path, monkeypatch, regions):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
    write_midi(tmp_path / "song.mid")
    (tmp_path / "tonematch.toml").write_text(TOML.format(regions=regions))
    messages: list[str] = []
    p = Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=messages.append)
    p.build()
    make_reference(server.live, tmp_path)
    p.analyze()
    return p, server, messages


def test_each_trial_records_every_tone_region(tmp_path, monkeypatch):
    p, server, messages = project(tmp_path, monkeypatch, regions=2)
    try:
        server.live.commands.clear()
        p.match(["tone"])
        assert server.live.commands.count("stop_playback") == 3 * 2          # 3 trials x 2 regions
        assert any("3 trials x 2 region(s)" in m and "6 takes" in m for m in messages)
        trials = [m for m in messages if m.startswith("  trial ")]
        assert len(trials) == 3
        assert all(m.count("[") >= 1 and "/" in m.split("[")[1] for m in trials)   # per-region losses
        entry = [e for e in p.state.data["log"] if e["stage"] == "tone"][-1]
        assert all(len(v) == 2 for v in entry["per_region"].values())
    finally:
        server.close()


def test_warns_when_fewer_regions_were_analysed(tmp_path, monkeypatch):
    p, server, messages = project(tmp_path, monkeypatch, regions=1)
    try:
        p.match(["tone"])
        assert any("only 1 region(s)" in m and "regions = 2" in m for m in messages)
        assert any("3 trials x 1 region(s)" in m for m in messages)
    finally:
        server.close()
