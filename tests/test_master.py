"""Master chain devices are found by name, whatever their position."""

import pytest

from tests.fake_live import FakeLiveServer, make_device
from tests.test_end_to_end import make_reference, write_midi
from tonematch.config import load_config
from tonematch.live.client import LiveClient, LiveError
from tonematch.pipeline import Project

TOML = """
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
"""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
    fake = server.live
    fake.master.append(make_device("Utility"))       # a device the user already had on the master
    fake.master_insert_front = True                  # Live inserting after the selected (first) device
    write_midi(tmp_path / "song.mid")
    (tmp_path / "tonematch.toml").write_text(TOML)
    messages: list[str] = []
    p = Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=messages.append)
    p.build()
    yield p, fake, tmp_path, messages
    server.close()


def master_names(fake):
    return [d[0] for d in fake.master]


def test_master_stage_finds_devices_in_any_order(setup):
    p, fake, tmp_path, messages = setup
    assert master_names(fake) == ["Limiter", "Glue Compressor", "EQ Eight", "Utility"]
    make_reference(fake, tmp_path)
    p.analyze()
    for t in fake.tracks:                      # mix 6 dB quieter than the reference -> limiter must work
        for name, params in t.devices:
            if name == "Utility":
                next(q for q in params if q.name == "Gain").value -= 6 / 70
    p.match(["master"])
    limiter = {q.name: q.value for q in fake.master[master_names(fake).index("Limiter")][1]}
    assert limiter["Gain"] > 3                                      # gain went to the real Limiter
    assert limiter["Ceiling"] == pytest.approx(-0.3, abs=0.05)
    assert any(m.startswith("  master EQ") for m in messages)       # EQ written to the real EQ Eight
    assert fake.master[master_names(fake).index("Utility")][1][0].value == 0.5   # user's device untouched


def test_build_twice_does_not_duplicate_master_devices(setup):
    p, fake, _, _ = setup
    p.build()
    assert sorted(master_names(fake)) == ["EQ Eight", "Glue Compressor", "Limiter", "Utility"]


def test_missing_master_device_is_skipped_or_explained(setup):
    p, fake, tmp_path, messages = setup
    fake.master = [d for d in fake.master if d[0] != "Limiter"]
    make_reference(fake, tmp_path)
    p.analyze()
    p.match(["master"])
    assert any("no Limiter on the Master track" in m for m in messages)
    with pytest.raises(LiveError, match="Master track has no 'Limiter'.*EQ Eight"):
        p.device_ref("Master", "Limiter")
