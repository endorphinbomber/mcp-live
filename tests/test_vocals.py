"""`match +vocals`: the master stage targets the original minus its vocals."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from tests.fake_live import SR, FakeLiveServer
from tests.test_end_to_end import make_reference, write_midi
from tonematch import cli
from tonematch.config import load_config
from tonematch.live.client import LiveClient
from tonematch.pipeline import Project

TOML = """
[project]
reference = "ref.wav"
midi = "song.mid"
[analysis]
separator = "none"
stems = { drums = "stems/drums.wav", bass = "stems/bass.wav", guitar = "stems/guitar.wav", vocals = "stems/vocals.wav" }
region_bars = 2
regions = 1
offset_s = 0.0
scale = 1.0
"""


def add_vocal(tmp_path):
    """A loud 'singer' (300 Hz-4 kHz noise) mixed into the reference, plus its stem."""
    ref, _ = sf.read(tmp_path / "ref.wav", dtype="float32", always_2d=True)
    noise = np.random.default_rng(0).standard_normal((len(ref), 1))
    voice = sosfilt(butter(4, [300, 4000], "bandpass", fs=SR, output="sos"), noise, axis=0)
    voice = np.repeat(voice * 1.5 * np.sqrt(np.mean(ref ** 2)) / np.sqrt(np.mean(voice ** 2)), 2, axis=1)
    sf.write(tmp_path / "stems" / "vocals.wav", voice.astype(np.float32), SR, subtype="FLOAT")
    sf.write(tmp_path / "ref.wav", (ref + voice).astype(np.float32), SR, subtype="FLOAT")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
    write_midi(tmp_path / "song.mid")
    (tmp_path / "tonematch.toml").write_text(TOML)
    messages: list[str] = []
    p = Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=messages.append)
    p.build()
    make_reference(server.live, tmp_path)
    add_vocal(tmp_path)
    p.analyze()
    yield p, server.live, messages
    server.close()


def test_instrumental_target_is_quieter_and_has_less_vocal_range(setup):
    p, _, _ = setup
    full = p.ref_mix_features(0)
    p.vocal_space = True
    inst = p.ref_mix_features(0)
    assert full.lufs - inst.lufs > 3
    assert inst.lufs == pytest.approx(p.ref_features(0, "instrumental").lufs)


def test_vocals_option_leaves_headroom(setup):
    p, fake, messages = setup
    master0, applied0 = copy.deepcopy(fake.master), copy.deepcopy(p.applied())

    def limiter_gain():
        return p.applied()["master"]["limiter_gain_db"]

    p.match(["master"])
    gain_full = limiter_gain()
    assert p.applied()["master"]["target"] == "mix"

    fake.master = master0
    p.state.data["applied"] = applied0
    p.vocal_space = True
    messages.clear()
    p.match(["master"])
    assert any("target = original instrumental" in m for m in messages)
    assert p.applied()["master"]["target"] == "instrumental"
    assert gain_full - limiter_gain() > 2


def test_older_analysis_without_instrumental_is_computed(setup):
    p, _, _ = setup
    features = p.state["analysis"]["features"]["0"]
    expected = features.pop("instrumental")["lufs"]
    p.vocal_space = True
    assert p.ref_mix_features(0).lufs == pytest.approx(expected, abs=0.05)
    assert "instrumental" in p.state["analysis"]["features"]["0"]


def test_missing_vocals_stem_explains(setup):
    p, _, _ = setup
    p.state["analysis"]["features"]["0"].pop("instrumental")
    p.state["analysis"]["stems"].pop("vocals")
    p.vocal_space = True
    with pytest.raises(ValueError, match="separated vocals"):
        p.ref_mix_features(0)


@pytest.mark.parametrize("argv, expected", [
    (["match", "+vocals"], True),
    (["match", "--stages", "master", "+Vocals"], True),
    (["match", "--vocals"], True),
    (["match"], False),
])
def test_cli_vocals_option(monkeypatch, argv, expected):
    seen = {}
    stub = SimpleNamespace(cfg=SimpleNamespace(match={}), vocal_space=False)
    stub.match = lambda stages: seen.update(vocals=stub.vocal_space)
    monkeypatch.setattr(cli, "_project", lambda args: stub)
    monkeypatch.setattr("tonematch.export.report.write_report", lambda p: "report.html")
    assert cli.main(argv) == 0
    assert seen["vocals"] is expected


def test_cli_unknown_plus_word(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_project", lambda args: pytest.fail("should not get this far"))
    assert cli.main(["match", "+drums"]) == 2
    assert "+vocals" in capsys.readouterr().err
