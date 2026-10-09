import tomllib

import pytest

from tests.fake_live import FakeLiveServer
from tests.test_end_to_end import write_midi
from tonematch.cli import main
from tonematch.config import ParamSpec, load_config
from tonematch.live.client import LiveClient
from tonematch.live.params import LiveParam, resolve_all
from tonematch.match.optimize import Knob, ToneSearch
from tonematch.pipeline import Project


def kontakt_params(values=None):
    values = values or {}
    return [LiveParam(i, f"#{n:03d}", 0.0, 1.0, values.get(n, 0.6), False, [])
            for i, n in enumerate((0, 2, 3, 4, 5, 7, 8))]


def test_group_knob_moves_all_members_together():
    spec = ParamSpec("toms", "^#00[345]$", group=True, span=0.2)
    resolved, missing = resolve_all([spec], kontakt_params())
    assert not missing and [m.name for m in resolved[0].members] == ["#003", "#004", "#005"]
    search = ToneSearch("Drums", [Knob("Drums", "kontakt_drums", spec, kontakt_params())], seed=1)
    for _ in range(5):
        cand = search.ask()
        values = {s.param_index: s.value for s in cand.values}
        assert len(values) == 3 and len(set(values.values())) == 1     # one value for all three
        search.tell(cand, 1.0)


def test_span_searches_around_the_current_value():
    spec = ParamSpec("kick", "^#000$", span=0.2)
    assert spec.bounds(0.6) == pytest.approx((0.4, 0.8))
    assert spec.bounds(0.95) == pytest.approx((0.75, 1.0))
    search = ToneSearch("Drums", [Knob("Drums", "kontakt_drums", spec, kontakt_params({0: 0.9}))], seed=2)
    first = search.ask()                       # first trial = current setting
    assert first.values[0].value == pytest.approx(0.9)
    search.tell(first, 1.0)
    for _ in range(10):
        cand = search.ask()
        assert 0.7 - 1e-9 <= cand.values[0].value <= 1.0
        search.tell(cand, 1.0)


def test_build_replaces_a_guitar_chain_but_keeps_kontakt(tmp_path, monkeypatch):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
    write_midi(tmp_path / "song.mid")
    old = """[project]
midi = "song.mid"
[[tracks]]
name = "Drums"
role = "drums"
instrument = "kontakt"
chain = ["EQ Eight", "Glue Compressor", "Utility"]
[[tracks]]
name = "Gtr R"
role = "guitar"
instrument = "metal_eclipse"
chain = ["archetype_gojira", "EQ Eight", "Utility"]
"""
    (tmp_path / "tonematch.toml").write_text(old)
    Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=lambda s: None).build()
    fake = server.live
    drums_kontakt = fake.tracks[0].devices[0]
    assert fake.tracks[1].devices[0][0] == "Metal Eclipse"

    (tmp_path / "tonematch.toml").write_text(old.replace('"metal_eclipse"', '"hellrazer"')
                                             .replace('"kontakt"', '"kontakt_drums"'))
    messages: list[str] = []
    Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=messages.append).build()
    assert [d[0] for d in fake.tracks[1].devices] == ["Metal Hellrazer", "Archetype Gojira X", "EQ Eight", "Utility"]
    assert fake.tracks[0].devices[0] is drums_kontakt                   # MixWave instance untouched
    assert any("Gtr R: chain is [Metal Eclipse" in m for m in messages)
    server.close()


def test_init_force_refreshes_defaults_and_keeps_project(tmp_path, capsys):
    cfg = tmp_path / "tonematch.toml"
    cfg.write_text('[project]\nreference = "my ref.mp3"\nscore = "REANIMATION.gp5"\n'
                   '[analysis]\nseparator = "none"\noffset_s = 0.5\n'
                   '[[tracks]]\nname = "Old"\nrole = "guitar"\ninstrument = "metal_eclipse"\n')
    assert main(["-c", str(cfg), "init"]) == 1
    assert main(["-c", str(cfg), "init", "--force"]) == 0
    new = tomllib.loads(cfg.read_text())
    assert new["project"]["reference"] == "my ref.mp3" and new["project"]["score"] == "REANIMATION.gp5"
    assert new["analysis"]["separator"] == "none" and new["analysis"]["offset_s"] == 0.5
    assert [t["name"] for t in new["tracks"]] == ["Drums", "Bass", "Gtr L", "Gtr R"]
    assert (tmp_path / "tonematch.toml.bak").read_text().count("Old") == 1
    load_config(cfg)                                                     # still a valid config
