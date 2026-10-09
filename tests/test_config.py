import pytest

from tests.fake_live import FakeLiveServer
from tests.gp_fixtures import write_gp5
from tests.test_end_to_end import write_midi
from tonematch.config import load_config
from tonematch.live.client import LiveClient
from tonematch.pipeline import Project


def cfg_in(tmp_path, project_toml: str = ""):
    (tmp_path / "tonematch.toml").write_text("[project]\n" + project_toml)
    return load_config(tmp_path / "tonematch.toml")


def test_only_guitar_pro_file_is_picked_up(tmp_path):
    (tmp_path / "REANIMATION.gp5").write_bytes(b"x")
    assert cfg_in(tmp_path).score_path.name == "REANIMATION.gp5"


def test_guitar_pro_preferred_over_configured_midi(tmp_path):
    (tmp_path / "song.mid").write_bytes(b"x")
    (tmp_path / "song.gp5").write_bytes(b"x")
    assert cfg_in(tmp_path, 'midi = "song.mid"\n').score_path.name == "song.gp5"


def test_explicit_score_always_wins(tmp_path):
    (tmp_path / "song.mid").write_bytes(b"x")
    (tmp_path / "song.gp5").write_bytes(b"x")
    assert cfg_in(tmp_path, 'score = "song.mid"\n').score_path.name == "song.mid"


def test_ambiguous_guitar_pro_files_keep_configured_midi(tmp_path):
    for n in ("a.gp5", "b.gp", "song.mid"):
        (tmp_path / n).write_bytes(b"x")
    assert cfg_in(tmp_path, 'midi = "song.mid"\n').score_path.name == "song.mid"
    with pytest.raises(FileNotFoundError, match="a.gp5, b.gp"):
        cfg_in(tmp_path).score_path


def test_midi_used_when_no_tab(tmp_path):
    (tmp_path / "song.mid").write_bytes(b"x")
    assert cfg_in(tmp_path).score_path.name == "song.mid"


def test_nothing_found_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Set score"):
        cfg_in(tmp_path, 'midi = "missing.mid"\n').score_path


def test_build_uses_tab_next_to_old_midi_config(tmp_path):
    server = FakeLiveServer()
    write_midi(tmp_path / "song.mid")
    write_gp5(tmp_path / "song.gp5")
    messages: list[str] = []
    proj = Project(cfg_in(tmp_path, 'midi = "song.mid"\n'), LiveClient(port=server.port), echo=messages.append)
    proj.state.data["analysis"] = {"song_file": "song.mid"}
    proj.build()
    assert "Song: song.gp5 (Guitar Pro, palm mutes from the tab)" in messages
    assert any(m.startswith("WARNING: analysis was made from song.mid") for m in messages)
    assert any("palm-muted (from the tab)" in m for m in messages)
    gtr = next(t for t in server.live.tracks if t.name == "Gtr L").session_clip["notes"]
    assert [n["velocity"] for n in gtr].count(20) == 16
    server.close()
