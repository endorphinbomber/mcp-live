"""Recordings: disarm/delete before reading, wait for Live to finish files, retake, log."""

import json

import pytest

from tests.fake_live import FakeLiveServer
from tests.test_end_to_end import write_midi
from tonematch.analysis.align import Region
from tonematch.config import load_config
from tonematch.live.client import LiveClient, LiveError
from tonematch.pipeline import Project

REGION = Region(start_beat=0.0, end_beat=8.0, ref_start_s=0.0, ref_end_s=2.0)


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
    write_midi(tmp_path / "song.mid")
    (tmp_path / "tonematch.toml").write_text('[project]\nmidi = "song.mid"\n')
    messages: list[str] = []
    p = Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=messages.append)
    p.build()
    p.cfg.match["finalize_timeout_s"] = 0.5       # keep never-finishing files quick in tests
    yield p, server.live, messages
    server.close()


def log_entries(p):
    path = p.cfg.workdir / "failed_takes.log"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_unreadable_take_is_recorded_again(proj):
    p, fake, messages = proj
    fake.broken_takes = {1}                        # first file of the first take is 0 bytes
    takes = p._take(["Gtr L", "Bass"], REGION)
    assert set(takes) == {"Gtr L", "Bass"} and all(len(a) > 0 for a in takes.values())
    assert any("recording again (1/2)" in m for m in messages)
    (entry,) = log_entries(p)
    (failure,) = entry["failures"]
    assert failure["source"] == "Gtr L" and failure["exists"] and failure["size"] == 0
    assert failure["error"].startswith("file is 0 bytes")
    assert all(not t.name.startswith("tm-bounce") for t in fake.tracks)


def test_files_live_is_still_writing_are_read_once_tracks_are_released(proj):
    p, fake, messages = proj
    fake.defer_finalize = True                   # file stays a 64 KiB-chunked stub until disarm/delete
    fake.commands.clear()
    takes = p._take(["Gtr L", "Bass"], REGION)
    assert set(takes) == {"Gtr L", "Bass"} and all(len(a) > 0 for a in takes.values())
    assert not any("recording again" in m for m in messages)
    cmds = fake.commands
    stop = cmds.index("stop_playback")
    disarms = [i for i, c in enumerate(cmds) if c == "set_track_arm" and i > stop]
    deletes = [i for i, c in enumerate(cmds) if c == "delete_track"]
    assert len(disarms) == 2 and max(disarms) < min(deletes)


def test_file_that_never_finishes_is_logged_with_sizes(proj):
    p, fake, _ = proj
    fake.defer_finalize = True
    fake.never_finalize = set(range(1, 100))
    with pytest.raises(LiveError, match="failed 3 times"):
        p._take(["Bass"], REGION)
    failure = log_entries(p)[0]["failures"][0]
    assert failure["size_first_seen"] == 65536 * 3 and failure["size"] == 65536 * 3
    assert failure["waited_s"] == 0.5


def test_persistent_failure_gives_up_with_log(proj):
    p, fake, _ = proj
    fake.missing_takes = set(range(1, 100))
    with pytest.raises(LiveError, match="failed 3 times.*failed_takes.log"):
        p._take(["Bass"], REGION)
    entries = log_entries(p)
    assert [e["attempt"] for e in entries] == [1, 2, 3]
    assert entries[0]["failures"][0]["error"].startswith("file missing")
    assert all(not t.name.startswith("tm-bounce") for t in fake.tracks)
