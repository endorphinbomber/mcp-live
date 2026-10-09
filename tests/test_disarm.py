"""No track is left record-armed after build or match."""

from tests.fake_live import FakeLiveServer
from tests.test_end_to_end import write_midi
from tonematch.config import load_config
from tonematch.live.client import LiveClient
from tonematch.live.session import LiveSession
from tonematch.pipeline import Project


def make(tmp_path, monkeypatch):
    monkeypatch.setattr("tonematch.live.session.time.sleep", lambda s: None)
    server = FakeLiveServer()
    write_midi(tmp_path / "song.mid")
    (tmp_path / "tonematch.toml").write_text('[project]\nmidi = "song.mid"\n')
    messages: list[str] = []
    p = Project(load_config(tmp_path / "tonematch.toml"), LiveClient(port=server.port), echo=messages.append)
    return p, server, messages


def test_build_leaves_no_track_armed(tmp_path, monkeypatch):
    p, server, messages = make(tmp_path, monkeypatch)
    try:
        p.build()
        assert [t.name for t in server.live.tracks if t.arm] == []
        assert "  disarmed: Gtr R" in messages             # the last track Live armed by itself
    finally:
        server.close()


def test_disarm_all_skips_tracks_that_cannot_be_armed(tmp_path, monkeypatch):
    p, server, _ = make(tmp_path, monkeypatch)
    try:
        p.build()
        fake = server.live
        fake.tracks[0].arm = fake.tracks[2].arm = True
        real = fake.handle

        def handle(cmd, params):                            # track 1 is a group: arming it fails
            if cmd == "set_track_arm" and params["track_index"] == 0:
                raise Exception("Track cannot be armed")
            return real(cmd, params)

        fake.handle = handle
        assert LiveSession(p.live.c).disarm_all() == [fake.tracks[2].name]
    finally:
        server.close()
