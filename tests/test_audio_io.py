import threading
import time

import numpy as np
import pytest
import soundfile as sf

from tonematch import audio
from tonematch.audio import load_audio, wait_until_readable


def test_wait_returns_once_a_file_finishes_writing(tmp_path):
    path = tmp_path / "take.wav"

    def writer():
        time.sleep(0.4)                                   # file appears late...
        with open(path, "wb") as f:                       # ...and grows for a while
            f.write(b"RIFF")
            f.flush()
            time.sleep(0.5)
        sf.write(path, np.zeros((4410, 2), np.float32), 44100)

    t = threading.Thread(target=writer)
    t.start()
    t0 = time.monotonic()
    wait_until_readable(path, timeout=10, interval=0.1)
    assert time.monotonic() - t0 >= 0.8
    assert sf.info(str(path)).frames == 4410
    t.join()


def test_wait_times_out_with_reason(tmp_path):
    path = tmp_path / "never.wav"
    path.write_bytes(b"RIFF....garbage")
    with pytest.raises(TimeoutError, match="never.wav"):
        wait_until_readable(path, timeout=0.5, interval=0.1)


def test_load_audio_retries_a_busy_file(tmp_path, monkeypatch):
    path = tmp_path / "busy.wav"
    sf.write(path, np.full((100, 2), 0.5, np.float32), 44100)
    real_read, calls = sf.read, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise sf.LibsndfileError(0, prefix="Error opening: System error ")
        return real_read(*a, **k)

    monkeypatch.setattr(audio.sf, "read", flaky)
    monkeypatch.setattr(audio.time, "sleep", lambda s: None)
    data = load_audio(path)
    assert calls["n"] == 3 and data.shape == (100, 2)


def test_load_audio_gives_up_with_clear_error(tmp_path, monkeypatch):
    path = tmp_path / "locked.wav"
    path.write_bytes(b"x")
    monkeypatch.setattr(audio.time, "sleep", lambda s: None)
    with pytest.raises(OSError, match="Cannot read"):
        load_audio(path, retries=2)
