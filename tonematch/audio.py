"""Audio file I/O helpers (soundfile first, ffmpeg fallback for MP3/AAC)."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ANALYSIS_SR = 44100
# Formats libsndfile reads; anything else (MP3 on old libsndfile, AAC...) goes through ffmpeg.
SNDFILE_FORMATS = {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg", ".caf", ".w64", ".rf64"}


def load_audio(path: str | Path, sr: int = ANALYSIS_SR, retries: int = 5) -> np.ndarray:
    """Load any audio file as float32 stereo (n, 2) at `sr`.

    A file another program still has open (Live finishing a recording; on Windows that
    is a hard lock) is retried with backoff instead of failing on the first attempt."""
    path = Path(path)
    delay = 0.25
    for attempt in range(retries + 1):
        try:
            data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
            break
        except (RuntimeError, OSError) as e:   # LibsndfileError is a RuntimeError
            if path.suffix.lower() not in SNDFILE_FORMATS:
                data, file_sr = _ffmpeg_decode(path)
                break
            if attempt == retries:
                raise OSError(f"Cannot read {path}: {e}") from None
            time.sleep(delay)
            delay = min(2.0, delay * 2)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    if file_sr != sr:
        g = np.gcd(int(file_sr), int(sr))
        data = resample_poly(data, sr // g, int(file_sr) // g, axis=0).astype(np.float32)
    return data


def wait_until_readable(path: str | Path, timeout: float = 30.0, interval: float = 0.3) -> None:
    """Block until a file that is being written (e.g. a Live recording) is complete:
    it exists, its size is stable across two checks and libsndfile can open it."""
    path = Path(path)
    deadline = time.monotonic() + timeout
    last_size, why = -1, "does not exist yet"
    while True:
        try:
            size = path.stat().st_size
            if size > 0 and size == last_size:
                if sf.info(str(path)).frames > 0:
                    return
                why = "has no audio frames yet"
            else:
                why = f"is still being written ({size} bytes)"
            last_size = size
        except FileNotFoundError:
            last_size, why = -1, "does not exist yet"
        except (RuntimeError, OSError) as e:
            why = f"cannot be opened yet ({e})"
        if time.monotonic() > deadline:
            raise TimeoutError(f"{path.name} {why} after {timeout:.0f}s")
        time.sleep(interval)


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"Cannot decode {path.name}: libsndfile does not support it and ffmpeg is not installed"
        )
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(ANALYSIS_SR), "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).reshape(-1, 2).copy(), ANALYSIS_SR


def save_wav(path: str | Path, data: np.ndarray, sr: int = ANALYSIS_SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="FLOAT")


def segment(data: np.ndarray, start_s: float, end_s: float, sr: int = ANALYSIS_SR) -> np.ndarray:
    a = max(0, int(round(start_s * sr)))
    b = min(len(data), int(round(end_s * sr)))
    return data[a:b]


def mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim == 2 else data
