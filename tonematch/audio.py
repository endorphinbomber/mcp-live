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


def load_audio(path: str | Path, sr: int = ANALYSIS_SR, retries: int = 0, delay: float = 0.3) -> np.ndarray:
    """Load any audio file as float32 stereo (n, 2) at `sr`.

    `retries` re-tries libsndfile `delay` seconds apart before the ffmpeg fallback (for
    files another program may still hold open for a moment)."""
    path = Path(path)
    for attempt in range(retries + 1):
        try:
            data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
            break
        except Exception:
            if attempt < retries:
                time.sleep(delay)
    else:
        data, file_sr = _ffmpeg_decode(path)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    if file_sr != sr:
        g = np.gcd(int(file_sr), int(sr))
        data = resample_poly(data, sr // g, int(file_sr) // g, axis=0).astype(np.float32)
    return data


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
