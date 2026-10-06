"""Reference stem separation with Demucs (htdemucs_6s has a dedicated guitar stem)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

STEMS_6S = ("drums", "bass", "guitar", "vocals", "piano", "other")


def separate(reference: Path, outdir: Path, model: str = "htdemucs_6s") -> dict[str, Path]:
    """Run Demucs once and return {stem: wav path}. Cached: skipped if stems exist."""
    target = outdir / model / reference.stem
    stems = {s: target / f"{s}.wav" for s in STEMS_6S}
    if all(p.exists() for p in stems.values()):
        return stems
    try:
        import demucs  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Demucs is not installed. `pip install 'tonematch[separate]'` (needs PyTorch), "
            "or set analysis.separator = \"none\" and point [analysis].stems at your own stems."
        ) from None
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(outdir), str(reference)]
    if shutil.which("nvidia-smi") is None:
        cmd[3:3] = ["-d", "cpu"]
    subprocess.run(cmd, check=True)
    missing = [s for s, p in stems.items() if not p.exists()]
    if missing:
        raise RuntimeError(f"Demucs finished but stems are missing: {missing} in {target}")
    return stems
