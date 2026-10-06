"""Checkpointed run state (work/state.json) so every stage is resumable."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


class State:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.path = workdir / "state.json"
        self.data: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def __getitem__(self, key: str) -> Any:
        if key not in self.data:
            raise KeyError(f"State has no '{key}' yet - run the earlier stage first")
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def save(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    def checkpoint(self, label: str) -> Path:
        """Snapshot the state after a stage, for rollback (`tonematch rollback <file>`)."""
        self.save()
        hist = self.workdir / "history"
        hist.mkdir(exist_ok=True)
        dest = hist / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.json"
        shutil.copy2(self.path, dest)
        return dest

    def log(self, stage: str, entry: dict) -> None:
        self.data.setdefault("log", []).append({"stage": stage, "time": time.time(), **entry})
        self.save()
