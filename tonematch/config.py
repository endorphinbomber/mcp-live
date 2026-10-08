"""Project configuration (TOML) with defaults shipped in default_config.toml."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.toml")

STOCK_DEVICES = {
    "EQ Eight", "Utility", "Compressor", "Glue Compressor", "Limiter", "Pedal", "Roar",
    "Saturator", "Amp", "Cabinet", "Gate", "Drum Buss",
}


@dataclass
class ParamSpec:
    key: str
    pattern: str
    kind: str = "continuous"          # "continuous" | "categorical"
    range: tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParamSpec":
        kind = d.get("kind", "continuous")
        if kind not in ("continuous", "categorical"):
            raise ValueError(f"param {d.get('key')!r}: kind must be continuous or categorical")
        lo, hi = d.get("range", (0.0, 1.0))
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"param {d.get('key')!r}: range must satisfy 0 <= lo < hi <= 1")
        return cls(key=d["key"], pattern=d["pattern"], kind=kind, range=(float(lo), float(hi)))


@dataclass
class PluginSpec:
    name: str
    search: str
    params: list[ParamSpec] = field(default_factory=list)
    uri: str = ""            # exact browser URI (from `tonematch plugins`); skips the search

    @property
    def is_stock(self) -> bool:
        return self.name in STOCK_DEVICES


@dataclass
class TrackSpec:
    name: str
    role: str
    stem: str
    instrument: str
    chain: list[str]
    pan: float = 0.0
    double_of: str = ""
    humanize: dict[str, float] = field(default_factory=dict)
    outputs: list[dict[str, str]] = field(default_factory=list)
    note_map: dict[int, int] = field(default_factory=dict)   # MIDI pitch remap (e.g. GM -> MixWave)
    palm_mute: dict = field(default_factory=dict)             # see default_config.toml
    transpose: int = 0                                        # semitones applied to every note

    @property
    def device_names(self) -> list[str]:
        """Instrument followed by the effect chain, as plugin keys."""
        return [self.instrument, *self.chain]


@dataclass
class Config:
    root: Path
    raw: dict[str, Any]
    tracks: list[TrackSpec]
    plugins: dict[str, PluginSpec]
    master_chain: list[str]

    # -- paths ---------------------------------------------------------
    def path(self, key: str) -> Path:
        p = Path(self.raw["project"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def score_path(self) -> Path:
        """The song as MIDI or Guitar Pro: [project] score = ... (or the older key `midi`)."""
        proj = self.raw["project"]
        key = "score" if proj.get("score") else "midi"
        return self.path(key)

    @property
    def workdir(self) -> Path:
        return self.path("workdir")

    @property
    def outdir(self) -> Path:
        return self.path("outdir")

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw["analysis"]

    @property
    def match(self) -> dict[str, Any]:
        return self.raw["match"]

    def plugin(self, key: str) -> PluginSpec:
        if key in self.plugins:
            return self.plugins[key]
        if key in STOCK_DEVICES:
            return PluginSpec(name=key, search=key)
        raise KeyError(f"No [plugins.{key}] section and '{key}' is not a known Live device")

    def track(self, name: str) -> TrackSpec:
        for t in self.tracks:
            if t.name == name:
                return t
        raise KeyError(f"No track named {name!r}")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> Config:
    with open(DEFAULT_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)
    root = Path.cwd()
    if path is not None:
        path = Path(path)
        with open(path, "rb") as f:
            raw = _deep_merge(raw, tomllib.load(f))
        root = path.resolve().parent
    return parse_config(raw, root)


def parse_config(raw: dict[str, Any], root: Path) -> Config:
    plugins = {
        key: PluginSpec(
            name=key,
            search=spec.get("search", key),
            params=[ParamSpec.from_dict(p) for p in spec.get("params", [])],
            uri=spec.get("uri", ""),
        )
        for key, spec in raw.get("plugins", {}).items()
    }
    tracks = [
        TrackSpec(
            name=t["name"],
            role=t["role"],
            stem=t.get("stem", t["role"]),
            instrument=t["instrument"],
            chain=list(t.get("chain", [])),
            pan=float(t.get("pan", 0.0)),
            double_of=t.get("double_of", ""),
            humanize=dict(t.get("humanize", {})),
            outputs=list(t.get("outputs", [])),
            note_map={int(k): int(v) for k, v in t.get("note_map", {}).items()},
            palm_mute=dict(t.get("palm_mute", {})),
            transpose=int(t.get("transpose", 0)),
        )
        for t in raw.get("tracks", [])
    ]
    names = [t.name for t in tracks]
    if len(set(names)) != len(names):
        raise ValueError("Track names must be unique")
    for t in tracks:
        if t.double_of and t.double_of not in names:
            raise ValueError(f"Track {t.name!r}: double_of {t.double_of!r} is not a track")
    cfg = Config(
        root=root,
        raw=raw,
        tracks=tracks,
        plugins=plugins,
        master_chain=list(raw.get("master", {}).get("chain", [])),
    )
    for t in tracks:
        for dev in t.device_names:
            cfg.plugin(dev)  # raises early on typos
    return cfg
