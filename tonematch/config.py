"""Project configuration (TOML) with defaults shipped in default_config.toml."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.toml")
GP_SUFFIXES = (".gp", ".gp5", ".gp4", ".gp3")
MIDI_SUFFIXES = (".mid", ".midi")

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
    group: bool = False               # one knob drives every parameter the pattern matches
    span: float | None = None         # range = current value +/- span (normalized), instead of `range`

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParamSpec":
        kind = d.get("kind", "continuous")
        if kind not in ("continuous", "categorical"):
            raise ValueError(f"param {d.get('key')!r}: kind must be continuous or categorical")
        lo, hi = d.get("range", (0.0, 1.0))
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"param {d.get('key')!r}: range must satisfy 0 <= lo < hi <= 1")
        span = d.get("span")
        if span is not None and not 0.0 < float(span) <= 1.0:
            raise ValueError(f"param {d.get('key')!r}: span must be between 0 and 1")
        if d.get("group") and kind != "continuous":
            raise ValueError(f"param {d.get('key')!r}: group only works for continuous knobs")
        return cls(key=d["key"], pattern=d["pattern"], kind=kind, range=(float(lo), float(hi)),
                   group=bool(d.get("group", False)), span=None if span is None else float(span))

    def bounds(self, current_u: float) -> tuple[float, float]:
        """Search range in normalized units, given the parameter's current value."""
        if self.span is None:
            return self.range
        lo, hi = max(0.0, current_u - self.span), min(1.0, current_u + self.span)
        return (lo, hi) if hi - lo > 1e-6 else (max(0.0, hi - 1e-3), hi)


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
        """The song file (Guitar Pro or MIDI).

        `score` wins when set. Otherwise the `midi` key is used, except that a single Guitar
        Pro file in the project folder is preferred over a MIDI file (it carries palm mutes).
        With neither set (or the file missing) the project folder is searched: Guitar Pro first,
        then MIDI."""
        proj = self.raw["project"]
        if proj.get("score"):
            return self.path("score")
        found = sorted(self.root.iterdir()) if self.root.is_dir() else []
        gps = [p for p in found if p.suffix.lower() in GP_SUFFIXES]
        mids = [p for p in found if p.suffix.lower() in MIDI_SUFFIXES]
        configured = self.path("midi") if proj.get("midi") else None
        if configured is not None and configured.exists():
            if configured.suffix.lower() in MIDI_SUFFIXES and len(gps) == 1:
                return gps[0]
            return configured
        if len(gps) == 1:
            return gps[0]
        if not gps and len(mids) == 1:
            return mids[0]
        names = ", ".join(p.name for p in gps + mids) or "none"
        what = f"'{configured.name}' not found; " if configured is not None else ""
        raise FileNotFoundError(f"{what}can't pick the song file in {self.root} (found: {names}). "
                                "Set score = \"yourfile.gp5\" under [project] in tonematch.toml.")

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


# ------------------------------------------------------------------ init --force
KEEP_SECTIONS = ("project", "analysis")


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f'"{k}" = {_toml_value(x)}' if not k.isidentifier() else f"{k} = {_toml_value(x)}"
                                for k, x in v.items()) + " }"
    raise TypeError(f"Can't write {type(v).__name__} to TOML")


def refresh_config_text(old_path: Path, default_text: str) -> str:
    """Default config text with the user's [project] and [analysis] values put back in."""
    with open(old_path, "rb") as f:
        old = tomllib.load(f)
    lines = default_text.splitlines()
    for section in KEEP_SECTIONS:
        values = dict(old.get(section, {}))
        if not values:
            continue
        start = next(i for i, ln in enumerate(lines) if ln.strip() == f"[{section}]")
        end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        for i in range(start + 1, end):
            raw = lines[i].lstrip("# ").split("=", 1)
            key = raw[0].strip()
            if len(raw) == 2 and key in values:
                comment = lines[i].split("#", 1)[1].strip() if "#" in lines[i].split("=", 1)[1] else ""
                if _toml_value(values[key]).count("#"):
                    comment = ""
                lines[i] = f"{key} = {_toml_value(values.pop(key))}" + (f"   # {comment}" if comment else "")
        insert = [f"{k} = {_toml_value(v)}" for k, v in values.items()]
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        lines[end:end] = insert
    return "\n".join(lines) + "\n"
