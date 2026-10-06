"""Export the matched settings: per-device parameter snapshots, mixer sheet, save checklist."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..live.params import resolve_all

_SAFE = re.compile(r"[^A-Za-z0-9 ._-]+")


def _safe(name: str) -> str:
    return _SAFE.sub("_", name).strip() or "unnamed"


def export_presets(project) -> Path:
    cfg, live = project.cfg, project.live
    out = cfg.outdir / "presets"
    out.mkdir(parents=True, exist_ok=True)
    build = project.state["build"]
    summary: list[str] = []
    for track, info in build["tracks"].items():
        for key in info["devices"]:
            ref = project.device_ref(track, key)
            params = live.device_params(ref)
            data = {"track": track, "device": key,
                    "parameters": [{"index": p.index, "name": p.name, "value": p.value, "display": p.display}
                                   for p in params]}
            (out / _safe(track)).mkdir(exist_ok=True)
            (out / _safe(track) / f"{_safe(key)}.json").write_text(json.dumps(data, indent=2))
            try:
                specs = cfg.plugin(key).params
            except KeyError:
                specs = []
            resolved, _ = resolve_all(specs, params)
            if resolved:
                summary.append(f"\n### {track} - {cfg.plugin(key).search}\n")
                summary += [f"- **{r.param.name}**: {r.param.display or round(r.param.value, 3)}" for r in resolved]
    for key, idx in build.get("master", {}).items():
        ref = project.device_ref("Master", key)
        params = live.device_params(ref)
        (out / "Master").mkdir(exist_ok=True)
        (out / "Master" / f"{_safe(key)}.json").write_text(json.dumps(
            {"track": "Master", "device": key,
             "parameters": [{"index": p.index, "name": p.name, "value": p.value, "display": p.display}
                            for p in params]}, indent=2))

    ap = project.applied()
    mixer = {"gain_db": ap.get("gain_db", {}), "pan": ap.get("pan", {}), "eq": ap.get("eq", {}),
             "master": ap.get("master", {})}
    (cfg.outdir / "mixer.json").write_text(json.dumps(mixer, indent=2))

    lines = [
        "# Saving the matched sound",
        "",
        "The Live API cannot save plug-in presets or Live Sets, so do these by hand once the match is done:",
        "",
        "1. **Archetype: Gojira** (both guitar tracks): open the plug-in, *Save As* a preset "
        "(e.g. `TM <song> L` / `TM <song> R`).",
        "2. **Ample Hellrazer / Metal Eclipse**: save via the Ample preset menu.",
        "3. **Kontakt (MixWave Gojira, Rickenbacker)**: *Save Multi As* (keeps output routing) or save each "
        "instrument as a snapshot.",
        "4. **Each track's chain**: select all devices, Cmd/Ctrl+G to group into a Rack, then save the Rack to "
        "your User Library - that stores plug-in states together with EQ/Utility settings (.adg).",
        "5. **The whole set**: *File > Save Live Set As Template* to reuse the routing, chains and mix.",
        "",
        "## Mixer",
        "",
        "| Track | Utility gain (dB) | Pan |",
        "|---|---|---|",
    ]
    for t in sorted(set(mixer["gain_db"]) | set(mixer["pan"])):
        g = mixer["gain_db"].get(t)
        p = mixer["pan"].get(t)
        lines.append(f"| {t} | {'' if g is None else f'{g:+.1f}'} | {'' if p is None else f'{p:+.2f}'} |")
    lines += ["", "## Key plug-in settings", *summary, ""]
    (cfg.outdir / "SAVE_PRESETS.md").write_text("\n".join(lines))
    return cfg.outdir
