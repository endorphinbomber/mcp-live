"""Command line: tonematch init|doctor|analyze|build|discover|match|export|report|reapply."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, load_config
from .live.client import LiveClient, LiveError

CONFIG_NAME = "tonematch.toml"


def _project(args):
    from .pipeline import Project
    cfg = load_config(args.config if Path(args.config).exists() else None)
    return Project(cfg, LiveClient(port=args.port))


def cmd_init(args) -> int:
    dest = Path(args.config)
    if dest.exists():
        print(f"{dest} already exists")
        return 1
    shutil.copy(DEFAULT_CONFIG_PATH, dest)
    print(f"Wrote {dest}. Set [project].reference and .midi, then run `tonematch analyze`.")
    return 0


def cmd_doctor(args) -> int:
    ok = True
    try:
        with LiveClient(port=args.port) as c:
            info = c.send("get_session_info")
            print(f"Live {info.get('live_version')} / bridge {info.get('bridge_version')}: "
                  f"{info.get('track_count')} tracks, {info.get('tempo')} BPM")
            from .config import load_config
            from .live.session import LiveSession
            cfg = load_config(args.config if Path(args.config).exists() else None)
            live = LiveSession(c)
            for key in sorted({d for t in cfg.tracks for d in t.device_names}):
                spec = cfg.plugin(key)
                if spec.is_stock:
                    continue
                try:
                    uri = live.find_browser_item(spec.search, stock=False, uri=spec.uri)
                    print(f"  plug-in '{spec.search}': found ({uri})")
                except LiveError as e:
                    print(f"  plug-in '{spec.search}': NOT FOUND - {e}")
                    ok = False
    except LiveError as e:
        print(e)
        return 2
    for mod in ("demucs",):
        try:
            __import__(mod)
            print(f"  {mod}: installed")
        except ImportError:
            print(f"  {mod}: not installed (pip install 'tonematch[separate]')")
    return 0 if ok else 1


def cmd_plugins(args) -> int:
    from .live.session import LiveSession
    with LiveClient(port=args.port) as c:
        items = LiveSession(c).walk_plugins(args.query or "")
    if not items:
        print("Live's Plug-ins browser shows nothing" + (f" matching '{args.query}'" if args.query else "")
              + ". Check Settings > Plug-Ins (VST3 system folders / VST2 custom folder) and Rescan.")
        return 1
    for i in items:
        print(f"{i['path']}\n    uri = \"{i['uri']}\"")
    print("\nPut the name in `search = \"...\"` (or the exact `uri = \"...\"`) under [plugins.<name>] in tonematch.toml.")
    return 0


def cmd_midi_info(args) -> int:
    from .midi import classify_parts, load_song, mark_palm_mutes
    cfg = load_config(args.config if Path(args.config).exists() else None)
    path = Path(args.file) if args.file else cfg.score_path
    song = load_song(path)
    roles = {id(p): r for r, p in classify_parts(song).items()}
    tempos = ", ".join(f"{bpm:g} BPM @ beat {b:g}" for b, bpm in song.tempos[:6])
    print(f"{path.name}: {len(song.parts)} parts, {song.end_beat / song.beats_per_bar():.0f} bars, "
          f"{song.time_signature[0]}/{song.time_signature[1]}, {tempos}")
    for line in song.notes_info:
        print(f"  note: {line}")
    for p in song.parts:
        vels = [n.velocity for n in p.notes]
        pitches = [n.pitch for n in p.notes]
        role = roles.get(id(p), "-")
        print(f"\n[{role}] {p.name} (channel {p.channel + 1}): {len(p.notes)} notes, pitch {min(pitches)}-{max(pitches)}, "
              f"velocity {min(vels)}-{max(vels)}")
        if p.strings:
            print(f"  tuning (low->high): {p.strings}")
        if p.ccs:
            ccs = sorted({c for _, c, _ in p.ccs})
            print(f"  controllers used: {ccs}")
        if role in ("guitar",):
            for mode in ("auto", "file", "velocity", "heuristic"):
                if mode == "file" and not any(n.palm_mute is not None for n in p.notes):
                    continue
                marked, used = mark_palm_mutes(p.notes, {"mode": mode}, song.beats_per_bar(), p.ccs)
                n_pm = sum(1 for n in marked if n.palm_mute)
                label = f"auto -> {used}" if mode == "auto" else mode
                print(f"  palm mutes with mode {label:<16} {n_pm:5d} of {len(marked)}")
    return 0


def cmd_analyze(args) -> int:
    _project(args).analyze()
    return 0


def cmd_build(args) -> int:
    todo = _project(args).build()
    print("\nManual steps:")
    for i, t in enumerate(todo, 1):
        print(f"  {i}. {t}")
    return 0


def cmd_discover(args) -> int:
    report = _project(args).discover()
    for dev, r in report.items():
        print(f"\n{dev}: {r.get('exposed', '?')} parameters exposed")
        for k, v in r.get("resolved", {}).items():
            extra = f"  options: {r['choices'][k]}" if k in r.get("choices", {}) else ""
            print(f"  {k:<10} -> {v}{extra}")
        for k in r.get("missing", []):
            print(f"  {k:<10} -> MISSING (fix its pattern in tonematch.toml, or Configure it in Live)")
    print("\nFull parameter dumps: work/params/*.json")
    return 0


def cmd_match(args) -> int:
    stages = args.stages.split(",") if args.stages else None
    p = _project(args)
    if args.trials:
        p.cfg.match["tone_trials"] = args.trials
    p.match(stages)
    from .export.report import write_report
    print(f"Report: {write_report(p)}")
    return 0


def cmd_export(args) -> int:
    from .export.presets import export_presets
    from .export.report import write_report
    p = _project(args)
    out = export_presets(p)
    write_report(p)
    print(f"Presets, mixer sheet and SAVE_PRESETS.md in {out}")
    return 0


def cmd_report(args) -> int:
    from .export.report import write_report
    print(write_report(_project(args)))
    return 0


def cmd_reapply(args) -> int:
    _project(args).reapply(Path(args.snapshot) if args.snapshot else None)
    print("Settings pushed to Live")
    return 0


def cmd_status(args) -> int:
    p = _project(args)
    st = p.state.data
    print(json.dumps({k: st[k] for k in ("analysis", "applied") if k in st}, indent=1, default=str)[:4000])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tonematch", description=__doc__)
    ap.add_argument("-c", "--config", default=CONFIG_NAME)
    ap.add_argument("--port", type=int, default=9877, help="AbletonMCP Remote Script port")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="write a tonematch.toml to edit").set_defaults(fn=cmd_init)
    sub.add_parser("doctor", help="check Live connection and plug-ins").set_defaults(fn=cmd_doctor)
    pl = sub.add_parser("plugins", help="list Live's Plug-ins browser (optionally filtered)")
    pl.add_argument("query", nargs="?")
    pl.set_defaults(fn=cmd_plugins)
    mi = sub.add_parser("midi-info", help="show parts, roles and palm-mute counts of the MIDI/Guitar Pro file")
    mi.add_argument("file", nargs="?")
    mi.set_defaults(fn=cmd_midi_info)
    sub.add_parser("analyze", help="separate stems, align MIDI, measure the reference").set_defaults(fn=cmd_analyze)
    sub.add_parser("build", help="create tracks/devices/MIDI in Live").set_defaults(fn=cmd_build)
    sub.add_parser("discover", help="list plug-in parameters and pattern matches").set_defaults(fn=cmd_discover)
    m = sub.add_parser("match", help="run matching stages (closed loop, real-time takes)")
    m.add_argument("--stages", help="comma list: levels,tone,eq,pan,master (default: full sequence)")
    m.add_argument("--trials", type=int, help="override tone_trials")
    m.set_defaults(fn=cmd_match)
    sub.add_parser("export", help="write presets, mixer sheet and save checklist").set_defaults(fn=cmd_export)
    sub.add_parser("report", help="write out/report.html").set_defaults(fn=cmd_report)
    r = sub.add_parser("reapply", help="push stored settings (or a history snapshot) back into Live")
    r.add_argument("snapshot", nargs="?")
    r.set_defaults(fn=cmd_reapply)
    sub.add_parser("status", help="print analysis and applied settings").set_defaults(fn=cmd_status)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (LiveError, ValueError, RuntimeError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
