"""MIDI file parsing: notes in beats, tempo map, part classification, humanizing."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import mido

DRUM_CHANNEL = 9  # GM channel 10
_ROLE_HINTS = {
    "drums": ("drum", "kit", "perc", "kick", "snare", "mixwave"),
    "bass": ("bass", "ricken"),
    "guitar": ("guitar", "gtr", "rhythm", "lead", "riff", "hellraz", "eclipse"),
}


@dataclass
class Note:
    pitch: int
    start: float      # beats
    duration: float   # beats
    velocity: int
    palm_mute: bool | None = None   # None = unknown (plain MIDI); Guitar Pro files set it

    def to_live(self) -> dict:
        return {
            "pitch": self.pitch,
            "start_time": round(self.start, 6),
            "duration": round(max(self.duration, 1e-3), 6),
            "velocity": int(min(127, max(1, self.velocity))),
            "mute": False,
        }


@dataclass
class Part:
    name: str
    channel: int
    notes: list[Note] = field(default_factory=list)
    ccs: list[tuple[float, int, int]] = field(default_factory=list)   # (beat, controller, value)
    is_percussion: bool = False
    strings: list[int] = field(default_factory=list)                   # open-string pitches (Guitar Pro)

    @property
    def end_beat(self) -> float:
        return max((n.start + n.duration for n in self.notes), default=0.0)

    @property
    def mean_pitch(self) -> float:
        return sum(n.pitch for n in self.notes) / len(self.notes) if self.notes else 0.0


@dataclass
class Song:
    parts: list[Part]
    tempos: list[tuple[float, float]]          # (beat, bpm)
    time_signature: tuple[int, int] = (4, 4)
    notes_info: list[str] = field(default_factory=list)   # loader remarks for `tonematch midi-info`

    @property
    def bpm(self) -> float:
        return self.tempos[0][1] if self.tempos else 120.0

    @property
    def end_beat(self) -> float:
        return max((p.end_beat for p in self.parts), default=0.0)

    def beats_to_seconds(self, beat: float) -> float:
        """Honour the tempo map (Live itself will play at the first tempo only)."""
        t, last_beat, bpm = 0.0, 0.0, self.bpm
        for b, new_bpm in self.tempos[1:]:
            if b >= beat:
                break
            t += (b - last_beat) * 60.0 / bpm
            last_beat, bpm = b, new_bpm
        return t + (beat - last_beat) * 60.0 / bpm

    def beats_per_bar(self) -> float:
        num, den = self.time_signature
        return num * 4.0 / den


def load_midi(path: str | Path) -> Song:
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    tempos: list[tuple[float, float]] = []
    time_sig = (4, 4)
    parts: dict[tuple[int, int], Part] = {}
    for ti, track in enumerate(mid.tracks):
        tick = 0
        open_notes: dict[tuple[int, int], tuple[int, int]] = {}
        name = track.name or f"Track {ti + 1}"
        for msg in track:
            tick += msg.time
            beat = tick / tpb
            if msg.type == "set_tempo":
                tempos.append((beat, round(mido.tempo2bpm(msg.tempo), 4)))
            elif msg.type == "time_signature" and tick == 0:
                time_sig = (msg.numerator, msg.denominator)
            elif msg.type == "control_change":
                cc_part = parts.setdefault((ti, msg.channel), Part(name=name, channel=msg.channel))
                cc_part.ccs.append((beat, msg.control, msg.value))
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes[(msg.channel, msg.note)] = (tick, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                start = open_notes.pop((msg.channel, msg.note), None)
                if start is None:
                    continue
                part = parts.setdefault((ti, msg.channel), Part(name=name, channel=msg.channel))
                part.notes.append(Note(msg.note, start[0] / tpb, (tick - start[0]) / tpb, start[1]))
    tempos.sort()
    dedup: list[tuple[float, float]] = []
    for b, bpm in tempos:
        if dedup and abs(dedup[-1][1] - bpm) < 1e-6:
            continue
        dedup.append((b, bpm))
    for p in parts.values():
        p.notes.sort(key=lambda n: (n.start, n.pitch))
    return Song(parts=[p for p in parts.values() if p.notes], tempos=dedup or [(0.0, 120.0)],
                time_signature=time_sig)


GP_EXTENSIONS = (".gp3", ".gp4", ".gp5", ".gp", ".gpx")


def load_song(path: str | Path) -> Song:
    """MIDI or Guitar Pro (.gp3/.gp4/.gp5/.gp), chosen by file extension."""
    if Path(path).suffix.lower() in GP_EXTENSIONS:
        from .gpfile import load_gp
        return load_gp(path)
    return load_midi(path)


def classify_parts(song: Song) -> dict[str, Part]:
    """Map roles (drums/bass/guitar) to parts by channel, track name, then register."""
    roles: dict[str, Part] = {}
    remaining = list(song.parts)
    for p in remaining:
        if p.is_percussion or p.channel == DRUM_CHANNEL or any(h in p.name.lower() for h in _ROLE_HINTS["drums"]):
            roles.setdefault("drums", p)
    remaining = [p for p in remaining if p is not roles.get("drums")]
    for role in ("bass", "guitar"):
        for p in remaining:
            if any(h in p.name.lower() for h in _ROLE_HINTS[role]):
                roles.setdefault(role, p)
        if role == "bass" and "bass" not in roles:
            # Guitar Pro: a 4-6 string instrument tuned down to E1 or lower is a bass.
            for p in remaining:
                if p.strings and len(p.strings) <= 6 and min(p.strings) <= 28:
                    roles.setdefault("bass", p)
        remaining = [p for p in remaining if p is not roles.get(role)]
    remaining.sort(key=lambda p: p.mean_pitch)
    if "bass" not in roles and remaining:
        roles["bass"] = remaining.pop(0)
    if "guitar" not in roles and remaining:
        roles["guitar"] = max(remaining, key=lambda p: len(p.notes))
    return roles


def humanize(notes: list[Note], bpm: float, timing_ms: float = 0.0, velocity: int = 0,
             seed: int = 0) -> list[Note]:
    """Return a copy with per-note timing/velocity jitter, for a believable double-track.

    A slowly drifting offset (random walk) plus small per-note noise sounds like a
    player, rather than pure white jitter which sounds like flamming.
    """
    rng = random.Random(seed)
    ms_to_beats = bpm / 60000.0
    drift = 0.0
    out = []
    for n in notes:
        drift = max(-timing_ms, min(timing_ms, drift + rng.gauss(0, timing_ms / 4)))
        jitter = rng.gauss(0, timing_ms / 6)
        shift = (drift * 0.7 + jitter) * ms_to_beats
        out.append(replace(
            n,
            start=max(0.0, n.start + shift),
            velocity=int(min(127, max(1, n.velocity + rng.randint(-velocity, velocity)))),
        ))
    return out


# ------------------------------------------------------------------ palm mutes
# Ample guitars play a note palm-muted when its velocity is below a threshold (40).
PM_MODES = ("auto", "file", "velocity", "cc", "heuristic", "none")


def parse_bars(spec: str | list | None) -> list[tuple[int, int]]:
    """'9-16, 33' -> [(9, 16), (33, 33)] (1-based, inclusive)."""
    if not spec:
        return []
    items = spec if isinstance(spec, list) else re.split(r"[,\s]+", str(spec).strip())
    out = []
    for item in items:
        item = str(item).strip()
        if not item:
            continue
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", item)
        if not m:
            raise ValueError(f"Bad bar range {item!r}; use e.g. \"9-16, 33\"")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        out.append((min(a, b), max(a, b)))
    return out


def _in_bars(beat: float, ranges: list[tuple[int, int]], beats_per_bar: float) -> bool:
    bar = int(beat // beats_per_bar) + 1
    return any(a <= bar <= b for a, b in ranges)


def _heuristic_flags(notes: list[Note], opts: dict) -> list[bool]:
    """Chugs: runs of short, low notes (single notes or small chords) are palm-muted;
    sustained notes, big chords and anything higher up stay open."""
    if not notes:
        return []
    max_len = float(opts.get("max_len", 0.5))
    low_limit = min(n.pitch for n in notes) + int(opts.get("low_range", 7))
    min_run = int(opts.get("min_run", 3))
    max_gap = float(opts.get("max_gap", 1.0))
    groups: dict[float, list[int]] = {}
    for i, n in enumerate(notes):
        groups.setdefault(round(n.start, 4), []).append(i)
    starts = sorted(groups)
    chug = [all(notes[i].duration <= max_len for i in groups[t]) and len(groups[t]) <= 3
            and min(notes[i].pitch for i in groups[t]) <= low_limit for t in starts]
    flags = [False] * len(notes)
    run: list[float] = []

    def close():
        if len(run) >= min_run:
            for t in run:
                for i in groups[t]:
                    flags[i] = True
        run.clear()

    for t, is_chug in zip(starts, chug):
        if is_chug and (not run or t - run[-1] <= max_gap):
            run.append(t)
        else:
            close()
            if is_chug:
                run.append(t)
    close()
    return flags


def _cc_flags(notes: list[Note], ccs: list[tuple[float, int, int]], controller: int) -> list[bool]:
    events = sorted((b, v) for b, c, v in ccs if c == controller)
    flags = []
    for n in notes:
        value = 0
        for b, v in events:
            if b > n.start + 1e-6:
                break
            value = v
        flags.append(value >= 64)
    return flags


def mark_palm_mutes(notes: list[Note], opts: dict, beats_per_bar: float,
                    ccs: list[tuple[float, int, int]] | None = None) -> tuple[list[Note], str]:
    """Decide palm mute per note (sets Note.palm_mute). Returns (notes, mode actually used)."""
    mode = str(opts.get("mode", "auto")).lower()
    if mode not in PM_MODES:
        raise ValueError(f"palm_mute mode must be one of {PM_MODES}")
    threshold = int(opts.get("threshold", 40))
    ccs = ccs or []
    controller = int(opts.get("cc", 64))
    if mode == "auto":
        if any(n.palm_mute is not None for n in notes):
            mode = "file"
        elif any(c == controller for _, c, _ in ccs) and "cc" in opts:
            mode = "cc"
        elif any(n.velocity < threshold for n in notes) and any(n.velocity >= threshold for n in notes):
            mode = "velocity"
        else:
            mode = "heuristic"
    if mode == "file":
        flags = [bool(n.palm_mute) for n in notes]
    elif mode == "velocity":
        below = int(opts.get("below", threshold))
        flags = [n.velocity < below for n in notes]
    elif mode == "cc":
        flags = _cc_flags(notes, ccs, controller)
    elif mode == "heuristic":
        flags = _heuristic_flags(notes, opts)
    else:
        flags = [False] * len(notes)
    pm_bars, open_bars = parse_bars(opts.get("pm_bars")), parse_bars(opts.get("open_bars"))
    out = []
    for n, f in zip(notes, flags):
        if _in_bars(n.start, open_bars, beats_per_bar):
            f = False
        elif _in_bars(n.start, pm_bars, beats_per_bar):
            f = True
        out.append(replace(n, palm_mute=f))
    return out, mode


def palm_mute_velocities(notes: list[Note], opts: dict) -> list[Note]:
    """Palm-muted notes get `velocity` (default 20); open notes are kept at or above
    `open_min` (default 64, never below the threshold) so nothing mutes by accident."""
    threshold = int(opts.get("threshold", 40))
    pm_vel = int(opts.get("velocity", 20))
    open_min = max(threshold, int(opts.get("open_min", 64)))
    if not 1 <= pm_vel < threshold:
        raise ValueError(f"palm_mute velocity must be between 1 and {threshold - 1}")
    return [replace(n, velocity=pm_vel) if n.palm_mute else replace(n, velocity=max(n.velocity, open_min))
            for n in notes]
