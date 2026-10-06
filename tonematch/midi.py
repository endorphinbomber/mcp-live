"""MIDI file parsing: notes in beats, tempo map, part classification, humanizing."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
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


def classify_parts(song: Song) -> dict[str, Part]:
    """Map roles (drums/bass/guitar) to parts by channel, track name, then register."""
    roles: dict[str, Part] = {}
    remaining = list(song.parts)
    for p in remaining:
        if p.channel == DRUM_CHANNEL or any(h in p.name.lower() for h in _ROLE_HINTS["drums"]):
            roles.setdefault("drums", p)
    remaining = [p for p in remaining if p is not roles.get("drums")]
    for role in ("bass", "guitar"):
        for p in remaining:
            if any(h in p.name.lower() for h in _ROLE_HINTS[role]):
                roles.setdefault(role, p)
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
        out.append(Note(
            pitch=n.pitch,
            start=max(0.0, n.start + shift),
            duration=n.duration,
            velocity=int(min(127, max(1, n.velocity + rng.randint(-velocity, velocity)))),
        ))
    return out
