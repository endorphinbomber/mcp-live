"""Guitar Pro reader: .gp3/.gp4/.gp5 (via PyGuitarPro) and .gp (Guitar Pro 7/8, zipped XML).

Produces the same Song/Part/Note model as the MIDI reader, with Note.palm_mute taken from
the tab. Repeats, alternate endings and the common D.C./D.S./Coda/Fine jumps are unrolled
so the timeline matches a recording of the song.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from .midi import DRUM_CHANNEL, Note, Part, Song

QUARTER_TICKS = 960
DYNAMICS = {"PPP": 15, "PP": 31, "P": 47, "MP": 63, "MF": 79, "F": 95, "FF": 111, "FFF": 127}


# ------------------------------------------------------------------ intermediate model
@dataclass
class RawNote:
    offset: Fraction        # beats from bar start
    duration: Fraction      # beats
    pitch: int
    velocity: int
    palm_mute: bool
    tie: bool               # continues the previous note of the same pitch


@dataclass
class Bar:
    numerator: int = 4
    denominator: int = 4
    repeat_open: bool = False
    repeat_jumps: int = 0              # times to jump back at the end of this bar
    endings: set[int] = field(default_factory=set)   # alternate-ending pass numbers
    target: str = ""                   # sign at this bar: coda, doublecoda, segno, segnosegno, fine
    jump: str = ""                     # instruction at the end of this bar: dacapo, dasegnoalcoda, dacoda ...
    tempos: list[tuple[Fraction, float]] = field(default_factory=list)   # (offset beats, bpm)

    @property
    def length(self) -> Fraction:
        return Fraction(self.numerator * 4, self.denominator)


@dataclass
class RawTrack:
    name: str
    percussion: bool
    strings: list[int]
    channel: int
    bars: list[list[RawNote]]
    dead_notes: int = 0


def _norm_direction(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


# ------------------------------------------------------------------ unrolling
def _resolve_endings(bars: list[Bar]) -> list[int]:
    """Spread each alternate-ending mark over the bars it covers (up to its repeat close) and
    return, per bar, the highest ending number of its ending group (0 = not in an ending)."""
    n = len(bars)
    for k in range(n):
        if not bars[k].endings:
            continue
        j = k
        while j < n and bars[j].repeat_jumps == 0:
            if j + 1 < n and (bars[j + 1].endings or bars[j + 1].repeat_open):
                j = -1
                break
            j += 1
        if 0 <= j < n:  # the ending reaches a repeat close: all bars up to it belong to it
            for m in range(k + 1, j + 1):
                bars[m].endings = set(bars[k].endings)
    group_max = [0] * n
    k = 0
    while k < n:
        if bars[k].endings:
            j = k
            while j + 1 < n and bars[j + 1].endings and not bars[j + 1].repeat_open:
                j += 1
            top = max(max(bars[m].endings) for m in range(k, j + 1))
            for m in range(k, j + 1):
                group_max[m] = top
            k = j + 1
        else:
            k += 1
    return group_max


def play_order(bars: list[Bar], info: list[str]) -> list[int]:
    """Bar indices in playback order (repeats, alternate endings, D.C./D.S./To Coda/Fine).
    After a D.C./D.S. jump repeats are played once and only the last ending is taken."""
    n = len(bars)
    group_max = _resolve_endings(bars)
    targets: dict[str, int] = {}
    for i, b in enumerate(bars):
        if b.target:
            targets.setdefault(b.target, i)
    order: list[int] = []
    i, section_start, pass_no = 0, 0, 1
    jumps_done: dict[int, int] = {}
    used_jumps: set[int] = set()
    jumped, al = False, ""
    for _ in range(20000):
        if i >= n:
            return order
        b = bars[i]
        if b.repeat_open and not (i == section_start and pass_no > 1):
            section_start, pass_no = i, 1
        if b.endings:
            wanted = group_max[i] if jumped else pass_no
            if wanted not in b.endings:
                i += 1
                continue
        order.append(i)
        if jumped and al == "fine" and b.target == "fine":
            return order
        if b.repeat_jumps > 0 and not jumped and jumps_done.get(i, 0) < b.repeat_jumps:
            jumps_done[i] = jumps_done.get(i, 0) + 1
            pass_no += 1
            i = section_start
            continue
        if b.repeat_jumps > 0:
            jumps_done[i] = 0
            section_start, pass_no = i + 1, 1
        j = b.jump
        if jumped and j in ("dacoda", "dadoublecoda"):
            want = j[2:]
            if al == want and want in targets:
                i, al = targets[want], ""
                continue
        if j.startswith(("dacapo", "dasegno")) and i not in used_jumps:
            used_jumps.add(i)
            if j.startswith("dacapo"):
                dest, rest = 0, j[len("dacapo"):]
            else:
                sign = "segnosegno" if j.startswith("dasegnosegno") else "segno"
                if sign not in targets:
                    info.append(f"'{j}' without a {sign} sign - ignored")
                    i += 1
                    continue
                dest, rest = targets[sign], j[len("da" + sign):]
            al = rest.removeprefix("al")
            jumped = True
            i, section_start, pass_no = dest, dest, 1
            continue
        i += 1
    raise ValueError("Guitar Pro navigation does not terminate (check repeats/directions)")


def assemble(bars: list[Bar], tracks: list[RawTrack], info: list[str], fallback_bpm: float) -> Song:
    order = play_order(bars, info)
    if len(order) != len(bars):
        info.append(f"Unrolled repeats/jumps: {len(bars)} written bars -> {len(order)} played bars")
    starts: list[Fraction] = []
    t = Fraction(0)
    for bi in order:
        starts.append(t)
        t += bars[bi].length
    tempos: list[tuple[float, float]] = []
    for bi, st in zip(order, starts):
        for off, bpm in bars[bi].tempos:
            tempos.append((float(st + off), bpm))
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0.0, tempos[0][1] if tempos else fallback_bpm))
    dedup: list[tuple[float, float]] = []
    for b, bpm in sorted(tempos):
        if dedup and abs(dedup[-1][1] - bpm) < 1e-6:
            continue
        dedup.append((b, bpm))
    parts = []
    for tr in tracks:
        notes: list[Note] = []
        last_by_pitch: dict[int, Note] = {}
        for bi, st in zip(order, starts):
            for rn in sorted(tr.bars[bi], key=lambda r: (r.offset, r.pitch)):
                start = st + rn.offset
                prev = last_by_pitch.get(rn.pitch)
                if rn.tie and prev is not None:
                    prev.duration = float(start + rn.duration) - prev.start
                    continue
                note = Note(rn.pitch, float(start), float(rn.duration), rn.velocity, rn.palm_mute)
                notes.append(note)
                last_by_pitch[rn.pitch] = note
        if tr.dead_notes:
            info.append(f"{tr.name}: {tr.dead_notes} dead (x) notes skipped")
        if notes:
            notes.sort(key=lambda n: (n.start, n.pitch))
            parts.append(Part(name=tr.name, channel=DRUM_CHANNEL if tr.percussion else tr.channel,
                              notes=notes, is_percussion=tr.percussion, strings=tr.strings))
    first = bars[order[0]] if order else Bar()
    return Song(parts=parts, tempos=dedup, time_signature=(first.numerator, first.denominator),
                notes_info=info)


# ------------------------------------------------------------------ .gp3/.gp4/.gp5
def _load_gp345(path: Path) -> Song:
    try:
        import guitarpro
    except ImportError:
        raise RuntimeError("Reading .gp3/.gp4/.gp5 needs PyGuitarPro: pip install pyguitarpro") from None
    gp = guitarpro.parse(str(path))
    info: list[str] = []
    bars: list[Bar] = []
    for h in gp.measureHeaders:
        bars.append(Bar(
            numerator=h.timeSignature.numerator,
            denominator=h.timeSignature.denominator.value,
            repeat_open=h.isRepeatOpen,
            repeat_jumps=max(0, h.repeatClose),
            endings={k + 1 for k in range(8) if h.repeatAlternative & (1 << k)},
            target=_norm_direction(h.direction.name) if h.direction else "",
            jump=_norm_direction(h.fromDirection.name) if h.fromDirection else "",
        ))
    tracks: list[RawTrack] = []
    for tr in gp.tracks:
        strings = sorted(s.value for s in tr.strings)
        rt = RawTrack(name=tr.name.strip() or f"Track {tr.number}", percussion=tr.isPercussionTrack,
                      strings=[] if tr.isPercussionTrack else strings,
                      channel=tr.channel.channel if tr.channel else 0, bars=[[] for _ in bars])
        for bi, measure in enumerate(tr.measures):
            header = measure.header
            for voice in measure.voices:
                for beat in voice.beats:
                    off = Fraction(beat.start - header.start, QUARTER_TICKS)
                    mix = beat.effect.mixTableChange if beat.effect else None
                    if mix is not None and mix.tempo is not None and getattr(mix.tempo, "value", -1) > 0:
                        bars[bi].tempos.append((off, float(mix.tempo.value)))
                    dur = Fraction(beat.duration.time, QUARTER_TICKS)
                    for note in beat.notes:
                        kind = note.type.name
                        if kind == "rest":
                            continue
                        if kind == "dead":
                            rt.dead_notes += 1
                            continue
                        pitch = note.value if tr.isPercussionTrack else note.realValue + int(tr.offset or 0)
                        rt.bars[bi].append(RawNote(off, dur, pitch, int(note.velocity),
                                                   bool(note.effect.palmMute), kind == "tie"))
        tracks.append(rt)
    for off, bpm in [(Fraction(0), float(gp.tempo))]:
        if bars and not any(o == 0 for o, _ in bars[0].tempos):
            bars[0].tempos.insert(0, (off, bpm))
    return assemble(bars, tracks, info, float(gp.tempo))


# ------------------------------------------------------------------ .gp (GP7/8)
_NOTE_VALUES = {"Whole": Fraction(4), "Half": Fraction(2), "Quarter": Fraction(1), "Eighth": Fraction(1, 2),
                "16th": Fraction(1, 4), "32nd": Fraction(1, 8), "64th": Fraction(1, 16),
                "128th": Fraction(1, 32), "256th": Fraction(1, 64), "DoubleWhole": Fraction(8)}
_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _ids(text: str | None) -> list[int]:
    return [int(x) for x in (text or "").split()]


def _props(el: ET.Element | None) -> dict[str, ET.Element]:
    if el is None:
        return {}
    return {p.get("name"): p for p in el.findall("Properties/Property")}


def _prop_int(props: dict, name: str, child: str) -> int | None:
    p = props.get(name)
    if p is None:
        return None
    v = p.findtext(child)
    return int(v) if v is not None and v.strip().lstrip("-").isdigit() else None


def _rhythm_beats(r: ET.Element) -> Fraction:
    value = _NOTE_VALUES.get(r.findtext("NoteValue", "Quarter"), Fraction(1))
    dot = r.find("AugmentationDot")
    if dot is not None:
        dots = int(dot.get("count", "1"))
        value *= 2 - Fraction(1, 2 ** dots)
    for tag in ("PrimaryTuplet", "SecondaryTuplet"):
        tup = r.find(tag)
        if tup is not None:
            num, den = int(tup.get("num", "1")), int(tup.get("den", "1"))
            if num:
                value *= Fraction(den, num)
    return value


def _drum_articulations(track: ET.Element) -> list[int]:
    """Flattened OutputMidiNumber per articulation (GP7/8 drum kits)."""
    return [int(a.findtext("OutputMidiNumber", "0"))
            for a in track.findall(".//InstrumentSet/Elements/Element/Articulations/Articulation")]


def _load_gp7(path: Path) -> Song:
    with zipfile.ZipFile(path) as z:
        name = next((n for n in z.namelist() if n.endswith("score.gpif")), None)
        if name is None:
            raise ValueError(f"{path.name}: no score.gpif inside - not a Guitar Pro 7/8 file")
        root = ET.fromstring(z.read(name))
    info: list[str] = []
    rhythms = {r.get("id"): _rhythm_beats(r) for r in root.findall("Rhythms/Rhythm")}
    notes_el = {n.get("id"): n for n in root.findall("Notes/Note")}
    beats_el = {b.get("id"): b for b in root.findall("Beats/Beat")}
    voices_el = {v.get("id"): v for v in root.findall("Voices/Voice")}
    bars_el = {b.get("id"): b for b in root.findall("Bars/Bar")}

    bars: list[Bar] = []
    master_bar_refs: list[list[int]] = []
    for mb in root.findall("MasterBars/MasterBar"):
        num, den = (mb.findtext("Time", "4/4").split("/") + ["4"])[:2]
        rep = mb.find("Repeat")
        bar = Bar(numerator=int(num), denominator=int(den))
        if rep is not None:
            bar.repeat_open = rep.get("start") == "true"
            if rep.get("end") == "true":
                bar.repeat_jumps = max(0, int(rep.get("count", "2")) - 1)
        bar.endings = set(_ids(mb.findtext("AlternateEndings")))
        bar.target = _norm_direction(mb.findtext("Directions/Target", ""))
        bar.jump = _norm_direction(mb.findtext("Directions/Jump", ""))
        bars.append(bar)
        master_bar_refs.append(_ids(mb.findtext("Bars")))

    first_tempo = 120.0
    for auto in root.findall("MasterTrack/Automations/Automation"):
        if auto.findtext("Type") != "Tempo":
            continue
        bi = int(auto.findtext("Bar", "0"))
        pos = Fraction(auto.findtext("Position", "0")).limit_denominator(1000)
        bpm = float((auto.findtext("Value", "120") or "120").split()[0])
        if bi < len(bars):
            bars[bi].tempos.append((pos * bars[bi].length, bpm))
        if bi == 0 and pos == 0:
            first_tempo = bpm

    tracks: list[RawTrack] = []
    for ti, tr in enumerate(root.findall("Tracks/Track")):
        tname = (tr.findtext("Name") or f"Track {ti + 1}").strip()
        percussion = (tr.findtext("InstrumentSet/Type") or "").lower() == "drumkit" or \
            "drum" in (tr.findtext("InstrumentSet/Name") or "").lower()
        staff_props = _props(tr.find("Staves/Staff"))
        if not staff_props:
            staff_props = _props(tr)
        tuning_el = staff_props.get("Tuning")
        tuning = _ids(tuning_el.findtext("Pitches")) if tuning_el is not None else []
        capo = _prop_int(staff_props, "CapoFret", "Fret") or 0
        drum_map = _drum_articulations(tr) if percussion else []
        rt = RawTrack(name=tname, percussion=percussion, strings=[] if percussion else sorted(tuning),
                      channel=ti, bars=[[] for _ in bars])
        for bi, refs in enumerate(master_bar_refs):
            if ti >= len(refs) or str(refs[ti]) not in bars_el:
                continue
            bar_el = bars_el[str(refs[ti])]
            for vid in _ids(bar_el.findtext("Voices")):
                if vid < 0 or str(vid) not in voices_el:
                    continue
                off = Fraction(0)
                for bid in _ids(voices_el[str(vid)].findtext("Beats")):
                    beat = beats_el.get(str(bid))
                    if beat is None:
                        continue
                    if beat.find("GraceNotes") is not None:
                        continue  # grace notes take no time in the bar
                    rhythm = beat.find("Rhythm")
                    dur = rhythms.get(rhythm.get("ref") if rhythm is not None else None, Fraction(1))
                    velocity = DYNAMICS.get((beat.findtext("Dynamic") or "MF").upper(), 79)
                    for nid in _ids(beat.findtext("Notes")):
                        note = notes_el.get(str(nid))
                        if note is None:
                            continue
                        props = _props(note)
                        if "Muted" in props or "DeadNote" in props:
                            rt.dead_notes += 1
                            continue
                        pitch = _note_pitch(props, tuning, capo, drum_map, percussion)
                        if pitch is None:
                            continue
                        tie = note.find("Tie")
                        rt.bars[bi].append(RawNote(off, dur, pitch, velocity, "PalmMuted" in props,
                                                   tie is not None and tie.get("destination") == "true"))
                    off += dur
        tracks.append(rt)
    return assemble(bars, tracks, info, first_tempo)


def _note_pitch(props: dict, tuning: list[int], capo: int, drum_map: list[int], percussion: bool) -> int | None:
    midi = _prop_int(props, "Midi", "Number")
    if midi is not None:
        return midi
    if percussion:
        art = _prop_int(props, "InstrumentArticulation", "Number")
        if art is not None and art < len(drum_map):
            return drum_map[art]
        el = _prop_int(props, "Element", "Element")
        return drum_map[el] if el is not None and el < len(drum_map) else None
    string, fret = _prop_int(props, "String", "String"), _prop_int(props, "Fret", "Fret")
    if string is not None and fret is not None and string < len(tuning):
        return tuning[string] + capo + fret
    cp = props.get("ConcertPitch")
    if cp is not None:
        step = cp.findtext("Pitch/Step")
        octave = cp.findtext("Pitch/Octave")
        if step in _STEPS and octave is not None:
            acc = {"#": 1, "b": -1, "##": 2, "bb": -2}.get(cp.findtext("Pitch/Accidental") or "", 0)
            return 12 * (int(octave) + 1) + _STEPS[step] + acc
    return None


# ------------------------------------------------------------------ entry point
def load_gp(path: str | Path) -> Song:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".gpx":
        raise ValueError(f"{path.name}: Guitar Pro 6 (.gpx) files aren't supported. Open it in Guitar Pro "
                         "and save as .gp (GP7/8) or export as .gp5.")
    if ext == ".gp":
        return _load_gp7(path)
    return _load_gp345(path)
