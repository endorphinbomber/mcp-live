"""Build small Guitar Pro files for tests (a .gp5 via PyGuitarPro and a GP7-style .gp zip)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import guitarpro as gp

DROP_C = [62, 57, 53, 48, 43, 36]      # strings 1..6 (high -> low), Gojira-style drop C
BASS_DROP_C = [48, 43, 38, 31]          # 4-string bass in drop C... (G2 D2 A1 G1 shifted) - low string <= E1 below
BASS_TUNING = [43, 38, 33, 28]          # standard 4-string bass E1 A1 D2 G2


def _measure_beats(measure, events):
    """events: list of (duration value, [(string, fret, palm_mute, tie)], velocity) ; empty notes = rest."""
    voice = measure.voices[0]
    for dur, notes, vel in events:
        beat = gp.Beat(voice, duration=gp.Duration(value=dur),
                       status=gp.BeatStatus.normal if notes else gp.BeatStatus.rest)
        for string, fret, pm, tie in notes:
            note = gp.Note(beat, value=fret, string=string, velocity=vel,
                           type=gp.NoteType.tie if tie else gp.NoteType.normal)
            note.effect.palmMute = pm
            beat.notes.append(note)
        voice.beats.append(beat)


def write_gp5(path: Path, tempo: int = 120) -> Path:
    """4 written bars: |: A | B :|  with endings - bar 3 is ending 1 (closes the repeat), bar 4 ending 2.
    Guitar: bar 1 = 8 palm-muted chugs on the low string; bar 2 = open power chord (whole, tied into
    nothing) ; bar 3 = 4 quarter open notes; bar 4 = half note + tied half.
    Bass: one whole note per bar.  Drums: kick on every quarter.
    Played order: 1 2 3 1 2 4  -> 6 bars."""
    song = gp.Song()
    song.tempo = tempo
    song.measureHeaders = []
    start = 960
    for k in range(4):
        h = gp.MeasureHeader(number=k + 1, start=start,
                             timeSignature=gp.TimeSignature(numerator=4, denominator=gp.Duration(value=4)))
        song.measureHeaders.append(h)
        start += h.length
    song.measureHeaders[0].isRepeatOpen = True
    song.measureHeaders[2].repeatAlternative = 1      # ending 1
    song.measureHeaders[2].repeatClose = 1            # jump back once
    song.measureHeaders[3].repeatAlternative = 2      # ending 2

    def track(number, name, tuning, percussion=False):
        t = gp.Track(song, number=number, name=name, isPercussionTrack=percussion,
                     strings=[gp.GuitarString(i + 1, v) for i, v in enumerate(tuning)])
        t.channel = gp.MidiChannel(channel=9 if percussion else number - 1, effectChannel=number)
        t.measures = [gp.Measure(t, h) for h in song.measureHeaders]
        return t

    gtr = track(1, "Rhythm Guitar", DROP_C)
    _measure_beats(gtr.measures[0], [(8, [(6, 0, True, False)], 95)] * 8)
    _measure_beats(gtr.measures[1], [(1, [(6, 0, False, False), (5, 2, False, False)], 111)])
    _measure_beats(gtr.measures[2], [(4, [(4, 2, False, False)], 95)] * 4)
    _measure_beats(gtr.measures[3], [(2, [(5, 0, False, False)], 95), (2, [(5, 0, False, True)], 95)])

    bass = track(2, "Bass", BASS_TUNING)
    for m in bass.measures:
        _measure_beats(m, [(1, [(4, 0, False, False)], 95)])

    drums = track(3, "Drums", [0] * 6, percussion=True)
    for m in drums.measures:
        _measure_beats(m, [(4, [(1, 36, False, False)], 111)] * 4)   # value = MIDI note on drum tracks

    song.tracks = [gtr, bass, drums]
    gp.write(song, str(path))
    return path


GPIF = """<?xml version="1.0" encoding="utf-8"?>
<GPIF>
  <MasterTrack><Tracks>0</Tracks><Automations>
    <Automation><Type>Tempo</Type><Linear>false</Linear><Bar>0</Bar><Position>0</Position><Value>100 2</Value></Automation>
    <Automation><Type>Tempo</Type><Linear>false</Linear><Bar>2</Bar><Position>0</Position><Value>150 2</Value></Automation>
  </Automations></MasterTrack>
  <Tracks>
    <Track id="0"><Name>Lead Guitar</Name>
      <Staves><Staff><Properties>
        <Property name="Tuning"><Pitches>36 43 48 53 57 62</Pitches></Property>
        <Property name="CapoFret"><Fret>0</Fret></Property>
      </Properties></Staff></Staves>
    </Track>
  </Tracks>
  <MasterBars>
    <MasterBar><Time>4/4</Time><Repeat start="true" end="false" count="0"/><Bars>0</Bars></MasterBar>
    <MasterBar><Time>4/4</Time><Repeat start="false" end="true" count="3"/><Bars>1</Bars></MasterBar>
    <MasterBar><Time>3/4</Time><Bars>2</Bars></MasterBar>
  </MasterBars>
  <Bars>
    <Bar id="0"><Voices>0 -1 -1 -1</Voices></Bar>
    <Bar id="1"><Voices>1 -1 -1 -1</Voices></Bar>
    <Bar id="2"><Voices>2 -1 -1 -1</Voices></Bar>
  </Bars>
  <Voices>
    <Voice id="0"><Beats>0 0 0 0 0 0 0 0</Beats></Voice>
    <Voice id="1"><Beats>1 2</Beats></Voice>
    <Voice id="2"><Beats>3 4</Beats></Voice>
  </Voices>
  <Beats>
    <Beat id="0"><Dynamic>F</Dynamic><Rhythm ref="0"/><Notes>0</Notes></Beat>
    <Beat id="1"><Dynamic>FF</Dynamic><Rhythm ref="1"/><Notes>1 2</Notes></Beat>
    <Beat id="2"><Dynamic>FF</Dynamic><Rhythm ref="1"/><Notes>3</Notes></Beat>
    <Beat id="3"><Dynamic>MF</Dynamic><Rhythm ref="2"/><Notes>4</Notes></Beat>
    <Beat id="4"><Dynamic>MF</Dynamic><Rhythm ref="3"/></Beat>
  </Beats>
  <Notes>
    <Note id="0"><Properties><Property name="String"><String>0</String></Property><Property name="Fret"><Fret>0</Fret></Property><Property name="PalmMuted"><Enable/></Property></Properties></Note>
    <Note id="1"><Properties><Property name="String"><String>0</String></Property><Property name="Fret"><Fret>0</Fret></Property></Properties></Note>
    <Note id="2"><Properties><Property name="String"><String>1</String></Property><Property name="Fret"><Fret>0</Fret></Property></Properties></Note>
    <Note id="3"><Tie origin="false" destination="true"/><Properties><Property name="String"><String>0</String></Property><Property name="Fret"><Fret>0</Fret></Property></Properties></Note>
    <Note id="4"><Properties><Property name="String"><String>2</String></Property><Property name="Fret"><Fret>3</Fret></Property><Property name="Muted"><Enable/></Property></Properties></Note>
  </Notes>
  <Rhythms>
    <Rhythm id="0"><NoteValue>Eighth</NoteValue></Rhythm>
    <Rhythm id="1"><NoteValue>Half</NoteValue></Rhythm>
    <Rhythm id="2"><NoteValue>Quarter</NoteValue><AugmentationDot count="1"/></Rhythm>
    <Rhythm id="3"><NoteValue>Quarter</NoteValue><AugmentationDot count="1"/></Rhythm>
  </Rhythms>
</GPIF>"""


def write_gp7(path: Path) -> Path:
    """|: 8 palm-muted 8ths | half chord + tied half :| x3 , then a 3/4 bar with a dead note + rest.
    Played: 1 2 1 2 1 2 3 ; tempo 100 then 150 from written bar 3."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("VERSION", "7.0")
        z.writestr("Content/score.gpif", GPIF)
    return path
