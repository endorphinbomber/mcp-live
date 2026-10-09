# tonematch

Closed-loop tone and mix matching in **Ableton Live**: give it a reference song (MP3) and a MIDI
version of it, and it builds a project with your plug-ins, then dials in amp settings, levels, EQ,
panning and master loudness so the MIDI rendition sounds as close as possible to the reference.

It drives Live through the Remote Script from
[wstierhout/ableton-live-mcp](https://github.com/wstierhout/ableton-live-mcp) (same socket, port 9877).

```
reference.mp3 ──► Demucs stems (drums / bass / guitar) ──► per-stem targets (spectrum, LUFS, width, dynamics)
song.mid ───────► aligned to the MP3 ──► comparison regions (densest 8-bar sections)
                                                   │
Live: build tracks + plug-ins + MIDI ◄─────────────┘
   └─► loop: set parameters ─► record the region (real time) ─► measure ─► compare ─► adjust
```

## Default template

| Track | Chain | Notes |
|---|---|---|
| **Drums** | Kontakt (MixWave Gojira) → EQ Eight → Glue Compressor → Utility | optional Kontakt multi-outs (Kick/Snare/OH/Room...), whose balance is then optimized too |
| **Bass** | Kontakt (NI Rickenbacker bass) → Pedal → EQ Eight → Compressor → Utility | |
| **Gtr L** | Ample Hellrazer → Archetype: Gojira → EQ Eight → Utility | the single MIDI guitar part |
| **Gtr R** | Ample Metal Eclipse → Archetype: Gojira → EQ Eight → Utility | same part, humanized (±12 ms drift, velocity jitter) - a different guitar on each side makes a convincing double |
| **Master** | EQ Eight → Glue Compressor → Limiter | |

Faders stay at 0 dB; all gain moves happen on each track's Utility, so the numbers are exact and
the faders are free for you.

## Setup

1. **Remote Script** (once): `uvx mcp-server-ableton-live install`, restart Live, then
   *Settings → Tempo & MIDI → Control Surface → AbletonMCP* (Input/Output: None).
2. **tonematch** (Python 3.11+):
   ```bash
   pip install -e ".[separate]"     # [separate] pulls in Demucs/PyTorch for stem separation
   ```
   Decoding MP3 uses libsndfile, falling back to `ffmpeg` if it's on your PATH.
3. In a folder with your files:
   ```bash
   tonematch init           # writes tonematch.toml - set reference = "song.mp3", midi = "song.mid" (or song.gp5)
   tonematch doctor         # checks Live is reachable and the plug-ins are found
   ```

## Workflow

```bash
tonematch analyze     # separate stems, align MIDI <-> MP3, pick regions, measure the reference
tonematch build       # create tracks, load plug-ins, write MIDI clips, set pans (prints manual steps)
```

Then do the manual steps `build` prints. The Live API can't do these:

- Open each **Kontakt** and load the library (MixWave Gojira on *Drums*, the Rickenbacker on *Bass*).
  For multi-outs, set Kontakt's outputs and list them under `outputs` on the Drums track in `tonematch.toml`.
- Click **Configure** on Archetype: Gojira (and the Ample plug-ins if needed) and touch the amp selector,
  gain/EQ knobs, cab/mic and gate so Live exposes them.

```bash
tonematch discover    # shows which configured knobs map to real plug-in parameters
```

`discover` writes every exposed parameter to `work/params/*.json`. If something shows `MISSING`, fix
its `pattern` (a case-insensitive regex) in `tonematch.toml`. In patterns, `{amp}` stands for the label of
the amp the optimizer picked, so only the active amp's knobs get tuned.

```bash
tonematch match                 # full sequence: levels, tone, levels, eq, pan, levels, master
tonematch match --stages levels # quick sanity check (2-3 takes)
tonematch match --stages tone --trials 60
tonematch export                # presets + mixer sheet + SAVE_PRESETS.md + report.html
```

What each stage does:

| Stage | How |
|---|---|
| `levels` | Records every track, sets each Utility so its LUFS hits the reference stem minus `headroom_db` (doubles share a stem at −3 dB each). Repeats until within 0.5 dB. |
| `tone` | One Optuna (TPE) search per track, run side by side: every take sets all tracks' candidates and records them in **one** real-time pass. Each amp option is tried first, then the search narrows in. Loss = third-octave spectral-shape distance + distortion density (spectral flatness) + crest/dynamics. |
| `eq` | Fits the remaining spectral difference to up to 6 EQ Eight bands per track (±6 dB, least squares). Re-measures and reverts any track it made worse. |
| `pan` | Doubles: solves the pan that reproduces the reference guitar stem's side/mid ratio. Single sources: pan from L/R balance. |
| `master` | Master EQ Eight (gentle, ±3 dB), then Limiter gain until the mix matches the reference loudness. |

Every stage saves a snapshot under `work/history/`. `tonematch reapply work/history/<file>.json` pushes
any earlier state back into Live.

### Time budget

Takes are recorded in real time. One 8-bar region at 120 BPM is 16 s, so 40 tone trials take about
11 minutes, and a full `match` usually takes 20-40 minutes. You can leave Live alone while it runs,
but don't touch the transport.

### Saving presets and the template

The Live API can't save plug-in presets or Live Sets. `export` writes `out/SAVE_PRESETS.md` with the
final values and the steps: save Archetype/Ample presets in the plug-ins, group each chain into a Rack and
save it (`.adg`, which keeps plug-in states), and save the set with *File → Save Live Set As Template*.

## Using it from Claude

Register both MCP servers (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ableton": { "command": "uvx", "args": ["mcp-server-ableton-live@1.8.1"] },
    "tonematch": {
      "command": "python",
      "args": ["-m", "tonematch.mcp_server"],
      "env": { "TONEMATCH_CONFIG": "/path/to/project/tonematch.toml" }
    }
  }
}
```

Claude can then run `tonematch_analyze` → `tonematch_build` → `tonematch_discover` → `tonematch_match`,
and use the ableton tools for anything by hand. Needs `pip install -e ".[mcp]"`.

## Guitar Pro files and palm mutes

**A Guitar Pro file is all you need; use MIDI only if you have no tab.** Leave `score = ""` and put the
`.gp5`/`.gp` file in the project folder, or set `score = "song.gp5"`. If a single Guitar Pro file sits next
to a MIDI file, tonematch uses the Guitar Pro file. Set `score = "song.mid"` to force MIDI. `analyze` and
`build` print which file they use. If you switch files, run `tonematch analyze` again. Supported formats are `.gp3`, `.gp4`, `.gp5` and `.gp` (Guitar Pro 7/8). For `.gpx`
(Guitar Pro 6), open the file in Guitar Pro and save it as `.gp`, or export it as `.gp5`.

From a Guitar Pro file, tonematch reads:
- each track's notes, with pitch from its tuning and capo, and ties merged
- dynamics as velocity
- tempo changes
- palm mutes, exactly as they're marked in the tab

It also unrolls repeats, alternate endings and the common D.C./D.S./To Coda/Fine jumps so the timeline
matches the recording. Dead (x) notes are skipped. Drum tracks are recognised as drums, and an
instrument tuned to E1 or lower is treated as the bass.

**Palm mutes.** The Ample guitars play a note palm-muted when its velocity is below 40. On the guitar
tracks, tonematch sets palm-muted notes to velocity 20 and keeps open notes at 64 or above, so the
humanized double can't mute a note by accident. Where the palm mutes come from depends on `mode`:
- `auto` (default): the tab's marks if the file has them. Otherwise, MIDI velocities if the part
  already separates muted from open notes. Otherwise a guess from the riffs: runs of short, low notes.
- `file`, `velocity`, `cc` (with `cc = 64`), `heuristic` or `none` choose one source explicitly.
- `pm_bars = "9-16, 33-40"` / `open_bars = "41-48"` force bar ranges either way.

```toml
palm_mute = { mode = "auto", velocity = 20, threshold = 40, open_min = 64 }
```

**Transposing.** `transpose = 12` on a track shifts all of its notes up an octave (negative numbers
shift down). The default config does this for the Bass, because the Rickenbacker plays an octave
higher than the tab or MIDI.

`tonematch midi-info` shows each part's role, tuning and how many notes each mode would palm-mute.
Check it before `build`.

## Troubleshooting

**"'Kontakt' not found in Live's browser"**: run `tonematch plugins kontakt`. It lists what Live's
Plug-ins browser actually contains, with each item's path and URI.
- **Nothing listed:** Live hasn't scanned the plug-in. Go to Settings → Plug-Ins, turn on *Use VST3
  Plug-in System Folders* (Kontakt's VST3 lives in `C:\Program Files\Common Files\VST3`), and/or
  set the VST2 custom folder. Click *Rescan*, then check it shows under Plug-ins in Live's browser.
- **Listed under another name** (e.g. "Kontakt 7"): set `search = "Kontakt 7"`, or copy the line
  `uri = "..."`, into `[plugins.kontakt]` in `tonematch.toml`.

The same applies to Archetype, Hellrazer and Metal Eclipse. `tonematch doctor` checks every plug-in in
your config. `build` can be re-run after a failure: tracks that already exist are reused.

**A take fails or is retaken ("take failed ... recording again")**: tonematch reads each recording
once Live has finished writing it, which happens when the temporary track is disarmed and removed. It
waits up to `finalize_timeout_s` (10 s) and records the take again up to `take_retries` times. Each
failure is logged in `work/failed_takes.log`. If this happens often, Live is writing files slowly:
- Keep the project out of folders synced by OneDrive, Proton Drive or Dropbox. The Desktop and
  Documents folders are often synced.
- Add the project folder to the Windows Defender exclusions.

## Limitations

- Stem separation of a mastered MP3 isn't perfect. Bass bleeds into guitar and cymbals into everything,
  so the loss weights favor each instrument's core range (guitar 90 Hz-7 kHz, bass 40 Hz-3 kHz).
- Live only plays one tempo. If the MIDI has tempo changes, Live uses the first one, and alignment
  fits a straight line through the rest. You can override `offset_s` / `scale` under `[analysis]`.
- Inside Kontakt, settings are only reachable if assigned to host automation. Otherwise MixWave's
  mic balance is matched through multi-outs, and the bass tone through the Live-side chain.
- Parameter names for third-party plug-ins come from regex patterns. Check `tonematch discover`
  before the first `match`.

## Development

```bash
pip install -e ".[dev]" && pytest
```

`tests/fake_live.py` is a fake Remote Script that uses the same socket protocol, with simulated
devices that really process audio: amp drive and EQ, Utility gain, EQ Eight, Limiter, and pan. The end-to-end test builds the
template, dials a hidden "true" sound into the fake, renders it as the reference, resets everything, and
checks that `match` finds its way back: it picks the same amp, lowers the tone loss, gets levels within
1.5 dB and pans the doubles wide.
