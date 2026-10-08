"""The tonematch workflow: analyze -> build -> discover -> match (stages) -> export/report."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis import align as align_mod
from .analysis.features import Features, analyze as analyze_audio
from .analysis.separate import separate
from .audio import ANALYSIS_SR, load_audio, segment
from .config import Config, ParamSpec, TrackSpec
from .live.client import LiveClient, LiveError
from .live.params import LiveParam, resolve, resolve_all
from .live.session import DeviceRef, LiveSession
from .match import levels as lv
from .match.eqfit import EqBand, fit_eq
from .match.loss import tone_loss, weights_for
from .match.optimize import Candidate, Knob, Setting, ToneSearch, run_lockstep
from dataclasses import replace

from .midi import Note, Part, Song, classify_parts, humanize, load_song, mark_palm_mutes, palm_mute_velocities
from .state import State

DEFAULT_STAGES = ["levels", "tone", "levels", "eq", "pan", "levels", "master"]
UTILITY_GAIN = ParamSpec(key="gain", pattern="^gain$")
EQ_BANDS = 8

Printer = Callable[[str], None]


def eq_param(band: int, what: str) -> str:
    """EQ Eight parameter names, e.g. '3 Gain A'. `what`: Filter On|Filter Type|Frequency|Gain|Resonance."""
    return f"{band} {what} A"


class Project:
    def __init__(self, cfg: Config, client: LiveClient | None = None, echo: Printer = print):
        self.cfg = cfg
        self.state = State(cfg.workdir)
        self._client = client
        self.echo = echo
        self._song: Song | None = None
        self._track_idx: dict[str, int] | None = None

    # ----------------------------------------------------------------- helpers
    @property
    def live(self) -> LiveSession:
        if self._client is None:
            self._client = LiveClient()
        return LiveSession(self._client)

    @property
    def song(self) -> Song:
        if self._song is None:
            self._song = load_song(self.cfg.score_path)
        return self._song

    def regions(self) -> list[align_mod.Region]:
        return [align_mod.Region(**r) for r in self.state["analysis"]["regions"]]

    def ref_features(self, region_i: int, stem: str) -> Features:
        return Features.from_dict(self.state["analysis"]["features"][str(region_i)][stem])

    def track_index(self, name: str, refresh: bool = False) -> int:
        if self._track_idx is None or refresh or name not in self._track_idx:
            self._track_idx = {n: i for i, n in enumerate(self.live.track_names())}
        if name not in self._track_idx:
            raise LiveError(f"Track {name!r} is not in the Live set - run `tonematch build`")
        return self._track_idx[name]

    def device_ref(self, track: str, key: str) -> DeviceRef:
        if track == "Master":
            idx = self.state["build"]["master"][key]
            return DeviceRef("Master", -1, idx, key)
        devs = self.state["build"]["tracks"][track]["devices"]
        if key not in devs:
            raise LiveError(f"Track {track!r} has no device {key!r} in the build map")
        return DeviceRef(track, self.track_index(track), devs[key], key)

    def out_tracks(self, t: TrackSpec) -> list[str]:
        return [f"{t.name} {o['name']}" for o in t.outputs]

    def sources(self, t: TrackSpec) -> list[str]:
        return [t.name, *self.out_tracks(t)]

    def applied(self) -> dict:
        return self.state.data.setdefault("applied", {"gain_db": {}, "pan": {}, "params": {}, "eq": {}})

    # ================================================================= analyze
    def analyze(self) -> dict:
        cfg, a = self.cfg, self.cfg.analysis
        ref_path = cfg.path("reference")
        self.echo(f"Loading reference {ref_path.name}")
        ref = load_audio(ref_path)
        if a.get("separator", "demucs") == "demucs":
            self.echo("Separating stems with Demucs (first run takes a few minutes)...")
            stems = separate(ref_path, cfg.workdir / "stems", a.get("demucs_model", "htdemucs_6s"))
        else:
            stems = {k: (cfg.root / v) for k, v in a.get("stems", {}).items()}
        needed = sorted({t.stem for t in cfg.tracks})
        missing = [s for s in needed if s not in stems]
        if missing:
            raise ValueError(f"No reference stem for {missing}; add them under [analysis].stems")
        stem_audio = {s: load_audio(stems[s]) for s in needed}

        song = self.song
        roles = classify_parts(song)
        for t in cfg.tracks:
            if t.role not in roles:
                raise ValueError(f"MIDI has no part for role {t.role!r} (parts: {[p.name for p in song.parts]})")
        if len(song.tempos) > 1:
            self.echo(f"WARNING: MIDI has {len(song.tempos)} tempo changes; Live will play at "
                      f"{song.bpm:g} BPM throughout and regions are matched with a linear fit.")
        anchor = stem_audio.get("drums", ref)
        alignment = align_mod.align(song, ref, drums_part=roles.get("drums"),
                                    ref_env=align_mod.onset_envelope(anchor))
        self.echo(f"Alignment: MIDI 0:00 = reference {alignment.offset_s:.2f}s, "
                  f"tempo scale {alignment.scale:.4f}, confidence {alignment.score:.2f}")
        if alignment.score < 0.25:
            self.echo("WARNING: weak alignment. Check that the MIDI tempo matches the song; "
                      "you can override [analysis] offset_s / scale in tonematch.toml.")
        if "offset_s" in a:
            alignment.offset_s = float(a["offset_s"])
        if "scale" in a:
            alignment.scale = float(a["scale"])
        regions = align_mod.pick_regions(song, alignment, len(ref) / ANALYSIS_SR,
                                         bars=int(a.get("region_bars", 8)), count=int(a.get("regions", 2)))
        features: dict[str, dict] = {}
        for i, r in enumerate(regions):
            per = {s: analyze_audio(segment(stem_audio[s], r.ref_start_s, r.ref_end_s)).to_dict() for s in needed}
            per["mix"] = analyze_audio(segment(ref, r.ref_start_s, r.ref_end_s)).to_dict()
            features[str(i)] = per
            self.echo(f"Region {i + 1}: beats {r.start_beat:g}-{r.end_beat:g} "
                      f"(ref {r.ref_start_s:.1f}-{r.ref_end_s:.1f}s), mix {per['mix']['lufs']:.1f} LUFS")
        self.state["analysis"] = {
            "alignment": alignment.to_dict(),
            "regions": [r.to_dict() for r in regions],
            "roles": {role: {"name": p.name, "channel": p.channel, "notes": len(p.notes)}
                      for role, p in roles.items()},
            "bpm": song.bpm,
            "time_signature": list(song.time_signature),
            "stems": {k: str(v) for k, v in stems.items()},
            "features": features,
        }
        self.state.checkpoint("analyze")
        return self.state["analysis"]

    # =================================================================== build
    def build(self) -> list[str]:
        """Create tracks, load plug-ins, write MIDI, set pans. Returns manual follow-ups."""
        cfg, live, song = self.cfg, self.live, self.song
        todo: list[str] = []
        num, den = song.time_signature
        live.c.batch([("set_tempo", {"tempo": song.bpm}),
                      ("set_time_signature", {"numerator": num, "denominator": den})])
        roles = classify_parts(song)
        bpb = song.beats_per_bar()
        length = math.ceil(song.end_beat / bpb) * bpb
        build: dict = {"tracks": {}, "master": {}}
        for t in cfg.tracks:
            idx = live.ensure_track(t.name, midi=True)
            devices = self._load_chain(idx, t.device_names, t.name)
            build["tracks"][t.name] = {"devices": devices}
            notes, summary = prepare_notes(song, roles[t.role], t)
            if summary:
                self.echo(f"  {t.name}: {summary}")
            live.write_midi(idx, [n.to_live() for n in notes], length)
            live.set_mixer(idx, pan=t.pan, unity=True)
            self.applied()["pan"][t.name] = t.pan
            if cfg.plugin(t.instrument).search.lower() == "kontakt":
                todo.append(f"'{t.name}': open Kontakt and load the {t.role} library "
                            f"({'MixWave Gojira' if t.role == 'drums' else 'NI Rickenbacker bass'}).")
            for out, o in zip(self.out_tracks(t), t.outputs):
                oidx = live.ensure_track(out, midi=False)
                build["tracks"][out] = {"devices": self._load_chain(oidx, o.get("chain", ["EQ Eight", "Utility"]), out)}
                try:
                    live.route_input(oidx, t.name, o.get("channel"))
                    live.c.send("set_track_monitoring", track_index=oidx, state=0)  # In
                except LiveError as e:
                    todo.append(f"'{out}': set its input to '{t.name}' / Kontakt output {o.get('channel')} "
                                f"and monitoring to In ({e}).")
                live.set_mixer(oidx, pan=0.0, unity=True)
        build["master"] = self._load_master(cfg.master_chain)
        self._track_idx = None
        self.state["build"] = build
        self.state.checkpoint("build")
        todo += [
            "Plug-ins with many parameters: click 'Configure' on the device title bar and touch the amp, "
            "cab/mic and gate controls so Live exposes them; then run `tonematch discover`.",
            "Save the set (File > Save Live Set As Template keeps it as your starting template).",
        ]
        return todo

    def _load_chain(self, track_index: int, keys: list[str], track_name: str) -> dict[str, int]:
        existing = self.live.devices(track_index)
        if len(existing) >= len(keys):
            self.echo(f"'{track_name}' already has {len(existing)} devices; keeping them")
            return {k: i for i, k in enumerate(keys)}
        if existing:
            raise LiveError(f"'{track_name}' has a partial chain ({[d['name'] for d in existing]}); "
                            "delete its devices or the track and rebuild")
        for k in keys:
            spec = self.cfg.plugin(k)
            self.echo(f"  {track_name}: loading {spec.search}")
            self.live.load_device(track_index, spec.search, stock=spec.is_stock, uri=spec.uri)
        loaded = self.live.devices(track_index)
        if len(loaded) != len(keys):
            raise LiveError(f"'{track_name}': expected {len(keys)} devices, Live shows "
                            f"{[d['name'] for d in loaded]}")
        return {k: i for i, k in enumerate(keys)}

    def _master_device_count(self) -> int:
        n = 0
        while True:
            try:
                self.live.c.send("get_master_device_parameters", device_index=n)
            except LiveError:
                return n
            n += 1

    def _load_master(self, keys: list[str]) -> dict[str, int]:
        start = self._master_device_count()
        prev = self.state.get("build", {}).get("master")
        if prev and start >= len(prev):
            return prev
        for k in keys:
            self.live.load_device(-1, self.cfg.plugin(k).search, stock=True)
        return {k: start + i for i, k in enumerate(keys)}

    # ================================================================ discover
    def discover(self) -> dict:
        report: dict = {}
        dump_dir = self.cfg.workdir / "params"
        dump_dir.mkdir(parents=True, exist_ok=True)
        for t in self.cfg.tracks:
            for key in t.device_names:
                spec = self.cfg.plugin(key)
                ref = self.device_ref(t.name, key)
                params = self.live.device_params(ref)
                (dump_dir / f"{t.name}__{key}.json").write_text(
                    json.dumps([p.__dict__ for p in params], indent=2))
                resolved, missing = resolve_all(spec.params, params)
                report[f"{t.name}/{key}"] = {
                    "exposed": len(params),
                    "resolved": {r.spec.key: r.param.name for r in resolved},
                    "missing": [m.key for m in missing],
                    "choices": {r.spec.key: [lbl for _, lbl in r.param.choices()]
                                for r in resolved if r.spec.kind == "categorical"},
                }
            for need in ("EQ Eight", "Utility"):
                if need in t.chain:
                    params = self.live.device_params(self.device_ref(t.name, need))
                    names = {p.name for p in params}
                    want = [eq_param(1, "Gain")] if need == "EQ Eight" else ["Gain"]
                    if not set(want) <= names:
                        report[f"{t.name}/{need}"] = {"missing": want, "exposed": len(params)}
        self.state["discover"] = report
        return report

    # ================================================================== render
    def render(self, tracks: list[TrackSpec], region_ids: list[int]) -> dict[str, list[Features]]:
        """Record every given track (with its multi-outs summed) over each region."""
        out: dict[str, list[Features]] = defaultdict(list)
        tempo = float(self.live.c.send("get_session_info").get("tempo", self.song.bpm))
        regions = self.regions()
        for ri in region_ids:
            r = regions[ri]
            sources = [s for t in tracks for s in self.sources(t)]
            files = self.live.record(sources, r.start_beat, r.end_beat)
            pre = LiveSession.preroll_seconds(r.start_beat, tempo)
            dur = (r.end_beat - r.start_beat) * 60.0 / tempo
            for t in tracks:
                audio = None
                for s in self.sources(t):
                    a = segment(load_audio(files[s]), pre, pre + dur)
                    audio = a if audio is None else audio[: len(a)] + a[: len(audio)]
                out[t.name].append(analyze_audio(audio))
        self.state.data["last_render"] = {k: [f.to_dict() for f in v] for k, v in out.items()}
        return out

    def render_master(self, region_ids: list[int]) -> list[Features]:
        tempo = float(self.live.c.send("get_session_info").get("tempo", self.song.bpm))
        feats = []
        for ri in region_ids:
            r = self.regions()[ri]
            f = self.live.record(["Resampling"], r.start_beat, r.end_beat)["Resampling"]
            pre = LiveSession.preroll_seconds(r.start_beat, tempo)
            feats.append(analyze_audio(segment(load_audio(f), pre, pre + (r.end_beat - r.start_beat) * 60 / tempo)))
        self.state.data.setdefault("last_render", {})["Master"] = [f.to_dict() for f in feats]
        return feats

    # ================================================================== stages
    def match(self, stages: list[str] | None = None) -> None:
        handlers = {"levels": self.stage_levels, "tone": self.stage_tone, "eq": self.stage_eq,
                    "pan": self.stage_pan, "master": self.stage_master}
        for st in stages or DEFAULT_STAGES:
            if st not in handlers:
                raise ValueError(f"Unknown stage {st!r}; choose from {sorted(handlers)}")
            self.echo(f"== stage: {st}")
            handlers[st]()
            self.state.checkpoint(st)

    # -- levels -----------------------------------------------------------
    def _set_gain(self, track: str, gain_db: float) -> float:
        gain_db = float(np.clip(gain_db, -35.0, 35.0))
        ref = self.device_ref(track, "Utility")
        p = resolve(UTILITY_GAIN, self.live.device_params(ref))
        if p is None:
            raise LiveError(f"'{track}': Utility has no 'Gain' parameter")
        achieved = self.live.set_display(ref, p, gain_db, "dB")
        self.applied()["gain_db"][track] = achieved if math.isfinite(achieved) else gain_db
        return self.applied()["gain_db"][track]

    def stage_levels(self) -> None:
        headroom = float(self.cfg.match.get("headroom_db", 6.0))
        tol = float(self.cfg.match.get("level_tolerance_db", 0.5))
        per_stem = defaultdict(int)
        for t in self.cfg.tracks:
            per_stem[t.stem] += 1
        regions = list(range(len(self.regions())))
        for attempt in range(3):
            feats = self.render(self.cfg.tracks, regions)
            worst = 0.0
            for t in self.cfg.tracks:
                render_lufs = float(np.mean([f.lufs for f in feats[t.name]]))
                target = float(np.mean([self.ref_features(i, t.stem).lufs for i in regions]))
                target += lv.share_db(per_stem[t.stem]) - headroom
                delta = lv.gain_delta_db(render_lufs, target)
                if render_lufs <= -60:
                    self.echo(f"  {t.name}: SILENT render - check the instrument/Kontakt library is loaded")
                worst = max(worst, abs(delta))
                for track in self.sources(t):
                    current = self.applied()["gain_db"].get(track, 0.0)
                    self._set_gain(track, current + delta)
                self.echo(f"  {t.name}: {render_lufs:.1f} -> target {target:.1f} LUFS ({delta:+.1f} dB)")
            self.state.log("levels", {"attempt": attempt, "max_delta_db": worst})
            if worst < tol:
                break

    # -- tone -------------------------------------------------------------
    def _knobs(self, t: TrackSpec) -> list[Knob]:
        knobs: list[Knob] = []
        for key in t.device_names:
            spec = self.cfg.plugin(key)
            if not spec.params:
                continue
            params = self.live.device_params(self.device_ref(t.name, key))
            for ps in spec.params:
                # Resolvable now, or after a categorical choice fills its {placeholder}.
                if resolve(ps, params) is not None or "{" in ps.pattern:
                    knobs.append(Knob(t.name, key, ps, params))
        for out in self.out_tracks(t):
            params = self.live.device_params(self.device_ref(out, "Utility"))
            spec = ParamSpec(key="balance", pattern="^gain$")
            cur = self.applied()["gain_db"].get(out, 0.0)
            knobs.append(Knob(out, "Utility", spec, params, db_range=(cur - 9.0, cur + 6.0)))
        return knobs

    def _apply(self, settings: list[Setting]) -> None:
        native = [(self.device_ref(s.track, s.device), s.param_index, s.value) for s in settings if not s.is_db]
        if native:
            self.live.set_params(native)
        for s in settings:
            if s.is_db:
                self._set_gain(s.track, s.value)
            else:
                self.applied()["params"].setdefault(s.track, {}).setdefault(s.device, {})[str(s.param_index)] = s.value

    def stage_tone(self) -> None:
        m = self.cfg.match
        n_trials = int(m.get("tone_trials", 40))
        region_ids = list(range(min(int(m.get("tone_regions", 1)), len(self.regions()))))
        searches, tracks = [], []
        for i, t in enumerate(self.cfg.tracks):
            knobs = self._knobs(t)
            if knobs:
                searches.append(ToneSearch(t.name, knobs, seed=int(m.get("seed", 7)) + i))
                tracks.append(t)
                self.echo(f"  {t.name}: optimizing {len(knobs)} parameters")
            else:
                self.echo(f"  {t.name}: nothing to optimize (no resolvable params) - skipped")
        if not searches:
            return
        best_seen: dict[str, float] = {}

        def evaluate(cands: dict[str, Candidate]) -> dict[str, float]:
            self._apply([s for c in cands.values() for s in c.values])
            feats = self.render(tracks, region_ids)
            losses = {}
            return {t.name: self._mean_loss(feats[t.name], t, region_ids) for t in tracks}

        def on_round(i: int, losses: dict[str, float]) -> None:
            for k, v in losses.items():
                best_seen[k] = min(best_seen.get(k, math.inf), v)
            self.state.log("tone", {"trial": i, "losses": losses})
            self.echo(f"  take {i + 1}/{n_trials}: " + ", ".join(
                f"{k} {v:.2f} (best {best_seen[k]:.2f})" for k, v in losses.items()))

        run_lockstep(searches, n_trials, evaluate, on_round)
        final = []
        for s in searches:
            best = s.best()
            if best is None:
                continue
            final += s.replay(best[1]).values
            self.echo(f"  {s.name}: best loss {best[0]:.2f}")
        self._apply(final)

    # -- eq ---------------------------------------------------------------
    def _write_eq(self, track: str, bands: list[EqBand]) -> None:
        ref = self.device_ref(track, "EQ Eight")
        params = {p.name: p for p in self.live.device_params(ref)}
        off = [(ref, eq_param(b, "Filter On"), 0.0) for b in range(1, EQ_BANDS + 1)
               if eq_param(b, "Filter On") in params]
        self.live.set_params(off)
        for i, band in enumerate(bands, start=1):
            self.live.set_params([(ref, eq_param(i, "Filter Type"), band.kind),
                                  (ref, eq_param(i, "Filter On"), 1.0)])
            self.live.set_display(ref, params[eq_param(i, "Frequency")], band.freq, "Hz")
            self.live.set_display(ref, params[eq_param(i, "Gain")], band.gain_db, "dB")
            self.live.set_display(ref, params[eq_param(i, "Resonance")], band.q, "")
        self.applied()["eq"][track] = [b.to_dict() for b in bands]

    def _eq_target(self, feats: list[Features], stem: str, region_ids: list[int]) -> np.ndarray:
        rend = np.mean([f.tone_curve() for f in feats], axis=0)
        ref = np.mean([self.ref_features(i, stem).tone_curve() for i in region_ids], axis=0)
        return ref - rend

    def stage_eq(self) -> None:
        m = self.cfg.match
        regions = list(range(len(self.regions())))
        tracks = [t for t in self.cfg.tracks if "EQ Eight" in t.chain]
        before = self.render(tracks, regions)
        prev: dict[str, list[EqBand]] = {}
        for t in tracks:
            bands = fit_eq(self._eq_target(before[t.name], t.stem, regions), weights_for(t.role),
                           int(m.get("eq_bands", 6)), float(m.get("eq_max_gain_db", 6.0)))
            prev[t.name] = [EqBand(**b) for b in self.applied()["eq"].get(t.name, [])]
            self._write_eq(t.name, _merge_eq(prev[t.name], bands))
            self.echo(f"  {t.name}: " + ", ".join(f"{b.kind} {b.freq:.0f}Hz {b.gain_db:+.1f}dB" for b in bands))
        after = self.render(tracks, regions)
        for t in tracks:
            l0 = self._mean_loss(before[t.name], t, regions)
            l1 = self._mean_loss(after[t.name], t, regions)
            self.state.log("eq", {"track": t.name, "before": l0, "after": l1})
            if l1 > l0:
                self.echo(f"  {t.name}: EQ made it worse ({l0:.2f} -> {l1:.2f}); reverting")
                self._write_eq(t.name, prev[t.name])
            else:
                self.echo(f"  {t.name}: loss {l0:.2f} -> {l1:.2f}")

    def _mean_loss(self, feats: list[Features], t: TrackSpec, region_ids: list[int]) -> float:
        return float(np.mean([tone_loss(f, self.ref_features(i, t.stem), t.role)["total"]
                              for f, i in zip(feats, region_ids)]))

    # -- pan --------------------------------------------------------------
    def stage_pan(self) -> None:
        regions = list(range(len(self.regions())))
        by_stem: dict[str, list[TrackSpec]] = defaultdict(list)
        for t in self.cfg.tracks:
            by_stem[t.stem].append(t)
        for stem, tracks in by_stem.items():
            feats = [self.ref_features(i, stem) for i in regions]
            role = tracks[0].role
            if len(tracks) == 2:
                p = float(np.mean([lv.double_pan(f, role) for f in feats]))
                sign = -1.0 if tracks[0].pan <= 0 else 1.0   # keep the configured sides
                pans = [sign * p, -sign * p]
            else:
                pans = [float(np.mean([lv.single_pan(f, role) for f in feats]))] * len(tracks)
            for t, p in zip(tracks, pans):
                self.live.set_mixer(self.track_index(t.name), pan=p)
                self.applied()["pan"][t.name] = p
                self.echo(f"  {t.name}: pan {p:+.2f}")
        self.state.save()

    # -- master -----------------------------------------------------------
    def stage_master(self) -> None:
        keys = self.state["build"]["master"]
        regions = list(range(len(self.regions())))
        ref_mix = [self.ref_features(i, "mix") for i in regions]
        feats = self.render_master(regions)
        if "EQ Eight" in keys:
            target = (np.mean([f.tone_curve() for f in ref_mix], axis=0)
                      - np.mean([f.tone_curve() for f in feats], axis=0))
            bands = fit_eq(target, weights_for("mix"), 4, 3.0)
            prev = [EqBand(**b) for b in self.applied()["eq"].get("Master", [])]
            self._write_eq("Master", _merge_eq(prev, bands))
            self.echo("  master EQ: " + ", ".join(f"{b.kind} {b.freq:.0f}Hz {b.gain_db:+.1f}dB" for b in bands))
        if "Limiter" in keys:
            ref = DeviceRef("Master", -1, keys["Limiter"], "Limiter")
            params = {p.name: p for p in self.live.device_params(ref)}
            if "Gain" not in params:
                raise LiveError(f"Limiter exposes no 'Gain' parameter (has {sorted(params)})")
            if "Ceiling" in params:
                self.live.set_display(ref, params["Ceiling"], -0.3, "dB")
            gain = self.applied().setdefault("master", {}).get("limiter_gain_db", 0.0)
            target_lufs = float(np.mean([f.lufs for f in ref_mix]))
            slope, last = 1.0, None   # observed dLUFS / dGain; < 1 once the limiter works
            for attempt in range(6):
                feats = self.render_master(regions)
                mix_lufs = float(np.mean([f.lufs for f in feats]))
                delta = target_lufs - mix_lufs
                self.echo(f"  master: {mix_lufs:.1f} LUFS vs reference {target_lufs:.1f} (limiter gain {gain:+.1f} dB)")
                if last is not None and abs(gain - last[0]) > 0.1:
                    slope = float(np.clip((mix_lufs - last[1]) / (gain - last[0]), 0.15, 1.0))
                if abs(delta) < 0.5:
                    break
                last = (gain, mix_lufs)
                gain = float(np.clip(gain + delta / slope, 0.0, 24.0))
                gain = self.live.set_display(ref, params["Gain"], gain, "dB")
                self.applied()["master"]["limiter_gain_db"] = gain
            if abs(delta) >= 0.5:
                self.echo(f"  master: stopped {delta:+.1f} dB short of the reference loudness - the "
                          "reference is likely more compressed; lower the Glue threshold and re-run `--stages master`.")
        self.state.log("master", {"lufs": feats[0].lufs, "crest_db": feats[0].crest_db})

    # ============================================================== rollback
    def reapply(self, snapshot: Path | None = None) -> None:
        """Push the stored settings (current state, or a history snapshot) back into Live."""
        if snapshot is not None:
            self.state.data = json.loads(Path(snapshot).read_text())
            self.state.save()
        ap = self.applied()
        for track, devs in ap["params"].items():
            items = [(self.device_ref(track, dev), int(i), v) for dev, vals in devs.items() for i, v in vals.items()]
            if items:
                self.live.set_params(items)
        for track, g in ap["gain_db"].items():
            self._set_gain(track, g)
        for track, p in ap["pan"].items():
            self.live.set_mixer(self.track_index(track), pan=p)
        for track, bands in ap["eq"].items():
            self._write_eq(track, [EqBand(**b) for b in bands])
        if "limiter_gain_db" in ap.get("master", {}):
            ref = self.device_ref("Master", "Limiter")
            p = {q.name: q for q in self.live.device_params(ref)}["Gain"]
            self.live.set_display(ref, p, ap["master"]["limiter_gain_db"], "dB")


def prepare_notes(song: Song, part: Part, t: TrackSpec) -> tuple[list[Note], str]:
    """Notes for one Live track: drum remap, palm-mute decision (on the source notes),
    humanize, then palm-mute velocities (so jitter can never cross the threshold)."""
    notes = list(part.notes)
    if t.note_map:
        notes = [replace(n, pitch=t.note_map.get(n.pitch, n.pitch)) for n in notes]
    summary, mode = "", ""
    if t.palm_mute:
        notes, mode = mark_palm_mutes(notes, t.palm_mute, song.beats_per_bar(), part.ccs)
    if t.humanize:
        h = t.humanize
        notes = humanize(notes, song.bpm, float(h.get("timing_ms", 0)), int(h.get("velocity", 0)),
                         int(h.get("seed", 0)))
    if t.palm_mute:
        notes = palm_mute_velocities(notes, t.palm_mute)
        n_pm = sum(1 for n in notes if n.palm_mute)
        source = {"file": "from the tab", "heuristic": "guessed from the riffs"}.get(mode, f"mode {mode}")
        summary = f"{n_pm} of {len(notes)} notes palm-muted ({source})"
    return notes, summary


def _merge_eq(prev: list[EqBand], new: list[EqBand], max_bands: int = EQ_BANDS) -> list[EqBand]:
    """Accumulate corrections from repeated EQ stages (EQ Eight has 8 bands)."""
    merged = list(prev) + list(new)
    if len(merged) <= max_bands:
        return merged
    return sorted(merged, key=lambda b: -abs(b.gain_db))[:max_bands]
