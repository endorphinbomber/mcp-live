"""High-level Live operations built on the Remote Script commands."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

from ..audio import wait_until_readable
from .client import LiveClient, LiveError
from .params import LiveParam

UNITY_FADER = 0.85          # Live mixer volume value for 0 dB
MONITOR_OFF = 2
MAX_RECORD_SECONDS = 300.0


@dataclass
class DeviceRef:
    track: str               # track name, or "Master"
    track_index: int         # -1 for master
    device_index: int
    name: str

    @property
    def is_master(self) -> bool:
        return self.track_index < 0


_NUM = re.compile(r"([+-]?inf|[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(k?)", re.I)


def parse_display(text: str) -> float:
    """'-3.2 dB' -> -3.2, '1.20 kHz' -> 1200, '-inf dB' -> -inf, '0.71' -> 0.71."""
    m = _NUM.search(str(text))
    if not m:
        raise ValueError(f"No number in display {text!r}")
    v = float(m.group(1))
    return v * 1000 if m.group(2).lower() == "k" else v


def _best_match(matches: list[dict], search: str) -> str | None:
    """Exact name first, then devices, then VST3 over VST2/AU duplicates, then shortest name."""
    matches = [m for m in matches if m.get("uri")]
    if not matches:
        return None
    s = search.lower()
    matches.sort(key=lambda m: (m["name"].lower() != s, not m.get("is_device", True),
                                "vst3" not in m["uri"].lower(), len(m["name"])))
    return matches[0]["uri"]


class LiveSession:
    def __init__(self, client: LiveClient):
        self.c = client

    # -- tracks --------------------------------------------------------
    def track_names(self) -> list[str]:
        n = self.c.send("get_session_info").get("track_count", 0)
        return [self.c.send("get_track_info", track_index=i).get("name", "") for i in range(n)]

    def track_index(self, name: str) -> int:
        names = self.track_names()
        if name not in names:
            raise LiveError(f"Track {name!r} not found in Live (have: {names})")
        return names.index(name)

    def devices(self, track_index: int) -> list[dict]:
        return self.c.send("get_track_info", track_index=track_index).get("devices", [])

    def create_track(self, name: str, midi: bool) -> int:
        r = self.c.send("create_midi_track" if midi else "create_audio_track", index=-1)
        idx = r.get("index")
        if idx is None:
            idx = self.c.send("get_session_info")["track_count"] - 1
        self.c.send("set_track_name", track_index=idx, name=name)
        return idx

    def ensure_track(self, name: str, midi: bool) -> int:
        names = self.track_names()
        return names.index(name) if name in names else self.create_track(name, midi)

    # -- devices -------------------------------------------------------
    def find_browser_item(self, search: str, stock: bool, uri: str = "") -> str:
        """Browser URI for a device/plug-in. Plug-in lookup does not rely on `is_device`
        (Live reports False for some VST3s) and falls back to walking the Plug-ins tree."""
        if uri:
            return uri
        categories = ["audio_effects", "instruments"] if stock else ["plugins"]
        errors: list[str] = []
        for cat in categories:
            try:
                r = self.c.send("search_browser", query=search, category=cat, max_results=50)
            except LiveError as e:
                errors.append(f"search in {cat}: {e}")
                continue
            hit = _best_match(r.get("matches", []), search)
            if hit:
                return hit
        if not stock:
            try:
                hit = _best_match([i for i in self.walk_plugins(search) if i["is_loadable"]], search)
            except LiveError as e:
                errors.append(f"walking plugins: {e}")
                hit = None
            if hit:
                return hit
        detail = f" ({'; '.join(errors)})" if errors else ""
        hint = (f" Run `tonematch plugins` to list what Live's Plug-ins browser shows, then set "
                f"`search` (or `uri`) for this plug-in in tonematch.toml. If Kontakt is missing there, "
                f"enable 'Use VST3 Plug-in System Folders' (and the VST2 custom folder if you use the "
                f"VST2 version) in Settings > Plug-Ins and click Rescan." if not stock else "")
        raise LiveError(f"'{search}' not found in Live's browser ({'/'.join(categories)}){detail}.{hint}")

    def walk_plugins(self, query: str = "", max_depth: int = 4, limit: int = 2000) -> list[dict]:
        """Breadth-first listing of Live's Plug-ins browser (path, name, uri, is_loadable)."""
        out: list[dict] = []
        queue = [("plugins", 0)]
        q = query.lower()
        while queue and len(out) < limit:
            path, depth = queue.pop(0)
            r = self.c.send("get_browser_items_at_path", path=path)
            if r.get("error"):
                if path == "plugins":
                    raise LiveError(f"{r['error']} (available: {r.get('available_categories')})")
                continue
            for item in r.get("items", []):
                name = item.get("name", "")
                child = f"{path}/{name}"
                if item.get("is_loadable") and (not q or q in name.lower()):
                    out.append({"path": child, "name": name, "uri": item.get("uri"),
                                "is_loadable": True, "is_device": bool(item.get("is_device"))})
                if item.get("is_folder") and depth + 1 < max_depth and "/" not in name:
                    queue.append((child, depth + 1))
        return out

    def load_device(self, track_index: int, search: str, stock: bool, uri: str = "") -> None:
        uri = self.find_browser_item(search, stock, uri)
        if track_index < 0:
            self.c.send("load_device_to_master", item_uri=uri)
        else:
            self.c.send("load_browser_item", track_index=track_index, item_uri=uri)

    def device_params(self, ref: DeviceRef) -> list[LiveParam]:
        if ref.is_master:
            r = self.c.send("get_master_device_parameters", device_index=ref.device_index)
        else:
            r = self.c.send("get_device_parameters", track_index=ref.track_index,
                            device_index=ref.device_index)
        return [LiveParam.from_live(p) for p in r.get("parameters", [])]

    def _set_cmd(self, ref: DeviceRef, parameter: int | str, value: float | str) -> tuple[str, dict]:
        if ref.is_master:
            return "set_master_device_parameter", {"device_index": ref.device_index,
                                                   "parameter": parameter, "value": value}
        return "set_device_parameter", {"track_index": ref.track_index, "device_index": ref.device_index,
                                        "parameter": parameter, "value": value}

    def set_params(self, items: list[tuple[DeviceRef, int | str, float | str]]) -> list[dict]:
        """Set many parameters in one batch (one undo step, one round trip)."""
        return self.c.batch([self._set_cmd(ref, p, v) for ref, p, v in items])

    def set_display(self, ref: DeviceRef, param: LiveParam, target: float, unit: str = "") -> float:
        """Set a parameter to a value in display units (dB, Hz, Q...).

        Tries Live's own display conversion first; plug-ins and ranges it cannot invert
        (e.g. Utility's "-inf dB" end) fall back to bisection on the native value using
        Live's readback. Returns the achieved display value.
        """
        if unit:
            try:
                cmd, kw = self._set_cmd(ref, param.index, f"{target:.3f} {unit}")
                return parse_display(self.c.send(cmd, **kw).get("display", ""))
            except (LiveError, ValueError):
                pass

        def probe(native: float) -> float:
            cmd, kw = self._set_cmd(ref, param.index, native)
            return parse_display(self.c.send(cmd, **kw).get("display", ""))

        lo, hi = param.min, param.max
        d_lo, d_hi = probe(lo), probe(hi)
        ascending = d_hi > d_lo
        best = (abs(d_hi - target), hi)
        for _ in range(16):
            mid = (lo + hi) / 2
            d = probe(mid)
            if math.isfinite(d):
                best = min(best, (abs(d - target), mid))
            if (d < target) == ascending:
                lo = mid
            else:
                hi = mid
        return probe(best[1])

    # -- mixer ---------------------------------------------------------
    def set_mixer(self, track_index: int, pan: float | None = None, unity: bool = False) -> None:
        cmds = []
        if pan is not None:
            cmds.append(("set_track_pan", {"track_index": track_index, "pan": max(-1.0, min(1.0, pan))}))
        if unity:
            cmds.append(("set_track_volume", {"track_index": track_index, "volume": UNITY_FADER}))
        self.c.batch(cmds)

    def route_input(self, track_index: int, source: str, channel_hint: str | None = None) -> None:
        routing = self.c.send("get_track_routing", track_index=track_index)
        types = [r.get("display_name") for r in routing.get("available_input_routing_types", [])]
        if source not in types:
            raise LiveError(f"Input '{source}' not available on track {track_index}: {types}")
        self.c.send("set_track_routing", track_index=track_index, field="input_routing_type",
                    display_name=source)
        if channel_hint:
            routing = self.c.send("get_track_routing", track_index=track_index)
            chans = [r.get("display_name") for r in routing.get("available_input_routing_channels", [])]
            hit = [c for c in chans if channel_hint.lower() in (c or "").lower()]
            if not hit:
                raise LiveError(f"No input channel matching '{channel_hint}' on '{source}': {chans}")
            self.c.send("set_track_routing", track_index=track_index, field="input_routing_channel",
                        display_name=hit[0])

    # -- MIDI ------------------------------------------------------------
    def write_midi(self, track_index: int, notes: list[dict], length_beats: float) -> None:
        """Put the notes in Session slot 0 and copy that clip to the arrangement at beat 0."""
        info = self.c.send("get_track_info", track_index=track_index)
        slots = info.get("clip_slots", [])
        if slots and slots[0].get("has_clip"):
            self.c.send("delete_clip", track_index=track_index, clip_index=0)
        self.c.send("create_clip", track_index=track_index, clip_index=0, length=float(length_beats))
        self.c.send("add_notes_to_clip", track_index=track_index, clip_index=0, notes=notes)
        self.c.send("set_clip_name", track_index=track_index, clip_index=0, name="tonematch")
        for i, clip in reversed(list(enumerate(
                self.c.send("get_arrangement_clips", track_index=track_index).get("clips", [])))):
            self.c.send("delete_arrangement_clip", track_index=track_index, arrangement_clip_index=i)
        self.c.send("duplicate_session_clip_to_arrangement", track_index=track_index, clip_index=0,
                    destination_time=0.0)

    # -- recording -------------------------------------------------------
    def record(self, sources: list[str], start_beat: float, end_beat: float,
               settle_s: float = 0.3, file_timeout_s: float = 30.0) -> dict[str, Path]:
        """Bounce several track outputs (or "Resampling" = master) in ONE real-time pass.

        One temporary audio track per source, input = that track's Post Mixer output,
        monitoring off, armed; arrangement record over [start_beat, end_beat].
        """
        info = self.c.send("get_session_info")
        tempo = float(info.get("tempo", 120.0))
        # Pre-roll one beat so plugin tails/gates settle; trimmed off by the caller's analysis.
        pre = 1.0 if start_beat >= 1.0 else 0.0
        duration = (end_beat - start_beat + pre) * 60.0 / tempo
        if duration > MAX_RECORD_SECONDS:
            raise LiveError(f"Region is {duration:.0f}s; keep it under {MAX_RECORD_SECONDS:.0f}s")
        self.c.send("stop_all_clips")
        self.c.send("back_to_arranger")
        bounce: dict[str, int] = {}
        try:
            for src in sources:
                idx = self.create_track(f"tm-bounce {src}", midi=False)
                bounce[src] = idx
                try:
                    self.route_input(idx, src, channel_hint=None if src == "Resampling" else "Post Mixer")
                except LiveError:
                    # No "Post Mixer" channel on this Live build: the default tap is fine.
                    self.route_input(idx, src)
                self.c.send("set_track_monitoring", track_index=idx, state=MONITOR_OFF)
                self.c.send("set_track_arm", track_index=idx, arm=True)
            self.c.send("set_current_song_time", time=start_beat - pre)
            self.c.send("set_record_mode", enabled=True)
            self.c.send("start_playback")
            time.sleep(duration + 0.3)
            self.c.send("set_record_mode", enabled=False)
            self.c.send("stop_playback")
            time.sleep(settle_s)
            out: dict[str, Path] = {}
            for src, idx in bounce.items():
                clips = self.c.send("get_arrangement_clips", track_index=idx).get("clips", [])
                files = [c["file_path"] for c in clips if c.get("is_audio_clip") and c.get("file_path")]
                if not files:
                    raise LiveError(f"Nothing was recorded from '{src}' (routing/arm failed?)")
                out[src] = Path(files[-1])
            # Live finishes writing (and on Windows unlocks) recordings shortly after the
            # transport stops; read nothing until every take is complete.
            for src, f in out.items():
                try:
                    wait_until_readable(f, timeout=file_timeout_s)
                except TimeoutError as e:
                    raise LiveError(f"Recording from '{src}' never became readable: {e}. Check free disk "
                                    "space and that no other program (antivirus, cloud sync) holds the "
                                    "project's Samples/Recorded folder.") from None
            return out
        finally:
            for idx in sorted(bounce.values(), reverse=True):
                try:
                    self.c.send("delete_track", track_index=idx)
                except LiveError:
                    pass

    @staticmethod
    def preroll_seconds(start_beat: float, tempo: float) -> float:
        return (1.0 if start_beat >= 1.0 else 0.0) * 60.0 / tempo
