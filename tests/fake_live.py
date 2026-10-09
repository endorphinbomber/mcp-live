"""A fake AbletonMCP Remote Script: same socket protocol, simulated devices and audio.

Devices process a deterministic noise source per track so that the closed loop can be
tested end to end: changing an amp's treble really changes the recorded spectrum, the
Utility gain really changes loudness, EQ Eight really filters, and so on.
"""

from __future__ import annotations

import json
import math
import re
import socket
import tempfile
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

from tonematch.match.eqfit import _biquad

SR = 44100
SECONDS = 4.0
MICS = ["Dynamic 57", "Condenser 414", "Ribbon 121"]
EQ_TYPES = ["Low Cut 48", "Low Cut 12", "Low Shelf", "Bell", "Notch", "High Shelf", "High Cut 12", "High Cut 48"]


@dataclass
class P:
    name: str
    min: float
    max: float
    value: float
    quantized: bool = False
    items: list[str] = field(default_factory=list)
    fmt: str = "plain"      # plain | db | hz | q | utility_db

    def display(self, v: float | None = None) -> str:
        v = self.value if v is None else v
        if self.quantized:
            return self.items[int(round(v))] if self.items else str(int(round(v)))
        if self.fmt == "db":
            return f"{v:.2f} dB"
        if self.fmt == "utility_db":
            db = -35 + 70 * v
            return "-inf dB" if v <= 0 else f"{db:.2f} dB"
        if self.fmt == "hz":
            hz = 10 * 2200 ** v
            return f"{hz / 1000:.2f} kHz" if hz >= 1000 else f"{hz:.1f} Hz"
        if self.fmt == "q":
            return f"{0.1 * 180 ** v:.2f}"
        return f"{v:.3f}"

    def as_dict(self, i: int) -> dict:
        d = {"index": i, "name": self.name, "value": self.value, "min": self.min, "max": self.max,
             "is_quantized": self.quantized, "is_enabled": True, "display": self.display()}
        if self.quantized:
            d["value_items"] = list(self.items)
        return d


def _num(text: str) -> float:
    m = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(k?)", text)
    v = float(m.group(1))
    return v * 1000 if m.group(2) else v


def make_device(kind: str) -> tuple[str, list[P]]:
    if kind == "Archetype Gojira X":
        # Names as Archetype Gojira X exposes them in Live (HOT amp configured).
        onoff = ["Off", "On"]
        ps = [P("Cab Type (unlinked)", 0, 2, 2, True, ["CLEAN", "RUST", "HOT"])]
        ps += [P(f"HOT Amp {k}", 0, 1, 0.5) for k in ("Gain", "Bass", "Mid", "Treble", "Presence", "Depth",
                                                        "Master", "Output")]
        ps += [P("Gate Active", 0, 1, 1, True, onoff), P("Gate Threshold", 0, 1, 0.3),
               P("OD Active", 0, 1, 0, True, onoff), P("OD Dist", 0, 1, 0.0), P("OD Tone", 0, 1, 0.5),
               P("OD Level", 0, 1, 0.5),
               P("DRT Active", 0, 1, 0, True, onoff), P("DRT Dist", 0, 1, 0.3), P("DRT Filter", 0, 1, 0.9),
               P("DRT Vol", 0, 1, 0.6)]
        for side in "LR":
            ps += [P(f"HOT Amp Cab {side} Type", 0, 2, 0, True, MICS),
                   P(f"Cab {side} Position", 0, 1, 0.3), P(f"Cab {side} Distance", 0, 1, 0.0),
                   P(f"Cab {side} Level", 0, 1, 0.8 if side == "L" else 0.5), P(f"Cab {side} Pan", 0, 1, 0.5),
                   P(f"Cab {side} Active", 0, 1, 1, True, onoff)]
        return kind, ps
    if kind in ("Metal Hellrazer", "Metal Eclipse"):
        return kind, [P("Pickup", 0, 1, 0, True, ["Neck", "Bridge"]), P("Tone", 0, 1, 0.5)]
    if kind == "Kontakt 8":
        # MixWave-style host-automation slots (#000 kick1, #002 snare, #014 overhead, ...).
        return kind, [P(f"#{i:03d}", 0, 1, 0.6) for i in (*range(16), 28, 29, 30, 32, 34, 50, 52)]
    if kind == "Utility":
        return kind, [P("Gain", 0, 1, 0.5, fmt="utility_db"), P("Mute", 0, 1, 0, True, ["Off", "On"])]
    if kind == "EQ Eight":
        ps = []
        for b in range(1, 9):
            ps += [P(f"{b} Filter On A", 0, 1, 0, True, ["Off", "On"]),
                   P(f"{b} Filter Type A", 0, 7, 3, True, EQ_TYPES),
                   P(f"{b} Frequency A", 0, 1, 0.5, fmt="hz"),
                   P(f"{b} Gain A", -15, 15, 0, fmt="db"),
                   P(f"{b} Resonance A", 0, 1, 0.42, fmt="q")]
        return kind, ps
    if kind == "Limiter":      # Live 12's Limiter
        return kind, [P("Device On", 0, 1, 1, True, ["Off", "On"]), P("Input Gain", -12, 24, 0, fmt="db"),
                      P("Ceiling", -24, 0, 0, fmt="db"), P("Maximize On", 0, 1, 1, True, ["Off", "On"]),
                      P("Threshold", -24, 0, 0, fmt="db"), P("Output", -24, 0, 0, fmt="db")]
    if kind == "Limiter (legacy)":
        return "Limiter", [P("Gain", 0, 24, 0, fmt="db"), P("Ceiling", -24, 0, 0, fmt="db")]
    if kind == "Pedal":
        return kind, [P("Type", 0, 2, 0, True, ["OD", "Distort", "Fuzz"]), P("Gain", 0, 1, 0.2),
                      P("Bass", 0, 1, 0.5), P("Mid", 0, 1, 0.5), P("Treble", 0, 1, 0.5), P("Dry/Wet", 0, 1, 1.0)]
    return kind, [P("Threshold", 0, 1, 0.5)]  # Glue / Compressor: pass-through


CATALOG = {
    "plugins": ["Archetype Gojira X", "Metal Hellrazer", "Metal Eclipse", "Kontakt 8"],
    "audio_effects": ["EQ Eight", "Utility", "Glue Compressor", "Compressor", "Limiter", "Pedal"],
}


@dataclass
class Track:
    name: str
    midi: bool
    devices: list[tuple[str, list[P]]] = field(default_factory=list)
    pan: float = 0.0
    volume: float = 0.85
    arm: bool = False
    monitoring: int = 1
    input_type: str = "Ext. In"
    input_channel: str = ""
    session_clip: dict | None = None
    arrangement: list[dict] = field(default_factory=list)


def _shelf(x, kind, f, gain_db, q=0.71):
    b, a = _biquad(kind, f, gain_db, q)
    return lfilter(b, a, x)


class FakeLive:
    def __init__(self):
        self.tracks: list[Track] = []
        self.master: list[tuple[str, list[P]]] = []
        self.tempo = 120.0
        self.sig = (4, 4)
        self.record_mode = False
        self.dir = Path(tempfile.mkdtemp(prefix="fakelive-"))
        self.takes = 0
        self.commands: list[str] = []
        self.seconds = SECONDS
        self.broken_takes: set[int] = set()      # take numbers written as 0-byte files
        self.missing_takes: set[int] = set()     # take numbers never written
        # Like real Live: a recording stays an unfinished 64 KiB-chunked file until its track
        # is disarmed or deleted. `never_finalize` takes never get completed.
        self.master_insert_front = False
        self.finalize_on_disarm = True          # False: only deleting the track finishes the file
        self.defer_finalize = False
        self.never_finalize: set[int] = set()
        self.pending: dict[str, tuple[Path, np.ndarray, int]] = {}   # track name -> (path, audio, take)
        self.search_hides: set[str] = set()      # plug-ins search_browser fails to return
        self.plugins_not_device: set[str] = set()  # plug-ins Live reports with is_device=False

    # ------------------------------------------------------------ audio model
    def source(self, t: Track) -> np.ndarray:
        rng = np.random.default_rng(zlib.crc32(t.name.encode()))
        x = rng.standard_normal(int(SR * self.seconds))
        inst = t.devices[0][0] if t.devices else ""
        if inst == "Kontakt 8" and "drum" in t.name.lower():
            env = np.exp(-np.arange(len(x)) % int(SR * 0.125) / (SR * 0.03))
            x = sosfilt(butter(2, [40, 12000], "bandpass", fs=SR, output="sos"), x * env)
            v = {p.name: p.value for p in t.devices[0][1]}
            x = _shelf(x, "Low Shelf", 120, (v["#000"] + v["#001"] - 1.2) * 12)      # kicks
            x = _shelf(x, "Bell", 2000, (v["#002"] - 0.6) * 12, 1.0)                  # snare
            x = _shelf(x, "High Shelf", 6000, (v["#014"] + v["#013"] - 1.2) * 10)    # overheads
            return x
        if inst == "Kontakt 8":
            return sosfilt(butter(4, 400, "lowpass", fs=SR, output="sos"), x) * 2
        return sosfilt(butter(2, [90, 7000], "bandpass", fs=SR, output="sos"), x) * 0.5

    def process(self, x: np.ndarray, devices) -> np.ndarray:
        for kind, ps in devices:
            v = {p.name: p for p in ps}
            if kind == "Archetype Gojira X":
                g = {name: p.value for name, p in v.items()}
                drive = 0.5 + 8 * g["HOT Amp Gain"] * (0.5 + g["HOT Amp Master"])
                if g["OD Active"] >= 0.5:
                    drive *= 1 + 3 * g["OD Dist"]
                    x = _shelf(x, "High Shelf", 1500, (g["OD Tone"] - 0.5) * 12)
                x = np.tanh(drive * x / (np.std(x) + 1e-9)) * 0.3
                x = _shelf(x, "Low Shelf", 90, (g["HOT Amp Depth"] - 0.5) * 12)
                x = _shelf(x, "Low Shelf", 200, (g["HOT Amp Bass"] - 0.5) * 18)
                x = _shelf(x, "Bell", 800, (g["HOT Amp Mid"] - 0.5) * 14, 0.8)
                x = _shelf(x, "High Shelf", 2500, (g["HOT Amp Treble"] - 0.5) * 18)
                x = _shelf(x, "High Shelf", 5000, (g["HOT Amp Presence"] - 0.5) * 10)
                mic_db = {"Dynamic 57": 0.0, "Condenser 414": 4.0, "Ribbon 121": -5.0}

                def mic(side, sig):
                    t_ = MICS[int(round(g[f"HOT Amp Cab {side} Type"]))]
                    sig = _shelf(sig, "High Shelf", 4000, mic_db[t_] - 8 * g[f"Cab {side} Position"])
                    return _shelf(sig, "Low Shelf", 150, -6 * g[f"Cab {side} Distance"])
                x = g["Cab L Level"] * mic("L", x) + g["Cab R Level"] * mic("R", x)
            elif kind in ("Metal Hellrazer", "Metal Eclipse"):
                x = sosfilt(butter(1, 1500 + 9000 * v["Tone"].value, "lowpass", fs=SR, output="sos"), x)
                if v["Pickup"].value >= 0.5:
                    x = _shelf(x, "High Shelf", 2000, 4)
            elif kind == "Pedal":
                drive = 1 + 10 * v["Gain"].value
                wet = np.tanh(drive * x / (np.std(x) + 1e-9)) * np.std(x)
                wet = _shelf(wet, "High Shelf", 2500, (v["Treble"].value - 0.5) * 24)
                x = (1 - v["Dry/Wet"].value) * x + v["Dry/Wet"].value * wet
            elif kind == "EQ Eight":
                for b in range(1, 9):
                    if v[f"{b} Filter On A"].value < 0.5:
                        continue
                    typ = EQ_TYPES[int(round(v[f"{b} Filter Type A"].value))]
                    if typ not in ("Bell", "Low Shelf", "High Shelf"):
                        continue
                    f = 10 * 2200 ** v[f"{b} Frequency A"].value
                    q = 0.1 * 180 ** v[f"{b} Resonance A"].value
                    x = _shelf(x, typ, min(f, SR / 2.2), v[f"{b} Gain A"].value, q)
            elif kind == "Utility":
                g = v["Gain"].value
                x = x * (0.0 if g <= 0 else 10 ** ((-35 + 70 * g) / 20))
            elif kind == "Limiter":
                drive = v["Gain"].value if "Gain" in v else v["Input Gain"].value
                if v.get("Maximize On") is not None and v["Maximize On"].value >= 0.5:
                    drive = -v["Threshold"].value           # Maximize: Threshold drives loudness
                x = x * 10 ** (drive / 20)
                ceil = 10 ** (v["Ceiling"].value / 20)
                x = np.clip(x, -ceil, ceil)
        return x

    def track_out(self, t: Track) -> np.ndarray:
        if not t.devices:
            return np.zeros((int(SR * self.seconds), 2))
        x = self.process(self.source(t), t.devices)
        # Balance law: panning left attenuates the right side, and vice versa.
        gl, gr = (1.0, 1.0 + t.pan) if t.pan <= 0 else (1.0 - t.pan, 1.0)
        fader = t.volume / 0.85
        return np.stack([x * gl, x * gr], axis=1) * fader

    def master_out(self) -> np.ndarray:
        mix = sum(self.track_out(t) for t in self.tracks if not t.name.startswith("tm-bounce"))
        return np.stack([self.process(mix[:, c], self.master) for c in range(2)], axis=1)

    # -------------------------------------------------------------- commands
    def t(self, i: int) -> Track:
        if not 0 <= i < len(self.tracks):
            raise IndexError("Track index out of range")
        return self.tracks[i]

    def dev(self, p: dict, master: bool = False):
        devs = self.master if master else self.t(p["track_index"]).devices
        i = p["device_index"]
        if not 0 <= i < len(devs):
            raise IndexError("Device index out of range")
        return devs[i]

    def set_param(self, dev, parameter, value) -> dict:
        name, ps = dev
        par = ps[parameter] if isinstance(parameter, int) else next(
            (q for q in ps if q.name == parameter), None)
        if par is None:
            raise Exception(f"Parameter not found: {parameter}")
        if isinstance(value, str):
            if par.quantized:
                hits = [i for i, it in enumerate(par.items) if it.lower() == value.lower()]
                if len(hits) != 1:
                    raise ValueError("Display label is missing or ambiguous")
                native = hits[0]
            else:
                if par.fmt in ("plain", "q") or "inf" in par.display(par.min):
                    raise ValueError("Use a native number")
                target = _num(value)
                lo, hi = par.min, par.max
                asc = _num(par.display(hi)) > _num(par.display(lo))
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if (_num(par.display(mid)) < target) == asc:
                        lo = mid
                    else:
                        hi = mid
                native = (lo + hi) / 2
        else:
            native = min(par.max, max(par.min, float(value)))
            if par.quantized:
                native = round(native)
        par.value = float(native)
        return {"device": name, "parameter": par.name, "value": par.value, "display": par.display()}

    def handle(self, cmd: str, p: dict):
        self.commands.append(cmd)
        if cmd == "batch":
            results = []
            for c in p["commands"]:
                results.append({"type": c["type"], "result": self.handle(c["type"], c.get("params", {}))})
            return {"executed": len(results), "results": results}
        if cmd == "get_session_info":
            return {"tempo": self.tempo, "track_count": len(self.tracks), "live_version": "12.2-fake",
                    "bridge_version": "1.8.1"}
        if cmd == "set_tempo":
            self.tempo = float(p["tempo"])
            return {"tempo": self.tempo}
        if cmd == "set_time_signature":
            self.sig = (p.get("numerator", 4), p.get("denominator", 4))
            return {}
        if cmd == "get_track_info":
            t = self.t(p["track_index"])
            return {"index": p["track_index"], "name": t.name,
                    "devices": [{"index": i, "name": d[0]} for i, d in enumerate(t.devices)],
                    "clip_slots": [{"index": 0, "has_clip": t.session_clip is not None}]}
        if cmd in ("create_midi_track", "create_audio_track"):
            self.tracks.append(Track(f"{len(self.tracks) + 1}-Track", cmd == "create_midi_track"))
            return {"index": len(self.tracks) - 1}
        if cmd == "set_track_name":
            self.t(p["track_index"]).name = p["name"]
            return {}
        if cmd == "delete_track":
            self._finalize(self.t(p["track_index"]).name)
            self.tracks.pop(p["track_index"])
            return {"deleted": True}
        if cmd == "search_browser":
            names = CATALOG.get(p.get("category") or "", sum(CATALOG.values(), []))
            hits = [{"name": n, "uri": f"query:{p.get('category')}#{n}", "is_device": n not in self.plugins_not_device}
                    for n in names if p["query"].lower() in n.lower() and n not in self.search_hides]
            return {"matches": hits}
        if cmd == "get_browser_items_at_path":
            parts = p["path"].split("/")
            if parts[0] != "plugins":
                return {"path": p["path"], "error": "Unknown or unavailable category", "items": []}
            folder = lambda n: {"name": n, "is_folder": True, "is_device": False, "is_loadable": False, "uri": None}
            if len(parts) == 1:
                return {"items": [folder("VST3")]}
            if parts[1:] == ["VST3"]:
                return {"items": [folder("Native Instruments"), folder("Other")]}
            vendor = {"Native Instruments": ["Kontakt 8"]}.get(parts[2], [n for n in CATALOG["plugins"] if n != "Kontakt 8"]) if len(parts) == 3 else []
            return {"items": [{"name": n, "is_folder": False, "is_device": n not in self.plugins_not_device,
                               "is_loadable": True, "uri": f"query:plugins#{n}"} for n in vendor]}
        if cmd == "load_browser_item":
            self.t(p["track_index"]).devices.append(make_device(p["item_uri"].split("#", 1)[1]))
            return {"loaded": True}
        if cmd == "load_device_to_master":
            dev = make_device(p["item_uri"].split("#", 1)[1])
            # Live inserts after the selected device; `master_insert_front` mimics having the
            # first device selected, which scrambles the order of a chain loaded one by one.
            if self.master_insert_front:
                self.master.insert(0, dev)
            else:
                self.master.append(dev)
            return {"loaded": True}
        if cmd == "delete_device":
            dev = self.dev(p)
            self.t(p["track_index"]).devices.remove(dev)
            return {"deleted": dev[0]}
        if cmd == "get_device_parameters":
            name, ps = self.dev(p)
            return {"device": name, "parameters": [q.as_dict(i) for i, q in enumerate(ps)]}
        if cmd == "get_master_device_parameters":
            name, ps = self.dev(p, master=True)
            return {"device": name, "parameters": [q.as_dict(i) for i, q in enumerate(ps)]}
        if cmd == "set_device_parameter":
            return self.set_param(self.dev(p), p["parameter"], p["value"])
        if cmd == "set_master_device_parameter":
            return self.set_param(self.dev(p, master=True), p["parameter"], p["value"])
        if cmd == "set_track_pan":
            self.t(p["track_index"]).pan = max(-1, min(1, float(p["pan"])))
            return {"panning": self.t(p["track_index"]).pan}
        if cmd == "set_track_volume":
            self.t(p["track_index"]).volume = float(p["volume"])
            return {"volume": self.t(p["track_index"]).volume}
        if cmd == "get_track_routing":
            t = self.t(p["track_index"])
            types = ["Ext. In", "Resampling"] + [x.name for x in self.tracks if x is not t]
            chans = ["Pre FX", "Post FX", "Post Mixer"] if t.input_type not in ("Ext. In", "Resampling") else []
            return {"input_routing_type": {"display_name": t.input_type},
                    "available_input_routing_types": [{"display_name": n} for n in types],
                    "available_input_routing_channels": [{"display_name": n} for n in chans]}
        if cmd == "set_track_routing":
            t = self.t(p["track_index"])
            if p["field"] == "input_routing_type":
                t.input_type = p["display_name"]
            else:
                t.input_channel = p["display_name"]
            return {}
        if cmd == "set_track_monitoring":
            self.t(p["track_index"]).monitoring = p["state"]
            return {}
        if cmd == "set_track_arm":
            self.t(p["track_index"]).arm = bool(p["arm"])
            if not p["arm"] and self.finalize_on_disarm:
                self._finalize(self.t(p["track_index"]).name)
            return {"arm": True}
        if cmd == "create_clip":
            self.t(p["track_index"]).session_clip = {"length": p["length"], "notes": []}
            return {}
        if cmd == "delete_clip":
            self.t(p["track_index"]).session_clip = None
            return {}
        if cmd == "add_notes_to_clip":
            self.t(p["track_index"]).session_clip["notes"] = p["notes"]
            return {}
        if cmd == "set_clip_name":
            return {}
        if cmd == "get_arrangement_clips":
            return {"clips": self.t(p["track_index"]).arrangement}
        if cmd == "delete_arrangement_clip":
            self.t(p["track_index"]).arrangement.pop(p["arrangement_clip_index"])
            return {}
        if cmd == "duplicate_session_clip_to_arrangement":
            t = self.t(p["track_index"])
            t.arrangement.append({"is_midi_clip": True, "is_audio_clip": False, "file_path": None,
                                  "notes": len(t.session_clip["notes"])})
            return {}
        if cmd == "set_record_mode":
            self.record_mode = bool(p["enabled"])
            return {}
        if cmd == "stop_playback":
            self._finish_recording()
            return {}
        if cmd in ("stop_all_clips", "back_to_arranger", "set_current_song_time", "start_playback"):
            return {}
        raise Exception(f"fake: unknown command {cmd}")

    def _finalize(self, track_name: str) -> None:
        if track_name in self.pending:
            path, audio, take = self.pending.pop(track_name)
            if take not in self.never_finalize:
                sf.write(path, audio.astype(np.float32), SR, subtype="FLOAT")

    def _finish_recording(self) -> None:
        for t in self.tracks:
            if not t.arm:
                continue
            if t.input_type == "Resampling":
                audio = self.master_out()
            else:
                audio = self.track_out(next(x for x in self.tracks if x.name == t.input_type))
            self.takes += 1
            path = self.dir / f"{t.name} take{self.takes}.wav"
            if self.takes in self.broken_takes:          # simulate Live leaving a file unreadable
                path.write_bytes(b"")
            elif self.defer_finalize:
                path.write_bytes(b"\0" * 65536 * 3)     # in progress: no valid header yet
                self.pending[t.name] = (path, audio, self.takes)
            elif self.takes not in self.missing_takes:
                sf.write(path, audio.astype(np.float32), SR, subtype="FLOAT")
            t.arrangement.append({"is_audio_clip": True, "is_midi_clip": False, "file_path": str(path)})


class FakeLiveServer:
    """Serve a FakeLive over TCP with the Remote Script's framing."""

    def __init__(self, live: FakeLive | None = None):
        self.live = live or FakeLive()
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn: socket.socket) -> None:
        buf = b""
        with conn:
            while True:
                data = conn.recv(1 << 16)
                if not data:
                    return
                buf += data
                try:
                    req = json.loads(buf.decode())
                except json.JSONDecodeError:
                    continue
                buf = b""
                try:
                    res = {"status": "success", "result": self.live.handle(req["type"], req.get("params", {}))}
                except Exception as e:  # mirror the Remote Script: errors become status=error
                    res = {"status": "error", "message": str(e)}
                conn.sendall(json.dumps(res, default=_jsonable).encode())

    def close(self) -> None:
        self.sock.close()


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if math.isnan(o):
        return None
    raise TypeError(type(o))
