"""Minimal client for the ableton-live-mcp Remote Script socket (default 127.0.0.1:9877).

Wire format (mirrors mcp-server-ableton-live 1.8.x `connection.py`): send one JSON
object `{"type": <command>, "params": {...}}`, receive one JSON object
`{"status": "success"|"error", "result": {...}, "message": "..."}`.
Only one request is in flight at a time.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

DEFAULT_HOST = os.environ.get("ABLETON_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("ABLETON_PORT", "9877"))
# The Remote Script gives each command a 10 s main-thread budget (60 s for create_audio_clip).
_SLOW = {"create_audio_clip": 60.0}


class LiveError(RuntimeError):
    pass


class LiveClient:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host, self.port = host, port
        self.sock: socket.socket | None = None

    # -- connection ------------------------------------------------------
    def connect(self) -> None:
        if self.sock is not None:
            return
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
        except OSError as e:
            self.sock = None
            raise LiveError(
                f"Cannot reach Ableton at {self.host}:{self.port} ({e}). Start Live and select "
                "AbletonMCP as a Control Surface (Settings > Tempo & MIDI). "
                "`uvx mcp-server-ableton-live install` installs the Remote Script."
            ) from None

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self) -> "LiveClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- protocol --------------------------------------------------------
    def send(self, command: str, **params: Any) -> dict:
        self.connect()
        timeout = _SLOW.get(command, 10.0) + 5.0
        if command == "batch":
            for c in params.get("commands", []):
                timeout = max(timeout, _SLOW.get(c.get("type", ""), 0.0) + 5.0)
        payload = json.dumps({"type": command, "params": params}).encode("utf-8")
        try:
            self.sock.settimeout(timeout)
            self.sock.sendall(payload)
            response = self._receive()
        except TimeoutError:
            self.close()
            raise LiveError(
                f"Timeout waiting for Live on '{command}'. A modal dialog in Live blocks the "
                "Remote Script - dismiss it and retry."
            ) from None
        except OSError as e:
            self.close()
            raise LiveError(f"Connection to Live lost during '{command}': {e}") from None
        if response.get("status") == "error":
            raise LiveError(f"{command}: {response.get('message', 'unknown error')}")
        return response.get("result", {}) or {}

    def _receive(self) -> dict:
        chunks: list[bytes] = []
        while True:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("connection closed mid-response")
            chunks.append(chunk)
            if not chunk.rstrip().endswith(b"}"):
                continue
            try:
                return json.loads(b"".join(chunks).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    def batch(self, commands: list[tuple[str, dict]]) -> list[dict]:
        """Run several commands in one main-thread task / one undo step."""
        if not commands:
            return []
        r = self.send("batch", commands=[{"type": t, "params": p} for t, p in commands])
        return [x.get("result", {}) for x in r.get("results", [])]
