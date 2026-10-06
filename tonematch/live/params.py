"""Resolve config ParamSpecs against the parameters a device actually exposes in Live."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import ParamSpec


@dataclass
class LiveParam:
    index: int
    name: str
    min: float
    max: float
    value: float
    is_quantized: bool
    value_items: list[str]
    display: str = ""

    @classmethod
    def from_live(cls, d: dict) -> "LiveParam":
        return cls(
            index=int(d["index"]), name=str(d["name"]), min=float(d["min"]), max=float(d["max"]),
            value=float(d["value"]), is_quantized=bool(d.get("is_quantized")),
            value_items=list(d.get("value_items") or []),
            display=str(d.get("display", "")),
        )

    # -- value mapping ---------------------------------------------------
    def native(self, u: float) -> float:
        """Normalized 0..1 -> native value (rounded for quantized params)."""
        v = self.min + min(1.0, max(0.0, u)) * (self.max - self.min)
        return float(round(v)) if self.is_quantized else v

    def normalized(self, native: float) -> float:
        span = self.max - self.min
        return (native - self.min) / span if span else 0.0

    def choices(self) -> list[tuple[int, str]]:
        """(native value, label) for each option of a quantized parameter."""
        lo, hi = int(round(self.min)), int(round(self.max))
        labels = self.value_items if len(self.value_items) == hi - lo + 1 else [str(v) for v in range(lo, hi + 1)]
        return list(zip(range(lo, hi + 1), labels))


@dataclass
class Resolved:
    spec: ParamSpec
    param: LiveParam


def split_alternatives(pattern: str) -> list[str]:
    """Split a regex on top-level '|' only (alternatives inside groups stay intact)."""
    out, depth, cur, escaped = [], 0, [], False
    for ch in pattern:
        if escaped:
            cur.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "|" and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return out


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def expand_pattern(pattern: str, choices: dict[str, str]) -> list[str]:
    """Substitute {key} with the chosen label of categorical `key`; alternatives whose
    placeholders are unresolved are dropped (so they never match too broadly)."""
    alts = []
    for alt in split_alternatives(pattern):
        keys = _PLACEHOLDER.findall(alt)
        if any(k not in choices for k in keys):
            continue
        alts.append(_PLACEHOLDER.sub(lambda m: re.escape(choices[m.group(1)]), alt))
    return alts


def resolve(spec: ParamSpec, params: list[LiveParam], choices: dict[str, str] | None = None,
            taken: set[int] | None = None) -> LiveParam | None:
    """First alternative that matches wins; ties go to the shortest parameter name."""
    taken = taken or set()
    for alt in expand_pattern(spec.pattern, choices or {}):
        rx = re.compile(alt, re.I)
        hits = [p for p in params if p.index not in taken and rx.search(p.name)]
        if spec.kind == "categorical":
            hits = [p for p in hits if p.is_quantized and p.max > p.min]
        if hits:
            return min(hits, key=lambda p: (len(p.name), p.index))
    return None


def resolve_all(specs: list[ParamSpec], params: list[LiveParam],
                choices: dict[str, str] | None = None) -> tuple[list[Resolved], list[ParamSpec]]:
    resolved, missing, taken = [], [], set()
    for spec in specs:
        p = resolve(spec, params, choices, taken)
        if p is None:
            missing.append(spec)
        else:
            taken.add(p.index)
            resolved.append(Resolved(spec, p))
    return resolved, missing
