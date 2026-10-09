"""Self-contained HTML report: reference vs. render tone curves, loss history, settings."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from ..analysis.features import BAND_CENTERS, Features

W, H, PAD = 640, 220, 36
CSS = """
:root{--bg:#fbfaf7;--fg:#1f2328;--muted:#6b7079;--grid:#e3e1db;--ref:#2f6fb3;--ren:#c2571a;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e8e6e3;--muted:#9aa0a8;--grid:#2c3036;
--ref:#79a9e0;--ren:#f0965a;--card:#1d2025}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:720px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px;margin:12px 0}
svg{width:100%;height:auto;display:block}text{fill:var(--muted);font-size:11px}
.legend span{display:inline-block;margin-right:16px}.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid)}
td.n{text-align:right;font-variant-numeric:tabular-nums}
pre{overflow:auto;font-size:12px;margin:0}
"""


def _x(i: int) -> float:
    return PAD + (W - 2 * PAD) * i / (len(BAND_CENTERS) - 1)


def _curve_svg(ref: np.ndarray, ren: np.ndarray | None) -> str:
    vals = np.concatenate([ref, ren if ren is not None else ref])
    lo, hi = float(np.floor(vals.min() / 6) * 6), float(np.ceil(vals.max() / 6) * 6)
    hi = hi if hi > lo else lo + 6

    def y(v: float) -> float:
        return PAD / 2 + (H - PAD) * (hi - v) / (hi - lo)

    parts = []
    for db in np.arange(lo, hi + 0.1, 6):
        parts.append(f'<line x1="{PAD}" x2="{W - PAD}" y1="{y(db):.1f}" y2="{y(db):.1f}" stroke="var(--grid)"/>'
                     f'<text x="{PAD - 6}" y="{y(db) + 4:.1f}" text-anchor="end">{db:+.0f}</text>')
    for i, f in enumerate(BAND_CENTERS):
        if i % 3 == 0:
            label = f"{f / 1000:.1f}k" if f >= 1000 else f"{f:.0f}"
            parts.append(f'<text x="{_x(i):.1f}" y="{H - 4}" text-anchor="middle">{label}</text>')

    def path(c: np.ndarray, color: str) -> str:
        d = " ".join(f"{'M' if i == 0 else 'L'}{_x(i):.1f},{y(v):.1f}" for i, v in enumerate(c))
        return f'<path d="{d}" fill="none" stroke="var({color})" stroke-width="2"/>'

    parts.append(path(ref, "--ref"))
    if ren is not None:
        parts.append(path(ren, "--ren"))
    return f'<svg viewBox="0 0 {W} {H}" role="img">{"".join(parts)}</svg>'


def _loss_svg(history: dict[str, list[float]]) -> str:
    if not history:
        return ""
    allv = [v for vs in history.values() for v in vs]
    lo, hi = 0.0, max(allv) * 1.05 or 1.0
    n = max(len(v) for v in history.values())
    colors = ["--ref", "--ren", "--muted", "--fg"]
    parts = [f'<line x1="{PAD}" x2="{W - PAD}" y1="{H - PAD / 2}" y2="{H - PAD / 2}" stroke="var(--grid)"/>']
    legend = []
    for k, (name, vs) in enumerate(history.items()):
        best = np.minimum.accumulate(vs)
        pts = " ".join(f"{PAD + (W - 2 * PAD) * i / max(1, n - 1):.1f},"
                       f"{PAD / 2 + (H - PAD) * (hi - v) / (hi - lo):.1f}" for i, v in enumerate(best))
        c = colors[k % len(colors)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var({c})" stroke-width="2"/>')
        legend.append(f'<span><i class="sw" style="background:var({c})"></i>{html.escape(name)}</span>')
    parts.append(f'<text x="{PAD}" y="12">best loss so far</text>')
    return f'<div class="legend">{"".join(legend)}</div><svg viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'


def write_report(project) -> Path:
    st = project.state.data
    cfg = project.cfg
    an = st.get("analysis", {})
    last = st.get("last_render", {})
    rows = []
    sections = []
    master_stem = "instrumental" if st.get("applied", {}).get("master", {}).get("target") == "instrumental" else "mix"
    for t in [*cfg.tracks, None]:
        name, stem = (t.name, t.stem) if t else ("Master", master_stem)
        ref_d = an.get("features", {}).get("0", {}).get(stem)
        if not ref_d:
            continue
        ref = Features.from_dict(ref_d)
        ren = Features.from_dict(last[name][0]) if last.get(name) else None
        sections.append(
            f'<h2>{html.escape(name)} <span class="muted">vs reference {html.escape(stem)}</span></h2>'
            f'<div class="card"><div class="legend"><span><i class="sw" style="background:var(--ref)"></i>reference</span>'
            f'<span><i class="sw" style="background:var(--ren)"></i>current render</span></div>'
            f'{_curve_svg(ref.tone_curve(), ren.tone_curve() if ren else None)}</div>')
        if ren:
            rows.append(f"<tr><td>{html.escape(name)}</td><td class=n>{ref.lufs:.1f}</td><td class=n>{ren.lufs:.1f}</td>"
                        f"<td class=n>{ref.crest_db:.1f}</td><td class=n>{ren.crest_db:.1f}</td>"
                        f"<td class=n>{ref.flatness:.3f}</td><td class=n>{ren.flatness:.3f}</td></tr>")
    history: dict[str, list[float]] = {}
    for e in st.get("log", []):
        if e["stage"] == "tone":
            for k, v in e["losses"].items():
                history.setdefault(k, []).append(v)
    ap = st.get("applied", {})
    mix_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td class=n>{ap.get('gain_db', {}).get(k, 0):+.1f}</td>"
        f"<td class=n>{ap.get('pan', {}).get(k, 0):+.2f}</td></tr>"
        for k in sorted(set(ap.get("gain_db", {})) | set(ap.get("pan", {}))))
    al = an.get("alignment", {})
    body = f"""<main>
<h1>Tone match report</h1>
<p class="muted">Alignment offset {al.get('offset_s', 0):.2f}s, tempo scale {al.get('scale', 1):.4f},
confidence {al.get('score', 0):.2f}. Curves are third-octave spectra with the overall level removed (region 1).</p>
{''.join(sections)}
<h2>Levels &amp; dynamics</h2><div class="card"><table>
<tr><th>Track</th><th>Ref LUFS</th><th>Render</th><th>Ref crest</th><th>Render</th><th>Ref flatness</th><th>Render</th></tr>
{''.join(rows)}</table></div>
<h2>Tone search</h2><div class="card">{_loss_svg(history) or '<p class="muted">No tone stage run yet.</p>'}</div>
<h2>Mixer</h2><div class="card"><table><tr><th>Track</th><th>Utility gain dB</th><th>Pan</th></tr>{mix_rows}</table></div>
<h2>EQ moves</h2><div class="card"><pre>{html.escape(json.dumps(ap.get('eq', {}), indent=1))}</pre></div>
</main>"""
    out = cfg.outdir / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html lang=en><head><meta charset=utf-8>"
                   f"<meta name=viewport content='width=device-width,initial-scale=1'>"
                   f"<title>Tone Match Report</title><style>{CSS}</style></head><body>{body}</body></html>")
    return out
