"""Optional MCP server so an assistant (Claude Desktop / Claude Code) can drive tonematch
next to ableton-live-mcp. Run: `python -m tonematch.mcp_server` (needs `pip install 'tonematch[mcp]'`).

Both servers talk to the same AbletonMCP Remote Script; tonematch opens its own socket
only while a tool runs, and commands are serialized by the Remote Script.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP
except ImportError:  # mcp 2.x renamed it
    from mcp.server.mcpserver import MCPServer as FastMCP

from .config import load_config
from .live.client import LiveClient
from .pipeline import DEFAULT_STAGES, Project

mcp = FastMCP("tonematch")
CONFIG = Path(os.environ.get("TONEMATCH_CONFIG", "tonematch.toml"))


def _run(fn) -> str:
    log = io.StringIO()
    with redirect_stdout(log):
        proj = Project(load_config(CONFIG if CONFIG.exists() else None), LiveClient(), echo=print)
        result = fn(proj)
    out = log.getvalue()
    if result is not None:
        out += "\n" + (result if isinstance(result, str) else json.dumps(result, indent=1, default=str))
    return out.strip()


@mcp.tool()
def tonematch_analyze() -> str:
    """Separate the reference into stems, align the MIDI to it, pick comparison regions and
    measure loudness/spectrum/width of each stem. Run first."""
    return _run(lambda p: {"regions": p.analyze()["regions"]})


@mcp.tool()
def tonematch_build() -> str:
    """Create the template in Live: tracks, plug-in chains, MIDI clips, pans, master chain.
    Returns the manual steps (loading Kontakt libraries, Configure, saving)."""
    return _run(lambda p: "\n".join(p.build()))


@mcp.tool()
def tonematch_discover() -> str:
    """Show which configured plug-in parameters resolve to real Live parameters."""
    return _run(lambda p: p.discover())


@mcp.tool()
def tonematch_match(stages: str = ",".join(DEFAULT_STAGES), trials: int = 0) -> str:
    """Run closed-loop matching stages (comma list of levels,tone,eq,pan,master). Each take is
    recorded in real time, so the tone stage takes about trials x region length."""
    def go(p: Project):
        if trials:
            p.cfg.match["tone_trials"] = trials
        p.match([s.strip() for s in stages.split(",") if s.strip()])
        return p.applied()
    return _run(go)


@mcp.tool()
def tonematch_export() -> str:
    """Write per-device parameter snapshots, mixer sheet, SAVE_PRESETS.md and report.html."""
    from .export.presets import export_presets
    from .export.report import write_report

    def go(p: Project):
        out = export_presets(p)
        return {"outdir": str(out), "report": str(write_report(p))}
    return _run(go)


@mcp.tool()
def tonematch_status() -> str:
    """Current analysis summary and applied settings."""
    return _run(lambda p: {k: p.state.data.get(k) for k in ("applied", "discover")})


if __name__ == "__main__":
    mcp.run()
