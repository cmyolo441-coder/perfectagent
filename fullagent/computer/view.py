"""Pure, bounded terminal dashboard rendering; no fabricated progress.

The same rendering is used by the live TUI and width/visual regression tests.
Values show actual checkpoints and events, not decorative simulations.
"""
from __future__ import annotations

import unicodedata
from .state import ROLES, safe_text

BLUE = "#5E9FE8"
GREEN = "#72BC8F"
ORANGE = "#DE9255"
RED = "#E97366"
DIM = "#999999"
FG = "#EAEAEA"


def cells(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def fit(value, width, pad=False):
    text = safe_text(value, 16000).replace("\n", " ").replace("\t", " ")
    width = max(0, width)
    if cells(text) > width:
        out, used = [], 0
        for c in text:
            size = cells(c)
            if used + size > max(0, width-1):
                break
            out.append(c)
            used += size
        text = "".join(out) + ("…" if width else "")
    if pad:
        text += " " * max(0, width-cells(text))
    return text


def tone(status):
    if status in ("done", "completed", "pass"):
        return GREEN
    if status in ("error", "needs_attention", "blocked", "budget_exhausted"):
        return RED
    if status in ("waiting", "paused", "planned", "cancelling", "cancelled"):
        return ORANGE
    if status == "idle":
        return DIM
    return BLUE


def dashboard_lines(snapshot, width=100, height=19):
    """Return (style, text) lines, each within terminal display-cell width."""
    width, height = max(12, int(width)), max(1, int(height))
    settings = snapshot.get("settings", {})
    status = snapshot.get("status", "ready")
    phase = snapshot.get("phase", "ready")
    elapsed = int(snapshot.get("elapsed_seconds", 0))
    title = f" COMPUTER  /on  ·  {status.upper()}  ·  {phase}  ·  {elapsed//60:02d}:{elapsed%60:02d}"
    out = [(f"bold {tone(status)}", title)]
    cap = settings.get("max_parallel", 8)
    out.append((DIM, f" 8 agents · parallel cap {cap} · {snapshot.get('root', '.')}") )
    budget = settings.get("token_budget", 0)
    used, reserved = snapshot.get("charged_tokens", 0), snapshot.get("reserved_tokens", 0)
    measured, estimated = snapshot.get("reported_tokens", 0), snapshot.get("estimated_tokens", 0)
    if height >= 13:
        out.append((FG, f" tokens {measured:,} reported + {estimated:,} est. | reserved {reserved:,} | cap {budget:,}"))
    if height >= 16:
        out.append((DIM, "─" * width))
    agents = snapshot.get("agents", {})
    tasks = snapshot.get("tasks", {})
    for agent_id, name, role, focus in ROLES:
        a = agents.get(agent_id, {})
        state = a.get("status", "idle")
        task_state = tasks.get(agent_id, {}).get("status")
        tools = int(a.get("tools", 0))
        tin, tout = int(a.get("tokens_in", 0)), int(a.get("tokens_out", 0))
        activity = a.get("error") or a.get("activity", focus)
        head = f" {agent_id} {name:<6} {state:<8}"
        if width >= 100:
            head += f" {role:<12} {tools:>3} tools {tin:>6}→{tout:<5} "
        elif width >= 70:
            head += f" {tools:>3} tools {tin+tout:>6} tok "
        else:
            head += f" {tools:>2}t "
        out.append((tone(state), head + fit(activity, max(0, width-cells(head)))))
    checks = snapshot.get("checks", [])
    passed = sum(bool(c.get("ok")) for c in checks)
    done = sum(t.get("status") == "done" for t in tasks.values())
    out.append((DIM, f" tasks {done}/8 reported done · checks {passed}/{len(checks)} passed · sources {len(snapshot.get('sources', []))}"))
    available = height - len(out) - 1
    events = snapshot.get("recent", [])
    if available > 0:
        if events:
            for event in events[-min(available, 5):]:
                message = event.get("message", "").replace("\n", " ⏎ ")
                actor = event.get("agent", "system")
                kind = event.get("kind", "event")
                out.append((ORANGE if "error" in kind else DIM, f" {actor} · {kind}: {message}"))
        else:
            out.append((DIM, " Type your project goal. Research uses your selected API model; no model call until you submit."))
    footer = " /computer pause|resume|cancel|report · Ctrl+C cancel · /off normal chat"
    if width < 80:
        footer = " /computer help · Ctrl+C cancel · /off"
    out.append((BLUE, footer))
    # Extremely short terminals show a compact status rather than overflow.
    if len(out) > height:
        out = out[:max(0, height-1)] + [out[-1]]
    return [(style, fit(text, width)) for style, text in out]


def fragments(snapshot, width=100, height=19):
    result = []
    for index, (style, line) in enumerate(dashboard_lines(snapshot, width, height)):
        if index:
            result.append(("", "\n"))
        result.append((style, line))
    return result


def status_text(snapshot):
    return "\n".join(line for _, line in dashboard_lines(snapshot, 120, 20))
