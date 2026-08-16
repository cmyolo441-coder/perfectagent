"""REPORT — enterprise export & forecasting.

Two capabilities, both pure folds over the event log (deterministic,
zero model calls):

  * EXPORT — a full audit report of the session: timeline, tool calls,
    judge verdicts, goal outcome, costs, subagent activity. Markdown or
    self-contained HTML — for handoff, review, or compliance.
  * FORECAST — projection from measured reality: tokens per turn, cost
    per turn, goal velocity -> estimated turns remaining; per-model
    usage breakdown. Numbers, not vibes.
"""

from __future__ import annotations

import html
import time
from datetime import datetime

from .kernel import EventLog, fold

INTERESTING = ("user.message", "assistant.message", "tool.call",
               "tool.result", "judge.verdict", "clause.proven",
               "clause.regressed", "goal.closed", "focus.stop",
               "workflow.done", "crew.done", "team.report",
               "swarm.report", "snapshot.taken", "autonomy.changed")


# ---------------------------------------------------------------------------
# Data gathering (one pass over the log)
# ---------------------------------------------------------------------------

def _gather(log: EventLog) -> dict:
    st = fold(log)
    events = log.events()
    tools: dict[str, int] = {}
    tool_errors = 0
    verdicts_pass = verdicts_fail = 0
    timeline: list[dict] = []
    models: dict[str, dict] = {}
    for ev in events:
        d = ev.data
        t = ev.type
        if t == "tool.call":
            name = d.get("name", "?")
            tools[name] = tools.get(name, 0) + 1
        if t == "tool.result" and d.get("status") == "error":
            tool_errors += 1
        if t == "judge.verdict":
            if d.get("passed"):
                verdicts_pass += 1
            else:
                verdicts_fail += 1
        if t == "cost.incurred":
            m = models.setdefault(str(d.get("model", "?")),
                                  {"tokens_in": 0, "tokens_out": 0,
                                   "usd": 0.0, "calls": 0})
            m["tokens_in"] += int(d.get("tokens_in", 0) or 0)
            m["tokens_out"] += int(d.get("tokens_out", 0) or 0)
            m["usd"] += float(d.get("usd", 0.0) or 0.0)
            m["calls"] += 1
        if t in INTERESTING:
            timeline.append({"seq": ev.seq, "ts": ev.ts, "type": t,
                             "data": d})
    return {"st": st, "events": events, "tools": tools,
            "tool_errors": tool_errors,
            "verdicts": (verdicts_pass, verdicts_fail),
            "timeline": timeline, "models": models}


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def export_markdown(log: EventLog, title: str = "FullAgent session report"
                    ) -> str:
    g = _gather(log)
    st = g["st"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# {title}",
             f"_generated {now} — every number below is a fold of the "
             f"event log; nothing is estimated or narrated._",
             "",
             "## Summary",
             f"- **events**: {len(g['events'])}   "
             f"**branch**: {st.branch}   **head seq**: {st.head_seq}",
             f"- **tool calls**: {st.tool_calls} "
             f"(errors: {g['tool_errors']})   "
             f"**commands run**: {st.commands_run}",
             f"- **judge verdicts**: {g['verdicts'][0]} passed / "
             f"{g['verdicts'][1]} failed",
             f"- **cost**: {st.cost_summary()}",
             f"- **episodes**: {len(st.episodes)}   "
             f"**dead-ends**: {len(st.dead_ends)}   "
             f"**facts**: {len(st.facts)}"]
    if st.files_touched:
        lines.append(f"- **files touched**: {len(st.files_touched)}")

    if g["models"]:
        lines += ["", "## Model usage", "| model | calls | tokens in | "
                  "tokens out |", "|---|---|---|---|"]
        for m, d in sorted(g["models"].items()):
            lines.append(f"| {m} | {d['calls']} | {d['tokens_in']:,} | "
                         f"{d['tokens_out']:,} |")

    if g["tools"]:
        lines += ["", "## Tool calls", "| tool | count |", "|---|---|"]
        for name, n in sorted(g["tools"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {n} |")

    goal = st.goal
    if goal:
        lines += ["", "## Goal",
                  f"- statement: {goal.get('statement', '')}",
                  f"- clauses: {len(goal.get('clauses', []))}",
                  f"- closed: {goal.get('closed_state', 'still active')}"]

    lines += ["", "## Timeline (key events)",
              "| seq | time | event | detail |", "|---|---|---|---|"]
    for row in g["timeline"][-200:]:
        stamp = time.strftime("%H:%M:%S", time.localtime(row["ts"]))
        d = row["data"]
        if row["type"] in ("user.message", "assistant.message"):
            detail = str(d.get("text", ""))[:80].replace("|", "\\|")
        elif row["type"] == "tool.call":
            detail = str(d.get("name", ""))
        elif row["type"] == "tool.result":
            detail = f"{d.get('name', '')} -> {d.get('status', '')}"
        elif row["type"] == "judge.verdict":
            detail = f"{'PASS' if d.get('passed') else 'FAIL'} " \
                     f"{d.get('kind', '')}: {str(d.get('detail', ''))[:60]}"
        else:
            detail = str(d)[:80].replace("|", "\\|")
        detail = detail.replace("\n", " ")
        lines.append(f"| {row['seq']} | {stamp} | {row['type']} | "
                     f"{detail} |")

    lines += ["", "---",
              "_FullAgent — the complete history (every event, including "
              "the parts trimmed above) remains in the append-only "
              "event log and can be replayed with `python main.py "
              "replay`._"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML export (self-contained, no external assets)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
      background:#1e1f29;color:#f8f8f2;margin:2rem auto;max-width:960px;
      padding:0 1rem;line-height:1.5}}
 h1{{color:#bd93f9}} h2{{color:#8be9fd;border-bottom:1px solid #44475a;
      padding-bottom:.3rem}}
 table{{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.85rem}}
 th,td{{border:1px solid #44475a;padding:.35rem .6rem;text-align:left}}
 th{{background:#282a36;color:#50fa7b}}
 tr:nth-child(even){{background:#232530}}
 code{{background:#282a36;padding:.1rem .3rem;border-radius:3px}}
 .pass{{color:#50fa7b}} .fail{{color:#ff5555}}
 em{{color:#6272a4}}
</style></head><body>
{body}
</body></html>
"""


def export_html(log: EventLog, title: str = "FullAgent session report"
                ) -> str:
    """Render the markdown report as self-contained HTML (dracula-ish)."""
    md = export_markdown(log, title)
    body: list[str] = []
    in_table = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("| "):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # separator row
            tag = "th" if not in_table else "td"
            if not in_table:
                body.append("<table>")
                in_table = True
            rendered = []
            for c in cells:
                cls = ""
                if c.startswith("PASS"):
                    cls = ' class="pass"'
                elif c.startswith("FAIL"):
                    cls = ' class="fail"'
                rendered.append(f"<{tag}{cls}>{html.escape(c)}</{tag}>")
            body.append("<tr>" + "".join(rendered) + "</tr>")
        else:
            if in_table:
                body.append("</table>")
                in_table = False
            if line.startswith("- "):
                content = html.escape(line[2:])
                content = content.replace("**", "")
                body.append(f"<p>•&nbsp;{content}</p>")
            elif line.startswith("_") and line.endswith("_"):
                body.append(f"<p><em>{html.escape(line.strip('_'))}"
                            f"</em></p>")
            elif line.strip():
                content = html.escape(line).replace("**", "")
                body.append(f"<p>{content}</p>")
    if in_table:
        body.append("</table>")
    return _HTML_TEMPLATE.format(title=html.escape(title),
                                 body="\n".join(body))


# ---------------------------------------------------------------------------
# Forecast — projection from measured reality
# ---------------------------------------------------------------------------

def forecast(log: EventLog) -> dict:
    """Deterministic projection: measured tokens/turn, cost/turn, and —
    when a goal is active — turns remaining from real velocity."""
    st = fold(log)
    events = log.events()
    user_turns = sum(1 for e in events if e.type == "user.message")
    result: dict = {"turns": user_turns,
                    "tokens_in": st.tokens_in,
                    "tokens_out": st.tokens_out,
                    "cost_usd": st.cost_usd}
    if user_turns:
        result["tokens_in_per_turn"] = round(st.tokens_in / user_turns)
        result["tokens_out_per_turn"] = round(st.tokens_out / user_turns)
        result["cost_per_turn_usd"] = round(st.cost_usd / user_turns, 6)

    goal = st.goal
    if goal and goal.get("clauses"):
        distance_measures = st.distance_measures
        if len(distance_measures) >= 2:
            first = float(distance_measures[0].get("distance", 1.0))
            last = float(distance_measures[-1].get("distance", first))
            span = max(1, len(distance_measures) - 1)
            delta_per_tick = (first - last) / span
            result["goal_distance"] = last
            result["velocity_per_tick"] = round(delta_per_tick, 4)
            if delta_per_tick > 1e-6:
                result["est_ticks_remaining"] = int(
                    last / delta_per_tick + 0.999)
            else:
                result["est_ticks_remaining"] = None  # stalled
        else:
            result["goal_distance"] = float(
                goal.get("distance", 1.0)) if "distance" in goal else None
    return result


def format_forecast(f: dict) -> str:
    lines = ["FORECAST — measured, not guessed",
             f"  turns so far      : {f['turns']}",
             f"  tokens            : {f['tokens_in']:,} in / "
             f"{f['tokens_out']:,} out"]
    if "tokens_in_per_turn" in f:
        lines.append(f"  per turn          : ~{f['tokens_in_per_turn']:,} "
                     f"in / ~{f['tokens_out_per_turn']:,} out"
                     + (f" · ${f['cost_per_turn_usd']:.4f}"
                        if f.get("cost_per_turn_usd") else ""))
    if "goal_distance" in f and f["goal_distance"] is not None:
        done = (1 - f["goal_distance"]) * 100
        lines.append(f"  goal progress     : {done:.0f}% "
                     f"(distance {f['goal_distance']:.2f})")
        est = f.get("est_ticks_remaining")
        if est is None:
            lines.append("  projection        : ⚠ no measurable velocity "
                         "— stalled or not enough data")
        else:
            lines.append(f"  projection        : ~{est} goal-tick(s) to "
                         f"done at current velocity")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EventLog(Path(td) / "report-test.jsonl")
            log.append("user.message", {"text": "build the thing"})
            log.append("tool.call", {"name": "write_file",
                                     "args": {"path": "x.py"}})
            log.append("tool.result", {"name": "write_file",
                                       "status": "done"})
            log.append("tool.call", {"name": "run_command",
                                     "args": {"command": "pytest"}})
            log.append("tool.result", {"name": "run_command",
                                       "status": "error"})
            log.append("judge.verdict", {"passed": True,
                                         "kind": "exit_code",
                                         "detail": "ok"})
            log.append("cost.incurred", {"usd": 0.0, "tokens_in": 500,
                                         "tokens_out": 120,
                                         "model": "test-model"})
            log.append("assistant.message", {"text": "done"})

            md = export_markdown(log)
            assert "# FullAgent session report" in md
            assert "write_file" in md and "run_command" in md
            assert "500" in md  # token counts rendered
            assert "PASS exit_code" in md

            page = export_html(log)
            assert page.startswith("<!DOCTYPE html>")
            assert "<table>" in page and "write_file" in page
            assert "FullAgent session report" in page

            f = forecast(log)
            assert f["turns"] == 1
            assert f["tokens_in"] == 500 and f["tokens_out"] == 120
            assert f["tokens_in_per_turn"] == 500
            text = format_forecast(f)
            assert "measured" in text

            print("REPORT SELF-TEST PASS")

    _self_test()
