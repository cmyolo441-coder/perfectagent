"""DASHBOARD — live observability over the Temporal Kernel.

The whole system is already event-sourced; the dashboard is simply the
X-ray: a real-time projection of the ledger into one screen — cost,
tokens, goal progress, active sub-agents, routing savings, speculation
hit-rate, dead-ends, verdicts, loop alerts, and the live event stream.

Design (pure Python, stdlib only):
  * Every panel is a pure fold over the EventLog — the dashboard keeps no
    state of its own, so it can never disagree with the kernel.
  * render() returns plain text (the TUI colours it); snapshot() returns
    the raw dict for programmatic consumers.
  * tail() streams the newest events since a seq cursor, so the TUI can
    poll cheaply without re-rendering everything.
"""

from __future__ import annotations

import time

from .kernel import EventLog, fold

# panels the dashboard renders, in display order
PANELS = ("cost", "goal", "agents", "router", "speculator", "memory",
          "health", "engineering", "stream")


class Dashboard:
    """Read-only live projection of the event log. Never writes."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    # -- snapshot (raw dict) ---------------------------------------------------

    def snapshot(self) -> dict:
        st = fold(self.log)
        goal = st.goal
        proven = len(st.goal_done)
        clauses = len(goal.get("clauses") or []) if goal else 0

        # sub-agent activity from the crew roster
        crew_done = sum(1 for e in self.log.events()
                        if e.type == "crew.done")

        # routing + speculation dividends
        routed = len(st.router_decisions)
        routed_cost = sum(float(d.get("est_cost", 0.0))
                          for d in st.router_decisions)
        prefetched = sum(1 for e in st.spec_events
                         if e.get("type") == "spec.prefetch")
        spec_hits = sum(1 for e in st.spec_events
                        if e.get("type") == "spec.hit")
        spec_misses = sum(1 for e in st.spec_events
                          if e.get("type") == "spec.miss")

        # healing + skills + council activity
        heals = sum(1 for e in st.heal_events
                    if e.get("type") == "heal.lesson")
        skills = sum(1 for e in st.skill_events
                     if e.get("type") == "skill.registered")
        councils = sum(1 for e in st.council_events
                       if e.get("type") == "council.verdict")

        # v4 engineering subsystem activity
        taint_findings = sum(len(e.get("findings") or [])
                             for e in st.analysis_events
                             if e.get("type") == "analysis.taint")
        graph_entities = 0
        for e in st.graph_events:
            graph_entities = max(graph_entities,
                                 int(e.get("entities", 0)))
        cov_runs = [e for e in st.coverage_events
                    if e.get("type") == "coverage.result"]
        cov_last = cov_runs[-1].get("percent", 0.0) if cov_runs else None
        fuzz_crashes = sum(1 for e in st.fuzz_events
                           if e.get("type") == "fuzz.crash")
        mut_reports = [e for e in st.mutation_events
                       if e.get("type") == "mutation.result"]
        mut_last = mut_reports[-1].get("score") if mut_reports else None

        return {
            "head_seq": st.head_seq,
            "cost_usd": st.cost_usd,
            "tokens_in": st.tokens_in,
            "tokens_out": st.tokens_out,
            "tool_calls": st.tool_calls,
            "tool_errors": st.tool_errors,
            "commands_run": st.commands_run,
            "files_touched": len(st.files_touched),
            "goal_active": bool(goal and goal.get("statement")),
            "goal_statement": (goal or {}).get("statement", ""),
            "clauses_proven": proven,
            "clauses_total": clauses,
            "crew_done": crew_done,
            "routed": routed,
            "routed_cost": round(routed_cost, 4),
            "spec_prefetched": prefetched,
            "spec_hits": spec_hits,
            "spec_misses": spec_misses,
            "episodes": len(st.episodes),
            "dead_ends": len(st.dead_ends),
            "facts": len(st.facts),
            "verdicts": len(st.verdicts),
            "verdicts_failed": sum(1 for v in st.verdicts
                                   if not v.get("passed")),
            "loop_alerts": len(st.loop_alerts),
            "budget_events": len(st.budget_events),
            "heals": heals,
            "skills": skills,
            "councils": councils,
            "taint_findings": taint_findings,
            "graph_entities": graph_entities,
            "cov_runs": len(cov_runs),
            "cov_last": cov_last,
            "fuzz_crashes": fuzz_crashes,
            "mut_runs": len(mut_reports),
            "mut_last": mut_last,
        }

    # -- render (text) -----------------------------------------------------------

    def render(self, width: int = 62) -> str:
        s = self.snapshot()
        bar = "─" * width
        lines = ["◆ FULLAGENT LIVE DASHBOARD", bar]

        # cost panel
        lines.append(
            f" COST   ${s['cost_usd']:.4f}   "
            f"{s['tokens_in']}→{s['tokens_out']} tok   "
            f"tools {s['tool_calls']} (err {s['tool_errors']})   "
            f"cmds {s['commands_run']}   files {s['files_touched']}")

        # goal panel
        if s["goal_active"]:
            pct = (s["clauses_proven"] / s["clauses_total"] * 100
                   if s["clauses_total"] else 0)
            filled = int(round(pct / 100 * 20))
            gbar = "█" * filled + "░" * (20 - filled)
            lines.append(f" GOAL   [{gbar}] {pct:.0f}%  "
                         f"{s['clauses_proven']}/{s['clauses_total']} "
                         f"clauses  \"{s['goal_statement'][:34]}\"")
        else:
            lines.append(" GOAL   none active")

        # agents panel
        lines.append(f" AGENTS crew done {s['crew_done']}   "
                     f"councils {s['councils']}")

        # router panel
        lines.append(f" ROUTER {s['routed']} routed   "
                     f"est ${s['routed_cost']:.4f}")

        # speculator panel
        total_spec = s["spec_hits"] + s["spec_misses"]
        rate = (s["spec_hits"] / total_spec) if total_spec else 0.0
        lines.append(f" SPEC   prefetched {s['spec_prefetched']}   "
                     f"hits {s['spec_hits']}   misses {s['spec_misses']}   "
                     f"rate {rate:.0%}")

        # memory panel
        lines.append(f" MEMORY episodes {s['episodes']}   "
                     f"facts {s['facts']}   "
                     f"dead-ends {s['dead_ends']}   "
                     f"heals {s['heals']}   skills {s['skills']}")

        # health panel
        lines.append(f" HEALTH verdicts {s['verdicts']} "
                     f"(failed {s['verdicts_failed']})   "
                     f"loop alerts {s['loop_alerts']}   "
                     f"budget events {s['budget_events']}")

        # engineering panel (v4: analysis, graph, coverage, fuzz, mutation)
        cov = (f"{s['cov_last']:.0f}%" if s["cov_last"] is not None
               else "—")
        mut = (f"{s['mut_last']:.0%}" if s["mut_last"] is not None
               else "—")
        lines.append(f" ENGINE taint {s['taint_findings']}   "
                     f"graph {s['graph_entities']} ent   "
                     f"cov {cov} ({s['cov_runs']} runs)   "
                     f"fuzz ⚠{s['fuzz_crashes']}   "
                     f"mut {mut} ({s['mut_runs']} runs)")

        lines.append(bar)
        return "\n".join(lines)

    # -- event stream --------------------------------------------------------------

    def tail(self, since_seq: int = -1, limit: int = 12) -> list[dict]:
        """Newest events with seq > since_seq (oldest first), for a live
        ticker. Cheap: walks the chain once."""
        out = []
        for ev in self.log.events():
            if ev.seq > since_seq:
                out.append({"seq": ev.seq, "type": ev.type,
                            "actor": ev.actor,
                            "summary": _summarise(ev.type, ev.data)})
        return out[-limit:]


def _summarise(type_: str, data: dict) -> str:
    """One-line human summary of an event for the ticker."""
    if type_ == "user.message":
        return str(data.get("text", ""))[:60]
    if type_ == "assistant.message":
        return str(data.get("text", ""))[:60]
    if type_ == "tool.call":
        return f"{data.get('name', '?')}"
    if type_ == "tool.result":
        return f"{data.get('name', '?')} -> {data.get('status', '?')}"
    if type_ == "cost.incurred":
        return f"${float(data.get('usd', 0)):.4f}"
    if type_ == "router.decision":
        return f"-> {data.get('model', '?')}"
    if type_ == "spec.hit":
        return f"cache hit {data.get('tool', '?')}"
    if type_ == "judge.verdict":
        return f"{'PASS' if data.get('passed') else 'FAIL'} " \
               f"{data.get('kind', '?')}"
    if type_ == "goal.distance":
        return f"distance {float(data.get('distance', 1)):.2f}"
    if type_ in ("heal.lesson", "skill.registered", "council.verdict"):
        return type_
    if type_ == "coverage.result":
        return f"coverage {data.get('percent', 0):.0f}% {data.get('path', '')}"
    if type_ == "fuzz.crash":
        return f"crash {data.get('error', '')[:40]}"
    if type_ == "mutation.result":
        return f"mutation score {data.get('score', 0):.0%}"
    if type_ == "analysis.taint":
        return f"taint {len(data.get('findings') or [])} finding(s)"
    return ""


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "dash.jsonl")
        dash = Dashboard(log)

        # empty log still renders every panel without crashing
        text = dash.render()
        assert "FULLAGENT LIVE DASHBOARD" in text
        assert "GOAL   none active" in text

        # seed a realistic session
        log.append("user.message", {"text": "fix the parser"}, actor="human")
        log.append("tool.call", {"name": "read_file",
                                 "args": {"path": "p.py"}})
        log.append("tool.result", {"name": "read_file", "status": "done"})
        log.append("tool.result", {"name": "edit_file", "status": "error"})
        log.append("cost.incurred", {"usd": 0.05, "tokens_in": 100,
                                     "tokens_out": 40})
        log.append("goal.set", {"statement": "fix parser",
                                "clauses": [{"id": "C1"}, {"id": "C2"}]})
        log.append("goal.clause.done", {"clause": "C1"})
        log.append("router.decision", {"model": "agnes-2.5-flash",
                                       "est_cost": 0.0})
        log.append("spec.prefetch", {"tool": "read_file"})
        log.append("spec.hit", {"tool": "read_file"})
        log.append("judge.verdict", {"passed": True, "kind": "exit_code"})
        log.append("judge.verdict", {"passed": False, "kind": "file_exists"})
        log.append("memory.episode", {"goal": "fix parser",
                                      "outcome": "success"})
        log.append("deadend.recorded", {"signature": "x", "reason": "y"})
        log.append("heal.lesson", {"root": "missing import"})
        log.append("skill.registered", {"name": "csv_clean"})
        log.append("council.verdict", {"decision": "thesis"})
        # v4 engineering events
        log.append("analysis.taint",
                   {"path": "p.py", "findings": [{"sink": "eval"}]})
        log.append("graph.entity", {"entities": 12, "relations": 9})
        log.append("coverage.result", {"path": "p.py", "percent": 83.0,
                                       "hit": 5, "total": 6})
        log.append("fuzz.run", {"target": "f", "iterations": 30})
        log.append("fuzz.crash", {"target": "f", "error": "TypeError"})
        log.append("mutation.result", {"path": "p.py", "score": 0.29,
                                       "killed": 2, "survived": 5})

        s = dash.snapshot()
        assert abs(s["cost_usd"] - 0.05) < 1e-9
        assert s["tool_calls"] == 1 and s["tool_errors"] == 1
        assert s["goal_active"] and s["clauses_proven"] == 1
        assert s["clauses_total"] == 2
        assert s["routed"] == 1 and s["spec_hits"] == 1
        assert s["verdicts"] == 2 and s["verdicts_failed"] == 1
        assert s["heals"] == 1 and s["skills"] == 1 and s["councils"] == 1
        assert s["taint_findings"] == 1 and s["graph_entities"] == 12
        assert s["cov_runs"] == 1 and s["cov_last"] == 83.0
        assert s["fuzz_crashes"] == 1
        assert s["mut_runs"] == 1 and abs(s["mut_last"] - 0.29) < 1e-9

        text = dash.render()
        assert "GOAL" in text and "50%" in text, text
        assert "ROUTER" in text and "SPEC" in text
        assert "HEALTH" in text
        assert "ENGINE" in text and "83%" in text and "29%" in text, text

        # tail streams only events after the cursor
        tail = dash.tail(since_seq=-1, limit=5)
        assert len(tail) == 5
        assert all(t["seq"] >= 0 for t in tail)
        head = log.head()
        assert dash.tail(since_seq=head) == []

    print("DASHBOARD SELF-TEST PASS")
