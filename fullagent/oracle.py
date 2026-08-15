"""ORACLE — self-improvement (§21).

No model training, no fine-tuning. Improvement through structured
accumulated experience mined from the event log corpus.

  * analyze()     — post-run report: wasted steps, dead ends hit, facts
                    learned, calibration error, cost by clause.
  * calibrate()   — estimated vs actual per node kind; the planner's
                    forecasts converge because every forecast is recorded.
  * learn_facts() — extract project facts into .fullagent/memory/facts.md
                    (human-readable, git-trackable).
  * constitution  — a user-editable standing-rules file always present in
                    L0. Human-authored, human-owned, NEVER auto-modified;
                    the Oracle may only PROPOSE an amendment.
"""

from __future__ import annotations

import json
from pathlib import Path

from .kernel import EventLog, fold


class Oracle:
    """Post-run analysis + calibration + facts + constitution."""

    def __init__(self, log: EventLog, memory_dir: Path | None = None) -> None:
        self.log = log
        self.memory_dir = Path(memory_dir) if memory_dir else None
        if self.memory_dir:
            self.memory_dir.mkdir(parents=True, exist_ok=True)

    # -- post-run analysis (§24.1) --------------------------------------------

    def analyze(self) -> dict:
        """Mine the current session's log into a structured report."""
        st = fold(self.log)
        wasted = 0
        for ev in self.log.events():
            if ev.type == "tool.result" and ev.data.get("status") == "error":
                wasted += 1
        dead_ends_hit = len(st.dead_ends)
        facts_learned = len(st.facts)
        cost_by_clause: dict[str, float] = {}
        for ev in self.log.events():
            if ev.type == "cost.incurred" and ev.correlation_id:
                cost_by_clause[ev.correlation_id] = \
                    cost_by_clause.get(ev.correlation_id, 0.0) + \
                    float(ev.data.get("usd", 0.0))
        cal = self.calibrate()
        return {
            "events": len(self.log),
            "tool_calls": st.tool_calls,
            "tool_errors": st.tool_errors,
            "wasted_steps": wasted,
            "dead_ends_hit": dead_ends_hit,
            "facts_learned": facts_learned,
            "cost_usd": st.cost_usd,
            "cost_by_clause": cost_by_clause,
            "calibration_error": cal.get("mean_abs_error"),
            "verdicts": len(st.verdicts),
            "loop_alerts": len(st.loop_alerts),
        }

    def format_report(self) -> str:
        a = self.analyze()
        lines = [
            "ORACLE — post-run analysis",
            f"  events {a['events']}   tool calls {a['tool_calls']}   "
            f"errors {a['tool_errors']}",
            f"  wasted steps {a['wasted_steps']}   "
            f"dead ends hit {a['dead_ends_hit']}   "
            f"facts learned {a['facts_learned']}",
            f"  cost ${a['cost_usd']:.4f}   verdicts {a['verdicts']}   "
            f"loop alerts {a['loop_alerts']}",
        ]
        if a["cost_by_clause"]:
            lines.append("  cost by clause:")
            for cid, usd in sorted(a["cost_by_clause"].items()):
                lines.append(f"    {cid}: ${usd:.4f}")
        if a["calibration_error"] is not None:
            lines.append(f"  calibration mean abs error: "
                         f"{a['calibration_error']:.2f}")
        return "\n".join(lines)

    # -- calibration (§24.2) ----------------------------------------------------

    def calibrate(self) -> dict:
        """Estimated vs actual per node kind. Real self-improvement without
        training: the system's PREDICTIONS get measurably better because it
        has a ground-truth record of every prediction it made."""
        samples = fold(self.log).calibration
        if not samples:
            return {"n": 0, "mean_abs_error": None, "by_kind": {}}
        errors: list[float] = []
        by_kind: dict[str, list[float]] = {}
        for s in samples:
            est = float(s.get("est", 0.0))
            actual = float(s.get("actual", 0.0))
            err = abs(est - actual)
            errors.append(err)
            kind = str(s.get("kind", "?"))
            by_kind.setdefault(kind, []).append(err)
        return {
            "n": len(samples),
            "mean_abs_error": sum(errors) / len(errors),
            "by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
        }

    def record_calibration(self, kind: str, est: float, actual: float) -> None:
        self.log.append("calibration.sample",
                        {"kind": kind, "est": float(est),
                         "actual": float(actual)}, actor="oracle")

    # -- facts (§24.1) -----------------------------------------------------------

    def learn_fact(self, fact: str, kind: str = "project") -> None:
        self.log.append("fact.learned", {"fact": fact, "kind": kind},
                        actor="oracle")

    def write_facts_md(self) -> Path | None:
        """Write learned facts to memory/facts.md — human-readable and
        git-trackable. Returns the path, or None if no memory dir."""
        if not self.memory_dir:
            return None
        facts = fold(self.log).facts
        path = self.memory_dir / "facts.md"
        lines = ["# Learned project facts", "",
                 "_Machine-written by the Oracle. Human-editable._", ""]
        for f in facts:
            lines.append(f"- [{f.get('kind', 'project')}] {f.get('fact', '')}")
        path.write_text("\n".join(lines) + "\n")
        return path

    # -- constitution (§24.4) ------------------------------------------------------

    def constitution_path(self) -> Path | None:
        if not self.memory_dir:
            return None
        return self.memory_dir.parent / "constitution.md"

    def read_constitution(self) -> str:
        """The standing-rules file, always present in L0. Returns '' if it
        does not exist yet."""
        p = self.constitution_path()
        if p and p.exists():
            try:
                return p.read_text()
            except OSError:
                return ""
        return ""

    def propose_constitution_amendment(self, text: str) -> str:
        """The Oracle may PROPOSE an amendment; only a human may accept it.
        The proposal is sealed as an event, never applied to the file."""
        self.log.append("goal.amendment",
                        {"kind": "CONSTITUTION", "rationale": text,
                         "verdict": "pending"}, actor="oracle")
        return "proposed (pending human approval)"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        log = EventLog(root / "oracle.jsonl")
        oracle = Oracle(log, memory_dir=root / "memory")

        # seed a session's worth of events
        log.append("tool.call", {"name": "read_file", "args": {"path": "a"}})
        log.append("tool.result", {"name": "read_file", "status": "error"})
        log.append("tool.result", {"name": "read_file", "status": "done"})
        log.append("cost.incurred", {"usd": 0.10}, correlation_id="C1")
        log.append("cost.incurred", {"usd": 0.05}, correlation_id="C2")
        log.append("deadend.recorded", {"signature": "abc", "reason": "x"})

        # analyze
        a = oracle.analyze()
        assert a["tool_calls"] >= 1
        assert a["wasted_steps"] == 1
        assert a["dead_ends_hit"] == 1
        assert abs(a["cost_usd"] - 0.15) < 1e-9
        assert abs(a["cost_by_clause"]["C1"] - 0.10) < 1e-9
        report = oracle.format_report()
        assert "ORACLE" in report and "cost by clause" in report

        # calibration
        oracle.record_calibration("WRITE", est=3.0, actual=5.0)
        oracle.record_calibration("WRITE", est=4.0, actual=4.0)
        cal = oracle.calibrate()
        assert cal["n"] == 2
        assert abs(cal["mean_abs_error"] - 1.0) < 1e-9
        assert abs(cal["by_kind"]["WRITE"] - 1.0) < 1e-9

        # facts
        oracle.learn_fact("tests live in tests/", "project")
        oracle.learn_fact("FLAKE: test_x is intermittent", "flake")
        path = oracle.write_facts_md()
        assert path and path.exists()
        content = path.read_text()
        assert "tests live in tests/" in content
        assert "[flake]" in content

        # constitution: never auto-modified, only proposed
        assert oracle.read_constitution() == ""
        result = oracle.propose_constitution_amendment("always run make check")
        assert "pending human" in result
        assert oracle.read_constitution() == ""  # still untouched
        ams = fold(log).amendments
        assert any(a.get("kind") == "CONSTITUTION" for a in ams)

    print("ORACLE SELF-TEST PASS")
