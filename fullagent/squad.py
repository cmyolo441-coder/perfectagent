"""SQUAD — eight advanced specialists on ONE project, all in parallel.

The next level above Team and Crew. When a job is big — a whole project,
a refactor across many modules, a release — you do not want eight generic
workers; you want eight DIFFERENT experts attacking it simultaneously,
each in its own lane, like a real engineering org thrown at a milestone:

    🗺️ planner       decomposes the goal into work packages + scopes
    🏛️ architect     system design, module contracts, ADR docs
    👨‍💻 debugger      reproduces + pinpoints the deepest open defect chain
    ⚡ optimizer      measures and ranks real performance bottlenecks
    🧹 refactorer    removes duplication/dead weight, tightens structure
    🔗 integrator    glues cross-module seams + proves it with a real run
    📝 documenter    README/docs/examples derived from the actual source
    🛠️ devops        build/test/tooling chain end-to-end

All eight launch at once on the SAME shared context (environment digest,
file map, known facts). Reads fan out freely; every write goes through
the ONE global write lock (invariant I7) — writes serialise across the
whole squad. Each specialist collapses into a compact structured report;
nothing pollutes the main conversation, and the whole run is sealed as
replayable squad.report events.

Mechanical, rung-1 guarantees (never prompt advice):
  * exactly eight REAL subagents (real model calls, real tool loops)
  * conflict detection — files touched by more than one write role are
    reported back so the lead agent can reconcile them
  * a failing specialist never kills the squad — it lands as an error
    report with everything else intact
  * token spend is metered per specialist but NEVER stops the run
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Effort, Model, Provider
from .kernel import EventLog, fold
from .team import Team, WorkerReport

# The eight specialists, in launch order. The planner runs FIRST-AMONG-
# EQUALS: its work-package decomposition lands in the same wave as the
# others (all eight start together), so the squad is one parallel wave,
# not a planning phase followed by work.
SQUAD_ROLES = ("planner", "architect", "debugger", "optimizer",
               "refactorer", "integrator", "documenter", "devops")

_WRITE_ROLES = {"architect", "refactorer", "documenter", "devops",
                "integrator"}

# Per-role mandate: what THIS specialist contributes to a shared goal.
# The role BRIEF (identity, rules) lives in systemprompt.ROLE_BRIEFS; the
# mandate here is the squad-specific assignment framing.
SQUAD_MANDATES: dict[str, str] = {
    "planner": (
        "SQUAD MANDATE (planner): decompose the goal into at most 8 "
        "concrete, independent work packages. For each: owner role, "
        "file/module scope, acceptance criterion, dependency order. "
        "Explicitly flag any file-scope overlap between packages. End "
        "with the critical path (what blocks what)."
    ),
    "architect": (
        "SQUAD MANDATE (architect): map the current architecture as it "
        "REALLY is (modules, data flow, dependency edges), then the "
        "design needed by the goal — module boundaries, interface "
        "contracts, file-level responsibilities. Write it to a design "
        "doc (docs/ or design/) so the squad builds against one plan."
    ),
    "debugger": (
        "SQUAD MANDATE (debugger): find the most damaging defect chain "
        "relevant to the goal. Reproduce it, form hypotheses, bisect, "
        "and pinpoint exact file:line with the evidence chain that "
        "proves the cause. Hand back the minimal fix recipe."
    ),
    "optimizer": (
        "SQUAD MANDATE (optimizer): measure the system's real hotspots "
        "relevant to the goal (time it, count it — never guess). Rank "
        "the top bottlenecks with before-numbers and the exact change "
        "that would fix each. Note any quick win under 30 minutes."
    ),
    "refactorer": (
        "SQUAD MANDATE (refactorer): find the highest-leverage structural "
        "cleanup relevant to the goal — duplication, dead code, "
        "accidental complexity, naming drift. Apply the safest subset "
        "while keeping behaviour identical; verify with existing checks."
    ),
    "integrator": (
        "SQUAD MANDATE (integrator): walk the cross-module seams touched "
        "by the goal — imports, call chains, interface mismatches, "
        "missing wiring. Fix the seams so the pieces work as one system "
        "and PROVE it with a real run (exact command + exit code)."
    ),
    "documenter": (
        "SQUAD MANDATE (documenter): derive documentation for what the "
        "goal changes — README section, usage examples, API notes — "
        "strictly from reading the real source. Fix any stale doc you "
        "pass. Never invent behaviour."
    ),
    "devops": (
        "SQUAD MANDATE (devops): make the project chain work end-to-end "
        "for the goal — install, build, lint, test — and fix the build/"
        "tooling breakage you find. Report the exact one-command chain "
        "and its exit codes."
    ),
}


@dataclass
class SquadRun:
    """The sealed outcome of one 8-specialist squad run."""
    goal: str
    reports: list[WorkerReport] = field(default_factory=list)
    elapsed_ms: int = 0
    conflicts: list[str] = field(default_factory=list)   # shared file paths
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def done(self) -> int:
        return sum(1 for r in self.reports if r.status == "done")

    @property
    def blocked(self) -> int:
        return sum(1 for r in self.reports if r.status == "blocked")

    @property
    def errors(self) -> int:
        return sum(1 for r in self.reports if r.status == "error")

    def to_dict(self) -> dict:
        return {"goal": self.goal, "elapsed_ms": self.elapsed_ms,
                "done": self.done, "blocked": self.blocked,
                "errors": self.errors, "conflicts": self.conflicts,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "specialists": [r.to_dict() for r in self.reports]}


class Squad:
    """Eight advanced specialists, one project, one parallel wave.

    Built on the Team machinery (real model calls, real tool loops,
    serialised writes); the squad layer adds role mandates, write-conflict
    detection, per-run event sealing, and a compact digest for the lead
    agent."""

    def __init__(self, log: EventLog, provider: Provider, model: Model,
                 effort: Effort, mastermind=None) -> None:
        self.log = log
        self.team = Team(log, provider, model, effort,
                         mastermind=mastermind)

    # -- public API ----------------------------------------------------------

    def run(self, goal: str, context: str = "", read_only: bool = False,
            on_progress=None) -> SquadRun:
        """Launch ALL EIGHT specialists in parallel on the shared goal.

        on_progress(finished, total, report) fires as each specialist
        lands — live squad status for the UI. Never raises: specialist
        failures land inside their reports."""
        goal = str(goal or "").strip()
        if not goal:
            return SquadRun(goal="")
        self.log.append("squad.launch",
                        {"goal": goal[:300],
                         "roles": list(SQUAD_ROLES),
                         "read_only": bool(read_only)},
                        actor="sovereign")

        tasks = [{
            "task": f"{SQUAD_MANDATES[role]}\n\nPROJECT GOAL: {goal}",
            "role": role,
        } for role in SQUAD_ROLES]

        t0 = time.monotonic()
        reports = self.team.run(tasks, context=context,
                                read_only=read_only,
                                on_progress=on_progress)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # deterministic write-conflict detection (I7 reconciliation hint):
        # a file touched by more than one WRITE-role specialist needs the
        # lead agent's attention — the lock serialised the writes, only
        # the ORDER is open.
        by_file: dict[str, set[str]] = {}
        for r in reports:
            if r.role in _WRITE_ROLES:
                for p in r.files_touched:
                    by_file.setdefault(p, set()).add(r.role)
        conflicts = sorted(p for p, roles in by_file.items()
                           if len(roles) > 1)

        run = SquadRun(goal=goal, reports=reports, elapsed_ms=elapsed_ms,
                       conflicts=conflicts,
                       tokens_in=sum(r.tokens_in for r in reports),
                       tokens_out=sum(r.tokens_out for r in reports))
        self.log.append("squad.report", run.to_dict(), actor="system")
        return run

    def runs(self, n: int = 5) -> list[dict]:
        """Last n squad runs from the fold, newest first."""
        if n <= 0:
            return []
        reps = fold(self.log).squad_reports
        return list(reversed(reps[-n:]))

    # -- rendering ------------------------------------------------------------

    def format(self, run: SquadRun) -> str:
        """Compact digest for the lead agent / TUI: every specialist's
        verdict, files, and the reconciliation warnings."""
        if not run.goal:
            return "ERROR: squad needs a non-empty goal"
        icon = {"done": "✓", "blocked": "◐", "error": "✗"}
        lines = [f"SQUAD RUN — goal: {run.goal}",
                 f"8 specialists · {run.done} done · {run.blocked} blocked "
                 f"· {run.errors} error · {run.elapsed_ms}ms · "
                 f"{run.tokens_in + run.tokens_out} tokens"]
        for r in run.reports:
            lines.append(
                f"{icon.get(r.status, '…')} [{r.role}] {r.status} · "
                f"{r.tool_calls} tools · {r.elapsed_ms}ms"
                + (f" · files: {', '.join(r.files_touched[:6])}"
                   if r.files_touched else ""))
            if r.error:
                lines.append(f"    error: {r.error[:200]}")
            if r.summary:
                lines.append("    " + r.summary.replace("\n", "\n    ")
                             [:1000])
        if run.conflicts:
            lines.append(
                "⚠ RECONCILE — these files were written by MORE THAN ONE "
                "specialist (writes were serialised, but verify the final "
                "state is coherent): " + ", ".join(run.conflicts))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — offline stand-in team drives the full squad mechanics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import tempfile
    import threading
    from pathlib import Path
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "squad-test.jsonl")
        provider = SimpleNamespace(key="t", name="T",
                                   base_url="http://t", api_key="sk-x",
                                   color="#fff")
        model = SimpleNamespace(id="stub", provider="t", label="Stub",
                                supports_tools=True)
        effort = SimpleNamespace(key="low", label="LOW", color="#fff",
                                 max_tokens=64, temperature=0.0,
                                 reasoning_effort=None)
        squad = Squad(log, provider, model, effort)

        # offline stand-in: every role lands after a real sleep so the
        # parallel claim is testable; two WRITE roles collide on app.py
        real_run = squad.team.run
        launch_threads = {"n": 0}

        def fake_team_run(tasks, context="", timeout=240.0, read_only=False,
                          on_progress=None):
            def _one(task):
                time.sleep(0.25)
                role = task["role"]
                files = ["app.py", "lib/util.py"] if role == "refactorer" \
                    else (["app.py"] if role == "integrator" else [])
                rep = WorkerReport(task=task["task"], role=role,
                                   status="done",
                                   summary=f"{role} handled its lane",
                                   files_touched=files, tool_calls=3,
                                   tokens_in=100, tokens_out=50,
                                   elapsed_ms=250)
                if on_progress:
                    try:
                        on_progress(1, len(tasks), rep)
                    except Exception:
                        pass
                return rep
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
                return list(ex.map(_one, tasks))

        squad.team.run = fake_team_run  # type: ignore[method-assign]

        progress: list[str] = []
        t0 = time.monotonic()
        run = squad.run("ship v2 of the platform",
                        context="ctx",
                        on_progress=lambda f, t, r: progress.append(
                            f"{f}/{t}:{r.role}"))
        elapsed = time.monotonic() - t0

        # all eight specialists really ran, in parallel, in input order
        assert [r.role for r in run.reports] == list(SQUAD_ROLES), run.reports
        assert all(r.status == "done" for r in run.reports)
        assert run.done == 8 and run.blocked == 0 and run.errors == 0
        assert elapsed < 1.4, f"squad not parallel ({elapsed:.2f}s)"
        assert run.tokens_in == 800 and run.tokens_out == 400

        # write-conflict detection: app.py was touched by 2 write roles
        assert run.conflicts == ["app.py"], run.conflicts

        # progress fired once per specialist
        assert len(progress) == 8, progress

        # events are sealed and foldable
        types = [e.type for e in log.events()]
        assert "squad.launch" in types and "squad.report" in types
        st = fold(log)
        assert len(st.squad_reports) == 1
        assert st.squad_reports[0]["done"] == 8
        json.dumps(st.squad_reports[0])  # JSON-serializable

        # digest renders with the reconcile warning
        text = squad.format(run)
        assert "SQUAD RUN" in text and "ship v2" in text
        assert "RECONCILE" in text and "app.py" in text

        # empty goal is rejected cleanly
        assert squad.run("").goal == ""

        # every mandate names its role and the goal lands in each task
        squad.team.run = lambda *a, **k: []  # type: ignore[method-assign]
        squad.run("quiet goal")
        tasks_seen = []
        log2 = EventLog(Path(td) / "squad-tasks.jsonl")
        squad2 = Squad(log2, provider, model, effort)
        squad2.team.run = lambda tasks, **k: tasks_seen.extend(tasks) or [] \
            # type: ignore[method-assign]
        squad2.run("build the api")
        assert len(tasks_seen) == 8
        assert all("PROJECT GOAL: build the api" in t["task"]
                   for t in tasks_seen)
        assert {t["role"] for t in tasks_seen} == set(SQUAD_ROLES)

        print("SQUAD SELF-TEST PASS")
