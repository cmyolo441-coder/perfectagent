"""WORKFLOWS — saved multi-step pipelines (enterprise automation).

A workflow is a reusable, version-controlled recipe: an ordered set of
steps, each with a task, a role, an optional model override, and an
optional machine-checkable `expect` predicate. Steps run one at a time,
grouped into phases that execute in order — real orchestration, not a
todo list.

    {"name": "ship-feature",
     "description": "build, test and review a feature",
     "steps": [
        {"task": "implement the parser", "role": "coder", "phase": 1},
        {"task": "write unit tests",     "role": "tester", "phase": 2,
         "expect": {"type": "exit_code", "command": "pytest -q"}},
        {"task": "review the diff",      "role": "reviewer", "phase": 2}
     ]}

Hard rules (mechanical):
  * Workflows are JSON files under <workflows_dir>/<name>.json — human
    editable, git-trackable, reusable across sessions.
  * Every run is sealed in the event log: workflow.start /
    workflow.step / workflow.done. Runs are auditable and replayable.
  * A step whose `expect` predicate FAILS blocks the workflow right
    there (never silently skipped) — the run report says exactly which
    step stopped and why.
  * Step execution is injected (`executor`), so the engine is fully
    testable without a live model; the agent binds it to the Crew.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import EventLog

_NAME_RE = re.compile(r"^[\w][\w.-]{0,63}$")

RUN_STATES = ("RUNNING", "DONE", "BLOCKED", "ABANDONED")


@dataclass
class WorkflowStep:
    task: str
    role: str = "coder"
    model: str = ""
    phase: int = 0
    expect: dict | None = None

    def to_dict(self) -> dict:
        d = {"task": self.task, "role": self.role}
        if self.model:
            d["model"] = self.model
        if self.phase:
            d["phase"] = self.phase
        if self.expect:
            d["expect"] = self.expect
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowStep":
        if not isinstance(d, dict) or not str(d.get("task", "")).strip():
            raise WorkflowError("every step needs a non-empty 'task'")
        expect = d.get("expect")
        if expect is not None and not isinstance(expect, dict):
            raise WorkflowError("'expect' must be a predicate dict")
        try:
            phase = int(d.get("phase", 0))
        except (TypeError, ValueError):
            raise WorkflowError("'phase' must be an integer")
        return cls(task=str(d["task"]).strip(),
                   role=str(d.get("role", "coder")).strip() or "coder",
                   model=str(d.get("model", "")).strip(),
                   phase=max(0, phase),
                   expect=expect)


@dataclass
class Workflow:
    name: str
    description: str = ""
    steps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Workflow":
        name = str(d.get("name", "")).strip()
        if not _NAME_RE.match(name):
            raise WorkflowError(
                f"invalid workflow name {name!r} — use letters, digits, "
                f"'_', '-', '.' (max 64 chars)")
        raw_steps = d.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise WorkflowError("a workflow needs at least one step")
        steps = [WorkflowStep.from_dict(s) for s in raw_steps]
        return cls(name=name,
                   description=str(d.get("description", "")).strip(),
                   steps=steps)


class WorkflowError(ValueError):
    """Raised for invalid definitions or unknown workflows."""


@dataclass
class StepResult:
    step: int
    task: str
    role: str
    status: str = "pending"      # pending | done | blocked | error
    summary: str = ""
    check: str = ""              # expect verdict detail
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {"step": self.step, "task": self.task[:200], "role": self.role,
                "status": self.status, "summary": self.summary[:400],
                "check": self.check[:200], "elapsed_ms": self.elapsed_ms}


class WorkflowEngine:
    """Saved pipelines over the event log.

    `executor(step_dict) -> dict` runs ONE step and returns
    {"status": "done|blocked|error", "summary": str}. The agent binds
    this to the Crew; tests inject a stub. `judge` (optional) verifies
    `expect` predicates after each step.
    """

    def __init__(self, log: EventLog, workflows_dir: Path,
                 executor=None, judge=None) -> None:
        self.log = log
        self.dir = Path(workflows_dir)
        self.executor = executor
        self.judge = judge

    # -- persistence -----------------------------------------------------------

    def _path(self, name: str) -> Path:
        if not _NAME_RE.match(str(name or "")):
            # the regex guards save(); load/delete/run receive raw user
            # text — without it, "../../etc/foo" reads or UNLINKS any
            # *.json outside this directory
            raise WorkflowError(f"invalid workflow name: {name!r}")
        return self.dir / f"{name}.json"

    def save(self, wf: Workflow) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(wf.name)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(wf.to_dict(), indent=2), encoding="utf-8")
        import os
        os.replace(tmp, path)
        self.log.append("workflow.saved",
                        {"name": wf.name, "steps": len(wf.steps)},
                        actor="sovereign")
        return path

    def load(self, name: str) -> Workflow:
        path = self._path(name)
        if not path.is_file():
            known = ", ".join(self.list()) or "none"
            raise WorkflowError(f"unknown workflow {name!r} (saved: {known})")
        try:
            return Workflow.from_dict(json.loads(path.read_text()))
        except (ValueError, KeyError, TypeError) as e:
            raise WorkflowError(f"workflow {name!r} is malformed: {e}")

    def list(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if path.is_file():
            path.unlink()
            return True
        return False

    # -- execution ---------------------------------------------------------------

    def run(self, name: str, timeout: float = 240.0) -> dict:
        """Execute a saved workflow: phases in order, steps within a
        phase one at a time (serial). A failed expect-predicate BLOCKS
        the run at that step. Returns the run report dict."""
        wf = self.load(name)
        if self.executor is None:
            raise WorkflowError("no executor bound — workflows cannot run")

        # group steps into phases (phase number, stable order)
        phases: dict[int, list[tuple[int, WorkflowStep]]] = {}
        for i, step in enumerate(wf.steps):
            phases.setdefault(step.phase, []).append((i + 1, step))
        ordered = sorted(phases.items())

        self.log.append("workflow.start",
                        {"name": wf.name, "steps": len(wf.steps),
                         "phases": len(ordered)},
                        actor="sovereign")
        results: list[StepResult] = []
        state = "RUNNING"
        t0 = time.monotonic()

        for phase_num, steps in ordered:
            if state != "RUNNING":
                break
            # steps within a phase run one at a time, in the order the
            # executor is bound to (the Crew's serial queue); a blocked
            # or failing step stops the remaining steps right there
            for n, step in steps:
                rep = self._run_step(
                    {"task": step.task, "role": step.role,
                     "model": step.model, "_n": n}, timeout)
                r = StepResult(step=n, task=step.task, role=step.role,
                               status=rep.get("status", "error"),
                               summary=str(rep.get("summary", "")),
                               elapsed_ms=int(rep.get("elapsed_ms", 0)))
                if r.status == "done" and step.expect is not None \
                        and self.judge is not None:
                    verdict = self.judge.check(step.expect)
                    r.check = verdict.detail
                    if not verdict.passed:
                        r.status = "blocked"
                        r.summary = (r.summary + f"\nEXPECT FAILED: "
                                     f"{verdict.detail}").strip()
                self.log.append("workflow.step",
                                {"name": wf.name, "phase": phase_num,
                                 **r.to_dict()},
                                actor="system")
                results.append(r)
                if r.status in ("blocked", "error"):
                    state = "BLOCKED"
                    break

        if state == "RUNNING":
            state = "DONE"
        report = {"name": wf.name, "state": state,
                  "steps": [r.to_dict() for r in results],
                  "elapsed_ms": int((time.monotonic() - t0) * 1000)}
        self.log.append("workflow.done",
                        {"name": wf.name, "state": state,
                         "steps_ok": sum(1 for r in results
                                         if r.status == "done"),
                         "steps_total": len(wf.steps),
                         "elapsed_ms": report["elapsed_ms"]},
                        actor="kernel")
        return report

    def _run_step(self, item: dict, timeout: float) -> dict:
        """Run ONE step through the bound executor — a blocked step never
        lets its phase's later steps execute; a hung executor is cut off
        at the step's timeout budget instead of stalling the workflow."""
        import threading
        started = time.monotonic()
        out: dict = {}

        def runner():
            try:
                rep = self.executor(item) or {}
                if not isinstance(rep, dict):
                    rep = {"status": "error", "summary": str(rep)}
            except Exception as e:  # noqa: BLE001 — a step never kills the engine
                rep = {"status": "error",
                       "summary": f"{type(e).__name__}: {e}"}
            out.update(rep)

        th = threading.Thread(target=runner, daemon=True,
                              name="workflow:step")
        th.start()
        th.join(max(0.0, float(timeout)))
        if not out and th.is_alive():
            rep = {"status": "error",
                   "summary": f"step timed out after {timeout:g}s"}
        else:
            rep = out or {"status": "error", "summary": "empty report"}
        rep.setdefault("elapsed_ms",
                       int((time.monotonic() - started) * 1000))
        rep.setdefault("status", "done")
        rep.setdefault("summary", "")
        return rep

    # -- rendering -----------------------------------------------------------------

    def format_list(self) -> str:
        names = self.list()
        if not names:
            return "no saved workflows — define one with /workflow define"
        lines = ["SAVED WORKFLOWS"]
        for n in names:
            try:
                wf = self.load(n)
            except WorkflowError as e:
                lines.append(f"  ✗ {n} — {e}")
                continue
            lines.append(f"  ◆ {n} — {len(wf.steps)} step(s)"
                         + (f" — {wf.description[:60]}"
                            if wf.description else ""))
            for i, s in enumerate(wf.steps, 1):
                check = " ⊛expect" if s.expect else ""
                model = f" [{s.model}]" if s.model else ""
                lines.append(f"      {i}. ({s.role}{model}) "
                             f"{s.task[:70]}{check}")
        return "\n".join(lines)

    def format_report(self, report: dict) -> str:
        icon = {"DONE": "✓", "BLOCKED": "⛔", "RUNNING": "…",
                "ABANDONED": "⊘"}.get(report["state"], "?")
        lines = [f"WORKFLOW {report['name']} — {icon} {report['state']} "
                 f"({report['elapsed_ms'] / 1000:.1f}s)"]
        for r in report["steps"]:
            glyph = {"done": "✓", "blocked": "⛔", "error": "✗",
                     "pending": "·"}.get(r["status"], "?")
            lines.append(f"  {glyph} step {r['step']} ({r['role']}) "
                         f"{r['task'][:80]}")
            if r["summary"]:
                lines.append("      " + r["summary"][:200].replace(
                    "\n", "\n      "))
            if r["check"]:
                lines.append(f"      expect: {r['check'][:120]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — stub executor + real judge predicates
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from .judge import Judge

    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log = EventLog(root / "wf-test.jsonl")
            ok_files = root / "ok.txt"

            def executor(item: dict) -> dict:
                if "fail" in item["task"]:
                    return {"status": "error", "summary": "exploded"}
                if "create" in item["task"]:
                    ok_files.write_text("shipped")
                return {"status": "done",
                        "summary": f"completed: {item['task'][:40]}"}

            judge = Judge(log)
            engine = WorkflowEngine(log, root / "workflows",
                                    executor=executor, judge=judge)

            # definition validation
            try:
                Workflow.from_dict({"name": "x", "steps": []})
                raise AssertionError("empty steps must fail")
            except WorkflowError:
                pass
            try:
                Workflow.from_dict({"name": "bad name!",
                                    "steps": [{"task": "t"}]})
                raise AssertionError("bad name must fail")
            except WorkflowError:
                pass

            # save / load / list round-trip
            wf = Workflow.from_dict({
                "name": "ship",
                "description": "build and verify",
                "steps": [
                    {"task": "create the artifact", "role": "coder",
                     "phase": 1},
                    {"task": "check it exists", "role": "tester",
                     "phase": 2,
                     "expect": {"type": "file_exists",
                                "path": str(ok_files)}},
                    {"task": "review it", "role": "reviewer",
                     "phase": 2}]})
            engine.save(wf)
            assert engine.list() == ["ship"]
            loaded = engine.load("ship")
            assert len(loaded.steps) == 3
            assert loaded.steps[1].expect["type"] == "file_exists"

            # successful run — all steps done, expect passes
            report = engine.run("ship")
            assert report["state"] == "DONE", report
            assert all(s["status"] == "done" for s in report["steps"])

            # failing expect blocks the run at that step
            wf2 = Workflow.from_dict({
                "name": "strict",
                "steps": [
                    {"task": "do work", "role": "coder"},
                    {"task": "impossible check", "role": "tester",
                     "expect": {"type": "file_exists",
                                "path": str(root / "missing.txt")}},
                    {"task": "never reached", "role": "reviewer"}]})
            engine.save(wf2)
            report = engine.run("strict")
            assert report["state"] == "BLOCKED", report
            assert report["steps"][1]["status"] == "blocked"
            assert len(report["steps"]) == 2  # step 3 never ran

            # executor error blocks too
            wf3 = Workflow.from_dict({"name": "boom",
                                      "steps": [{"task": "fail hard"}]})
            engine.save(wf3)
            report = engine.run("boom")
            assert report["state"] == "BLOCKED"
            assert report["steps"][0]["status"] == "error"

            # events sealed
            types = [e.type for e in log.events()]
            assert types.count("workflow.start") == 3
            assert types.count("workflow.done") == 3
            assert "workflow.step" in types and "workflow.saved" in types

            # rendering
            assert "ship" in engine.format_list()
            assert "BLOCKED" in engine.format_report(report)

            # delete
            assert engine.delete("boom") and not engine.delete("boom")

            print("WORKFLOWS SELF-TEST PASS")

    _self_test()
