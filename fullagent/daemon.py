"""DAEMON — Mission Control (autonomous long-running missions).

Missions that outlive a single turn: the daemon owns a mission record in
the event log, advances it one TICK at a time, checkpoints progress after
every tick, and can be resumed from the last checkpoint after any restart.
A mission is a queue of steps; each tick executes the next pending step
through a caller-supplied executor, records the outcome, and checkpoints.

Hard rules (mechanical):
  * All mission state lives in the event log — daemon.mission /
    daemon.checkpoint / daemon.tick / daemon.wake / daemon.done. A daemon
    object keeps no authoritative state; resume() rebuilds everything from
    the fold, so a crash loses at most one in-flight tick.
  * A step that fails is retried up to max_retries, then the mission is
    BLOCKED (never silently skipped) and says exactly where it stopped.
  * Self-wake: wake_conditions are deterministic predicates over the fold
    (e.g. "a verdict failed", "budget event sealed"). due() reports which
    conditions currently hold — the scheduler (or a human) decides when to
    actually run the next tick. The daemon never sleeps in-process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .kernel import EventLog, fold

MISSION_STATES = ("RUNNING", "BLOCKED", "DONE", "ABANDONED")
STEP_STATES = ("PENDING", "RUNNING", "DONE", "FAILED", "SKIPPED")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Step:
    id: str
    task: str
    state: str = "PENDING"
    attempts: int = 0
    result: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "task": self.task, "state": self.state,
                "attempts": self.attempts, "result": self.result[:300]}


@dataclass
class Mission:
    mission_id: str
    statement: str
    steps: list[Step] = field(default_factory=list)
    state: str = "RUNNING"
    checkpoint_seq: int = -1     # event seq of the last checkpoint
    ticks: int = 0

    def pending(self) -> Step | None:
        for s in self.steps:
            if s.state == "PENDING":
                return s
        return None

    def progress(self) -> float:
        if not self.steps:
            return 1.0
        done = sum(1 for s in self.steps if s.state == "DONE")
        return done / len(self.steps)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class Daemon:
    """Advance missions one tick at a time over the event log.

    `executor` runs one step: executor(step.task) -> str result. A result
    starting with 'ERROR:' counts as a failed attempt."""

    def __init__(self, log: EventLog, executor=None,
                 max_retries: int = 2) -> None:
        self.log = log
        self.executor = executor
        self.max_retries = max(0, int(max_retries))

    # -- mission lifecycle -----------------------------------------------------

    def start(self, statement: str, tasks: list[str]) -> Mission:
        """Seal a new mission. Step ids are M1..Mn."""
        # SPEED-EXPOSURE FIX: the old millisecond-clock id could collide
        # when missions are started back-to-back (now that everything
        # runs faster); seq is strictly monotonic, so this never can.
        mission_id = f"mission-{self.log.head() + 1}"
        steps = [Step(id=f"M{i + 1}", task=t) for i, t in enumerate(tasks)]
        self.log.append("daemon.mission",
                        {"mission_id": mission_id, "statement": statement,
                         "steps": [s.to_dict() for s in steps],
                         "state": "RUNNING"},
                        actor="daemon")
        return Mission(mission_id, statement, steps)

    def tick(self, mission_id: str) -> dict:
        """Execute the next pending step, seal the outcome, checkpoint.

        Returns {mission_id, step, state, result, progress}. A step that
        exhausts its retries BLOCKS the mission — it is never skipped."""
        m = self.resume(mission_id)
        if m is None:
            return {"mission_id": mission_id, "error": "no such mission"}
        if m.state != "RUNNING":
            return {"mission_id": mission_id, "state": m.state,
                    "error": f"mission is {m.state}, not RUNNING"}
        step = m.pending()
        if step is None:
            self._close(m, "DONE")
            return {"mission_id": mission_id, "state": "DONE",
                    "progress": 1.0}

        step.attempts += 1
        result = ""
        if self.executor is not None:
            try:
                result = str(self.executor(step.task))
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"
        else:
            result = "ERROR: no executor attached"

        failed = result.startswith("ERROR:")
        step.state = "FAILED" if failed else "DONE"
        step.result = result
        m.ticks += 1

        self.log.append("daemon.tick",
                        {"mission_id": mission_id, "step": step.to_dict(),
                         "failed": failed, "attempt": step.attempts},
                        actor="daemon")

        if failed and step.attempts > self.max_retries:
            # retries exhausted — the mission blocks here, visibly
            step.state = "FAILED"
            self._patch_state(m, "BLOCKED")
            self._checkpoint(m)
            return {"mission_id": mission_id, "step": step.id,
                    "state": "BLOCKED", "result": result,
                    "progress": round(m.progress(), 3)}

        if failed:
            step.state = "PENDING"  # retry on the next tick
        self._checkpoint(m)
        if m.pending() is None and not failed:
            self._close(m, "DONE")
            return {"mission_id": mission_id, "step": step.id,
                    "state": "DONE", "result": result, "progress": 1.0}
        return {"mission_id": mission_id, "step": step.id,
                "state": "RUNNING", "result": result,
                "progress": round(m.progress(), 3)}

    def abandon(self, mission_id: str, reason: str = "") -> bool:
        m = self.resume(mission_id)
        if m is None or m.state == "DONE":
            return False
        self._patch_state(m, "ABANDONED")
        self.log.append("daemon.done",
                        {"mission_id": mission_id, "state": "ABANDONED",
                         "reason": reason}, actor="human")
        return True

    # -- checkpoints + resume ----------------------------------------------------

    def _checkpoint(self, m: Mission) -> None:
        ev = self.log.append("daemon.checkpoint",
                             {"mission_id": m.mission_id,
                              "steps": [s.to_dict() for s in m.steps],
                              "ticks": m.ticks, "state": m.state},
                             actor="daemon")
        m.checkpoint_seq = ev.seq

    def _patch_state(self, m: Mission, state: str) -> None:
        m.state = state
        self.log.append("daemon.checkpoint",
                        {"mission_id": m.mission_id,
                         "steps": [s.to_dict() for s in m.steps],
                         "ticks": m.ticks, "state": state},
                        actor="daemon")

    def _close(self, m: Mission, state: str) -> None:
        m.state = state
        self.log.append("daemon.done",
                        {"mission_id": m.mission_id, "state": state,
                         "ticks": m.ticks,
                         "progress": round(m.progress(), 3)},
                        actor="daemon")

    def resume(self, mission_id: str) -> Mission | None:
        """Rebuild a mission purely from the fold — the daemon's crash
        recovery. Latest checkpoint wins; ticks replay step states."""
        evs = fold(self.log).daemon_events
        base = None
        for e in evs:
            if e.get("type") == "daemon.mission" and \
                    e.get("mission_id") == mission_id:
                base = e
        if base is None:
            return None
        steps = [Step(id=s["id"], task=s["task"], state=s.get("state"),
                      attempts=int(s.get("attempts", 0)),
                      result=s.get("result", ""))
                 for s in base.get("steps") or []]
        m = Mission(mission_id, str(base.get("statement", "")), steps,
                    state="RUNNING")
        # replay checkpoints and ticks in sealed order
        for e in evs:
            if e.get("mission_id") != mission_id:
                continue
            if e.get("type") == "daemon.checkpoint":
                m.ticks = int(e.get("ticks", m.ticks))
                m.state = str(e.get("state", m.state))
                by_id = {s["id"]: s for s in e.get("steps") or []}
                for s in m.steps:
                    if s.id in by_id:
                        s.state = by_id[s.id].get("state", s.state)
                        s.attempts = int(by_id[s.id].get("attempts",
                                                         s.attempts))
                        s.result = by_id[s.id].get("result", s.result)
            elif e.get("type") == "daemon.done":
                m.state = str(e.get("state", m.state))
        return m

    def missions(self) -> list[dict]:
        """One summary row per mission, newest first."""
        evs = fold(self.log).daemon_events
        ids: list[str] = []
        for e in evs:
            if e.get("type") == "daemon.mission":
                mid = e.get("mission_id", "")
                if mid and mid not in ids:
                    ids.append(mid)
        rows = []
        for mid in reversed(ids):
            m = self.resume(mid)
            if m is None:
                continue
            rows.append({"mission_id": mid, "statement": m.statement,
                         "state": m.state, "ticks": m.ticks,
                         "progress": round(m.progress(), 3),
                         "steps": len(m.steps)})
        return rows

    # -- self-wake -----------------------------------------------------------------

    def wake_conditions(self) -> list[str]:
        """Deterministic predicates over the fold that justify waking the
        daemon for another tick. The daemon itself never sleeps or polls —
        this only REPORTS what currently holds."""
        st = fold(self.log)
        reasons: list[str] = []
        running = [r for r in self.missions() if r["state"] == "RUNNING"]
        if running:
            reasons.append(f"{len(running)} mission(s) RUNNING with "
                           "pending steps")
        failed_verdicts = sum(1 for v in st.verdicts if not v.get("passed"))
        if failed_verdicts:
            reasons.append(f"{failed_verdicts} failed verdict(s) to react to")
        if st.budget_events:
            reasons.append("budget event sealed — re-evaluate missions")
        return reasons

    def due(self) -> bool:
        return bool(self.wake_conditions())

    def format_status(self) -> str:
        rows = self.missions()
        lines = ["DAEMON — mission control"]
        if not rows:
            lines.append("  no missions")
        for r in rows[:8]:
            bar = int(r["progress"] * 12)
            lines.append(f"  {r['mission_id']}  [{r['state']:<9}] "
                         f"{'█' * bar}{'░' * (12 - bar)} "
                         f"{r['progress']:.0%}  {r['ticks']} ticks  "
                         f"{r['statement'][:36]}")
        wakes = self.wake_conditions()
        if wakes:
            lines.append("  wake: " + "; ".join(wakes))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "daemon.jsonl")

        results = {"step two": "ERROR: transient failure"}

        def executor(task: str) -> str:
            return results.get(task, f"OK: {task}")

        d = Daemon(log, executor, max_retries=1)

        # start a 3-step mission
        m = d.start("ship the parser", ["step one", "step two", "step three"])
        assert m.mission_id and len(m.steps) == 3

        # tick 1: step one done
        r = d.tick(m.mission_id)
        assert r["state"] == "RUNNING" and r["step"] == "M1", r

        # tick 2: step two fails (attempt 1) -> stays RUNNING for retry
        r = d.tick(m.mission_id)
        assert r["result"].startswith("ERROR:"), r
        m2 = d.resume(m.mission_id)
        assert m2.state == "RUNNING", m2.state

        # tick 3: transient failure cleared -> step two done
        results["step two"] = "OK: recovered"
        r = d.tick(m.mission_id)
        assert r["step"] == "M2" and not r["result"].startswith("ERROR:"), r

        # tick 4: step three done -> mission DONE
        r = d.tick(m.mission_id)
        assert r["state"] == "DONE" and r["progress"] == 1.0, r

        # resume rebuilds the finished mission from the fold alone
        m3 = d.resume(m.mission_id)
        assert m3.state == "DONE" and m3.progress() == 1.0
        assert all(s.state == "DONE" for s in m3.steps)

        # a permanently failing step BLOCKS the mission after retries
        d2 = Daemon(log, lambda t: "ERROR: permanent", max_retries=1)
        m4 = d2.start("doomed", ["only step"])
        d2.tick(m4.mission_id)          # attempt 1 fails -> retry
        r = d2.tick(m4.mission_id)      # attempt 2 fails -> BLOCKED
        assert r["state"] == "BLOCKED", r
        assert d2.resume(m4.mission_id).state == "BLOCKED"

        # abandon works on a blocked mission
        assert d2.abandon(m4.mission_id, "giving up")
        assert d2.resume(m4.mission_id).state == "ABANDONED"

        # wake conditions reflect running missions + failed verdicts
        d3 = Daemon(log, executor)
        m5 = d3.start("long mission", ["a", "b"])
        assert d3.due() is True
        assert any("RUNNING" in w for w in d3.wake_conditions())

        # missions() lists newest first with progress
        rows = d3.missions()
        assert rows[0]["mission_id"] == m5.mission_id
        assert "DAEMON" in d3.format_status()

        # everything is sealed in the ledger
        evs = fold(log).daemon_events
        types = {e["type"] for e in evs}
        assert {"daemon.mission", "daemon.checkpoint", "daemon.tick",
                "daemon.done"} <= types

    print("DAEMON SELF-TEST PASS")
