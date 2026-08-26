"""RACE — racing universes: first verified strategy wins.

One problem, N strategies, tried ONE AT A TIME in lane order (no
parallel subagents). Each universe pursues a different approach (role +
instructions); the moment one produces a result that passes the
verifier, it wins — every remaining lane is skipped and sealed as
cancelled. Losers' partial work is discarded by design: you bought
thoroughness, you paid the compute.

    strategies  default three lanes — the DIRECT lane (just do it),
                the CAREFUL lane (analyse first, then act), and the
                SPLIT lane (break the work into parts) — or supply
                your own
    race        lanes run serially; after each one lands the verifier
                (injectable — production uses the Judge) checks it
    finish      first PASS wins (race.winner); remaining lanes are
                sealed cancelled without ever running; if NOTHING
                passes, the best failure is returned with all
                universes' evidence (race.cancel)

The runner and verifier are injectable; the self-test races three
universe-runners with different speeds and proves: the fast-but-wrong
universe does NOT win, the verified one does, and the slowest is
cancelled before finishing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .kernel import EventLog

DEFAULT_STRATEGIES = (
    {"id": "direct", "role": "coder",
     "instructions": "Solve it directly and verify as you go."},
    {"id": "careful", "role": "architect",
     "instructions": "Map the problem first, then implement the safest "
                     "solution."},
    {"id": "split", "role": "planner",
     "instructions": "Break the work into parts and assemble them."},
)


@dataclass
class UniverseOutcome:
    strategy: str
    result: str = ""
    passed: bool = False
    cancelled: bool = False
    elapsed_ms: int = 0


@dataclass
class RaceResult:
    task: str
    winner: str = ""
    answer: str = ""
    outcomes: list[UniverseOutcome] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {"task": self.task[:200], "winner": self.winner,
                "elapsed_ms": self.elapsed_ms,
                "outcomes": [o.__dict__ for o in self.outcomes]}


class RacingUniverses:
    """Launch N strategies; first verified pass takes the race."""

    def __init__(self, log: EventLog, runner, verifier,
                 strategies: list[dict] | None = None) -> None:
        """runner(strategy: dict, task: str, cancel: threading.Event)
        -> str — must poll `cancel` and bail out when set.
        verifier(task: str, result: str) -> bool."""
        self.log = log
        self.runner = runner
        self.verifier = verifier
        self.strategies = list(strategies or DEFAULT_STRATEGIES)

    def race(self, task: str, timeout: float = 300.0) -> RaceResult:
        """Run the strategies ONE AT A TIME in lane order — no parallel
        universes. The first verified pass wins; every lane after the
        winner is sealed cancelled without ever running."""
        task = str(task or "").strip()
        result = RaceResult(task=task)
        if not task or not self.strategies:
            return result
        t0 = time.monotonic()
        self.log.append("race.start",
                        {"task": task[:300],
                         "strategies": [s["id"] for s in
                                        self.strategies]},
                        actor="kernel")

        cancel = threading.Event()
        deadline = t0 + timeout
        winner: UniverseOutcome | None = None
        # arm the deadline as an EVENT, not a between-lanes check — a hung
        # runner that never consults `cancel` would otherwise block forever
        timer: threading.Timer | None = None
        if timeout and timeout > 0:
            timer = threading.Timer(timeout, cancel.set)
            timer.daemon = True
            timer.start()

        try:
            for s in self.strategies:
                if time.monotonic() >= deadline:
                    break
                started = time.monotonic()
                try:
                    out = self.runner(s, task, cancel)
                    outcome = UniverseOutcome(
                        strategy=s["id"], result=str(out or ""),
                        elapsed_ms=int((time.monotonic() - started) * 1000))
                except Exception as e:    # a crashing universe just loses
                    outcome = UniverseOutcome(
                        strategy=s["id"],
                        result=f"ERROR: {e}",
                        elapsed_ms=int((time.monotonic() - started) * 1000))
                if cancel.is_set() and \
                        time.monotonic() >= deadline:
                    result.outcomes.append(outcome)
                    break
                try:
                    outcome.passed = self.verifier(task, outcome.result)
                except Exception as e:
                    # a raising verifier must fail the lane, not kill the race
                    outcome.passed = False
                    outcome.result = (f"VERIFIER ERROR: "
                                      f"{type(e).__name__}: {e}")
                result.outcomes.append(outcome)
                if outcome.passed:
                    winner = outcome
                    cancel.set()
                    break
        finally:
            if timer is not None:
                timer.cancel()

        # lanes after the winner (or timeout) never ran — sealed cancelled
        landed = {o.strategy for o in result.outcomes}
        for s in self.strategies:
            if s["id"] not in landed:
                result.outcomes.append(UniverseOutcome(
                    strategy=s["id"], cancelled=True))

        if winner is not None:
            result.winner = winner.strategy
            result.answer = winner.result
        else:
            # nobody passed: surface the least-bad failure as evidence
            ranked = sorted(result.outcomes,
                            key=lambda o: (o.cancelled, o.elapsed_ms))
            if ranked:
                result.answer = ranked[0].result
        result.elapsed_ms = int((time.monotonic() - t0) * 1000)
        self.log.append("race.winner" if winner else "race.cancel",
                        result.to_dict(), actor="kernel")
        return result

    def format(self, result: RaceResult) -> str:
        head = result.winner or "NONE (best failure shown)"
        lines = [f"RACE — {result.task}",
                 f"  winner: {head} · {result.elapsed_ms}ms"]
        for o in result.outcomes:
            icon = "✓" if o.passed else ("⊘" if o.cancelled else "✗")
            lines.append(f"  {icon} [{o.strategy}] {o.elapsed_ms}ms · "
                         f"{o.result[:120]}")
        if result.answer:
            lines.append("  ANSWER: " + result.answer[:800])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — fast-but-wrong loses, verified wins, slow is cancelled
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "race.jsonl")

        def runner(strategy: dict, task: str,
                   cancel: threading.Event) -> str:
            if strategy["id"] == "direct":
                time.sleep(0.10)
                return "42 is probably the answer, maybe"   # fast, wrong
            if strategy["id"] == "careful":
                time.sleep(0.35)
                if cancel.is_set():
                    return "CANCELLED before finishing"
                return "VERIFIED: the answer is 42 with proof"  # right
            time.sleep(1.0)                                  # too slow
            if cancel.is_set():
                return "CANCELLED"
            return "late answer"

        def verifier(task: str, result: str) -> bool:
            return "VERIFIED:" in result

        race = RacingUniverses(log, runner, verifier)
        result = race.race("meaning of life", timeout=10.0)

        by_id = {o.strategy: o for o in result.outcomes}
        # the fast-but-unverified lane ran first and LOST
        assert "direct" in by_id and not by_id["direct"].passed
        # the careful lane verified and won
        assert result.winner == "careful", result.to_dict()
        assert "42" in result.answer
        # the slow lane never ran — sealed cancelled after the winner
        assert by_id["split"].cancelled and by_id["split"].result == ""
        assert result.elapsed_ms < 900     # slow lane never ran

        # when NOTHING passes, the best failure surfaces with evidence
        def never_pass(t, r):
            return False

        race2 = RacingUniverses(log, runner, never_pass)
        r2 = race2.race("impossible", timeout=5.0)
        assert r2.winner == "" and r2.answer                # evidence

        # a crashing universe loses without killing the race
        def boom(strategy, task, cancel):
            if strategy["id"] == "direct":
                raise RuntimeError("universe exploded")
            time.sleep(0.05)
            return "VERIFIED: fine"

        race3 = RacingUniverses(log, boom, verifier)
        r3 = race3.race("survive", timeout=5.0)
        assert r3.winner == "careful" or r3.winner == "split", \
            r3.to_dict()
        crashed = next(o for o in r3.outcomes if o.strategy == "direct")
        assert "exploded" in crashed.result and not crashed.passed

        # empty task is clean; events sealed
        assert race.race("").winner == ""
        kinds = {e.type for e in log.events()}
        assert {"race.start", "race.winner", "race.cancel"} <= kinds

        print("RACE SELF-TEST PASS")
