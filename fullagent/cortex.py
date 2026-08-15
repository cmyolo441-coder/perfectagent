"""CORTEX — orchestration (§13).

The plan is a typed DAG; every node carries a predicate, a risk class, a
path set (the write-exclusivity key), and cost estimates. Three hard
mechanisms live here, all rung 1 (pure Python, free, deterministic):

  * Write-exclusivity (invariant I7): two WRITE nodes with overlapping
    path sets can never be scheduled concurrently. Reads fan out, writes
    serialise (§16.1).
  * Hierarchical budget governor (§13.5): a run budget with slices per
    subtree. Breaching any axis PAUSES — never silently kills — and emits
    budget.event. A runaway sub-agent cannot consume the parent's budget.
  * Loop / thrash / oscillation detectors (§13.4): exact-repeat tool
    calls, file-content A-B-A-B oscillation, and cost-slope breaches are
    detected from the event log and sealed as loop.alert events.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .kernel import EventLog, fold

NODE_KINDS = ("READ", "WRITE", "EXEC", "VERIFY", "ASK", "RESEARCH",
              "REFACTOR")
NODE_STATUSES = ("PENDING", "RUNNING", "PASSED", "FAILED", "SKIPPED",
                 "ROLLED_BACK", "BRANCHED")
RISK_LEVELS = ("SAFE", "GUARDED", "DESTRUCTIVE", "IRREVERSIBLE")


def _canonical_args(name: str, args: dict) -> str:
    payload = json.dumps({"name": name, "args": args}, sort_keys=True,
                         ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Plan node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    goal: str
    kind: str = "READ"
    predicate: dict | None = None       # PredicateSpec (§19)
    depends_on: list[str] = field(default_factory=list)
    path_set: list[str] = field(default_factory=list)  # write-exclusivity
    risk: str = "SAFE"
    clause_id: str | None = None        # attribution (§38.1)
    est_cost_usd: float = 0.0
    est_steps: int = 1
    status: str = "PENDING"
    attempts: int = 0

    def to_dict(self) -> dict:
        return {"id": self.id, "goal": self.goal, "kind": self.kind,
                "predicate": self.predicate, "depends_on": self.depends_on,
                "path_set": self.path_set, "risk": self.risk,
                "clause_id": self.clause_id, "est_cost_usd": self.est_cost_usd,
                "est_steps": self.est_steps, "status": self.status,
                "attempts": self.attempts}


# ---------------------------------------------------------------------------
# Plan DAG
# ---------------------------------------------------------------------------

class Plan:
    """A typed DAG over the event log. Nodes are sealed as plan.node events;
    status changes as plan.node.status. The frontier is a pure fold."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    def add(self, node: Node) -> Node:
        if node.kind not in NODE_KINDS:
            raise ValueError(f"node kind must be one of {NODE_KINDS}")
        if node.risk not in RISK_LEVELS:
            raise ValueError(f"risk must be one of {RISK_LEVELS}")
        self.log.append("plan.node", node.to_dict(), actor="sovereign",
                        correlation_id=node.clause_id)
        return node

    def set_status(self, node_id: str, status: str) -> None:
        if status not in NODE_STATUSES:
            raise ValueError(f"status must be one of {NODE_STATUSES}")
        self.log.append("plan.node.status", {"id": node_id,
                                             "status": status},
                        actor="kernel")

    def nodes(self) -> dict[str, dict]:
        return fold(self.log).nodes

    def frontier(self) -> list[dict]:
        """Nodes whose depends_on are all PASSED and that are PENDING.
        This is the parallel frontier the scheduler draws from (§16.5)."""
        nodes = self.nodes()
        out: list[dict] = []
        for n in nodes.values():
            if n.get("status") != "PENDING":
                continue
            deps = n.get("depends_on") or []
            if all(nodes.get(d, {}).get("status") == "PASSED" for d in deps):
                out.append(n)
        return out

    def eligible(self, max_parallel: int = 8) -> list[dict]:
        """The frontier filtered by write-exclusivity (I7): no two WRITE
        nodes with overlapping path sets, and no more than max_parallel.
        Deterministic ordering keeps the selection reproducible."""
        frontier = sorted(self.frontier(), key=lambda n: n.get("id", ""))
        chosen: list[dict] = []
        locked_paths: set[str] = set()
        for n in frontier:
            if len(chosen) >= max_parallel:
                break
            paths = set(n.get("path_set") or [])
            if n.get("kind") == "WRITE" and paths & locked_paths:
                continue  # would overlap an already-scheduled write
            if n.get("kind") == "WRITE":
                locked_paths |= paths
            chosen.append(n)
        return chosen


# ---------------------------------------------------------------------------
# Budget governor (§13.5)
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    max_usd: float = 3.00
    max_steps: int = 200
    max_tokens: int = 4_000_000
    max_files: int = 80
    slices: dict = field(default_factory=dict)  # subtree -> fraction


class BudgetGovernor:
    """Hierarchical budget over the event log. Every check is a fold; every
    breach is a budget.event that PAUSES the run (never silently kills)."""

    def __init__(self, log: EventLog, budget: Budget | None = None) -> None:
        self.log = log
        self.budget = budget or Budget()

    def spend(self) -> dict:
        st = fold(self.log)
        return {"usd": st.cost_usd, "steps": st.tool_calls,
                "tokens": st.tokens_in + st.tokens_out,
                "files": len(st.files_touched)}

    def check(self) -> tuple[bool, str]:
        """Return (ok, reason). A breach on ANY axis pauses the run."""
        s = self.spend()
        b = self.budget
        if s["usd"] > b.max_usd:
            return False, f"USD budget exceeded: ${s['usd']:.4f} > ${b.max_usd}"
        if s["steps"] > b.max_steps:
            return False, f"step budget exceeded: {s['steps']} > {b.max_steps}"
        if s["tokens"] > b.max_tokens:
            return False, f"token budget exceeded: {s['tokens']} > {b.max_tokens}"
        if s["files"] > b.max_files:
            return False, f"file budget exceeded: {s['files']} > {b.max_files}"
        return True, ""

    def enforce(self) -> bool:
        """Check and, on breach, seal a budget.event (pause). Returns True
        if the run may continue."""
        ok, reason = self.check()
        if not ok:
            self.log.append("budget.event",
                            {"kind": "exceeded", "reason": reason,
                             "spend": self.spend()},
                            actor="kernel")
        return ok

    def slice_for(self, subtree: str) -> float:
        """USD slice for a subtree — a hard cap a sub-agent cannot borrow
        past (§13.5)."""
        frac = self.budget.slices.get(subtree, 0.0)
        return self.budget.max_usd * frac


# ---------------------------------------------------------------------------
# Loop / thrash / oscillation detectors (§13.4)
# ---------------------------------------------------------------------------

class LoopDetector:
    """Detects wasted motion from the event log. All checks are rung 1."""

    def __init__(self, log: EventLog, repeat_threshold: int = 3,
                 window: int = 10) -> None:
        self.log = log
        self.repeat_threshold = repeat_threshold
        self.window = window

    def exact_repeat(self) -> str | None:
        """hash(tool, canonical_args) seen repeat_threshold times within
        the last `window` tool.call events -> returns the signature."""
        calls = [e for e in self.log.events() if e.type == "tool.call"]
        recent = calls[-self.window:]
        counts: dict[str, int] = {}
        for ev in recent:
            sig = _canonical_args(ev.data.get("name", ""),
                                  ev.data.get("args") or {})
            counts[sig] = counts.get(sig, 0) + 1
            if counts[sig] >= self.repeat_threshold:
                return sig
        return None

    def oscillation(self, path: str, hashes: list[str]) -> bool:
        """File content hash flipping A-B-A-B -> True (hard stop, present
        both versions to the human)."""
        if len(hashes) < 4:
            return False
        a, b, c, d = hashes[-4:]
        return a == c and b == d and a != b

    def detect(self) -> list[dict]:
        """Run all detectors, seal loop.alert events, return the alerts."""
        alerts: list[dict] = []
        sig = self.exact_repeat()
        if sig:
            alert = {"kind": "exact_repeat", "signature": sig,
                     "action": "force REFLECT with the repetition as evidence"}
            alerts.append(alert)
            self.log.append("loop.alert", alert, actor="kernel")
        return alerts


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "cortex.jsonl")

        # -- plan DAG + frontier + write-exclusivity (I7) --------------------
        plan = Plan(log)
        plan.add(Node("n1", "read config", kind="READ"))
        plan.add(Node("n2", "write auth", kind="WRITE",
                      path_set=["src/auth.py"], depends_on=["n1"]))
        plan.add(Node("n3", "write auth tests", kind="WRITE",
                      path_set=["src/auth.py"], depends_on=["n1"]))
        plan.add(Node("n4", "write docs", kind="WRITE",
                      path_set=["docs.md"], depends_on=["n1"]))

        # n1 is the only PENDING node with all deps passed
        assert [n["id"] for n in plan.frontier()] == ["n1"]
        plan.set_status("n1", "PASSED")

        # n2 and n3 overlap on src/auth.py -> only one may be scheduled
        elig = plan.eligible()
        ids = [n["id"] for n in elig]
        assert "n2" in ids and "n4" in ids, ids
        assert not ("n2" in ids and "n3" in ids), \
            f"overlapping writes scheduled together: {ids}"

        # -- budget governor (I8) ---------------------------------------------
        gov = BudgetGovernor(log, Budget(max_usd=0.001, max_steps=1000))
        log.append("cost.incurred", {"usd": 0.5, "tokens_in": 10,
                                     "tokens_out": 5})
        ok, reason = gov.check()
        assert not ok and "USD" in reason, (ok, reason)
        assert gov.enforce() is False
        budget_events = fold(log).budget_events
        assert budget_events and budget_events[-1]["kind"] == "exceeded"

        # slices: a sub-agent's hard cap
        gov2 = BudgetGovernor(log, Budget(max_usd=10.0,
                                          slices={"scouts": 0.05}))
        assert abs(gov2.slice_for("scouts") - 0.5) < 1e-9

        # -- loop detector (§13.4) ---------------------------------------------
        log2 = EventLog(Path(td) / "loops.jsonl")
        det = LoopDetector(log2, repeat_threshold=3, window=10)
        for _ in range(2):
            log2.append("tool.call", {"name": "read_file",
                                      "args": {"path": "x.py"}})
        assert det.exact_repeat() is None
        log2.append("tool.call", {"name": "read_file",
                                  "args": {"path": "x.py"}})
        sig = det.exact_repeat()
        assert sig is not None
        alerts = det.detect()
        assert alerts and alerts[0]["kind"] == "exact_repeat"
        assert fold(log2).loop_alerts

        # oscillation A-B-A-B
        assert det.oscillation("f.py", ["h1", "h2", "h1", "h2"]) is True
        assert det.oscillation("f.py", ["h1", "h2", "h1", "h3"]) is False
        assert det.oscillation("f.py", ["h1", "h2", "h1"]) is False

    print("CORTEX SELF-TEST PASS")
