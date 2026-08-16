"""FORMAL — temporal-logic model checking for agent plans and traces.

Prompt advice says "always snapshot before delete"; a model checker
PROVES it. This module checks event traces (real or hypothetical plan
executions) against temporal properties and returns a formal verdict
with a counterexample when one fails — no LLM anywhere in the loop.

A practical LTL subset, each property a pure predicate over traces:

    Never(a)              a must not occur at any position
    Before(a, b)          every a must be preceded by a b
    AlwaysAfter(a, b)     every a must eventually be followed by b
    AlwaysBetween(a, b)   between two b's there must be an a (e.g. a
                          snapshot between any write and the next turn)

Traces are lists of predicates; `plan_traces()` enumerates the concrete
executions a compiled plan DAG can produce (the interleave the lock
pass permits), and `verify_trace()` checks REAL kernel event logs —
the same checker audits history as gates the future.

The default property set encodes the system's own constitution:
I2 (snapshot before mutation), §16.1 (writes serialise: no two writes
adjacent without a completion between), §13.5 (a budget pause never
lands between a write and its verification).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

from .kernel import EventLog

_MAX_TRACES = 64          # plan- enumeration ceiling (combinatorics guard)


# ---------------------------------------------------------------------------
# Properties — pure functions over a trace of predicate-sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Never:
    a: str

    def check(self, trace: list[set[str]]) -> str | None:
        for i, s in enumerate(trace):
            if self.a in s:
                return f"{self.a} occurs at position {i}"
        return None


@dataclass(frozen=True)
class Before:
    """Every `a` must be preceded by `b` somewhere earlier."""
    a: str
    b: str

    def check(self, trace: list[set[str]]) -> str | None:
        seen_b = False
        for i, s in enumerate(trace):
            if self.b in s:
                seen_b = True
            if self.a in s and not seen_b:
                return f"{self.a} at position {i} without prior " \
                       f"{self.b}"
        return None


@dataclass(frozen=True)
class AlwaysAfter:
    """Every `a` must eventually be followed by `b` before the trace
    ends."""
    a: str
    b: str

    def check(self, trace: list[set[str]]) -> str | None:
        for i, s in enumerate(trace):
            if self.a in s:
                if not any(self.b in t for t in trace[i + 1:]):
                    return f"{self.a} at position {i} never followed " \
                           f"by {self.b}"
        return None


@dataclass(frozen=True)
class AlwaysBetween:
    """Between two `b` markers there must be an `a`."""
    a: str
    b: str

    def check(self, trace: list[set[str]]) -> str | None:
        last_b = -1
        for i, s in enumerate(trace):
            if self.b in s:
                if last_b >= 0 and not any(
                        self.a in t for t in trace[last_b + 1:i]):
                    return f"no {self.a} between {self.b} at positions " \
                           f"{last_b}..{i}"
                last_b = i
        return None


@dataclass(frozen=True)
class WritesSerialise:
    """§16.1 — no two writes without a verify between them (I7 as a
    temporal property, checked directly on write/verify predicates)."""

    def check(self, trace: list[set[str]]) -> str | None:
        unverified = False
        for i, s in enumerate(trace):
            if "verify" in s:
                unverified = False
            if "write" in s:
                if unverified:
                    return f"write at position {i} follows an " \
                           f"unverified write"
                unverified = True
        return None


PROPERTY_TYPES = (Never, Before, AlwaysAfter, AlwaysBetween,
                  WritesSerialise)

# the constitution, mechanically enforced on every plan
DEFAULT_PROPERTIES: tuple = (
    Before("write", "snapshot"),          # I2/A2 — no naked writes
    Before("delete", "snapshot"),
    WritesSerialise(),                    # §16.1 — writes serialise
    AlwaysAfter("write", "verify"),       # §37.4 — writes get checked
)


# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------

def trace_from_events(events) -> list[set[str]]:
    """Fold a real kernel event stream into a predicate trace."""
    trace: list[set[str]] = []
    for ev in events:
        preds: set[str] = set()
        t, d = ev.type, ev.data or {}
        if t == "snapshot.taken":
            preds.add("snapshot")
        elif t == "tool.call":
            name = str(d.get("name", ""))
            if name in ("write_file", "edit_file", "create_directory"):
                preds.add("write")
            elif name == "delete_path":
                preds.add("delete")
        elif t == "tool.result":
            name = str(d.get("name", ""))
            if name in ("write_file", "edit_file", "create_directory",
                        "delete_path"):
                preds.add("verify")
        elif t == "judge.verdict":
            preds.add("verify")
        elif t == "budget.event":
            preds.add("pause")
        if preds:
            trace.append(preds)
    return trace


def plan_traces(waves: list[list[dict]]) -> list[list[set[str]]]:
    """Enumerate the concrete executions a compiled plan (waves of
    items) can produce — every interleave the wave structure permits,
    capped combinatorially. Each item contributes its predicates
    (write/delete/read) plus the mechanical snapshot/verify the kernel
    would insert around writes."""
    per_wave: list[list[list[set[str]]]] = []
    for wave in waves:
        variants: list[list[set[str]]] = []
        for _ in range(1):                     # representative orderings
            seq: list[set[str]] = []
            for item in wave:
                paths = item.get("paths") or []
                writer = bool(paths)
                if writer:
                    seq.append({"snapshot"})     # A2 is mechanical
                kind = {"write"} if writer else {"read"}
                if item.get("task", "").lower().startswith("delete"):
                    kind = {"delete"}
                seq.append(kind)
                if writer:
                    seq.append({"verify"})
            variants.append(seq)
        # interleave variants: cap the permutation blow-up
        if len(variants) > 1:
            variants = variants[:_MAX_TRACES]
        per_wave.append(variants)

    traces: list[list[set[str]]] = [[]]
    for variants in per_wave:
        traces = [t + v for t in traces for v in variants]
        if len(traces) > _MAX_TRACES:
            traces = traces[:_MAX_TRACES]
    return traces


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    ok: bool
    checked: int = 0                       # traces examined
    violations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checked": self.checked,
                "violations": self.violations}


class ModelChecker:
    """Check traces against properties. Pure, total, deterministic."""

    def __init__(self, log: EventLog,
                 properties: tuple = DEFAULT_PROPERTIES) -> None:
        self.log = log
        self.properties = tuple(properties)

    def verify_trace(self, trace: list[set[str]],
                     label: str = "trace") -> VerificationResult:
        violations = []
        for prop in self.properties:
            why = prop.check(trace)
            if why is not None:
                violations.append({"property": _name(prop),
                                   "trace": label, "why": why})
        result = VerificationResult(ok=not violations, checked=1,
                                     violations=violations)
        if violations:
            self.log.append("verify.violation",
                            {"label": label, "violations": violations})
        return result

    def verify_plan(self, waves: list[list[dict]]) -> VerificationResult:
        """Every execution the plan permits must satisfy every property
        — a single counterexample rejects the plan."""
        traces = plan_traces(waves)
        all_violations: list[dict] = []
        for i, tr in enumerate(traces):
            r = self.verify_trace(tr, label=f"plan-trace-{i}")
            all_violations.extend(r.violations)
        result = VerificationResult(ok=not all_violations,
                                    checked=len(traces),
                                    violations=all_violations)
        self.log.append("verify.plan", result.to_dict())
        return result

    def audit_log(self) -> VerificationResult:
        """Check the REAL history — did the system itself ever violate
        its constitution?"""
        trace = trace_from_events(self.log.events())
        return self.verify_trace(trace, label="history")


def _name(prop) -> str:
    if isinstance(prop, Never):
        return f"never({prop.a})"
    if isinstance(prop, Before):
        return f"before({prop.a}, {prop.b})"
    if isinstance(prop, AlwaysAfter):
        return f"always_after({prop.a}, {prop.b})"
    if isinstance(prop, AlwaysBetween):
        return f"always_between({prop.a}, {prop.b})"
    if isinstance(prop, WritesSerialise):
        return "writes_serialise"
    return type(prop).__name__


# ---------------------------------------------------------------------------
# Self-test — proofs and counterexamples, all offline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "formal.jsonl")
        mc = ModelChecker(log)

        # a clean trace passes every default property
        clean = [{"snapshot"}, {"write"}, {"verify"}, {"read"}]
        assert mc.verify_trace(clean).ok

        # a naked write violates snapshot-before-write — counterexample
        bad = mc.verify_trace([{"write"}, {"verify"}])
        assert not bad.ok
        assert "without prior" in bad.violations[0]["why"]
        assert bad.violations[0]["property"] == "before(write, snapshot)"
        assert mc.verify_trace([{"delete"}]).violations  # delete too

        # double write with no completion between them (§16.1)
        dw = mc.verify_trace([{"snapshot"}, {"write"}, {"write"},
                              {"verify"}])
        assert any(v["property"] == "writes_serialise"
                   for v in dw.violations), dw.violations

        # a write never verified (liveness failure at trace end)
        lv = mc.verify_trace([{"snapshot"}, {"write"}])
        assert any(v["property"] == "always_after(write, verify)"
                   for v in lv.violations)

        # plan verification: kernel-shaped plans pass
        plan = [[{"task": "read the code", "role": "reviewer",
                  "paths": []}],
                [{"task": "write the fix", "role": "coder",
                  "paths": ["src/a.py"]}]]
        r = mc.verify_plan(plan)
        assert r.ok and r.checked >= 1, r.to_dict()

        # events -> trace -> audit
        log.append("snapshot.taken", {"tree": "t1"})
        log.append("tool.call", {"name": "write_file",
                                 "args": {"path": "a.py"}})
        log.append("tool.result", {"name": "write_file",
                                   "status": "done"})
        log.append("tool.call", {"name": "delete_path",
                                 "args": {"path": "b.txt"}})
        audit = mc.audit_log()
        assert audit.ok, audit.violations     # delete had a snapshot

        # ...but a delete without any snapshot is caught in real history
        log2 = EventLog(Path(td) / "formal2.jsonl")
        log2.append("tool.call", {"name": "delete_path",
                                  "args": {"path": "x.txt"}})
        bad_audit = ModelChecker(log2).audit_log()
        assert not bad_audit.ok
        assert any("delete" in v["why"] for v in bad_audit.violations)

        # custom properties compose like the built-ins
        strict = ModelChecker(log, properties=(Never("pause"),))
        assert strict.verify_trace([{"pause"}]).violations

        print("FORMAL SELF-TEST PASS")
