"""FABRIC — the bitemporal knowledge graph: facts with a lifespan.

A normal knowledge base answers "what is true NOW". The fabric answers
BOTH "what is true now" and "what did we know THEN" — two time axes:

    valid time     when the fact was true in the WORLD
    transaction    when the fact entered the FABRIC (the log's clock)

Mechanics:
    assert(s, p, o)      a new fact; if it CONTRADICTS a live fact on
                         the same (s, p), the old fact's valid-time is
                         closed at the new fact's start — the old truth
                         is not deleted, it EXPIRED, with full
                         provenance (fabric.retract event)
    query(s, p, at=)     as-of queries: at=None → the live truth; at=t
                         → what the fabric believed was valid at t
    since(when)          everything learned after a transaction time —
                         "what's new since I last looked"
    contradiction()      live facts that somehow overlap (can only
                         happen through explicit same-timestamp
                         asserts) — surfaced, never hidden

Every assert/retract is a sealed kernel event, so the fabric's own
history is itself replayable — a knowledge base that can be debugged
with the Theater.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .kernel import EventLog

_INFINITY = float("inf")


@dataclass
class Fact:
    subject: str
    predicate: str
    obj: str
    valid_from: float
    valid_to: float = _INFINITY
    txn_time: float = field(default_factory=time.time)
    confidence: float = 1.0
    superseded_by: str | None = None      # id of the replacing fact

    @property
    def fid(self) -> str:
        return f"{self.subject}|{self.predicate}|{self.obj}"

    def live(self, at: float | None = None) -> bool:
        t = time.time() if at is None else at
        return self.valid_from <= t < self.valid_to

    def to_dict(self) -> dict:
        return {"s": self.subject, "p": self.predicate, "o": self.obj,
                "valid_from": self.valid_from,
                "valid_to": None if self.valid_to == _INFINITY
                else self.valid_to,
                "txn": round(self.txn_time, 3),
                "confidence": self.confidence}


class KnowledgeFabric:
    """Bitemporal triples with automatic contradiction expiry."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.facts: list[Fact] = []

    # -- writes -----------------------------------------------------------------

    def assert_fact(self, subject: str, predicate: str, obj: str,
                    valid_from: float | None = None,
                    confidence: float = 1.0) -> Fact:
        """Assert (s, p, o). Any live fact on the same (s, p) with a
        DIFFERENT object expires at this fact's valid_from."""
        subject = str(subject).strip()
        predicate = str(predicate).strip()
        obj = str(obj).strip()
        if not subject or not predicate or not obj:
            raise ValueError("subject, predicate and object are "
                             "all required")
        now = time.time()
        start = valid_from if valid_from is not None else now
        fact = Fact(subject=subject, predicate=predicate, obj=obj,
                    valid_from=start, txn_time=now,
                    confidence=confidence)
        # expire contradictory live predecessors
        for old in self.facts:
            if (old.subject, old.predicate) == (subject, predicate) \
                    and old.obj != obj and old.live(max(now, start)) \
                    and old.valid_to == _INFINITY:
                old.valid_to = max(now, start)
                old.superseded_by = fact.fid
                self.log.append("fabric.retract",
                                {"retracted": old.to_dict(),
                                 "reason": "superseded",
                                 "by": fact.fid}, actor="kernel")
        self.facts.append(fact)
        self.log.append("fabric.assert", fact.to_dict(),
                        actor="sovereign")
        return fact

    # -- reads ----------------------------------------------------------------------

    def query(self, subject: str, predicate: str,
              at: float | None = None) -> list[Fact]:
        """Live facts for (s, p) — now, or as of a valid-time."""
        return sorted(
            [f for f in self.facts
             if f.subject == subject and f.predicate == predicate
             and f.live(at)],
            key=lambda f: f.valid_from)

    def ask(self, subject: str, predicate: str,
            at: float | None = None) -> str | None:
        hits = self.query(subject, predicate, at)
        return hits[-1].obj if hits else None

    def since(self, txn_time: float) -> list[Fact]:
        """Everything learned after a transaction time."""
        return [f for f in self.facts if f.txn_time > txn_time]

    def live_all(self) -> list[Fact]:
        return [f for f in self.facts if f.live()]

    def contradictions(self, at: float | None = None
                       ) -> list[tuple[Fact, Fact]]:
        """Live facts on the same (s, p) with different objects — only
        possible via explicit overlapping valid-times (e.g. a future-
        dated assert). Surfaced, never hidden."""
        live = [f for f in self.facts if f.live(at)]
        out = []
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if (a.subject, a.predicate) == (b.subject, b.predicate) \
                        and a.obj != b.obj:
                    out.append((a, b))
        return out

    # -- reporting ---------------------------------------------------------------------

    def history(self, subject: str, predicate: str) -> str:
        rows = sorted([f for f in self.facts
                       if f.subject == subject
                       and f.predicate == predicate],
                      key=lambda f: f.valid_from)
        if not rows:
            return f"no history for {subject}·{predicate}"
        lines = [f"HISTORY — {subject} {predicate}:"]
        for f in rows:
            vfrom = time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime(f.valid_from))
            if f.live():
                state = "live"
            elif f.valid_to == _INFINITY:
                state = "not yet valid"
            else:
                state = ("expired @ " + time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(f.valid_to)))
            lines.append(f"  {f.obj:<30} valid from {vfrom} — "
                         f"{state}  (conf {f.confidence:.2f})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — time travel over contradicting truths
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "fabric.jsonl")
        fabric = KnowledgeFabric(log)

        t1 = time.time() - 90 * 86400          # 90 days ago
        t2 = time.time() - 30 * 86400          # 30 days ago

        # the dependency was flask, then became fastapi
        fabric.assert_fact("api", "framework", "flask", valid_from=t1)
        fabric.assert_fact("api", "framework", "fastapi", valid_from=t2)

        # NOW: fastapi is the live truth, flask expired
        assert fabric.ask("api", "framework") == "fastapi"
        # AS OF 60 DAYS AGO: flask was the truth
        assert fabric.ask("api", "framework", at=t1 + 60 * 86400) == \
            "flask"
        # before anything: unknown
        assert fabric.ask("api", "framework", at=t1 - 1) is None

        # non-conflicting facts on different predicates coexist
        fabric.assert_fact("api", "language", "python",
                           valid_from=t1)
        assert fabric.ask("api", "language") == "python"
        assert fabric.ask("api", "framework") == "fastapi"

        # a FUTURE-dated assert first: it cannot expire anything (its
        # window hasn't opened) and nothing expires IT — a present-
        # dated rival afterwards leaves both live tomorrow, and the
        # contradiction is surfaced, not hidden
        tomorrow = time.time() + 86400
        fabric.assert_fact("db", "driver", "postgres",
                           valid_from=tomorrow)
        assert fabric.contradictions() == []          # not yet
        marker = time.time()
        time.sleep(0.01)
        fabric.assert_fact("db", "driver", "mysql", valid_from=t1)
        learned = fabric.since(marker)
        assert len(learned) == 1, [f.to_dict() for f in learned]
        assert learned[0].obj == "mysql"
        cons = fabric.contradictions(at=tomorrow + 1)
        assert len(cons) == 1 and cons[0][0].obj != cons[0][1].obj, cons

        # history renders both eras
        text = fabric.history("api", "framework")
        assert "flask" in text and "fastapi" in text and "expired" \
            in text

        # validation + events
        try:
            fabric.assert_fact("", "x", "y")
            raise AssertionError("empty subject must raise")
        except ValueError:
            pass
        kinds = [e.type for e in log.events()]
        assert "fabric.assert" in kinds and "fabric.retract" in kinds

        print("FABRIC SELF-TEST PASS")
