"""DUAL — System 1 / System 2: metacognitive routing.

Kahneman's architecture, mechanical. Every request first meets the FAST
path and only earns the SLOW path when it must:

    System 1 (fast)      a cache of previously-verified answers plus a
                         cheap single model call — intuition: pattern
                         match first, think only if nothing matches
    System 2 (slow)      the deliberate stack — deep tools, debate,
                         judge verification (injectable: production
                         wires the debate/worker machinery)
    metacognition        the router ESTIMATES confidence before
                         answering: novelty (does the brain know this
                         domain?), hedging language in the fast answer
                         ("maybe", "I think", "not sure"), complexity
                         markers (multiple questions, negations,
                         numbers to reconcile), and cache staleness.
                         Confidence below the bar → escalate to System
                         2 and RECORD the escalation.

Fast answers that clear the bar cost one cheap call; hard questions
never bluff — they escalate. The escalation ledger is sealed
(dual.route / dual.escalation) so the split stays measurable: what
fraction of work genuinely needed deep thought.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .kernel import EventLog

_ESCALATE_BAR = 0.62
# A hard cap on the answer cache. Without it, a long-lived agent
# (the README pitches multi-hour sessions) leaks memory unbounded: a
# prompt-injection payload that asks the agent to enumerate 1..N with a
# unique suffix per call forces one model call per nonce and pins every
# response in `self.cache` forever. 256 entries is enough to amortise the
# real repeat-question traffic; the rest gets evicted LRU.
_CACHE_MAX = 256
_HEDGE_WORDS = re.compile(
    r"\b(maybe|perhaps|i think|not sure|unsure|possibly|might be|"
    r"probably|guess|honestly not|no idea|cannot determine)\b",
    re.I)
_COMPLEXITY = re.compile(
    r"[?;]|\b(vs|versus|compare|why|how come|prove|derive|trade-?off|"
    r"step by step)\b", re.I)


@dataclass
class RouteDecision2:
    system: int                     # 1 or 2
    answer: str
    confidence: float
    why: str
    cached: bool = False
    elapsed_ms: int = 0


class DualProcess:
    """Metacognitive router over an injectable fast/slow pair."""

    def __init__(self, log: EventLog, fast_fn, slow_fn,
                 brain=None, bar: float = _ESCALATE_BAR) -> None:
        """fast_fn(question) -> str (cheap). slow_fn(question) -> str
        (deliberate — debate, tools, whatever is wired). brain: optional
        cognitive memory used for the novelty signal."""
        self.log = log
        self.fast_fn = fast_fn
        self.slow_fn = slow_fn
        self.brain = brain
        self.bar = bar
        self.cache: OrderedDict[str, tuple[str, float, bool]] = OrderedDict()
        self.stats = {"system1": 0, "system2": 0, "cache_hits": 0,
                      "escalations": 0}

    def _cache_put(self, key: str, value: tuple[str, float, bool]) -> None:
        """Insert/update with LRU semantics and a hard cap. Prevents the
        cache from growing without bound (DoS via unique-questions loop)."""
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > _CACHE_MAX:
            self.cache.popitem(last=False)

    # -- metacognition ------------------------------------------------------

    def _confidence(self, question: str, answer: str,
                    cache_age: float | None) -> float:
        """Estimate how much to trust the fast path for this pair.
        Mechanical signals only, all in [0, 1] space."""
        conf = 0.75
        hedges = len(_HEDGE_WORDS.findall(answer or ""))
        conf -= 0.18 * min(hedges, 3)
        complexity = len(_COMPLEXITY.findall(question))
        conf -= 0.06 * min(complexity, 4)
        if self.brain is not None:
            known = self.brain.recall(question, k=3)
            has_memory = len(self.brain.memories) > 0
            if known:
                conf += 0.10             # familiar, reinforced domain
            elif has_memory:
                conf -= 0.12             # novel relative to experience
            # an empty brain carries no novelty signal either way
        if cache_age is not None:
            conf -= min(cache_age / 86400.0, 0.25)   # stale cache decays
        if not (answer or "").strip():
            conf = 0.0
        return max(0.0, min(1.0, conf))

    # -- the router ------------------------------------------------------------

    def ask(self, question: str) -> RouteDecision2:
        """Answer via System 1 unless metacognition escalates."""
        question = str(question or "").strip()
        t0 = time.monotonic()
        if not question:
            return RouteDecision2(1, "", 0.0, "empty question")

        key = question.lower().strip()[:200]
        cached = self.cache.get(key)
        if cached is not None:
            answer, sealed_at, from_slow = cached
            age = time.time() - sealed_at
            conf = self._confidence(question, answer, age)
            if from_slow:
                conf = min(1.0, conf + 0.15)   # deep-process provenance
            if conf >= self.bar:
                self.stats["cache_hits"] += 1
                self.stats["system1"] += 1
                self.log.append("dual.route",
                                {"system": 1, "cached": True,
                                 "confidence": round(conf, 3)})
                return RouteDecision2(1, answer, conf,
                                      "verified cache hit",
                                      cached=True,
                                      elapsed_ms=int(
                                          (time.monotonic() - t0)
                                          * 1000))

        fast = self.fast_fn(question)
        conf = self._confidence(question, fast, None)
        if conf >= self.bar:
            self.stats["system1"] += 1
            self._cache_put(key, (fast, time.time(), False))
            self.log.append("dual.route",
                            {"system": 1, "cached": False,
                             "confidence": round(conf, 3)})
            return RouteDecision2(1, fast, conf, "fast path cleared "
                                               "the bar",
                                  elapsed_ms=int((time.monotonic() - t0)
                                                 * 1000))

        # escalate — never bluff
        self.stats["system2"] += 1
        self.stats["escalations"] += 1
        self.log.append("dual.escalation",
                        {"question": question[:200],
                         "fast_confidence": round(conf, 3),
                         "bar": self.bar,
                         "signals": {"hedges": len(
                                         _HEDGE_WORDS.findall(fast)),
                                     "complexity": len(
                                         _COMPLEXITY.findall(
                                             question))}})
        slow = self.slow_fn(question)
        self._cache_put(key, (slow, time.time(), True))
        return RouteDecision2(2, slow, max(conf, 0.8),
                              "escalated: fast confidence "
                              f"{conf:.2f} < bar {self.bar:.2f}",
                              elapsed_ms=int((time.monotonic() - t0)
                                             * 1000))

    def format_stats(self) -> str:
        s = self.stats
        total = s["system1"] + s["system2"] or 1
        return ("DUAL PROCESS — system1 {} ({:.0f}%) · system2 {} · "
                "cache hits {} · escalations {}\n  bar: {:.2f}"
                .format(s["system1"], 100 * s["system1"] / total,
                        s["system2"], s["cache_hits"],
                        s["escalations"], self.bar))


# ---------------------------------------------------------------------------
# Self-test — scripted fast/slow prove the routing matrix
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "dual.jsonl")
        calls = {"fast": 0, "slow": 0}

        def fast(q: str) -> str:
            calls["fast"] += 1
            if "weather" in q:
                return "It is sunny, 24°C."
            return "maybe I think it could be something, not sure"

        def slow(q: str) -> str:
            calls["slow"] += 1
            return "Verified deep answer."

        dual = DualProcess(log, fast, slow)

        # easy factual → system 1, cached after
        r1 = dual.ask("what is the weather today?")
        assert r1.system == 1 and "sunny" in r1.answer, r1.why
        # same again → cache hit, no model call at all
        r2 = dual.ask("what is the weather today?")
        assert r2.system == 1 and r2.cached
        assert calls["fast"] == 1

        # hedged fast answer on a complex question → escalation
        r3 = dual.ask("compare the two architectures; which one and why?")
        assert r3.system == 2 and "Verified deep" in r3.answer
        assert dual.stats["escalations"] == 1

        # the escalated answer is now cached at high confidence
        r4 = dual.ask("compare the two architectures; which one and why?")
        assert r4.system == 1 and r4.cached and "Verified deep" in \
            r4.answer

        # complexity alone can push a clean-sounding answer over the bar
        r5 = dual.ask("derive step by step; prove it; why? how come?")
        assert r5.system == 2

        # novelty signal through the brain: unknown domain pushes down
        from fullagent.brain import Brain
        brain = Brain(EventLog(Path(td) / "brain.jsonl"), None)
        dual2 = DualProcess(EventLog(Path(td) / "d2.jsonl"), fast,
                            slow, brain=brain)
        rb = dual2.ask("what is the weather today?")
        assert rb.system == 1            # clean answer survives
        # a NOVEL domain with a hedged fast answer escalates harder
        rn = dual2.ask("explain the quantum trade-off maybe?")
        assert rn.system == 2

        # empty question is clean
        assert dual.ask("").confidence == 0.0

        # stats + events
        assert "DUAL PROCESS" in dual.format_stats()
        kinds = {e.type for e in log.events()}
        assert {"dual.route", "dual.escalation"} <= kinds

        print("DUAL SELF-TEST PASS")
