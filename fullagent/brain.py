"""BRAIN — a four-store cognitive memory with forgetting and sleep.

The Hippocampus (memory.py) remembers everything forever; the Brain
remembers like an animal does:

    working     this session's scratchpad — dies with the session
    episodic    what happened, when (auto-ingested from kernel events)
    semantic    distilled facts, each with IMPORTANCE and a strength
    procedural  stable, repeatedly-useful knowledge promoted to skills

Mechanics (all rung 1 — deterministic math, zero tokens):
  * IMPORTANCE = base(kind) × recency × verification boost (judge-proven
    facts matter more) × access frequency
  * RETENTION follows an Ebbinghaus-style forgetting curve:
        R(t) = exp(-Δt / S),  S = S0 × (1 + strength)
    every recall REVIEWS the memory — its strength grows and the curve
    flattens (spaced repetition, mechanically)
  * SLEEP() is the consolidation pass, run between sessions or on
    demand: near-duplicate semantics merge (token-Jaccard), repeated
    episodic patterns distill into semantic facts, semantic facts that
    survived many reviews promote to procedural skills, and memories
    whose retention fell below the floor are forgotten (sealed as
    brain.forgotten — forgetting is an event, never silent loss)

Persistence is a JSON file under the app dir; the event log carries the
audit trail (remembered/recalled/consolidated/forgotten).
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .kernel import EventLog, fold

STORES = ("working", "episodic", "semantic", "procedural")

# base importance by memory kind
_BASE_IMPORTANCE = {"fact": 0.8, "lesson": 0.7, "episode": 0.5,
                    "dead_end": 0.9, "skill": 0.9, "note": 0.4}
S0 = 3600.0 * 24 * 2          # two days baseline strength horizon
RETENTION_FLOOR = 0.15        # below this, sleep() forgets
REVIEW_BOOST = 1.6            # each review multiplies strength
PROMOTE_THRESHOLD = 5         # reviews needed for semantic -> procedural
MERGE_SIMILARITY = 0.72       # token-Jaccard for near-duplicates
MAX_PER_STORE = 512

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class Memory:
    """One memory in any store."""
    id: str
    store: str
    text: str
    kind: str = "note"
    created: float = field(default_factory=time.time)
    last_review: float = field(default_factory=time.time)
    strength: float = 1.0            # grows with reviews
    reviews: int = 0
    verified: bool = False           # judge-proven facts
    tags: list[str] = field(default_factory=list)

    # -- forgetting curve -----------------------------------------------------

    @property
    def importance(self) -> float:
        base = _BASE_IMPORTANCE.get(self.kind, 0.4)
        age = max(0.0, time.time() - self.created)
        recency = 1.0 / (1.0 + age / (3600.0 * 24))   # halves each day-ish
        freq = 1.0 + 0.2 * min(self.reviews, 10)
        boost = 1.25 if self.verified else 1.0
        return base * recency * freq * boost

    def retention(self, now: float | None = None) -> float:
        """R(t) = exp(-Δt / S) with S = S0 × (1 + strength)."""
        now = now if now is not None else time.time()
        delta = max(0.0, now - self.last_review)
        return math.exp(-delta / (S0 * (1.0 + self.strength)))

    def review(self) -> None:
        """A recall is a review: strength up, curve flattened, schedule
        pushed out (spaced repetition, mechanically)."""
        self.reviews += 1
        self.strength *= REVIEW_BOOST
        self.last_review = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Memory":
        return Memory(**{k: d[k] for k in
                         ("id", "store", "text", "kind", "created",
                          "last_review", "strength", "reviews",
                          "verified", "tags") if k in d})


class Brain:
    """Four stores, one forgetting curve, one consolidation pass."""

    def __init__(self, log: EventLog, path: Path | None = None) -> None:
        self.log = log
        self.path = Path(path) if path else None
        self.memories: dict[str, Memory] = {}
        self._counter = 0
        if self.path is not None:
            self._load()

    # -- persistence ------------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for d in data.get("memories", []):
                try:
                    m = Memory.from_dict(d)
                except (TypeError, KeyError):
                    continue
                self.memories[m.id] = m
                n = int(m.id[1:]) if m.id[1:].isdigit() else 0
                self._counter = max(self._counter, n)
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"memories": [m.to_dict() for m in
                                         self.memories.values()]},
                           indent=1), encoding="utf-8")
        except OSError:
            pass  # persistence is a convenience, never a crash path

    # -- write paths -------------------------------------------------------------

    def remember(self, text: str, store: str = "semantic",
                 kind: str = "note", verified: bool = False,
                 tags: list[str] | None = None) -> Memory:
        """Store one memory. Duplicates (same store, high similarity)
        reinforce the existing memory instead of piling up."""
        text = str(text or "").strip()
        if not text:
            raise ValueError("cannot remember empty text")
        if store not in STORES:
            raise ValueError(f"store must be one of {STORES}")
        for m in self.memories.values():
            if m.store == store and jaccard(m.text, text) >= \
                    MERGE_SIMILARITY:
                m.review()
                m.verified = m.verified or verified
                self.log.append("brain.remembered",
                                {"id": m.id, "store": store,
                                 "merged": True})
                self._save()
                return m
        self._counter += 1
        m = Memory(id=f"m{self._counter}", store=store, text=text,
                   kind=kind, verified=verified, tags=tags or [])
        self.memories[m.id] = m
        self._cap(store)
        self.log.append("brain.remembered",
                        {"id": m.id, "store": store, "kind": kind,
                         "verified": verified, "chars": len(text)})
        self._save()
        return m

    def ingest_kernel(self) -> int:
        """Pull episodic material from the event log: assistant summaries,
        learned facts, dead ends, judge verdicts. Idempotent — only
        events newer than the last ingest are read."""
        last = max((int(e.data.get("brain_marker", 0))
                    for e in self.log.events()
                    if e.type == "brain.consolidated"), default=0)
        added = 0
        for ev in self.log.events():
            if ev.seq <= last:
                continue
            if ev.type == "fact.learned":
                self.remember(str(ev.data.get("fact", ""))[:400],
                              store="semantic", kind="fact",
                              verified=ev.data.get("kind") == "goal")
                added += 1
            elif ev.type == "deadend.recorded":
                self.remember(
                    f"dead end: {ev.data.get('reason', '')}"[:400],
                    store="semantic", kind="dead_end")
                added += 1
            elif ev.type == "assistant.message":
                text = str(ev.data.get("text", ""))
                if len(text) > 120:            # substantive replies only
                    self.remember(text[:300], store="episodic",
                                  kind="episode")
                    added += 1
        return added

    # -- read paths ----------------------------------------------------------------

    def recall(self, query: str, k: int = 4,
               store: str | None = None) -> list[Memory]:
        """Rank by (retention × importance × relevance) — only what the
        curve has kept alive can surface. A surfaced memory is REVIEWED."""
        q = _tokens(query)
        scored: list[tuple[float, Memory]] = []
        for m in self.memories.values():
            if store and m.store != store:
                continue
            t = _tokens(m.text)
            overlap = len(q & t)
            relevance = overlap / (len(q) + 1) if q else 0.0
            score = m.retention() * m.importance * (0.25 + relevance)
            scored.append((score, m))
        scored.sort(key=lambda p: -p[0])
        top = [m for _, m in scored[:max(0, k)]]
        for m in top:
            m.review()
        if top:
            self.log.append("brain.recalled",
                            {"query": query[:200],
                             "ids": [m.id for m in top]})
            self._save()
        return top

    def context_block(self, query: str = "", k: int = 4) -> str:
        """A compact recall block for the model's context."""
        mems = self.recall(query or "current work", k=k)
        if not mems:
            return ""
        lines = [f"MEMORY ({len(mems)} recalled, ranked by the "
                 f"forgetting curve):"]
        for m in mems:
            v = " ✓verified" if m.verified else ""
            lines.append(f"- [{m.kind}{v}] {m.text[:200]}")
        return "\n".join(lines)

    # -- sleep: consolidation --------------------------------------------------------

    def sleep(self) -> dict:
        """The consolidation pass. Merges near-duplicates, distills
        repeated episodes into semantic facts, promotes stable semantics
        to procedural skills, forgets what fell below the floor."""
        now = time.time()
        stats = {"merged": 0, "distilled": 0, "promoted": 0,
                 "forgotten": 0}

        # 1. forget what the curve has killed
        dead = [m for m in self.memories.values()
                if m.retention(now) < RETENTION_FLOOR and m.reviews == 0]
        for m in dead:
            del self.memories[m.id]
            stats["forgotten"] += 1
            self.log.append("brain.forgotten",
                            {"id": m.id, "store": m.store,
                             "text": m.text[:200]})

        # 2. merge near-duplicates within each store (keep the stronger)
        for store in STORES:
            items = sorted((m for m in self.memories.values()
                            if m.store == store),
                           key=lambda m: -m.strength)
            kept: list[Memory] = []
            for m in items:
                if any(jaccard(m.text, k.text) >= MERGE_SIMILARITY
                       for k in kept):
                    del self.memories[m.id]
                    stats["merged"] += 1
                    continue
                kept.append(m)

        # 3. distill: episodic themes repeated across >=3 distinct
        # episodes become one semantic fact
        episodes = [m for m in self.memories.values()
                    if m.store == "episodic"]
        seen: dict[str, int] = {}
        for m in episodes:
            for tag in (m.tags or _signature_tags(m.text)):
                seen[tag] = seen.get(tag, 0) + 1
        for tag, n in seen.items():
            if n >= 3:
                siblings = [m for m in episodes
                            if tag in (m.tags or _signature_tags(m.text))]
                digest = siblings[0].text[:160]
                exists = any(jaccard(digest, m.text) >= MERGE_SIMILARITY
                             for m in self.memories.values()
                             if m.store == "semantic")
                if not exists:
                    self._counter += 1
                    fact = Memory(id=f"m{self._counter}", store="semantic",
                                  text=f"[distilled from {n} episodes] "
                                       f"{digest}",
                                  kind="fact", strength=2.0,
                                  reviews=n)
                    self.memories[fact.id] = fact
                    stats["distilled"] += 1

        # 4. promote: semantic memories with enough reviews become
        # procedural skills (knowledge that has proven repeatedly useful)
        for m in list(self.memories.values()):
            if m.store == "semantic" and m.reviews >= PROMOTE_THRESHOLD:
                m.store = "procedural"
                m.kind = "skill"
                stats["promoted"] += 1

        self._cap_all()
        self.log.append("brain.consolidated",
                        {**stats, "brain_marker": self.log.head(),
                         "remaining": len(self.memories)})
        self._save()
        return stats

    def _cap(self, store: str) -> None:
        items = [m for m in self.memories.values() if m.store == store]
        if len(items) <= MAX_PER_STORE:
            return
        items.sort(key=lambda m: -(m.retention() * m.importance))
        for m in items[MAX_PER_STORE:]:
            del self.memories[m.id]

    def _cap_all(self) -> None:
        for store in STORES:
            self._cap(store)

    # -- reporting ----------------------------------------------------------------------

    def stats(self) -> dict:
        by_store: dict[str, int] = {s: 0 for s in STORES}
        for m in self.memories.values():
            by_store[m.store] = by_store.get(m.store, 0) + 1
        alive = [m for m in self.memories.values()
                 if m.retention() >= RETENTION_FLOOR]
        return {"total": len(self.memories), "by_store": by_store,
                "alive": len(alive),
                "avg_retention": round(
                    sum(m.retention() for m in self.memories.values())
                    / max(1, len(self.memories)), 3)}

    def format_stats(self) -> str:
        s = self.stats()
        return ("BRAIN — {total} memories ({alive} above the retention "
                "floor)\n  working    {working}\n  episodic   {episodic}"
                "\n  semantic   {semantic}\n  procedural {procedural}"
                "\n  avg retention {avg_retention}").format(**{
                    **s, **s["by_store"]})


def _signature_tags(text: str) -> list[str]:
    """Crude theme tags for distillation: the longest few tokens."""
    toks = sorted(_tokens(text), key=len, reverse=True)
    return toks[:3]


# ---------------------------------------------------------------------------
# Self-test — the whole cognitive loop, offline and fast (clock injected)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path as P
        store_path = P(td) / "brain.json"
        log = EventLog(P(td) / "brain-log.jsonl")
        brain = Brain(log, store_path)

        # remember + spaced repetition
        m = brain.remember("the parser lives in src/parser.py", kind="fact",
                           verified=True)
        assert m.store == "semantic" and m.reviews == 0
        brain.remember("parser lives in src/parser.py")    # near-dupe merges
        assert len(brain.memories) == 1 and m.reviews == 1

        # forgetting curve: retention decays, review flattens it
        before = m.retention()
        m.last_review -= 3600 * 24 * 10                # 10 days stale
        stale = m.retention()
        assert stale < before < 1.0
        m.review()
        assert m.retention() > 0.99                    # reviewed -> fresh

        # unreviewed stale memory is forgotten by sleep
        gone = brain.remember("totally irrelevant fluff", kind="note")
        gone.last_review -= 3600 * 24 * 365            # a year silent
        gone.created -= 3600 * 24 * 365
        stats = brain.sleep()
        assert stats["forgotten"] >= 1, stats
        assert gone.id not in brain.memories

        # recall ranks relevant above irrelevant and reviews winners
        noise = brain.remember("cooking pasta needs salt water", kind="note")
        hits = brain.recall("where is the parser file?", k=2)
        assert hits and hits[0].id == m.id, [h.text for h in hits]
        assert m.reviews >= 3                            # recall reviews

        # promotion: semantic with >= threshold reviews -> procedural
        while m.reviews < PROMOTE_THRESHOLD:
            m.review()
        brain.sleep()
        assert m.store == "procedural" and m.kind == "skill"

        # distillation: 3 same-theme episodes -> one semantic fact
        for i, txt in enumerate((
                "wired retry logic into charge calls",
                "added idempotency keys for safety",
                "replayed events through the ledger twice")):
            brain.remember(txt, store="episodic", kind="episode",
                           tags=["payment", "gateway"])
        # make them dissimilar enough not to merge, but same theme
        stats = brain.sleep()
        assert any("distilled from 3 episodes" in mm.text
                   for mm in brain.memories.values()), stats

        # kernel ingestion pulls facts + dead ends
        log.append("fact.learned", {"fact": "tests run with pytest -q",
                                    "kind": "goal"})
        log.append("deadend.recorded", {"reason": "sed -i broke on BSD"})
        log.append("assistant.message", {"text": "short"})
        log.append("assistant.message",
                   {"text": "we shipped the new parser after fixing the "
                            "tokenizer bug and adding property tests "
                            "across the whole token pipeline" * 2})
        added = brain.ingest_kernel()
        assert added >= 3, added
        assert any("pytest" in mm.text for mm in brain.memories.values())

        # persistence round-trip
        brain2 = Brain(EventLog(P(td) / "b2.jsonl"), store_path)
        assert set(brain2.memories) == set(brain.memories)

        # context block + stats render
        block = brain.context_block("parser")
        assert "MEMORY" in block and "parser" in block
        assert "BRAIN" in brain.format_stats()

        # fold carries the audit trail
        st = fold(log)
        kinds = {e["type"] for e in st.advanced_events}
        assert {"brain.remembered", "brain.recalled",
                "brain.consolidated", "brain.forgotten"} <= kinds, kinds

        print("BRAIN SELF-TEST PASS")
