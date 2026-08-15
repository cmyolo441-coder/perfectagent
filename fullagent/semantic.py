"""SEMANTIC MEMORY — Hippocampus 2.0 (meaning-based recall).

The Hippocampus (memory.py) recalls by recency: the last N episodes. This
module adds recall by MEANING: every episode, fact and dead-end is embedded
into a vector, and a query retrieves whatever is semantically closest —
"how did we solve a similar problem before?"

Design (pure Python, stdlib only — no numpy, no model calls):
  * Embedding = feature hashing. Tokens are hashed into a fixed-width
    sparse vector (signed hashing to cancel collisions), L2-normalised.
    Deterministic, free, and good enough for nearest-neighbour recall over
    a few hundred records.
  * Similarity = cosine over the sparse vectors.
  * The corpus is ALWAYS rebuilt from the event-log fold — no separate
    state to drift. Indexing seals a 'semantic.indexed' event so recall
    quality is auditable.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

from .kernel import EventLog, fold

DIM = 256          # feature-hash width — plenty for a session corpus
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP = frozenset("a an the and or of to in for is are was were be been "
                  "this that with on at by from as it its".split())


# ---------------------------------------------------------------------------
# Embedding — signed feature hashing (deterministic, stdlib only)
# ---------------------------------------------------------------------------

def _token_hash(token: str) -> tuple[int, int]:
    """Map a token to (bucket, sign) via sha256 — collision-tolerant."""
    h = hashlib.sha256(token.encode("utf-8")).digest()
    bucket = int.from_bytes(h[:2], "big") % DIM
    sign = 1 if h[2] & 1 else -1
    return bucket, sign


def embed(text: str) -> dict[int, float]:
    """Sparse L2-normalised vector for a text (dict bucket -> weight).

    Unigrams plus bigrams, stop-words dropped, signed feature hashing so
    hash collisions partially cancel instead of always adding."""
    tokens = [t for t in _TOKEN_RE.findall(text.lower())
              if t not in _STOP and len(t) > 1]
    grams = list(tokens) + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    vec: dict[int, float] = {}
    for g in grams:
        bucket, sign = _token_hash(g)
        vec[bucket] = vec.get(bucket, 0.0) + sign
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        vec = {k: v / norm for k, v in vec.items()}
    return vec


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """Cosine similarity of two sparse vectors (both pre-normalised)."""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


# ---------------------------------------------------------------------------
# Corpus records
# ---------------------------------------------------------------------------

@dataclass
class MemoryItem:
    kind: str            # episode | fact | dead_end
    text: str            # the searchable rendering of the record
    payload: dict        # the original record from the fold
    vector: dict[int, float] = field(default_factory=dict)


def _episode_text(ep: dict) -> str:
    parts = [str(ep.get("goal", "")), str(ep.get("approach", "")),
             str(ep.get("outcome", "")), str(ep.get("lesson", "") or "")]
    parts += [str(f) for f in ep.get("facts") or []]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """Meaning-based recall over the episodic corpus. The index is a pure
    projection of the event log — rebuild it any time, it cannot drift."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self._items: list[MemoryItem] = []
        self._indexed = 0
        self._indexed_head = -1   # log head at last reindex

    def reindex(self) -> int:
        """Rebuild the whole index from the fold. Returns item count."""
        st = fold(self.log)
        items: list[MemoryItem] = []
        for ep in st.episodes:
            text = _episode_text(ep)
            if text.strip():
                items.append(MemoryItem("episode", text, ep, embed(text)))
        for f in st.facts:
            text = str(f.get("fact", ""))
            if text.strip():
                items.append(MemoryItem("fact", text, f, embed(text)))
        for d in st.dead_ends:
            text = f"{d.get('signature', '')} {d.get('reason', '')}"
            if text.strip():
                items.append(MemoryItem("dead_end", text, d, embed(text)))
        self._items = items
        self._indexed = len(items)
        self.log.append("semantic.indexed", {"items": len(items)},
                        actor="librarian")
        # capture the head AFTER the index event is sealed, so _ensure_fresh
        # does not immediately reindex on its own append
        self._indexed_head = self.log.head()
        return len(items)

    def _ensure_fresh(self) -> None:
        """Reindex if the log has grown since the last index, so recall
        always sees the current corpus (the index is a pure projection)."""
        if not self._items or self.log.head() != self._indexed_head:
            self.reindex()

    def recall(self, query: str, k: int = 3,
               min_similarity: float = 0.10) -> list[dict]:
        """The k corpus items closest in meaning to the query.

        Returns [{kind, similarity, text, payload}] sorted by similarity.
        Dead-ends are included — remembering what FAILED is recall too."""
        self._ensure_fresh()
        qv = embed(query)
        scored = [(cosine(qv, it.vector), it) for it in self._items]
        scored.sort(key=lambda p: (-p[0], p[1].kind, p[1].text))
        out = []
        for sim, it in scored[:k]:
            if sim < min_similarity:
                break
            out.append({"kind": it.kind, "similarity": round(sim, 3),
                        "text": it.text[:300], "payload": it.payload})
        return out

    def recall_block(self, query: str, k: int = 3) -> str:
        """Prompt-injectable rendering of a recall — empty string when
        nothing is similar enough (never inject noise)."""
        hits = self.recall(query, k=k)
        if not hits:
            return ""
        lines = [f"SEMANTIC RECALL for: {query[:80]}"]
        for h in hits:
            lines.append(f"- [{h['kind']} {h['similarity']:.2f}] {h['text']}")
        return "\n".join(lines)

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for it in self._items:
            kinds[it.kind] = kinds.get(it.kind, 0) + 1
        return {"items": len(self._items), "kinds": kinds, "dim": DIM}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from .memory import Hippocampus

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "semantic.jsonl")
        hip = Hippocampus(log)
        sem = SemanticMemory(log)

        # seed a corpus: two unrelated episodes + facts + a dead end
        hip.record_episode(
            goal="fix the CSV parser crash",
            approach="handle quoted fields in the tokenizer",
            actions=["read parser.py", "edit tokenizer"],
            outcome="success",
            facts=["csv parser lives in parser.py"],
            lesson="quoted fields need a state machine")
        hip.record_episode(
            goal="deploy the docker image",
            approach="build with multi-stage Dockerfile",
            actions=["write Dockerfile", "docker build"],
            outcome="success")
        hip.record_dead_end(signature="sha256:regex-csv",
                            reason="regex cannot parse quoted csv fields")

        n = sem.reindex()
        assert n == 3, n  # 2 episodes + 1 dead end (episode facts ride
        # inside their episode's embedded text)

        # meaning-based recall: parser question finds parser episode, not docker
        hits = sem.recall("how did we fix the csv parsing bug", k=2)
        assert hits, "expected recall hits"
        assert "csv" in hits[0]["text"].lower(), hits
        assert hits[0]["similarity"] > 0.1
        assert all("docker" not in h["text"].lower() for h in hits[:1])

        # dead ends are recallable — remembering failure is memory too
        hits = sem.recall("can we parse csv with a regex", k=3)
        assert any(h["kind"] == "dead_end" for h in hits), hits

        # unrelated query returns nothing above the noise floor
        assert sem.recall("quantum entanglement poetry", k=3) == []

        # prompt block renders only when there is signal
        assert sem.recall_block("quantum entanglement poetry") == ""
        block = sem.recall_block("csv parser")
        assert "SEMANTIC RECALL" in block

        # the index is a pure projection — reindex gives identical recall
        before = sem.recall("csv parser crash", k=2)
        sem.reindex()
        assert sem.recall("csv parser crash", k=2) == before

        # indexing is sealed in the log
        assert fold(log).semantic_index

        s = sem.stats()
        assert s["items"] == 3 and s["kinds"]["episode"] == 2

    print("SEMANTIC SELF-TEST PASS")
