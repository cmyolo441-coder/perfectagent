"""Hippocampus — episodic memory + dead-end ledger.

Closed task nodes are compressed into STRUCTURED episode records (never
prose) and appended to the event log as `memory.episode` events. Approaches
known not to work are appended as `deadend.recorded` events. This module
keeps no state of its own: every query is a pure fold over the log, so
memory survives rewind, replay, and resume for free.

Design (pure Python, stdlib only):
  * Records are plain JSON-serializable dicts — they live in the JSONL log.
  * Dead-end checks are deterministic Python lookups over the fold. No LLM.
  * context_block() renders a compact, prompt-injectable summary.
"""

from __future__ import annotations

from .kernel import EventLog, fold

# Rough cap for one field rendered into context_block (~400 tokens total).
_FIELD_CHARS = 120


def _clip(text: str, limit: int = _FIELD_CHARS) -> str:
    """Clip a string for prompt injection without breaking the line format."""
    text = str(text).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class Hippocampus:
    """Episodic memory projected from `memory.episode` / `deadend.recorded`
    events in the log. All reads fold the log; writes append to it."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    # -- writes ------------------------------------------------------------

    def record_episode(self, *, goal: str, approach: str,
                       actions: list, outcome: str,
                       artifacts: list | None = None,
                       facts: list | None = None,
                       lesson: str | None = None,
                       dead_ends: list | None = None,
                       cost_usd: float = 0.0, steps: int = 0) -> dict:
        """Compress a closed task node into a STRUCTURED record (never prose)
        and emit a 'memory.episode' event. Returns the record dict.
        If dead_ends is given, also call record_dead_end for each entry
        (a signature string, or a dict with signature/reason/scope/confidence)."""
        record = {
            "goal": goal,
            "approach": approach,
            "actions": list(actions),
            "outcome": outcome,
            "artifacts": list(artifacts or []),
            "facts": [str(f) for f in (facts or [])],
            "lesson": lesson,
            "dead_ends": list(dead_ends or []),
            "cost_usd": float(cost_usd),
            "steps": int(steps),
        }
        self.log.append("memory.episode", record)
        for entry in record["dead_ends"]:
            if isinstance(entry, dict):
                self.record_dead_end(
                    signature=str(entry.get("signature", "")),
                    reason=str(entry.get("reason", "")),
                    scope=str(entry.get("scope", "session")),
                    confidence=str(entry.get("confidence", "definitive")),
                )
            else:
                self.record_dead_end(
                    signature=str(entry),
                    reason=f"failed while pursuing: {goal}",
                )
        return record

    def record_dead_end(self, *, signature: str, reason: str,
                        scope: str = "session",
                        confidence: str = "definitive") -> dict:
        """Record an approach known NOT to work. Emit 'deadend.recorded'.
        signature = canonical hash/id of the approach. Returns the dict."""
        record = {
            "signature": signature,
            "reason": reason,
            "scope": scope,
            "confidence": confidence,
        }
        self.log.append("deadend.recorded", record)
        return record

    # -- reads (pure folds, no LLM) -----------------------------------------

    def is_dead_end(self, signature: str) -> bool:
        """Deterministic check (no LLM): is this signature in the ledger?
        Fold the log and scan State.dead_ends."""
        st = fold(self.log)
        return any(d.get("signature") == signature for d in st.dead_ends)

    def recent_episodes(self, n: int = 5) -> list[dict]:
        """Return the last n episode records from the fold, newest first."""
        if n <= 0:
            return []
        st = fold(self.log)
        return list(reversed(st.episodes[-n:]))

    def facts(self) -> list[str]:
        """Aggregate all 'facts' across episodes, deduplicated, order-preserving."""
        st = fold(self.log)
        seen: set[str] = set()
        out: list[str] = []
        for ep in st.episodes:
            for fact in ep.get("facts") or []:
                fact = str(fact)
                if fact not in seen:
                    seen.add(fact)
                    out.append(fact)
        return out

    def context_block(self, max_episodes: int = 3) -> str:
        """Render a compact, prompt-injectable text block summarizing recent
        episodes, learned facts, and active dead-ends. Kept <= ~400 tokens
        by clipping each rendered field."""
        st = fold(self.log)
        lines: list[str] = ["MEMORY"]

        episodes = st.episodes[-max_episodes:] if max_episodes > 0 else []
        if episodes:
            lines.append("RECENT EPISODES (newest first):")
            for ep in reversed(episodes):
                lines.append(
                    f"- [{_clip(ep.get('outcome', '?'), 20)}] "
                    f"goal: {_clip(ep.get('goal', ''))} | "
                    f"approach: {_clip(ep.get('approach', ''))} | "
                    f"steps: {int(ep.get('steps', 0))} | "
                    f"cost: ${float(ep.get('cost_usd', 0.0)):.4f}"
                )
                if ep.get("lesson"):
                    lines.append(f"  lesson: {_clip(ep['lesson'])}")

        facts: list[str] = []
        seen_facts: set[str] = set()
        for ep in st.episodes:
            for fact in ep.get("facts") or []:
                fact = str(fact)
                if fact not in seen_facts:
                    seen_facts.add(fact)
                    facts.append(fact)
        # top-level learned facts (goal proofs, team worker results)
        for f in st.facts:
            fact = str(f.get("fact", ""))
            if fact and fact not in seen_facts:
                seen_facts.add(fact)
                facts.append(fact)
        if facts:
            lines.append("FACTS:")
            lines.extend(f"- {_clip(f)}" for f in facts[-12:])

        if st.dead_ends:
            lines.append("DEAD ENDS (do not retry):")
            seen_sigs: set[str] = set()
            for d in reversed(st.dead_ends):
                sig = str(d.get("signature", ""))
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                lines.append(
                    f"- {sig} — {_clip(d.get('reason', ''))} "
                    f"({_clip(d.get('confidence', 'definitive'), 20)})"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "memory-test.jsonl")
        hip = Hippocampus(log)

        rec = hip.record_episode(
            goal="fix the parser",
            approach="rewrite tokenizer",
            actions=["read parser.py", "edit tokenizer", "run tests"],
            outcome="success",
            artifacts=["parser.py"],
            facts=["tokenizer is line-based", "tests live in tests/",
                   "tokenizer is line-based"],
            lesson="run the test suite after every edit",
            dead_ends=[{"signature": "sha256:abc123",
                        "reason": "regex approach hit catastrophic backtracking"}],
            cost_usd=0.0123,
            steps=4,
        )
        assert rec["goal"] == "fix the parser"
        assert rec["steps"] == 4
        assert rec["dead_ends"][0]["signature"] == "sha256:abc123"
        json.dumps(rec)  # record must be JSON-serializable

        hip.record_dead_end(signature="sha256:def456",
                            reason="API requires a token we do not have")

        assert hip.is_dead_end("sha256:abc123")      # via episode dead_ends
        assert hip.is_dead_end("sha256:def456")      # recorded directly
        assert not hip.is_dead_end("sha256:unknown")

        recent = hip.recent_episodes(5)
        assert len(recent) == 1 and recent[0]["goal"] == "fix the parser"

        assert hip.facts() == ["tokenizer is line-based",
                               "tests live in tests/"]  # deduped, ordered

        block = hip.context_block()
        assert "MEMORY" in block
        assert "fix the parser" in block
        assert "sha256:def456" in block
        assert len(block) < 1600  # ~400-token budget

        print("MEMORY SELF-TEST PASS")
