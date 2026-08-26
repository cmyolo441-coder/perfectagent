"""CASSETTE — record/replay of model calls (§20.2, §35).

Every request/response pair is recorded; a session replays against the
cassette with ZERO API cost. This is how the test suite runs, and it is
the dividend 'deterministic testing' from the Mul Bindu table (§0.2).

Design (pure Python, stdlib only):
  * Key = sha256(canonical(model, messages, tools)) — the request hash.
  * record mode: real calls pass through and are stored keyed by hash.
  * replay mode: matching requests return the stored response; a miss is a
    hard error (never a silent live call), so replays are deterministic.
  * Cassettes are JSONL, one line per pair, human-inspectable.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any
import copy


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def request_key(model: str, messages: list[dict],
                tools: list[dict] | None,
                effort_key: str | None = None) -> str:
    """The cassette key for one request (§8.1: blake3(request) — here
    sha256, same role).

    `effort_key` participates in the hash: sampling params (max_tokens,
    temperature) change with effort but NOT with messages, so two requests
    that differ only in effort must not collide on one recorded response."""
    payload = {"model": model, "messages": messages, "tools": tools or [],
               "effort": effort_key or ""}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


class Cassette:
    """Record or replay model request/response pairs."""

    def __init__(self, path: Path, mode: str = "off") -> None:
        """mode: 'off' | 'record' | 'replay'."""
        if mode not in ("off", "record", "replay"):
            raise ValueError("mode must be off | record | replay")
        self.path = Path(path)
        self.mode = mode
        self._lock = threading.Lock()
        self._store: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        if mode in ("record", "replay") and self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pair = json.loads(line)
                except ValueError:
                    continue
                key = pair.get("key")
                if key:
                    self._store[key] = pair.get("response")

    def _persist(self, key: str, response: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": response},
                               ensure_ascii=False) + "\n")

    # -- record ---------------------------------------------------------------

    def record(self, model: str, messages: list[dict],
               tools: list[dict] | None, response: dict,
               effort_key: str | None = None) -> None:
        """Store a real response (record mode only)."""
        if self.mode != "record":
            return
        key = request_key(model, messages, tools, effort_key)
        with self._lock:
            # deep-copy IN: a caller mutating the response afterwards must
            # not rewrite what the cassette (and the JSONL) recorded
            stored = copy.deepcopy(response)
            self._store[key] = stored
            self._persist(key, stored)

    # -- replay ---------------------------------------------------------------

    def replay(self, model: str, messages: list[dict],
               tools: list[dict] | None,
               effort_key: str | None = None) -> dict | None:
        """Return the stored response for a request (replay mode only).
        A miss returns None and counts as a miss — the caller must treat it
        as a hard error, never fall back to a live call, or the replay is
        no longer deterministic."""
        if self.mode != "replay":
            return None
        key = request_key(model, messages, tools, effort_key)
        with self._lock:
            if key in self._store:
                self.hits += 1
                # deep-copy OUT: annotating a replayed response must not
                # corrupt every future replay of the same key
                return copy.deepcopy(self._store[key])
            self.misses += 1
            return None

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cassette.jsonl"
        msgs = [{"role": "user", "content": "hello"}]
        resp = {"content": "hi there", "usage": {"prompt_tokens": 5,
                                                 "completion_tokens": 3}}

        # record mode stores the pair
        rec = Cassette(path, mode="record")
        rec.record("model-x", msgs, None, resp)
        assert len(rec) == 1

        # replay mode returns it with zero API cost
        play = Cassette(path, mode="replay")
        got = play.replay("model-x", msgs, None)
        assert got == resp, got
        assert play.hits == 1 and play.misses == 0

        # a different request is a miss (deterministic: never a live call)
        other = [{"role": "user", "content": "different"}]
        assert play.replay("model-x", other, None) is None
        assert play.misses == 1

        # off mode neither records nor replays
        off = Cassette(Path(td) / "off.jsonl", mode="off")
        off.record("model-x", msgs, None, resp)
        assert off.replay("model-x", msgs, None) is None
        assert len(off) == 0

        # the cassette file is human-inspectable JSONL
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        pair = json.loads(lines[0])
        assert pair["response"] == resp and "key" in pair

        # replay survives reload from disk
        play2 = Cassette(path, mode="replay")
        assert play2.replay("model-x", msgs, None) == resp

    print("CASSETTE SELF-TEST PASS")
