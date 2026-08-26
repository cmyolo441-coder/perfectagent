"""SPECULATOR — speculative execution (zero-latency feel).

While the model is thinking, the agent PREDICTS which read-only tool calls
it is likely to make next and runs them ahead of time in a background
thread. When the model actually asks, the answer is served from the
speculative cache instantly — a cache hit instead of a real execution.

Hard safety rules (mechanical, not advisory):
  * ONLY whitelisted read-only tools are ever prefetched. A speculative
    write is structurally impossible — the whitelist is the gate.
  * Predictions are deterministic Python over the conversation (rung 1):
    paths mentioned in the last user message, files referenced by recent
    tool calls, directory listings after a cd-like command. No model call.
  * Every prefetch, hit, miss and eviction is sealed into the event log,
    so the hit-rate — the actual dividend — is auditable.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .kernel import EventLog, fold

# The ONLY tools speculation may ever run. Read-only by construction.
SPECULATIVE_TOOLS = ("read_file", "list_dir", "file_info",
                     "search_files", "glob_files")

MAX_PREFETCH = 4          # predictions executed per speculation round
CACHE_TTL_TURNS = 6       # a prefetched result expires after this many turns
# a path: starts with ./~ OR contains a slash OR ends with a file extension
_PATH_RE = re.compile(
    r"(?:^|[\s'\"`(])"
    r"((?:[./~][\w./~-]{0,200})|(?:[\w~-]+(?:/[\w.-]+)+)|(?:[\w~-]+\.\w{1,6}))"
    r"(?=$|[\s'\"`),:;])")


# ---------------------------------------------------------------------------
# Prediction (rung 1 — deterministic, no model)
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    tool: str
    args: dict
    score: float = 0.0
    why: str = ""

    def key(self) -> str:
        return f"{self.tool}:{sorted(self.args.items())}"


def predict(user_text: str, recent_tools: list[dict],
            cwd_listing: list[str] | None = None) -> list[Prediction]:
    """Predict the read-only calls the model is likely to make next.

    Signals, in strength order:
      1. explicit paths in the user message -> read_file / file_info
      2. a directory mentioned -> list_dir
      3. a search-ish verb + term -> search_files
      4. after recent file reads -> read the sibling files
    """
    preds: list[Prediction] = []
    seen: set[str] = set()

    def add(p: Prediction) -> None:
        k = p.key()
        if k not in seen:
            seen.add(k)
            preds.append(p)

    # 1+2: paths named by the user
    for m in _PATH_RE.finditer(user_text or ""):
        path = m.group(1)
        if path.endswith(("/", ".")):
            add(Prediction("list_dir", {"path": path.rstrip("/") or "."},
                           0.9, "path in user message"))
        else:
            add(Prediction("read_file", {"path": path},
                           0.9, "path in user message"))
            add(Prediction("file_info", {"path": path},
                           0.4, "path in user message"))

    # 3: search intent
    low = (user_text or "").lower()
    sm = re.search(r"\b(?:search|find|grep|dhundo)\b\s+(?:for\s+)?['\"]?([\w .-]{2,40})", low)
    if sm:
        add(Prediction("search_files", {"pattern": sm.group(1).strip()},
                       0.6, "search verb in user message"))

    # 4: siblings of recently read files
    for call in recent_tools[-3:]:
        if call.get("name") == "read_file":
            p = str(call.get("args", {}).get("path", ""))
            if "/" in p:
                parent = p.rsplit("/", 1)[0]
                add(Prediction("list_dir", {"path": parent},
                               0.5, "sibling of recent read"))

    # 5: default — list the cwd when nothing else is signalled
    if not preds:
        add(Prediction("list_dir", {"path": "."}, 0.2, "default cwd probe"))

    preds.sort(key=lambda p: -p.score)
    return preds[:MAX_PREFETCH]


# ---------------------------------------------------------------------------
# Speculator
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    tool: str
    args: dict
    result: str
    born_turn: int


class Speculator:
    """Prefetch predicted read-only calls; serve real calls from the cache.

    `runner` executes one tool call: runner(name, args) -> str. In the
    agent it is bound to the real read-only handlers; in tests it is a stub.
    """

    def __init__(self, log: EventLog, runner=None) -> None:
        self.log = log
        self.runner = runner
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.turn = 0
        self.hits = 0
        self.misses = 0

    # -- speculation round ---------------------------------------------------

    def speculate(self, user_text: str, recent_tools: list[dict]) -> int:
        """Predict + prefetch in a background pool. Returns # prefetched."""
        self.turn += 1
        self._expire()
        preds = predict(user_text, recent_tools)
        with self._lock:
            fresh = [p for p in preds if p.key() not in self._cache]
        if not fresh or self.runner is None:
            return 0

        def _run(p: Prediction) -> tuple[Prediction, str]:
            try:
                return p, self.runner(p.tool, dict(p.args))
            except Exception as e:  # speculation must never surface errors
                return p, f"ERROR: {type(e).__name__}: {e}"

        done = 0
        with ThreadPoolExecutor(max_workers=min(4, len(fresh))) as ex:
            for p, result in ex.map(_run, fresh):
                if result.startswith("ERROR:"):
                    continue
                with self._lock:
                    self._cache[p.key()] = CacheEntry(
                        p.tool, p.args, result, self.turn)
                self.log.append("spec.prefetch",
                                {"tool": p.tool, "args": p.args,
                                 "score": p.score, "why": p.why,
                                 "chars": len(result)},
                                actor="speculator")
                done += 1
        return done

    # -- serving ---------------------------------------------------------------

    def serve(self, tool: str, args: dict) -> str | None:
        """If this exact call was prefetched and is fresh, return the cached
        result (a hit). Otherwise None — the caller runs it for real."""
        if tool not in SPECULATIVE_TOOLS:
            return None
        key = f"{tool}:{sorted(args.items())}"
        # Counter increment AND the spec.hit event log BOTH live inside
        # the lock now. The previous code released the lock between
        # `del self._cache[key]` and `self.hits += 1`, so two threads
        # asking for the same prefetched call could both miss-then-hit
        # and the audit log would record spec.miss before spec.hit for
        # what was actually a single legitimate cache hit.
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.misses += 1
                self.log.append("spec.miss", {"tool": tool, "args": args},
                                actor="speculator")
                return None
            del self._cache[key]
            self.hits += 1
            self.log.append("spec.hit", {"tool": tool, "args": args,
                                         "chars": len(entry.result)},
                            actor="speculator")
        return entry.result

    # -- housekeeping ------------------------------------------------------------

    def _expire(self) -> None:
        with self._lock:
            stale = [k for k, e in self._cache.items()
                     if self.turn - e.born_turn > CACHE_TTL_TURNS]
            for k in stale:
                del self._cache[k]
            if stale:
                self.log.append("spec.evict", {"count": len(stale)},
                                actor="speculator")

    def stats(self) -> dict:
        evs = fold(self.log).spec_events
        prefetched = sum(1 for e in evs if e["type"] == "spec.prefetch")
        hits = sum(1 for e in evs if e["type"] == "spec.hit")
        misses = sum(1 for e in evs if e["type"] == "spec.miss")
        rate = hits / max(1, hits + misses)
        return {"prefetched": prefetched, "hits": hits, "misses": misses,
                "hit_rate": round(rate, 3), "cached": len(self._cache)}

    def format_status(self) -> str:
        s = self.stats()
        return ("SPECULATOR — speculative execution\n"
                f"  prefetched {s['prefetched']}   hits {s['hits']}   "
                f"misses {s['misses']}   hit-rate {s['hit_rate']:.0%}\n"
                f"  only read-only tools are ever prefetched: "
                f"{', '.join(SPECULATIVE_TOOLS)}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "spec.jsonl")

        calls: list[tuple[str, dict]] = []

        def runner(name: str, args: dict) -> str:
            calls.append((name, args))
            if name == "read_file" and "missing" in str(args.get("path")):
                return "ERROR: file not found"
            return f"OK:{name}:{args}"

        spec = Speculator(log, runner)

        # prediction: a path in the user message is the strongest signal
        preds = predict("please look at src/main.py and fix it", [])
        assert any(p.tool == "read_file" and p.args["path"] == "src/main.py"
                   for p in preds), preds

        # speculation round prefetches the predicted read
        n = spec.speculate("please look at src/main.py and fix it", [])
        assert n >= 1, n
        assert any(c[0] == "read_file" for c in calls), calls

        # the model then asks for exactly that file -> cache HIT
        got = spec.serve("read_file", {"path": "src/main.py"})
        assert got is not None and got.startswith("OK:read_file"), got

        # asking again is a miss (the entry was consumed)
        assert spec.serve("read_file", {"path": "src/main.py"}) is None

        # a write tool can NEVER be served speculatively
        assert spec.serve("write_file", {"path": "x", "content": "y"}) is None

        # erroring prefetches are dropped, never cached
        spec.speculate("read missing/file.txt", [])
        assert spec.serve("read_file", {"path": "missing/file.txt"}) is None

        # stats + ledger
        s = spec.stats()
        assert s["hits"] == 1 and s["misses"] >= 2, s
        assert fold(log).spec_events

        text = spec.format_status()
        assert "SPECULATOR" in text and "hit-rate" in text

    print("SPECULATOR SELF-TEST PASS")
