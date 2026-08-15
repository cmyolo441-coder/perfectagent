"""Swarm — parallel read-only scout sub-agents.

"Reads fan out. Writes serialise." Each scout gets a fresh minimal context
(its question + a little shared context) and a strictly read-only tool
whitelist (files, listing, search — no writes, no shell, no network
mutations), and returns a small structured report — never a transcript.
Running N scouts in parallel costs about what one bloated conversation
costs, and nothing pollutes the main agent's context: reports land in the
event log as 'swarm.report' events.
"""

from __future__ import annotations

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import systemprompt
from .client import APIError, chat_blocking
from .config import Effort, Model, Provider
from .kernel import EventLog, fold
from .tools import Tool, build_registry, parse_tool_arguments

# Scout answers stay small (~400 tokens); clip anything larger.
MAX_ANSWER_CHARS = 1600
# A scout may take at most this many read-only tool steps before it must
# answer with what it has.
MAX_SCOUT_TOOL_STEPS = 6
# Rate-limit retry: parallel scouts on a free-tier API must wait, not die.
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_WAIT = 2.0
# The ONLY tools a scout may ever call. Reads fan out; writes serialise —
# and a scout simply has no write path at all.
SCOUT_TOOL_NAMES = ("read_file", "list_dir", "file_info",
                    "search_files", "glob_files")
# The scout's system prompt lives in systemprompt.py — single source.

_CONFIDENCE_RE = re.compile(r"confidence:\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


@dataclass
class ScoutReport:
    question: str
    answer: str
    confidence: float      # 0.0..1.0 (parsed from the model if present, else 0.5)
    ok: bool               # False if the scout errored
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        """JSON-serializable report payload for the 'swarm.report' event."""
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "ok": self.ok,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


def _parse_confidence(text: str) -> float:
    """Parse 'Confidence: NN%' from the answer; clamp to 0..1, default 0.5."""
    m = _CONFIDENCE_RE.search(text)
    if not m:
        return 0.5
    return max(0.0, min(1.0, float(m.group(1)) / 100.0))


def _strip_confidence_line(text: str) -> str:
    """Drop the trailing 'Confidence: NN%' line (it is stored separately)."""
    return re.sub(r"\n?\s*confidence:\s*\d+(?:\.\d+)?\s*%\.?\s*$", "",
                  text, flags=re.IGNORECASE).strip()


class Swarm:
    """Fan-out of parallel read-only scouts over a shared EventLog."""

    def __init__(self, log: EventLog, provider: Provider, model: Model,
                 effort: Effort, max_parallel: int = 8,
                 mastermind=None) -> None:
        self.log = log
        self.provider = provider
        self.model = model
        self.effort = effort
        self.max_parallel = max(1, int(max_parallel))
        # Mastermind gate: when attached, every scout prompt is served
        # hash-sealed through the single door to the model. Standalone
        # (self-tests) falls back to the module source.
        self.mastermind = mastermind
        # read-only whitelist carved out of the main registry
        registry = build_registry()
        self.tools: dict[str, Tool] = {
            name: registry[name] for name in SCOUT_TOOL_NAMES
            if name in registry
        }

    def scout(self, questions: list[str], context: str = "",
              timeout: float = 120.0) -> list[ScoutReport]:
        """Run one read-only scout per question, in parallel; return the
        reports in input order and append one 'swarm.report' event each."""
        if not questions:
            return []

        def _safe(question: str) -> ScoutReport:
            try:
                return self._run_one(question, context, timeout)
            except Exception as e:  # a failing scout must not kill the swarm
                return ScoutReport(question=question, answer="",
                                   confidence=0.0, ok=False, error=str(e))

        workers = min(len(questions), self.max_parallel)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            reports = list(ex.map(_safe, questions))

        # Writes serialise: one event per scout, appended in input order.
        for r in reports:
            self.log.append("swarm.report", r.to_dict())
        return reports

    def _chat(self, messages: list[dict], schemas: list[dict] | None,
              timeout: float):
        """chat_blocking with rate-limit retry + exponential backoff."""
        last_err: Exception | None = None
        for attempt in range(RATE_LIMIT_RETRIES):
            try:
                return chat_blocking(self.provider, self.model, self.effort,
                                     messages, schemas, timeout=timeout)
            except APIError as e:
                msg = str(e).lower()
                rate_limited = (e.status == 429 or "rate limit" in msg
                                or "too many requests" in msg)
                if not rate_limited:
                    raise
                last_err = e
                wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt) \
                    + random.uniform(0, 1.5)
                time.sleep(wait)
        raise last_err  # type: ignore[misc]

    def _run_one(self, question: str, context: str,
                 timeout: float) -> ScoutReport:
        """One scout: fresh [system, user] context, read-only tools only,
        bounded tool loop, then a final answer."""
        t0 = time.monotonic()
        user = (f"Context:\n{context}\n\nQuestion: {question}" if context
                else f"Question: {question}")
        messages: list[dict] = []
        if self.mastermind is not None:
            messages, _ = self.mastermind.gate.dispatch("scout", messages)
        else:
            systemprompt.with_system(messages, systemprompt.scout())
        messages.append({"role": "user", "content": user})
        schemas = ([t.openai_schema() for t in self.tools.values()]
                   if self.model.supports_tools else None)
        try:
            for _ in range(MAX_SCOUT_TOOL_STEPS):
                result = self._chat(messages, schemas, timeout)
                if not result.tool_calls:
                    break
                messages.append({"role": "assistant",
                                 "content": result.content or None,
                                 "tool_calls": result.tool_calls})
                for tc in result.tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args = parse_tool_arguments(fn.get("arguments"))
                    tool = self.tools.get(name)
                    if tool is None:
                        out = (f"ERROR: tool '{name}' is not available to "
                               f"scouts. Read-only tools: "
                               + ", ".join(self.tools))
                    else:
                        try:
                            out = tool.handler(**args)
                        except Exception as e:  # report, never raise
                            out = f"ERROR: {type(e).__name__}: {e}"
                    messages.append({"role": "tool",
                                     "tool_call_id": tc.get("id", "call_0"),
                                     "content": out})
            else:
                # step budget spent — force a final answer
                messages.append({"role": "user",
                                 "content": "Tool budget exhausted. Answer "
                                            "now with what you have."})
                result = self._chat(messages, None, timeout)
        except Exception as e:
            return ScoutReport(question=question, answer="", confidence=0.0,
                               ok=False, error=str(e),
                               elapsed_ms=int((time.monotonic() - t0) * 1000))

        content = (result.content or "").strip()
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        answer = _strip_confidence_line(content)
        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS] + " …[truncated]"
        return ScoutReport(question=question, answer=answer,
                           confidence=_parse_confidence(content),
                           ok=True, elapsed_ms=elapsed_ms)

    def reports(self, n: int = 10) -> list[dict]:
        """Last n swarm report dicts from the fold, newest first."""
        if n <= 0:
            return []
        reps = fold(self.log).swarm_reports
        return list(reversed(reps[-n:]))

    def format(self, reports: list[ScoutReport]) -> str:
        """Compact TUI text: one scannable block per scout."""
        blocks: list[str] = []
        for i, r in enumerate(reports, 1):
            filled = max(0, min(5, round(r.confidence * 5)))
            meter = "●" * filled + "○" * (5 - filled)
            head = f"◆ scout {i} · {meter} {int(r.confidence * 100)}% · {r.elapsed_ms}ms"
            q = f"  Q: {r.question}"
            if r.ok:
                a = "  A: " + r.answer.replace("\n", "\n     ")
            else:
                a = f"  ✗ error: {r.error}"
            blocks.append("\n".join((head, q, a)))
        return "\n\n".join(blocks)


if __name__ == "__main__":
    import json
    import tempfile
    from pathlib import Path

    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EventLog(Path(td) / "swarm-test.jsonl")
            provider = Provider(key="fake", name="Fake",
                                base_url="http://invalid.local",
                                api_key="sk-fake", color="#ffffff")
            model = Model(id="fake-model", provider="fake", label="Fake")
            effort = Effort(key="low", label="LOW", color="#6272a4",
                            max_tokens=256, temperature=0.2,
                            reasoning_effort=None, description="test")
            sw = Swarm(log, provider, model, effort, max_parallel=8)

            # Offline stand-in: sleeps so serial execution would take >= 0.9s.
            def fake_run_one(question: str, context: str,
                             timeout: float) -> ScoutReport:
                time.sleep(0.3)
                return ScoutReport(question=question,
                                   answer=f"answer to {question}",
                                   confidence=0.75, ok=True, elapsed_ms=300)

            sw._run_one = fake_run_one  # type: ignore[method-assign]

            questions = ["q1", "q2", "q3"]
            t0 = time.monotonic()
            reports = sw.scout(questions, context="ctx")
            elapsed = time.monotonic() - t0

            assert len(reports) == 3, "expected 3 reports"
            assert [r.question for r in reports] == questions, \
                "input order must be preserved"
            assert all(r.ok for r in reports)
            assert elapsed < 0.75, \
                f"scouts did not run in parallel ({elapsed:.2f}s)"

            st = fold(log)
            assert len(st.swarm_reports) == 3, "expected 3 swarm.report events"
            assert [d["question"] for d in st.swarm_reports] == questions
            for d in st.swarm_reports:
                json.dumps(d)  # every report dict must be JSON-serializable
                assert set(d) == {"question", "answer", "confidence",
                                  "ok", "error", "elapsed_ms"}

            assert sw.scout([]) == [], "empty questions -> empty list"
            assert len(fold(log).swarm_reports) == 3, "no events for no scouts"

            # A raising scout yields ok=False, not an exception.
            def boom(question: str, context: str,
                     timeout: float) -> ScoutReport:
                raise RuntimeError("kaboom")

            sw._run_one = boom  # type: ignore[method-assign]
            errs = sw.scout(["bad"])
            assert len(errs) == 1 and errs[0].ok is False
            assert "kaboom" in errs[0].error

            newest = sw.reports(1)
            assert newest and newest[0]["question"] == "bad"
            assert len(sw.reports()) == 4

            text = sw.format(reports)
            assert "scout 1" in text and "q1" in text

            print("SWARM SELF-TEST PASS")

    _self_test()
