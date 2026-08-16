"""Team — parallel worker sub-agents (§16 Swarm, worker grade).

Reads fan out. Writes serialise. Up to MAX_WORKERS (8) professional
specialist workers run simultaneously, each with its own fresh context,
its own role brief, and a bounded tool loop. Workers are real agents —
they read, search, run commands, fetch the web, and (builders) write
files — but every write passes through ONE global lock, so two workers
can never mutate the world at the same time (invariant I7).

Nothing a worker does pollutes the main conversation: each worker
collapses into a compact structured WorkerReport, and one 'team.report'
event per worker lands in the event log.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import systemprompt
from .client import APIError, chat_blocking, shrink_tool_outputs
from .config import Effort, Model, Provider
from .kernel import EventLog, fold
from .tools import Tool, build_registry, parse_tool_arguments

MAX_WORKERS = 8            # hard ceiling on parallel workers
MAX_WORKER_STEPS = 14      # tool-loop budget per worker
MAX_SUMMARY_CHARS = 1800
RATE_LIMIT_RETRIES = 6     # retries when the provider rate-limits a worker
RATE_LIMIT_BASE_WAIT = 2.0  # seconds; doubles each retry (+ jitter)
STAGGER_SECONDS = 0.6      # launch wave: worker i starts ~i*0.6s in, so 8
                           # workers never hit the API in one instant burst

# One global write lock: writes serialise across ALL workers (§16.1).
_WRITE_LOCK = threading.Lock()

# Role -> (tool whitelist, brief). Reads fan out freely; only builder roles
# get write tools, and those go through _WRITE_LOCK.
# Role -> (tool whitelist, write permission). The role BRIEFS — the words
# the model actually reads — live in systemprompt.ROLE_BRIEFS, so every
# prompt is defined in one file.
ROLES: dict[str, dict] = {
    "researcher": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "web_search", "web_fetch"),
        "writes": False,
    },
    "coder": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file",
                  "create_directory"),
        "writes": True,
    },
    "tester": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "run_command"),
        "writes": False,
    },
    "reviewer": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files"),
        "writes": False,
    },
    "analyst": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "web_search", "web_fetch", "run_command"),
        "writes": False,
    },
}
DEFAULT_ROLE = "coder"
# The worker system prompt template lives in systemprompt.py — single source.


@dataclass
class WorkerReport:
    task: str
    role: str
    summary: str = ""
    status: str = "running"      # done | blocked | error
    files_touched: list[str] = field(default_factory=list)
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "task": self.task, "role": self.role, "summary": self.summary,
            "status": self.status, "files_touched": self.files_touched,
            "tool_calls": self.tool_calls,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "error": self.error, "elapsed_ms": self.elapsed_ms,
        }


def _parse_final(text: str) -> tuple[str, str]:
    """Split the worker's final reply into (status, summary)."""
    status = "done"
    lines = text.strip().splitlines()
    summary_lines: list[str] = []
    for line in lines:
        low = line.strip().upper()
        if low.startswith("STATUS:"):
            val = line.split(":", 1)[1].strip().upper()
            status = "blocked" if val.startswith("BLOCK") else "done"
        elif low.startswith("SUMMARY:"):
            summary_lines.append(line.split(":", 1)[1].strip())
        elif summary_lines:
            summary_lines.append(line.strip())
    summary = "\n".join(s for s in summary_lines if s).strip()
    if not summary:  # model ignored the format — keep the whole reply
        summary = text.strip()
    return status, summary


class Team:
    """Parallel fan-out of worker sub-agents over a shared EventLog."""

    def __init__(self, log: EventLog, provider: Provider, model: Model,
                 effort: Effort, mastermind=None) -> None:
        self.log = log
        self.provider = provider
        self.model = model
        self.effort = effort
        # Mastermind gate: when attached, every worker prompt is served
        # hash-sealed through the single door to the model. Standalone
        # (self-tests) falls back to the module source.
        self.mastermind = mastermind
        registry = build_registry()
        self._toolsets: dict[str, dict[str, Tool]] = {}
        for role, spec in ROLES.items():
            self._toolsets[role] = {
                name: registry[name] for name in spec["tools"]
                if name in registry
            }

    # -- public API ---------------------------------------------------------

    def run(self, tasks: list[dict], context: str = "",
            timeout: float = 240.0,
            read_only: bool = False) -> list[WorkerReport]:
        """Run one worker per task IN PARALLEL (max MAX_WORKERS).

        Each task is {'task': str, 'role': str?}. With read_only=True no
        worker gets any write tool (the autonomy ladder applied to the
        team). Returns reports in input order and appends one
        'team.report' event per worker."""
        tasks = tasks[:MAX_WORKERS]
        if not tasks:
            return []

        def _safe(index: int, task: dict) -> WorkerReport:
            try:
                return self._run_one(task, context, timeout, read_only,
                                     stagger_index=index)
            except Exception as e:  # one failing worker never kills the team
                return WorkerReport(task=task.get("task", ""),
                                    role=task.get("role", DEFAULT_ROLE),
                                    status="error", error=str(e))

        workers = min(len(tasks), MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            reports = list(ex.map(lambda it: _safe(it[0], it[1]),
                                  enumerate(tasks)))

        # Writes serialise: reports are sealed one at a time, input order.
        for r in reports:
            self.log.append("team.report", r.to_dict(), actor="system")
        return reports

    def reports(self, n: int = 10) -> list[dict]:
        reps = fold(self.log).team_reports
        return list(reversed(reps[-n:])) if n > 0 else []

    def format(self, reports: list[WorkerReport]) -> str:
        blocks: list[str] = []
        for i, r in enumerate(reports, 1):
            icon = {"done": "✓", "blocked": "◐", "error": "✗"}.get(
                r.status, "…")
            head = (f"◆ worker {i} [{r.role}] {icon} {r.status} · "
                    f"{r.tool_calls} tool calls · {r.elapsed_ms}ms")
            lines = [head, f"  task: {r.task}"]
            if r.files_touched:
                lines.append("  files: " + ", ".join(r.files_touched[:8]))
            if r.error:
                lines.append(f"  error: {r.error}")
            if r.summary:
                lines.append("  " + r.summary.replace("\n", "\n  "))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # -- one worker ----------------------------------------------------------

    def _chat(self, messages: list[dict], schemas: list[dict] | None,
              timeout: float):
        """chat_blocking with rate-limit retry + exponential backoff.

        8 parallel workers on a free-tier API will hit rate limits; a
        worker must wait and retry, not die. Backoff doubles each attempt
        with jitter so the workers naturally de-synchronise.

        Also carries context-overflow protection: if a worker's own tool
        loop bloats its context past the window, the oldest tool results
        are truncated and the call retried — a worker never dies with a
        context-length error."""
        last_err: Exception | None = None
        for attempt in range(RATE_LIMIT_RETRIES):
            try:
                return chat_blocking(self.provider, self.model, self.effort,
                                     messages, schemas,
                                     on_overflow=lambda: shrink_tool_outputs(
                                         messages),
                                     timeout=timeout)
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

    def _run_one(self, task: dict, context: str,
                 timeout: float, read_only: bool = False,
                 stagger_index: int = 0) -> WorkerReport:
        role = task.get("role") or DEFAULT_ROLE
        if role not in ROLES:
            role = DEFAULT_ROLE
        spec = ROLES[role]
        tools = dict(self._toolsets[role])
        if read_only:
            tools = {n: t for n, t in tools.items()
                     if n not in ("write_file", "edit_file",
                                  "create_directory", "run_command")}
        schemas = ([t.openai_schema() for t in tools.values()]
                   if self.model.supports_tools else None)

        # Launch wave: worker i waits ~i*STAGGER before its first call so
        # 8 workers spread across ~4s instead of one instant burst. This
        # keeps free-tier APIs from rate-limiting the whole team at t=0.
        if stagger_index > 0:
            time.sleep(stagger_index * STAGGER_SECONDS
                       + random.uniform(0, 0.3))

        user = (f"Shared context:\n{context}\n\nYOUR TASK: {task['task']}"
                if context else f"YOUR TASK: {task['task']}")
        messages: list[dict] = []
        if self.mastermind is not None:
            messages, _ = self.mastermind.gate.dispatch(
                f"worker:{role}", messages)
        else:
            systemprompt.with_system(
                messages, systemprompt.worker(role, MAX_WORKERS))
        messages.append({"role": "user", "content": user})

        t0 = time.monotonic()
        report = WorkerReport(task=task["task"], role=role)
        result = None
        for _ in range(MAX_WORKER_STEPS):
            result = self._chat(messages, schemas, timeout)
            if result.usage:
                report.tokens_in += int(result.usage.get(
                    "prompt_tokens", 0) or 0)
                report.tokens_out += int(result.usage.get(
                    "completion_tokens", 0) or 0)
            if not result.tool_calls:
                break
            messages.append({"role": "assistant",
                             "content": result.content or None,
                             "tool_calls": result.tool_calls})
            for tc in result.tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = parse_tool_arguments(fn.get("arguments"))
                report.tool_calls += 1
                tool = tools.get(name)
                if tool is None:
                    out = (f"ERROR: tool '{name}' is not available to a "
                           f"{role} worker. Available: "
                           + ", ".join(tools))
                else:
                    # THE GOLDEN RULE: writes serialise across workers.
                    lock = _WRITE_LOCK if spec["writes"] and name in (
                        "write_file", "edit_file", "create_directory",
                        "run_command") else None
                    try:
                        if lock:
                            with lock:
                                out = tool.handler(**args)
                        else:
                            out = tool.handler(**args)
                        if name in ("write_file", "edit_file") and \
                                out.startswith("OK"):
                            p = str(args.get("path", ""))
                            if p and p not in report.files_touched:
                                report.files_touched.append(p)
                    except Exception as e:  # report, never raise
                        out = f"ERROR: {type(e).__name__}: {e}"
                messages.append({"role": "tool",
                                 "tool_call_id": tc.get("id", "call_0"),
                                 "content": out})
        else:
            messages.append({"role": "user",
                             "content": "Step budget exhausted. Produce your "
                                        "final STATUS/SUMMARY report now."})
            result = self._chat(messages, None, timeout)
            if result.usage:
                report.tokens_in += int(result.usage.get(
                    "prompt_tokens", 0) or 0)
                report.tokens_out += int(result.usage.get(
                    "completion_tokens", 0) or 0)

        content = (result.content or "") if result else ""
        report.elapsed_ms = int((time.monotonic() - t0) * 1000)
        status, summary = _parse_final(content)
        report.status = status
        report.summary = (summary[:MAX_SUMMARY_CHARS] + " …[truncated]"
                          if len(summary) > MAX_SUMMARY_CHARS else summary)
        return report


if __name__ == "__main__":
    import tempfile

    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EventLog(Path(td) / "team-test.jsonl")
            provider = Provider(key="fake", name="Fake",
                                base_url="http://invalid.local",
                                api_key="sk-fake", color="#ffffff")
            model = Model(id="fake-model", provider="fake", label="Fake")
            effort = Effort(key="low", label="LOW", color="#6272a4",
                            max_tokens=256, temperature=0.2,
                            reasoning_effort=None, description="test")
            team = Team(log, provider, model, effort)

            # every role's toolset is a strict subset of the registry,
            # and only writer roles hold write tools
            reg = build_registry()
            for role, spec in ROLES.items():
                ts = team._toolsets[role]
                assert set(ts) <= set(reg), role
                assert set(ts) == set(spec["tools"]) & set(reg), role
                if not spec["writes"]:
                    assert not (set(ts) & {"write_file", "edit_file",
                                           "delete_path", "move_path",
                                           "copy_path"}), role
            assert "delete_path" not in team._toolsets["coder"]

            # read-only mode strips every write/execute tool
            ro = {n for n in team._toolsets["coder"]
                  if n not in ("write_file", "edit_file",
                               "create_directory", "run_command")}
            assert not (ro & {"write_file", "edit_file", "run_command"})

            # offline stand-in: sleeps so serial execution would take >= 0.9s
            def fake_run_one(task: dict, context: str,
                             timeout: float, read_only: bool = False,
                             stagger_index: int = 0) -> WorkerReport:
                time.sleep(0.3)
                return WorkerReport(task=task["task"],
                                    role=task.get("role", DEFAULT_ROLE),
                                    status="done",
                                    summary=f"did: {task['task']}",
                                    tool_calls=2, elapsed_ms=300)

            team._run_one = fake_run_one  # type: ignore[method-assign]
            tasks = [{"task": f"task {i}", "role": "coder"} for i in range(8)]
            t0 = time.monotonic()
            reports = team.run(tasks, context="ctx")
            elapsed = time.monotonic() - t0

            assert len(reports) == 8, "expected 8 reports"
            assert [r.task for r in reports] == [t["task"] for t in tasks]
            assert all(r.status == "done" for r in reports)
            assert elapsed < 1.2, f"workers not parallel ({elapsed:.2f}s)"

            st = fold(log)
            assert len(st.team_reports) == 8

            # a 9th task is clipped to the ceiling
            assert len(team.run(tasks + [{"task": "extra"}])) == MAX_WORKERS
            assert team.run([]) == []

            # a raising worker yields status=error, not an exception
            def boom(task: dict, context: str,
                     timeout: float, read_only: bool = False,
                     stagger_index: int = 0) -> WorkerReport:
                raise RuntimeError("kaboom")

            team._run_one = boom  # type: ignore[method-assign]
            errs = team.run([{"task": "bad"}])
            assert len(errs) == 1 and errs[0].status == "error"
            assert "kaboom" in errs[0].error

            # final-report parsing
            s, summ = _parse_final("STATUS: DONE\nSUMMARY: line one\nline two")
            assert s == "done" and "line one" in summ and "line two" in summ
            s, summ = _parse_final("STATUS: BLOCKED\nSUMMARY: need access")
            assert s == "blocked" and summ == "need access"
            s, summ = _parse_final("just some plain text")
            assert s == "done" and summ == "just some plain text"

            text = team.format(reports)
            assert "worker 1" in text and "task 0" in text

            print("TEAM SELF-TEST PASS")

    _self_test()
