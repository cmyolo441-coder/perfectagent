"""Team — the shared worker-subagent substrate.

The parallel fan-out machinery is gone: every subagent now runs through
the persistent CREW (crew.py), which is the ONLY way to execute a worker.
This file keeps the pieces the whole system shares:

    ROLES               role -> tool whitelist + write permission
    WorkerReport        the compact structured report a worker collapses into
    parse_worker_final  split a worker's final STATUS/SUMMARY reply
    chat_with_retry     rate-limit-hardened model call (crew + evaluators)
    _WRITE_LOCK         ONE global write lock — the concurrent Crew's
                        agents can never mutate the world at the same
                        time (invariant I7)

Nothing a worker does pollutes the main conversation: each worker
collapses into a compact structured WorkerReport, and its lifecycle is
sealed as crew.* events in the log.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

from .client import APIError, chat_blocking, shrink_tool_outputs

MAX_WORKER_STEPS = 96      # tool-loop budget per worker (big-project grade)
MAX_WORKERS = 8            # roster ceiling; baked into worker prompts
MAX_SUMMARY_CHARS = 1800
RATE_LIMIT_RETRIES = 8     # retries when the provider rate-limits a worker
RATE_LIMIT_BASE_WAIT = 2.0  # seconds; doubles each retry (+ jitter)

# One global write lock: writes serialise across ALL subagents (§16.1).
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
    # -- advanced specialists (big-project grade) --------------------------
    "architect": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "create_directory"),
        "writes": True,
    },
    "debugger": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "run_command"),
        "writes": False,
    },
    "optimizer": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "run_command"),
        "writes": False,
    },
    "refactorer": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file",
                  "create_directory"),
        "writes": True,
    },
    "documenter": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file",
                  "create_directory"),
        "writes": True,
    },
    "devops": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file",
                  "create_directory", "run_command"),
        "writes": True,
    },
    "integrator": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file",
                  "create_directory", "run_command"),
        "writes": True,
    },
    "planner": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files"),
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


def parse_worker_final(text: str) -> tuple[str, str]:
    """Split a worker's final reply into (status, summary). Shared by
    the persistent Crew and every subsystem that reads worker reports.

    Status taxonomy: a worker says ``STATUS: DONE|BLOCKED|ERROR``. Earlier
    this code only recognised BLOCKED and silently swallowed every other
    failure label (ERROR, FAILED, TIMEOUT, ...) as ``done`` — a worker that
    timed out would be marked successful, the contract would close, and
    the dead-end would be lost. We now default to ``error`` for any
    non-DONE / non-BLOCKED label so genuine failures stay visible."""
    status = "done"
    lines = text.strip().splitlines()
    summary_lines: list[str] = []
    for line in lines:
        low = line.strip().upper()
        if low.startswith("STATUS:"):
            val = line.split(":", 1)[1].strip().upper()
            if val.startswith("DONE") or not val:
                status = "done"
            elif val.startswith("BLOCK"):
                status = "blocked"
            else:
                # explicit ERROR / FAILED / TIMEOUT / anything unknown:
                # surface as error so callers (Crew, Healer) can act on it
                status = "error"
        elif low.startswith("SUMMARY:"):
            summary_lines.append(line.split(":", 1)[1].strip())
        elif summary_lines:
            summary_lines.append(line.strip())
    summary = "\n".join(s for s in summary_lines if s).strip()
    if not summary:  # model ignored the format — keep the whole reply
        summary = text.strip()
    return status, summary


def chat_with_retry(provider, model, effort, messages: list[dict],
                    schemas: list[dict] | None, timeout: float):
    """chat_blocking with rate-limit retry + exponential backoff.

    Shared by the persistent Crew and every blocking subsystem evaluator.
    Free-tier APIs rate-limit; a worker must wait and retry, not die.
    Backoff doubles each attempt with jitter.

    Also carries context-overflow protection: if a worker's own tool
    loop bloats its context past the window, the oldest tool results
    are truncated and the call retried — a worker never dies with a
    context-length error."""
    last_err: Exception | None = None
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return chat_blocking(provider, model, effort,
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
            if attempt >= RATE_LIMIT_RETRIES - 1:
                break  # last attempt — no point sleeping after the verdict
            wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt) \
                + random.uniform(0, 1.5)
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


if __name__ == "__main__":
    def _self_test() -> None:
        from .tools import build_registry

        reg = build_registry()
        for role, spec in ROLES.items():
            ts = set(spec["tools"])
            assert ts <= set(reg), role
            if not spec["writes"]:
                assert not (ts & {"write_file", "edit_file",
                                  "delete_path", "move_path",
                                  "copy_path"}), role
        assert "delete_path" not in ROLES["coder"]["tools"]

        # final-report parsing
        s, summ = parse_worker_final("STATUS: DONE\nSUMMARY: line one\nline two")
        assert s == "done" and "line one" in summ and "line two" in summ
        s, summ = parse_worker_final("STATUS: BLOCKED\nSUMMARY: need access")
        assert s == "blocked" and summ == "need access"
        s, summ = parse_worker_final("just some plain text")
        assert s == "done" and summ == "just some plain text"

        print("TEAM SELF-TEST PASS")

    _self_test()
