"""Agent loop: LLM <-> tools, with events for the TUI."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import config
from .client import APIError, TurnCancelled, chat_blocking, chat_stream
from .config import Config, Effort, Model, Provider, PROVIDERS, model_by_id
from .tools import RISK_CONFIRM, Tool, build_registry, parse_tool_arguments

SYSTEM_PROMPT = """You are FullAgent, an elite terminal AI agent running inside the user's shell.

You accomplish tasks end to end using your tools:
- read_file / write_file / edit_file / list_dir / file_info / create_directory / copy_path / move_path / delete_path for file work
- search_files (regex over contents) and glob_files to find things
- run_command to execute shell commands (builds, tests, git, installs, running programs)
- web_fetch and web_search for information from the internet

Working style:
- Be decisive: inspect before editing, make the change, then verify it (run the code/tests) instead of guessing.
- Prefer several small correct steps over one big guess. Keep going until the task is genuinely done.
- edit_file requires an exact, unique old_string — read the file first if unsure.
- For risky or destructive operations, be careful and explain what you are doing.
- Keep replies concise and factual; show results, not narration. Use markdown sparingly.
- When the task is complete, summarize what was done and the outcome in a few lines."""


@dataclass
class ToolEvent:
    name: str
    args: dict
    result: str = ""
    status: str = "running"   # running | done | error | denied
    duration: float = 0.0


@dataclass
class Turn:
    user_text: str
    assistant_text: str = ""
    reasoning: str = ""
    tools: list[ToolEvent] = field(default_factory=list)
    model_id: str = ""
    effort: str = ""
    error: str = ""
    usage: dict | None = None
    duration: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class Agent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tools = build_registry()
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}]
        self.session_id = uuid.uuid4().hex[:8]
        self.turns: list[Turn] = []

    # -- model / effort ----------------------------------------------------

    @property
    def model(self) -> Model:
        m = model_by_id(self.cfg.model_id)
        assert m is not None
        return m

    @property
    def provider(self) -> Provider:
        return PROVIDERS[self.model.provider]

    @property
    def effort(self) -> Effort:
        e = config.effort_by_key(self.cfg.effort)
        assert e is not None
        return e

    # -- conversation ------------------------------------------------------

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.turns = []
        self.session_id = uuid.uuid4().hex[:8]

    def _tool_schemas(self) -> list[dict] | None:
        if not self.model.supports_tools:
            return None
        return [t.openai_schema() for t in self.tools.values()]

    def run_turn(self, user_text: str,
                 on_token: Callable[[str], None],
                 on_reasoning: Callable[[str], None],
                 on_tool_call: Callable[[ToolEvent], None],
                 on_tool_update: Callable[[ToolEvent], None],
                 on_status: Callable[[str], None],
                 approve: Callable[[Tool, dict], bool],
                 should_cancel: Callable[[], bool] | None = None,
                 ) -> Turn:
        """Run one user turn through the full agent loop."""
        turn = Turn(user_text=user_text, model_id=self.model.id,
                    effort=self.cfg.effort)
        started = time.time()
        self.messages.append({"role": "user", "content": user_text})

        iterations = 0
        try:
            while iterations < config.MAX_TOOL_ITERATIONS:
                iterations += 1
                on_status("thinking")
                result = self._complete(on_token, on_reasoning, on_status,
                                        should_cancel)

                if result.reasoning:
                    turn.reasoning += result.reasoning

                if result.tool_calls:
                    # record assistant message with tool calls
                    self.messages.append({
                        "role": "assistant",
                        "content": result.content or None,
                        "tool_calls": result.tool_calls,
                    })
                    if result.content:
                        turn.assistant_text += result.content
                    for tc in result.tool_calls:
                        fn = tc["function"]
                        name = fn.get("name", "")
                        args = parse_tool_arguments(fn.get("arguments"))
                        ev = ToolEvent(name=name, args=args)
                        turn.tools.append(ev)
                        on_tool_call(ev)
                        self._execute_tool(ev, approve, on_status)
                        on_tool_update(ev)
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": ev.result,
                        })
                    continue

                # plain assistant reply — done
                turn.assistant_text += result.content
                turn.usage = result.usage
                self.messages.append(
                    {"role": "assistant", "content": result.content})
                break
            else:
                turn.error = f"stopped after {config.MAX_TOOL_ITERATIONS} tool iterations"
        except APIError as e:
            turn.error = str(e)
            self.messages.pop()  # drop the failed user message
        except TurnCancelled:
            turn.error = "cancelled"
            # keep whatever was generated so far in the conversation
            self.messages.append(
                {"role": "assistant",
                 "content": turn.assistant_text or "(cancelled by user)"})
        except KeyboardInterrupt:
            turn.error = "interrupted"
            self.messages.append(
                {"role": "assistant",
                 "content": turn.assistant_text or "(interrupted)"})

        turn.duration = time.time() - started
        self.turns.append(turn)
        return turn

    def _complete(self, on_token, on_reasoning, on_status, should_cancel=None):
        schemas = self._tool_schemas()
        try:
            return chat_stream(self.provider, self.model, self.effort,
                               self.messages, schemas,
                               on_token=on_token,
                               on_reasoning=on_reasoning,
                               on_tool_start=lambda n: on_status(f"tool:{n}"),
                               should_cancel=should_cancel)
        except APIError as e:
            # some providers refuse streaming or tool params — degrade gracefully
            msg = str(e).lower()
            if e.status == 400 and "tool" in msg and schemas:
                on_status("retrying (no tools)")
                return chat_stream(self.provider, self.model, self.effort,
                                   self.messages, None,
                                   on_token=on_token,
                                   on_reasoning=on_reasoning,
                                   should_cancel=should_cancel)
            if e.status == 400 and "stream" in msg:
                on_status("retrying (non-stream)")
                return chat_blocking(self.provider, self.model, self.effort,
                                     self.messages, schemas)
            raise

    def _execute_tool(self, ev: ToolEvent,
                      approve: Callable[[Tool, dict], bool],
                      on_status: Callable[[str], None]) -> None:
        tool = self.tools.get(ev.name)
        started = time.time()
        if tool is None:
            ev.status = "error"
            ev.result = f"ERROR: unknown tool '{ev.name}'. Available: " \
                        + ", ".join(self.tools)
            return
        if tool.risk == RISK_CONFIRM and not self.cfg.auto_approve:
            if not approve(tool, ev.args):
                ev.status = "denied"
                ev.result = "ERROR: user denied this action. Ask the user " \
                            "how to proceed or choose another approach."
                return
        on_status(f"running:{ev.name}")
        try:
            ev.result = tool.handler(**ev.args)
            ev.status = "done"
        except TypeError as e:
            ev.result = f"ERROR: bad arguments for {ev.name}: {e}"
            ev.status = "error"
        except Exception as e:  # noqa: BLE001 — tool errors go back to the LLM
            ev.result = f"ERROR: {type(e).__name__}: {e}"
            ev.status = "error"
        ev.duration = time.time() - started

    # -- persistence -------------------------------------------------------

    def save_session(self) -> Path | None:
        try:
            config.ensure_dirs()
            path = config.SESSIONS_DIR / f"{self.session_id}.json"
            path.write_text(json.dumps({
                "session_id": self.session_id,
                "model_id": self.cfg.model_id,
                "saved_at": datetime.now().isoformat(),
                "messages": self.messages,
            }, indent=1))
            return path
        except OSError:
            return None
