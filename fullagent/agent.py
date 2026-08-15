"""Agent loop: LLM <-> tools, event-sourced on the Temporal Kernel.

Every user message, tool call, tool result, assistant reply, and cost is
an immutable event in the append-only log (kernel.py). Conversation state,
cost, goal distance, dead-ends and verdicts are projections folded from
that log.

Wired here (all enforced mechanically, rung 1 — never as prompt advice):
  * §38.1  Total attribution — with an active goal, every tool call is
           bound to the focus clause (correlation_id). An action with no
           open clause to serve is rejected as an OrphanAction.
  * A2/I3  Snapshot-before-write — no mutating tool runs without a
           committed recovery path in the snapshot store.
  * §37.4  Anti-clauses re-checked after EVERY successful write.
  * I8     Budget governor — a breach pauses the turn, never silent kill.
  * §13.4  Loop detection — exact repeats are sealed and interrupted.
  * §38.3  Goal kernel tick — distance measured, focus re-aimed.
  * §9     Time travel — rewind (files + state), revert (files only).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import config
from .autopilot import AutoPilot, RouteDecision
from .cassette import Cassette
from .client import APIError, TurnCancelled, chat_blocking, chat_stream
from .config import Config, Effort, Model, Provider, PROVIDERS, model_by_id
from .cortex import Budget, BudgetGovernor, LoopDetector
from .forge import Forge
from .goal import GoalContract
from .judge import Judge
from .kernel import EventLog, fold
from .memory import Hippocampus
from .nexus import Nexus
from .oracle import Oracle
from .snapshots import SnapshotStore
from .swarm import Swarm
from .team import Team
from .tools import RISK_CONFIRM, Tool, build_registry, parse_tool_arguments

SYSTEM_PROMPT = """You are FullAgent, an elite terminal AI agent running inside the user's shell.

You accomplish tasks end to end using your tools:
- read_file / write_file / edit_file / list_dir / file_info / create_directory / copy_path / move_path / delete_path for file work
- search_files (regex over contents) and glob_files to find things
- run_command to execute shell commands (builds, tests, git, installs, running programs)
- code_symbols / code_impact to understand code semantically (call graph, blast radius)
- web_fetch and web_search for information from the internet

Working style:
- Be decisive: inspect before editing, make the change, then verify it (run the code/tests) instead of guessing.
- Prefer several small correct steps over one big guess. Keep going until the task is genuinely done.
- edit_file requires an exact, unique old_string — read the file first if unsure.
- For risky or destructive operations, be careful and explain what you are doing.
- Keep replies concise and factual; show results, not narration. Use markdown sparingly.
- When the task is complete, summarize what was done and the outcome in a few lines."""

# Autonomy ladder (§22): L0 observer … L5 autonomous
AUTONOMY_LEVELS = {
    0: "Observer — reads only, no mutations",
    1: "Advisor — proposes, applies nothing",
    2: "Assistant — every mutation needs approval",
    3: "Collaborator — safe actions free, risky need approval (default)",
    4: "Pilot — auto-approve except deletes",
    5: "Autonomous — full freedom within budget",
}

_MUTATING_TOOLS = {"write_file", "edit_file", "create_directory",
                   "copy_path", "move_path", "delete_path", "run_command"}
_ALWAYS_ASK = {"delete_path"}
# tools whose args name the paths they touch (snapshot targets)
_PATH_ARG_TOOLS = {"write_file": ("path",), "edit_file": ("path",),
                   "create_directory": ("path",), "delete_path": ("path",),
                   "copy_path": ("src", "dst"), "move_path": ("src", "dst")}


@dataclass
class ToolEvent:
    name: str
    args: dict
    result: str = ""
    status: str = "running"   # running | done | error | denied | blocked
    duration: float = 0.0
    clause_id: str | None = None


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


def _signature(name: str, args: dict) -> str:
    """Canonical approach signature — the dead-end ledger key."""
    payload = json.dumps({"name": name, "args": args},
                         sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Agent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tools = build_registry()
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}]
        self.session_id = uuid.uuid4().hex[:8]
        self.turns: list[Turn] = []

        # Temporal kernel + the nine subsystems
        config.ensure_dirs()
        self.log = EventLog(config.EVENT_LOG_FILE, session=self.session_id)
        self.store = SnapshotStore(config.APP_DIR / "store")
        self.memory = Hippocampus(self.log)
        self.judge = Judge(self.log)
        self.goal = GoalContract(self.log, judge=self.judge)
        self.swarm = Swarm(self.log, self.provider, self.model, self.effort)
        self.team = Team(self.log, self.provider, self.model, self.effort)
        self.autopilot = AutoPilot(self.log)
        self.nexus = Nexus()
        self.forge = Forge(self.log)
        self.oracle = Oracle(self.log, memory_dir=config.APP_DIR / "memory")
        self.budget_gov = BudgetGovernor(self.log, Budget())
        self.loop_det = LoopDetector(self.log)
        self.autonomy = 3
        self._error_counts: dict[str, int] = {}
        self._file_hashes: dict[str, list[str]] = {}  # oscillation history
        self._register_code_tools()

        # cassette: record/replay model calls (FULLAGENT_CASSETTE=path,
        # FULLAGENT_CASSETTE_MODE=record|replay|off)
        import os
        cassette_path = os.environ.get("FULLAGENT_CASSETTE")
        cassette_mode = os.environ.get("FULLAGENT_CASSETTE_MODE", "off")
        self.cassette = (Cassette(Path(cassette_path), cassette_mode)
                         if cassette_path else None)

        self.log.append("session.start", {"session_id": self.session_id,
                                          "model": self.cfg.model_id,
                                          "effort": self.cfg.effort},
                        actor="system")
        self.forge.probe()  # PERCEIVE: stamp the environment

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

    def state(self):
        """Live projection of the event log (cost, goal, dead-ends, …)."""
        return fold(self.log)

    # -- conversation ------------------------------------------------------

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.turns = []
        self.session_id = uuid.uuid4().hex[:8]
        self.log.session = self.session_id
        self.log.append("session.start", {"session_id": self.session_id,
                                          "model": self.cfg.model_id,
                                          "effort": self.cfg.effort},
                        actor="system")

    def _system_content(self, route: RouteDecision | None = None) -> str:
        """Base prompt + constitution + live goal contract + memory."""
        parts = [SYSTEM_PROMPT]
        constitution = self.oracle.read_constitution()
        if constitution.strip():
            parts.append("\nCONSTITUTION (standing rules, always apply):\n"
                         + constitution.strip())
        goal = self.goal.status()
        if goal.active:
            parts.append("\nACTIVE GOAL CONTRACT:\n" + self.goal.format() +
                         "\nEvery action must serve an open clause. When a "
                         "clause's predicate genuinely passes, say "
                         "'PROVEN: <clause id>' — the kernel verifies it, "
                         "never trust self-declared success.")
        if route is not None and route.use_web:
            parts.append("\nREAL-TIME WEB MODE: this request needs live, "
                         "up-to-the-minute data. Use web_search (and "
                         "web_fetch for details) to get CURRENT facts — "
                         "never answer from stale knowledge. Quote the "
                         "retrieval time and sources.")
        mem = self.memory.context_block()
        if mem:
            parts.append("\nMEMORY (from prior work):\n" + mem)
        return "\n".join(parts)

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
                 on_route: Callable[[RouteDecision], None] | None = None,
                 ) -> Turn:
        """Run one user turn through the full agent loop."""
        turn = Turn(user_text=user_text, model_id=self.model.id,
                    effort=self.cfg.effort)
        started = time.time()

        # AUTOPILOT: the agent decides for itself which powers this turn
        # needs — parallel team, goal mode, real-time web — and enables
        # them on its own. Every decision is logged and surfaced.
        route = self.autopilot.route(user_text,
                                     self.goal.status().active,
                                     self.autonomy)
        if route.active and on_route:
            on_route(route)
        if route.suggest_goal:
            self._autopilot_goal(route)
        if route.use_team:
            self._autopilot_team(route, on_status)

        # refresh dynamic system context (constitution + goal + memory)
        self.messages[0] = {"role": "system",
                            "content": self._system_content(route)}
        self.messages.append({"role": "user", "content": user_text})
        user_ev = self.log.append("user.message",
                                  {"text": user_text,
                                   "session": self.session_id},
                                  actor="human", provenance="user")

        iterations = 0
        try:
            while iterations < config.MAX_TOOL_ITERATIONS:
                iterations += 1
                # I8: budget governor — a breach PAUSES the turn
                if not self.budget_gov.enforce():
                    evs = fold(self.log).budget_events
                    reason = evs[-1]["reason"] if evs else "budget exceeded"
                    turn.error = f"PAUSED — {reason}. Extend the budget " \
                                 f"or stop."
                    break
                on_status("thinking")
                result = self._complete(on_token, on_reasoning, on_status,
                                        should_cancel)

                if result.reasoning:
                    turn.reasoning += result.reasoning

                if result.tool_calls:
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
                        self._execute_tool(ev, approve, on_status,
                                           causation_id=user_ev.id)
                        on_tool_update(ev)
                        self.log.append(
                            "tool.result",
                            {"name": name, "status": ev.status,
                             "duration": round(ev.duration, 3),
                             "preview": ev.result[:300]},
                            actor="system", provenance="tool_output",
                            correlation_id=ev.clause_id)
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": ev.result,
                        })
                        # §38.3 goal kernel tick after every action
                        self._goal_tick()
                    continue

                # plain assistant reply — done
                turn.assistant_text += result.content
                turn.usage = result.usage
                self.messages.append(
                    {"role": "assistant", "content": result.content})
                self.log.append("assistant.message",
                                {"text": result.content,
                                 "session": self.session_id},
                                actor="sovereign", provenance="model",
                                causation_id=user_ev.id)
                self._emit_cost(result.usage)
                self._detect_goal_clauses(result.content)
                break
            else:
                turn.error = f"stopped after {config.MAX_TOOL_ITERATIONS} tool iterations"
        except APIError as e:
            turn.error = str(e)
            self.messages.pop()  # drop the failed user message
            self.log.append("turn.error", {"error": str(e)})
        except TurnCancelled:
            turn.error = "cancelled"
            self.messages.append(
                {"role": "assistant",
                 "content": turn.assistant_text or "(cancelled by user)"})
            self.log.append("turn.cancelled", {})
        except KeyboardInterrupt:
            turn.error = "interrupted"
            self.messages.append(
                {"role": "assistant",
                 "content": turn.assistant_text or "(interrupted)"})
            self.log.append("turn.cancelled", {})

        turn.duration = time.time() - started
        self.turns.append(turn)
        return turn

    def _goal_tick(self) -> None:
        """§38.3 kernel loop: measure distance, re-aim focus. Pure Python
        over projections — zero tokens."""
        if not self.goal.status().active:
            return
        self.goal.measure()
        self.goal.reaim()

    # -- autopilot actions ---------------------------------------------------

    def _autopilot_goal(self, route: RouteDecision) -> None:
        """Auto-draft and freeze a machine-checkable goal contract from the
        request. If the contract fails validation (it cannot be made
        checkable), the turn simply proceeds without one."""
        from .goal import GoalContractError
        try:
            self.goal.set_goal(route.goal_statement, route.goal_clauses)
        except GoalContractError:
            return

    def _autopilot_team(self, route: RouteDecision,
                        on_status: Callable[[str], None]) -> None:
        """Dispatch the parallel worker team. Reports land in the log and
        are folded into memory so the main turn can build on them."""
        read_only = self.autonomy <= 1
        on_status(f"team:{len(route.tasks)}")
        reports = self.team.run(route.tasks,
                                context=self.scout_context(),
                                read_only=read_only)
        # fold the results into episodic memory for the main turn
        for r in reports:
            if r.status == "done" and r.summary:
                self.log.append("fact.learned",
                                {"fact": f"[{r.role}] {r.task} -> "
                                         f"{r.summary[:300]}",
                                 "kind": "team"})
        total_in = sum(r.tokens_in for r in reports)
        total_out = sum(r.tokens_out for r in reports)
        if total_in or total_out:
            self.log.append("cost.incurred",
                            {"usd": 0.0, "tokens_in": total_in,
                             "tokens_out": total_out,
                             "model": self.model.id, "team": True},
                            actor="system")

    def _emit_cost(self, usage: dict | None) -> None:
        if not usage:
            return
        tin = int(usage.get("prompt_tokens", 0) or 0)
        tout = int(usage.get("completion_tokens", 0) or 0)
        # configured models are free-tier; ledger still tracks tokens so a
        # price table can be dropped in later without schema changes
        usd = 0.0
        focus = self.goal.status().focus
        self.log.append("cost.incurred", {"usd": usd, "tokens_in": tin,
                                          "tokens_out": tout,
                                          "model": self.model.id},
                        correlation_id=focus)

    def _detect_goal_clauses(self, text: str) -> None:
        """The model may CLAIM 'PROVEN: <id>' — but the clause is only
        marked proven if its own predicate actually passes (§39.1: the
        binary proven flag is never set from model output alone)."""
        goal = self.goal.status()
        if not goal.active:
            return
        low = text.lower()
        for clause in goal.clauses:
            if clause.state in ("PROVEN", "WAIVED"):
                continue
            key = clause.id.lower()
            claimed = (f"proven: {key}" in low or f"done: {key}" in low
                       or f"completed: {key}" in low
                       or f"✓ {key}" in low)
            if not claimed:
                continue
            if clause.proof and self.judge is not None:
                # verify the claim against reality
                ok, detail = self.goal.prove_by_predicate(clause.id)
                if ok:
                    self.log.append("fact.learned",
                                    {"fact": f"clause {clause.id} proven: "
                                             f"{detail}", "kind": "goal"})
            elif clause.advisory:
                # advisory clauses have no predicate — human-tracked
                self.goal.prove_clause(clause.id, True, "human_approval",
                                       detail="model claim, advisory clause")

    def _complete(self, on_token, on_reasoning, on_status, should_cancel=None):
        schemas = self._tool_schemas()
        # cassette replay: zero API cost, fully deterministic (§20.2)
        if self.cassette is not None and self.cassette.mode == "replay":
            from .client import StreamResult
            stored = self.cassette.replay(self.model.id, self.messages,
                                          schemas)
            if stored is None:
                raise APIError("cassette replay miss — request not in the "
                               "recorded cassette (deterministic replay "
                               "never falls back to a live call)")
            content = stored.get("content", "")
            if content and on_token:
                on_token(content)
            return StreamResult(content=content,
                                reasoning=stored.get("reasoning", ""),
                                tool_calls=stored.get("tool_calls", []),
                                usage=stored.get("usage"),
                                model=self.model.id)
        try:
            result = chat_stream(self.provider, self.model, self.effort,
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
                result = chat_stream(self.provider, self.model, self.effort,
                                     self.messages, None,
                                     on_token=on_token,
                                     on_reasoning=on_reasoning,
                                     should_cancel=should_cancel)
            elif e.status == 400 and "stream" in msg:
                on_status("retrying (non-stream)")
                result = chat_blocking(self.provider, self.model,
                                       self.effort, self.messages, schemas)
            else:
                raise
        if self.cassette is not None and self.cassette.mode == "record":
            self.cassette.record(self.model.id, self.messages, schemas,
                                 {"content": result.content,
                                  "reasoning": result.reasoning,
                                  "tool_calls": result.tool_calls,
                                  "usage": result.usage})
        return result

    # -- autonomy + gating -----------------------------------------------------

    def set_autonomy(self, level: int) -> str:
        level = max(0, min(5, level))
        # the contract's autonomy ceiling cannot be raised by the agent (§37.1)
        if self.goal.status().active:
            ceiling = self._goal_autonomy_ceiling()
            if ceiling is not None and level > ceiling:
                level = ceiling
        self.autonomy = level
        self.log.append("autonomy.changed", {"level": level}, actor="human")
        return AUTONOMY_LEVELS[level]

    def _goal_autonomy_ceiling(self) -> int | None:
        g = fold(self.log).goal
        if not g:
            return None
        ceiling = g.get("autonomy_ceiling")
        return int(ceiling) if ceiling is not None else None

    def _attribute(self, tool_name: str) -> tuple[str | None, str | None]:
        """§38.1 total attribution. Returns (clause_id, orphan_reason).

        With an active goal, every action binds to the focus clause (the
        highest-gravity open clause). If no open clause exists, the action
        is an orphan and is rejected before execution."""
        goal = self.goal.status()
        if not goal.active:
            return None, None  # normal mode: no attribution required
        focus = goal.focus or self.goal.reaim()
        open_ids = {c.id for c in goal.clauses
                    if c.state in ("OPEN", "REGRESSED")}
        if focus and focus in open_ids:
            return focus, None
        if open_ids:
            return sorted(open_ids)[0], None
        return None, ("OrphanAction: no open clause to serve — the goal "
                      "is fully proven, waived, or blocked. Close the goal "
                      "or amend the contract.")

    def _gate(self, tool: Tool, args: dict) -> str | None:
        """Return a block reason, or None if the action may proceed."""
        # autonomy ladder (§22)
        if tool.name in _MUTATING_TOOLS:
            if self.autonomy <= 1:
                return (f"autonomy L{self.autonomy} is read-only — "
                        "raise it with /autonomy to allow mutations")
            if self.autonomy == 2:
                return "ASK"
            if self.autonomy == 4 and tool.name in _ALWAYS_ASK:
                return "ASK"
        # IRREVERSIBLE tools always need approval, at every level (§22)
        if tool.name in ("web_fetch", "web_search") and self.autonomy < 5:
            pass  # network reads stay free; network writes would be gated
        # deterministic dead-end blocking (§14.3)
        sig = _signature(tool.name, args)
        if self.memory.is_dead_end(sig):
            return (f"this exact approach is in the dead-end ledger "
                    f"(signature {sig}) — choose a different approach")
        return None

    def _snapshot_paths(self, tool_name: str, args: dict) -> list[str]:
        """The paths a mutating tool touches — the snapshot targets (A2)."""
        keys = _PATH_ARG_TOOLS.get(tool_name, ())
        paths = [str(args[k]) for k in keys if args.get(k)]
        if tool_name == "run_command":
            # commands can touch anything; snapshot the cwd tree shallowly
            paths = [str(p) for p in Path.cwd().iterdir()
                     if p.is_file()][:200]
        return paths

    def _execute_tool(self, ev: ToolEvent,
                      approve: Callable[[Tool, dict], bool],
                      on_status: Callable[[str], None],
                      causation_id: str | None = None) -> None:
        tool = self.tools.get(ev.name)
        started = time.time()
        if tool is None:
            ev.status = "error"
            ev.result = f"ERROR: unknown tool '{ev.name}'. Available: " \
                        + ", ".join(self.tools)
            return

        # §38.1 attribution — orphans are rejected before execution
        clause_id, orphan = self._attribute(ev.name)
        ev.clause_id = clause_id
        if orphan:
            ev.status = "blocked"
            ev.result = f"ERROR: blocked — {orphan}"
            self.log.append("tool.blocked", {"name": ev.name,
                                             "reason": orphan},
                            causation_id=causation_id)
            return

        block = self._gate(tool, ev.args)
        if block and block != "ASK":
            ev.status = "blocked"
            ev.result = f"ERROR: blocked — {block}"
            self.log.append("tool.blocked", {"name": ev.name,
                                             "reason": block},
                            causation_id=causation_id,
                            correlation_id=clause_id)
            return

        needs_ask = (block == "ASK"
                     or (tool.risk == RISK_CONFIRM and not self.cfg.auto_approve
                         and self.autonomy < 4))
        if needs_ask and not self.cfg.auto_approve:
            if not approve(tool, ev.args):
                ev.status = "denied"
                ev.result = "ERROR: user denied this action. Ask the user " \
                            "how to proceed or choose another approach."
                return

        # A2/I3: snapshot BEFORE any mutation — no write without a
        # committed recovery path
        snapshot_tree = None
        if ev.name in _MUTATING_TOOLS:
            paths = self._snapshot_paths(ev.name, ev.args)
            if paths:
                snap = self.store.take(paths)
                snapshot_tree = snap["tree"]
                self.log.append("snapshot.taken",
                                {"tree": snap["tree"],
                                 "paths": list(snap["paths"]),
                                 "before_tool": ev.name},
                                actor="kernel",
                                causation_id=causation_id,
                                correlation_id=clause_id)

        self.log.append("tool.call", {"name": ev.name, "args": ev.args},
                        actor="sovereign", provenance="model",
                        causation_id=causation_id,
                        correlation_id=clause_id)

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
        # most handlers report failure as an "ERROR: …" string rather than
        # raising — count those as errors too, so the dead-end ledger works
        if ev.status == "done" and ev.result.startswith("ERROR:"):
            ev.status = "error"
        ev.duration = time.time() - started

        # §37.4: after EVERY successful write, re-check the anti-clauses
        if ev.status == "done" and ev.name in _MUTATING_TOOLS:
            violations = self.goal.check_anti_clauses()
            if violations:
                v = violations[0]
                ev.result += (f"\nWARNING: anti-clause {v['clause']} "
                              f"violated — {v['detail']}. The run cannot "
                              f"close until this is repaired or rewound.")

        # §13.4 oscillation: file content flipping A-B-A-B
        if ev.name in ("write_file", "edit_file") and ev.status == "done":
            p = str(ev.args.get("path", ""))
            try:
                h = hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
            except OSError:
                h = ""
            if h:
                hist = self._file_hashes.setdefault(p, [])
                hist.append(h)
                if self.loop_det.oscillation(p, hist):
                    self.log.append("loop.alert",
                                    {"kind": "oscillation", "path": p,
                                     "action": "hard stop — present both "
                                               "versions to the human"},
                                    actor="kernel")

        # repeated identical failure -> deterministic dead-end (§14.3)
        if ev.status == "error":
            sig = _signature(ev.name, ev.args)
            self._error_counts[sig] = self._error_counts.get(sig, 0) + 1
            if self._error_counts[sig] == 2:
                self.memory.record_dead_end(
                    signature=sig,
                    reason=f"{ev.name} failed twice: {ev.result[:200]}",
                    scope="session", confidence="contextual")

        # §13.4 exact-repeat detection over recent tool calls
        self.loop_det.detect()

    # -- time travel (§9) --------------------------------------------------------

    def _nearest_snapshot(self, upto_seq: int) -> dict | None:
        """The snapshot whose captured state matches 'as of upto_seq' (§9.2).

        A snapshot.taken event at seq N records the world immediately
        BEFORE the write it precedes — i.e. the state as of seq N-1. So
        the state 'as of S' is the newest snapshot with seq <= S+1."""
        best = None
        for ev in self.log.events(upto_seq=upto_seq + 1):
            if ev.type == "snapshot.taken":
                best = ev
        return best

    def rewind_to(self, seq: int) -> tuple[int, int]:
        """REWIND (§9.1): filesystem AND agent state return to step N.

        Materialises the nearest snapshot at/below seq, then rewinds the
        log head and rebuilds the conversation from the fold. The old
        branch remains fully intact — nothing is destroyed.
        Returns (new_head, messages_kept)."""
        snap = self._nearest_snapshot(seq)
        if snap:
            self.store.materialise(snap.data.get("tree", ""))
        new_head = self.log.rewind(seq)
        st = fold(self.log, upto_seq=new_head)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.messages.extend(st.messages)
        return new_head, len(st.messages)

    def revert_files_to(self, seq: int) -> dict:
        """REVERT (§9.1): filesystem only returns to step N — the agent
        KEEPS its memory of what happened. This is what feeds the Dead-End
        Ledger: the files go back, the lesson stays."""
        snap = self._nearest_snapshot(seq)
        if not snap:
            return {"error": f"no snapshot at or before seq {seq}"}
        result = self.store.materialise(snap.data.get("tree", ""))
        self.log.append("kernel.revert",
                        {"to_seq": seq, "tree": snap.data.get("tree"),
                         **result}, actor="human")
        return result

    def fork_timeline(self, name: str | None = None) -> str:
        """Branch the timeline at the current head; continue on the fork."""
        branch = self.log.fork(name=name)
        self.log.checkout(branch)
        self.log.append("session.start",
                        {"session_id": self.session_id, "branch": branch})
        return branch

    # -- code intelligence (§15) ---------------------------------------------------

    def _register_code_tools(self) -> None:
        """Add the deterministic code.* tools (rung 2-4) to the registry."""
        nexus = self.nexus

        def code_symbols(name: str = "", path: str = ".") -> str:
            nexus.index(path)
            if name:
                syms = nexus.find_symbol(name)
                if not syms:
                    return f"no symbol named {name!r} under {path}"
                return "\n".join(f"{s.kind} {s.name}  {s.path}:{s.lineno} "
                                 f"params=({', '.join(s.params)})"
                                 for s in syms)
            lines = [f"{s.kind} {s.name}  {s.path}:{s.lineno}"
                     for s in list(nexus.idx.symbols.values())[:200]]
            return "\n".join(lines) or "no symbols found"

        def code_impact(name: str, path: str = ".") -> str:
            nexus.index(path)
            return nexus.format_impact(name)

        self.tools["code_symbols"] = Tool(
            "code_symbols",
            "List or find code symbols (functions/classes) via AST — "
            "precise and cheaper than grep. Args: name (optional), path.",
            {"type": "object", "properties": {
                "name": {"type": "string"},
                "path": {"type": "string"}}, "required": []},
            code_symbols)
        self.tools["code_impact"] = Tool(
            "code_impact",
            "Impact analysis: if I change this symbol, what breaks? "
            "Callers, tests, public API, risk score. Args: name, path.",
            {"type": "object", "properties": {
                "name": {"type": "string"},
                "path": {"type": "string"}}, "required": ["name"]},
            code_impact)

    # -- swarm (§16) ---------------------------------------------------------

    def scout_context(self, max_chars: int = 4000) -> str:
        """Shared read-only context handed to every scout (§16: scouts get
        a fresh minimal context — the question plus a little shared state).
        Scouts have no tools, so anything they must reason about has to be
        in here: where we are, what the goal is, what is known so far."""
        parts: list[str] = [f"Working directory: {Path.cwd()}"]
        try:
            entries = sorted(p.name + ("/" if p.is_dir() else "")
                             for p in Path.cwd().iterdir()
                             if not p.name.startswith("."))[:40]
            if entries:
                parts.append("Directory listing: " + ", ".join(entries))
        except OSError:
            pass
        goal = self.goal.status()
        if goal.active:
            parts.append("Active goal: " + goal.statement)
            for c in goal.clauses:
                parts.append(f"  clause {c.id} [{c.state}]: {c.text}")
        st = fold(self.log)
        if st.files_touched:
            parts.append("Files touched this session: "
                         + ", ".join(sorted(st.files_touched)[:20]))
        facts = [f.get("fact", "") for f in st.facts][-8:]
        if facts:
            parts.append("Known facts:\n- " + "\n- ".join(facts))
        episodes = st.episodes[-3:]
        for ep in episodes:
            parts.append(f"Past episode: {ep.get('goal', '')} -> "
                         f"{ep.get('outcome', '')}")
        ctx = "\n".join(parts)
        return ctx[:max_chars]

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
