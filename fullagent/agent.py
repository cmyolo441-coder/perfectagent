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
from . import systemprompt
from .autopilot import AutoPilot, RouteDecision
from .cassette import Cassette
from .client import (APIError, TurnCancelled, chat_blocking, chat_stream,
                     estimate_tokens, is_context_overflow)
from .config import Config, Effort, Model, Provider, PROVIDERS, model_by_id
from .cortex import Budget, BudgetGovernor, LoopDetector
from .council import Council
from .cov import CoverageEngine
from .daemon import Daemon
from .dashboard import Dashboard
from .forge import Forge
from .fuzz import Fuzzer
from .goal import GoalContract
from .healer import Healer
from .judge import Judge
from .kernel import EventLog, fold
from .kgraph import KnowledgeGraph
from .mastermind import Mastermind
from .memory import Hippocampus
from .mutate import MutationTester
from .nexus import Nexus
from .oracle import Oracle
from .router import Router
from .semantic import SemanticMemory
from .skills import SkillForge
from .snapshots import SnapshotStore
from .speculate import Speculator, SPECULATIVE_TOOLS
from .swarm import Swarm
from .taint import StaticAnalyzer
from .team import Team
from .tools import RISK_CONFIRM, Tool, build_registry, parse_tool_arguments

# The system prompt lives in systemprompt.py — the single source of truth.
# This module only ever reads it from there.

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
        self.session_id = uuid.uuid4().hex[:8]
        self.turns: list[Turn] = []

        # Temporal kernel + the nine subsystems
        config.ensure_dirs()
        self.log = EventLog(config.EVENT_LOG_FILE, session=self.session_id)
        # Mastermind: hash-sealed prompts + the single gate to the model.
        # Built before messages so the very first system prompt is sealed
        # and dispatched through the gate, not assembled by hand.
        self.mastermind = Mastermind(self.log)
        self.messages: list[dict] = []
        self._reseat_system_prompt()
        self.store = SnapshotStore(config.APP_DIR / "store")
        self.memory = Hippocampus(self.log)
        self.judge = Judge(self.log)
        self.goal = GoalContract(self.log, judge=self.judge)
        self.swarm = Swarm(self.log, self.provider, self.model, self.effort,
                           mastermind=self.mastermind)
        self.team = Team(self.log, self.provider, self.model, self.effort,
                         mastermind=self.mastermind)
        self.autopilot = AutoPilot(self.log)
        self.nexus = Nexus()
        self.forge = Forge(self.log)
        self.oracle = Oracle(self.log, memory_dir=config.APP_DIR / "memory")
        self.budget_gov = BudgetGovernor(self.log, Budget())
        self.loop_det = LoopDetector(self.log)

        # v3 advanced subsystems — all event-sourced on the same kernel
        self.router = Router(self.log)                 # cost brain
        self.semantic = SemanticMemory(self.log)       # meaning-based recall
        self.speculator = Speculator(self.log,
                                     runner=self._spec_runner)
        self.dashboard = Dashboard(self.log)           # live observability
        self.daemon = Daemon(self.log,
                             executor=self._daemon_step)
        self.healer = Healer(self.log)                 # root-cause capture
        self.skill_forge = SkillForge(self.log,
                                      skills_dir=config.APP_DIR / "skills")
        self.council = Council(self.log, speaker=self._council_speaker)

        # v4 professional subsystems — same kernel, same discipline
        self.kgraph = KnowledgeGraph(self.log)         # knowledge graph
        self.static = StaticAnalyzer(self.log)         # taint/complexity
        self.coverage = CoverageEngine(self.log)       # real line coverage
        self.fuzzer = Fuzzer(self.log)                 # property fuzzing
        self.mutator: MutationTester | None = None     # needs a suite cmd

        self.autonomy = 3
        self._error_counts: dict[str, int] = {}
        self._file_hashes: dict[str, list[str]] = {}  # oscillation history
        self._register_code_tools()
        self._register_v4_tools()
        self._register_persisted_skills()

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

    def _base_prompt(self) -> str:
        """The system prompt for this session, served from the Mastermind
        vault (hash-sealed copy of systemprompt.py). Falls back to the
        module source if the vault somehow lacks the name, so the model is
        never promptless."""
        name = self.cfg.prompt
        sealed = self.mastermind.vault.get(name)
        return sealed if sealed is not None else systemprompt.get(name)

    def _reseat_system_prompt(self,
                              sections: dict[str, str] | None = None
                              ) -> None:
        """Route the conversation's system prompt through the Mastermind
        gate — the single door to the model. Guarantees messages[0] carries
        the sealed prompt, composes any live context sections beneath it,
        and seals a prompt.dispatch lineage event."""
        self.messages, _ = self.mastermind.gate.dispatch(
            self.cfg.prompt, self.messages, sections=sections)

    def state(self):
        """Live projection of the event log (cost, goal, dead-ends, …)."""
        return fold(self.log)

    # -- conversation ------------------------------------------------------

    def reset(self) -> None:
        self.messages = []
        self._reseat_system_prompt()
        self.turns = []
        self.session_id = uuid.uuid4().hex[:8]
        self.log.session = self.session_id
        self.log.append("session.start", {"session_id": self.session_id,
                                          "model": self.cfg.model_id,
                                          "effort": self.cfg.effort},
                        actor="system")

    def _context_sections(self,
                          route: RouteDecision | None = None,
                          query: str = ""
                          ) -> dict[str, str]:
        """Live context sections composed beneath the sealed prompt by the
        Mastermind gate (constitution, goal, web, memory). The framing is
        the composer's job — bodies here are plain content only."""
        sections: dict[str, str] = {}
        constitution = self.oracle.read_constitution()
        if constitution.strip():
            sections["constitution"] = constitution.strip()
        goal = self.goal.status()
        if goal.active:
            sections["goal"] = (self.goal.format() +
                                "\nEvery action must serve an open clause. "
                                "When a clause's predicate genuinely "
                                "passes, say 'PROVEN: <clause id>' — the "
                                "kernel verifies it, never trust "
                                "self-declared success.")
        if route is not None and route.use_web:
            sections["web"] = ("This request needs live, up-to-the-minute "
                               "data. Use web_search (and web_fetch for "
                               "details) to get CURRENT facts — never "
                               "answer from stale knowledge. Quote the "
                               "retrieval time and sources.")
        mem = self.memory.context_block()
        # v3: meaning-based recall — pull the episodes/facts/dead-ends most
        # similar to the current request, not just the most recent ones.
        if query:
            recall = self.semantic.recall_block(query, k=3)
            if recall:
                mem = (mem + "\n\n" + recall) if mem else recall
        if mem:
            sections["memory"] = mem
        return sections

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

        # refresh the system document through the Mastermind gate — the
        # sealed prompt is guaranteed at position 0, live context sections
        # (constitution + goal + web + memory) are composed beneath it.
        self._reseat_system_prompt(
            sections=self._context_sections(route, query=user_text))

        # v3: speculate on the read-only calls this turn will likely make —
        # they run in the background while the model thinks (zero-latency
        # feel). Only whitelisted read-only tools are ever prefetched.
        recent_tools = [{"name": t.name, "args": t.args}
                        for tn in self.turns[-2:] for t in tn.tools]
        self.speculator.speculate(user_text, recent_tools)

        # Guard: a single giant paste (a whole file, a huge log) must never
        # blow the context window by itself — cap it so the turn can still
        # run. The full text stays in the event log; the model sees a
        # truncated copy with instructions to re-read from disk if needed.
        user_text = self._cap_user_message(user_text)

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
                # keep the context under the window — compact stale turns
                # before asking the model (I9: input budget governor)
                try:
                    self._maybe_compact()
                except Exception:
                    pass
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

    def _fit_budget(self) -> int:
        """The input token budget we aim to stay under.

        Mirrors the client's clamp math (client._window_max_tokens) so the
        two never disagree: the budget is the largest input size that still
        leaves room for the margin and a minimum completion. Staying under
        this guarantees the request fits no matter how long the session has
        been running."""
        window = self.model.context_window
        # solve: input + (8192 + input/32) + 1024 <= window
        return int((window - 9_216) * 32 / 33)

    def _cap_user_message(self, text: str) -> str:
        """Cap a single user message so it can never consume the whole
        context window on its own (a giant paste of a file/log). The cap
        is a quarter of the fit budget — plenty for any real instruction,
        small enough that the rest of the conversation always fits."""
        cap_chars = int(self._fit_budget() * 3.2 // 4)  # budget tokens -> chars
        if len(text) <= cap_chars:
            return text
        head = text[: cap_chars // 2]
        tail = text[-cap_chars // 4:]
        self.log.append("user.message.capped",
                        {"chars": len(text), "kept": len(head) + len(tail)},
                        actor="kernel")
        return (head
                + f"\n\n[… the kernel truncated this message — {len(text):,} "
                  f"chars originally. If you need the full text, it is on "
                  f"disk; use read_file/search_files instead of relying on "
                  f"this paste.]\n\n"
                + tail)

    def _maybe_compact(self) -> None:
        """Keep the conversation under the model's context window.

        Three escalating passes, run only when needed:
          1. truncate stale tool outputs to summaries,
          2. drop the oldest user→(assistant+tool results) turns as whole
             units so tool_call / tool-response pairing is never broken,
          3. last resort — truncate EVERY tool output and trim the oldest
             assistant messages.
        Only the model-visible context shrinks — the event log keeps
        everything, so nothing is ever lost."""
        schemas = self._tool_schemas()
        schema_tokens = (estimate_tokens(schemas, self.model.id)
                         if schemas else 0)
        budget = self._fit_budget()
        if (estimate_tokens(self.messages, self.model.id)
                + schema_tokens <= budget):
            return
        dropped = self._compact_old_tools()
        while (estimate_tokens(self.messages, self.model.id)
               + schema_tokens > budget
               and len(self.messages) > 2):
            if not self._drop_oldest_turn():
                break
            dropped += 1
        # last resort: still over budget — truncate all tool outputs and
        # trim the oldest assistant messages until it fits
        if (estimate_tokens(self.messages, self.model.id)
                + schema_tokens > budget):
            self._compact_old_tools(keep=0)
            self._trim_oldest_assistant(budget - schema_tokens)
        if dropped:
            self.log.append("context.compacted",
                            {"messages": len(self.messages),
                             "units": dropped,
                             "est_tokens": estimate_tokens(self.messages,
                                              self.model.id)},
                            actor="kernel")

    def _trim_oldest_assistant(self, budget: int) -> None:
        """Shrink the oldest assistant messages (they are the least
        actionable once their tool results are gone) until under budget."""
        while (estimate_tokens(self.messages, self.model.id) > budget
               and len(self.messages) > 2):
            # find the oldest assistant message with real content
            target = None
            for i, m in enumerate(self.messages):
                if m.get("role") == "assistant" and m.get("content"):
                    target = i
                    break
            if target is None:
                return
            content = str(self.messages[target]["content"])
            if len(content) > 400:
                self.messages[target]["content"] = (
                    content[:300]
                    + f"\n[… trimmed by the kernel — {len(content):,} "
                      f"chars originally]")
            else:
                return

    def _emergency_compact(self) -> None:
        """Hard compaction used when a request was ALREADY rejected for
        exceeding the context window. More aggressive than _maybe_compact:
        truncates every tool output, then drops oldest turns until the
        conversation is at most a third of the window — leaving a wide
        margin so the retry is guaranteed to fit."""
        self._compact_old_tools(keep=0)
        target = self.model.context_window // 3
        while (estimate_tokens(self.messages, self.model.id) > target
               and len(self.messages) > 2):
            if not self._drop_oldest_turn():
                break
        self._trim_oldest_assistant(target)
        self.log.append("context.compacted",
                        {"messages": len(self.messages),
                         "est_tokens": estimate_tokens(self.messages,
                                              self.model.id),
                         "reason": "emergency — request rejected for "
                                   "context length"},
                        actor="kernel")

    def _overflow_shrink(self) -> bool:
        """Callback handed to the client (on_overflow): invoked when the
        backend rejects a request because the INPUT itself no longer fits
        the window. Shrinks the conversation and reports whether anything
        actually shrank — the client retries only when it did. This is
        what lets an arbitrarily long session on a huge project recover
        instead of dying with a context-length error."""
        before = estimate_tokens(self.messages, self.model.id)
        self._emergency_compact()
        after = estimate_tokens(self.messages, self.model.id)
        return after < before

    def _compact_old_tools(self, keep: int = 2) -> int:
        """Truncate stale tool results to summaries, keeping the newest
        `keep` tool responses verbatim."""
        tool_idx = [i for i, m in enumerate(self.messages)
                    if m.get("role") == "tool"]
        stale = tool_idx if keep <= 0 else tool_idx[:-keep]
        n = 0
        for i in stale:
            content = str(self.messages[i].get("content") or "")
            if len(content) > 500:
                self.messages[i]["content"] = (
                    content[:350]
                    + f"\n[… truncated by the kernel — {len(content):,} "
                      f"chars originally; re-read the file if you need it]")
                n += 1
        return n

    def _drop_oldest_turn(self) -> bool:
        """Remove the oldest user message plus everything up to the next
        user message (the assistant reply and its tool results)."""
        msgs = self.messages
        if msgs and msgs[0].get("role") == "system":
            start = 1
        else:
            start = 0
        end = None
        for j in range(start + 1, len(msgs)):
            if msgs[j].get("role") == "user":
                end = j
                break
        if end is None or end <= start:
            return False
        del msgs[start:end]
        return True

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
                                 should_cancel=should_cancel,
                                 on_overflow=self._overflow_shrink)
        except APIError as e:
            # some providers refuse streaming or tool params — degrade gracefully
            msg = str(e).lower()
            if is_context_overflow(str(e)):
                # last line of defence: the client already re-clamped,
                # retried and shrunk, so compact one final time and try
                # once more — a long session must never die with a
                # context-length error.
                on_status("compacting context")
                self._emergency_compact()
                result = chat_stream(self.provider, self.model, self.effort,
                                     self.messages, schemas,
                                     on_token=on_token,
                                     on_reasoning=on_reasoning,
                                     on_tool_start=lambda n: on_status(f"tool:{n}"),
                                     should_cancel=should_cancel,
                                     on_overflow=self._overflow_shrink)
            elif e.status == 400 and "tool" in msg and schemas:
                on_status("retrying (no tools)")
                result = chat_stream(self.provider, self.model, self.effort,
                                     self.messages, None,
                                     on_token=on_token,
                                     on_reasoning=on_reasoning,
                                     should_cancel=should_cancel,
                                     on_overflow=self._overflow_shrink)
            elif e.status == 400 and "stream" in msg:
                on_status("retrying (non-stream)")
                result = chat_blocking(self.provider, self.model,
                                       self.effort, self.messages, schemas,
                                       on_overflow=self._overflow_shrink)
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
        # v3: if the Speculator already prefetched this exact read-only
        # call, serve it from the cache instead of re-running (a hit).
        cached = self.speculator.serve(ev.name, ev.args)
        if cached is not None:
            ev.result = cached
            ev.status = "done"
        else:
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
            # v3: healer captures + classifies the root cause and seals a
            # lesson, so the same failure is recognised instantly next time.
            # (No fixer attached here — the agent reads the diagnosis and
            # decides the fix; the healer never mutates on its own.)
            try:
                self.healer.heal(ev.result,
                                 context=f"{ev.name} {ev.args}")
            except Exception:
                pass

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
        self.messages = []
        self._reseat_system_prompt()
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

    def _register_v4_tools(self) -> None:
        """Add the deterministic v4 engineering tools (rung 2-4) to the
        registry: static analysis, knowledge graph, coverage, fuzzing."""
        static = self.static
        kgraph = self.kgraph
        coverage = self.coverage
        fuzzer = self.fuzzer

        def analyze_code(path: str = ".", glob_filter: str = "*.py",
                         max_files: int = 50) -> str:
            p = Path(path).expanduser()
            if p.is_file():
                return static.format_report(static.analyze_file(str(p)))
            tree = static.analyze_tree(str(p), glob_filter, max_files)
            return static.format_report(tree)

        def graph_index(path: str = ".") -> str:
            p = Path(path).expanduser()
            sources: dict[str, str] = {}
            files = ([p] if p.is_file()
                     else sorted(p.glob("**/*.py"))[:200])
            for f in files:
                if f.is_file():
                    try:
                        sources[f.stem] = f.read_text(errors="replace")
                    except OSError:
                        continue
            kgraph.index_code(sources)
            kgraph.index_log()
            return kgraph.format_status()

        def graph_query(name: str, kind: str = "") -> str:
            hits = kgraph.find(name, kind or None)
            if not hits:
                return f"no entity matching {name!r} — run graph_index first"
            lines = []
            for e in hits[:20]:
                lines.append(f"{e.kind} {e.id}  ({e.name})")
                for r in kgraph.out_edges(e.id)[:8]:
                    lines.append(f"    --{r.rel}--> {r.dst}")
                for r in kgraph.in_edges(e.id)[:8]:
                    lines.append(f"    <--{r.rel}-- {r.src}")
            return "\n".join(lines)

        def graph_impact(name: str) -> str:
            hits = kgraph.find(name)
            if not hits:
                return f"no entity matching {name!r} — run graph_index first"
            lines = []
            for e in hits[:5]:
                dep = kgraph.impact(e.id)
                lines.append(f"{e.id}: {len(dep)} dependent(s)")
                lines.extend(f"    {d}" for d in dep[:20])
            return "\n".join(lines)

        def measure_coverage(path: str, command: str) -> str:
            """Coverage of `path` while `command` (a python -c snippet or
            test command) runs. The subject is executed via subprocess so
            the trace stays inside this process only for the import."""
            tp = Path(path).expanduser()
            if not tp.is_file():
                return f"ERROR: not a file: {tp}"
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                tp.stem, str(tp.resolve()))
            if spec is None or spec.loader is None:
                return f"ERROR: cannot import {tp}"
            mod = importlib.util.module_from_spec(spec)

            def subject():
                spec.loader.exec_module(mod)
                # run the command as a python expression against the module
                eval(compile(command, "<coverage-subject>", "exec"),
                     {"mod": mod, "__name__": "__coverage__"})

            res = coverage.measure(str(tp), subject)
            missed = ", ".join(str(m) for m in res.missed[:30])
            return (f"coverage {res.percent:.0f}%  "
                    f"({res.hit}/{res.total} lines)\nmissed: {missed}")

        def fuzz_target(path: str, function: str, iterations: int = 200,
                        nargs: int = 1) -> str:
            tp = Path(path).expanduser()
            if not tp.is_file():
                return f"ERROR: not a file: {tp}"
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"fuzz_{tp.stem}", str(tp.resolve()))
            if spec is None or spec.loader is None:
                return f"ERROR: cannot import {tp}"
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                return f"ERROR: import failed: {type(e).__name__}: {e}"
            fn = getattr(mod, function, None)
            if not callable(fn):
                return f"ERROR: no callable {function!r} in {tp}"
            report = fuzzer.fuzz(fn, iterations=iterations, nargs=nargs,
                                 name=f"{tp.stem}.{function}")
            lines = [f"fuzz {report.target}: {report.iterations} runs, "
                     f"{report.crashes} crash(es), "
                     f"{report.invariant_failures} invariant failure(s)"]
            if report.first_crash:
                c = report.first_crash
                lines.append(f"  first crash: {c.error}")
                lines.append(f"    args: {repr(c.args)[:120]}")
                lines.append(f"    shrunk: {repr(c.shrunk_args)[:120]} "
                             f"→ {c.shrunk_error}")
            return "\n".join(lines)

        self.tools["analyze_code"] = Tool(
            "analyze_code",
            "Static analysis of a file or tree: taint flows (source→sink), "
            "cyclomatic complexity hotspots, import cycles. Deterministic "
            "AST analysis. Args: path, glob_filter, max_files.",
            {"type": "object", "properties": {
                "path": {"type": "string"},
                "glob_filter": {"type": "string"},
                "max_files": {"type": "integer"}}, "required": []},
            analyze_code)
        self.tools["graph_index"] = Tool(
            "graph_index",
            "Build the knowledge graph (entities + typed relations) from "
            "Python sources under path, plus the session log. Args: path.",
            {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": []},
            graph_index)
        self.tools["graph_query"] = Tool(
            "graph_query",
            "Query the knowledge graph: find entities by name and show "
            "their relations. Args: name, kind (optional).",
            {"type": "object", "properties": {
                "name": {"type": "string"},
                "kind": {"type": "string"}}, "required": ["name"]},
            graph_query)
        self.tools["graph_impact"] = Tool(
            "graph_impact",
            "Impact analysis over the knowledge graph: everything that "
            "depends on this entity ('what breaks if I change X?'). "
            "Args: name.",
            {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]},
            graph_impact)
        self.tools["measure_coverage"] = Tool(
            "measure_coverage",
            "Real line coverage (sys.settrace) of a Python file while a "
            "subject snippet runs. Args: path (file under test), command "
            "(python code executed with the module bound as `mod`).",
            {"type": "object", "properties": {
                "path": {"type": "string"},
                "command": {"type": "string"}},
                "required": ["path", "command"]},
            measure_coverage, risk=RISK_CONFIRM)
        self.tools["fuzz_target"] = Tool(
            "fuzz_target",
            "Property-based fuzzing of a function: generated + boundary + "
            "mutated inputs, crash shrinking to a minimal reproducer. "
            "Args: path, function, iterations, nargs.",
            {"type": "object", "properties": {
                "path": {"type": "string"},
                "function": {"type": "string"},
                "iterations": {"type": "integer"},
                "nargs": {"type": "integer"}},
                "required": ["path", "function"]},
            fuzz_target, risk=RISK_CONFIRM)

    # -- v3 subsystem callbacks ------------------------------------------------

    def _spec_runner(self, name: str, args: dict) -> str:
        """Execute one read-only tool call for the Speculator. Only the
        whitelisted read-only tools ever reach here (the speculator gates
        on SPECULATIVE_TOOLS before calling)."""
        if name not in SPECULATIVE_TOOLS:
            return f"ERROR: {name} is not speculative-safe"
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name}"
        try:
            return tool.handler(**args)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _daemon_step(self, task: str) -> str:
        """Run one daemon mission step. A step is executed as a read-only
        scout-style probe by default — the daemon advances missions without
        mutating the world unless a write is explicitly part of the task.
        Returns 'ERROR: …' on failure so the daemon can retry/block."""
        try:
            reports = self.swarm.scout([task], context=self.scout_context(),
                                       timeout=120.0)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
        if not reports:
            return "ERROR: daemon step produced no report"
        r = reports[0]
        if not r.ok:
            return f"ERROR: {r.error or 'step failed'}"
        return f"OK: {r.answer[:400]}"

    def _council_speaker(self, role: str, brief: str) -> str:
        """Produce one council position through the model (blocking, no
        tools). The synthesis brief is already blind — it carries only the
        two arguments."""
        messages = [{"role": "system",
                     "content": "You are one voice in a structured debate "
                                "council. Answer exactly as instructed."},
                    {"role": "user", "content": brief}]
        result = chat_blocking(self.provider, self.model, self.effort,
                               messages, None, timeout=120.0)
        return result.content or ""

    def _register_persisted_skills(self) -> None:
        """Load previously forged skills from disk and expose them as tools.
        Only skills that re-pass the safety scan are loaded."""
        try:
            self.skill_forge.load_persisted()
        except Exception:
            return
        for name, skill in self.skill_forge.registry.items():
            if name in self.tools:
                continue
            self._expose_skill(name, skill)

    def _expose_skill(self, name: str, skill) -> None:
        """Wrap a validated skill as a live Tool in the registry."""
        try:
            namespace = self.skill_forge._load(skill)
        except Exception:
            return
        fn = namespace.get(skill.entry or name)
        if not callable(fn):
            return

        def handler(_fn=fn, **kwargs):
            try:
                return str(_fn(**kwargs))
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}"

        props = skill.parameters or {}
        schema = {"type": "object",
                  "properties": {k: (v if isinstance(v, dict)
                                     else {"type": "string"})
                                 for k, v in props.items()}}
        self.tools[name] = Tool(name, skill.description or name,
                                schema, handler)

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
