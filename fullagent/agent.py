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
from .attention import AttentionEconomy
from .bandit import BanditRouter
from .brain import Brain
from .causal import CausalEngine
from .ci import CIPilot
from .compiler import IntentCompiler, default_drafter
from .cortex import Budget, BudgetGovernor, LoopDetector
from .council import Council
from .crew import Crew, CrewError
from .debate import DebateTournament
from .dual import DualProcess
from .evolution import EvolutionEngine, default_benchmark
from .fabric import KnowledgeFabric
from .formal import ModelChecker
from .homeo import Homeostasis
from .market import TaskMarket
from .mcts import TreeSearch
from .merge import TimelineMerger
from .mesh import MeshNode
from .meta import RoleForge, default_drafter as role_drafter
from .goal import TERMINAL_STATES
from .report import export_html, export_markdown, forecast, format_forecast
from .workflows import WorkflowEngine, WorkflowError
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
from .race import RacingUniverses
from .synth import ProgramSynthesizer, SynthSpec, default_generator
from .taint import StaticAnalyzer
from .team import MAX_SUMMARY_CHARS, WorkerReport, parse_worker_final
from .theater import Theater
from .tools import RISK_CONFIRM, Tool, build_registry, parse_tool_arguments
from .tower import Tower
from .tuner import ParzenTuner
from .world import WorldModel

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
    scorecard: dict = field(default_factory=dict)
    duration: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


def _signature(name: str, args: dict) -> str:
    """Canonical approach signature — the dead-end ledger key."""
    payload = json.dumps({"name": name, "args": args},
                         sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]



NOTIFY_EVENTS = ("goal.closed", "focus.stop", "workflow.done",
                 "crew.done", "provider.failover")

# provider errors that justify an automatic model failover
_FAILOVER_STATUSES = {408, 429, 500, 502, 503, 504}


class Notifier:
    """Enterprise event notifications — fire selected kernel events to a
    file sink (JSONL) or an HTTP webhook. Never raises: notifications are
    a courtesy, never a crash path. Configure via /notify."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.sink: str = ""          # "" off | "file:<path>" | http(s) URL
        self.sent = 0
        self.last_error = ""

    def configure(self, sink: str) -> str:
        sink = str(sink or "").strip()
        if sink in ("", "off"):
            self.sink = ""
            return "off"
        if sink.startswith("file:"):
            path = Path(sink[5:]).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.sink = f"file:{path}"
            return self.sink
        if sink.startswith(("http://", "https://")):
            self.sink = sink
            return sink
        raise ValueError("sink must be 'off', 'file:<path>', or an "
                         "http(s):// URL")

    def emit(self, event_type: str, payload: dict) -> bool:
        if not self.sink:
            return False
        record = {"event": event_type, "ts": time.time(),
                  "app": "fullagent", **payload}
        try:
            if self.sink.startswith("file:"):
                with open(self.sink[5:], "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            else:
                import requests
                requests.post(self.sink, json=record, timeout=5)
            self.sent += 1
            return True
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    def status(self) -> str:
        state = self.sink or "off"
        extra = f" · sent {self.sent}" if self.sent else ""
        extra += f" · last error: {self.last_error}" if self.last_error else ""
        return f"notifications: {state}{extra} · events: " \
               + ", ".join(NOTIFY_EVENTS)


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
        # v5 advanced subsystems — same kernel, same discipline
        self.brain = Brain(self.log, config.APP_DIR / "brain.json")
        self.compiler = IntentCompiler(
            self.log, default_drafter(self.provider, self.model,
                                      self.effort),
            executor=self._compile_wave)
        self.evolution = EvolutionEngine(
            self.log, self._evolution_mutator, self._evolution_evaluator)
        self.merger = TimelineMerger(self.log)
        self.theater = Theater(self.log)
        self.debate = DebateTournament(
            self.log, self._debate_speaker,
            [m.id for m in config.MODELS if m.supports_tools][:4])
        self.market = TaskMarket(self.log, executor=self._market_exec)
        self.tower = Tower(self)
        # v6 frontier subsystems — same kernel, same discipline
        self.formal = ModelChecker(self.log)
        self.mcts = TreeSearch(self.log, self._mcts_evaluator)
        self.causal = CausalEngine(self.log)
        self.bandit = BanditRouter(
            self.log, [m.id for m in config.MODELS[:8]])
        self.mesh = MeshNode(self.log, self.session_id,
                             executor=self._mesh_exec)
        self.roleforge = RoleForge(
            self.log,
            role_drafter(self.provider, self.model, self.effort),
            self._role_audition)
        self.synth = ProgramSynthesizer(
            self.log, default_generator(self.provider, self.model,
                                        self.effort),
            registry=self.tools)
        self.ci = CIPilot(self.log, Path.cwd(),
                          runner=self._ci_runner)
        self.tuner = ParzenTuner(self.log, {
            "effort": [e.key for e in config.EFFORTS],
            "worker_steps": ["low", "medium", "high"],
        })
        self.dual = DualProcess(self.log, self._dual_fast,
                                self._dual_slow, brain=self.brain)
        self.world = WorldModel(self.log, root=Path.cwd())
        self.racer = RacingUniverses(self.log, self._race_runner,
                                     self._race_verify)
        self.homeo = Homeostasis(self.log, repairs={
            "tool_error_rate": self._repair_reseed_prompts,
            "loop_alerts": self._repair_consolidate,
            "tool_latency_ms": self._repair_warm_caches,
        })
        self.attention = AttentionEconomy(self.log)
        self.fabric = KnowledgeFabric(self.log)
        self.crew = Crew(self.log, self.provider, self.model, self.effort,
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
        # live status pipe for the current turn (set in run_turn) — lets
        # long-running tools (crew) stream progress into the UI border
        self._turn_status: Callable[[str], None] | None = None
        # Focus Mode (deep work): distance history drives stall detection
        self._focus_history: list[float] = []
        # enterprise extras
        self.workflows = WorkflowEngine(self.log,
                                        config.APP_DIR / "workflows",
                                        executor=self._workflow_step,
                                        judge=self.judge)
        self.notifier = Notifier(self.log)
        self._notify_seq = self.log.head()
        self.health = {"model_errors": {}, "failovers": 0}
        self._failed_over = False            # at most one failover per turn
        self._turn_start_seq = 0
        self._compact_digests: list[str] = []  # knowledge kept on compaction
        self._error_counts: dict[str, int] = {}
        self._file_hashes: dict[str, list[str]] = {}  # oscillation history
        # SPEED: cached tool schemas — rebuilt only when the registry size
        # changes (was re-serialized before every model call)
        self._schemas_cache: list[dict] | None = None
        self._schemas_tool_count = -1
        # SPEED: compaction gate — the full token re-estimate only runs
        # after ~15% conversation growth, not before every model call
        self._compact_check_chars = 0
        self._register_code_tools()
        self._register_v4_tools()
        self._register_crew_tools()
        self._register_advanced_tools()
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
        # v5: the cognitive brain — what survived the forgetting curve
        if query:
            brain_block = self.brain.context_block(query, k=3)
            if brain_block:
                mem = (mem + "\n\n" + brain_block) if mem else brain_block
        if mem:
            sections["memory"] = mem
        if self._compact_digests:
            sections["compacted"] = (
                "COMPACTED HISTORY — knowledge preserved from turns that "
                "were compressed to fit the context window:\n- "
                + "\n- ".join(self._compact_digests[-12:]))
        return sections

    def _tool_schemas(self) -> list[dict] | None:
        if not self.model.supports_tools:
            return None
        # SPEED: the schema list is rebuilt only when the registry changes
        # (new tools/skills) — it used to be re-serialized before EVERY
        # model call inside a turn.
        n = len(self.tools)
        if (self._schemas_cache is not None
                and self._schemas_tool_count == n):
            return self._schemas_cache
        schemas = [t.openai_schema() for t in self.tools.values()]
        self._schemas_cache = schemas
        self._schemas_tool_count = n
        return schemas

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
        self._turn_start_seq = self.log.head()
        self._failed_over = False

        # AUTOPILOT: the agent decides for itself which powers this turn
        # needs — goal mode, real-time web — and enables
        # them on its own. Every decision is logged and surfaced.
        route = self.autopilot.route(user_text,
                                     self.goal.status().active,
                                     self.autonomy)
        if route.active and on_route:
            on_route(route)
        if route.suggest_goal:
            self._autopilot_goal(route)

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
        self._turn_status = on_status
        try:
            while iterations < config.MAX_TOOL_ITERATIONS:
                iterations += 1
                # I8: budget governor — a breach PAUSES the turn
                if not self.budget_gov.enforce():
                    evs = fold(self.log).budget_events
                    reason = evs[-1]["reason"] if evs else "budget exceeded"
                    turn.error = f"PAUSED — {reason}. Extend the budget " \
                                 f"with /budget (e.g. /budget steps 500 " \
                                 f"or /budget reset) or stop."
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

        self._turn_status = None
        turn.duration = time.time() - started
        try:
            self._score_turn(turn)
        except Exception:
            pass  # scorecard is insight, never a crash path
        try:
            self._flush_notifications()
        except Exception:
            pass
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
        everything, so nothing is ever lost.

        SPEED: the expensive token estimate (a JSON dump of the whole
        conversation) runs only when the conversation has measurably
        grown since the last check — the cheap char-count is the gate."""
        schemas = self._tool_schemas()
        schema_tokens = (estimate_tokens(schemas, self.model.id)
                         if schemas else 0)
        budget = self._fit_budget()
        chars = self._messages_chars()
        if self._compact_check_chars > 0 and \
                chars <= self._compact_check_chars * 1.15:
            return  # not yet grown enough to justify a full re-estimate
        fits = (estimate_tokens(self.messages, self.model.id)
                + schema_tokens <= budget)
        self._compact_check_chars = max(chars, 1)
        if fits:
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
            self._compact_check_chars = 0  # force re-estimate next time
            self.log.append("context.compacted",
                            {"messages": len(self.messages),
                             "units": dropped,
                             "digests": len(self._compact_digests),
                             "est_tokens": estimate_tokens(self.messages,
                                              self.model.id)},
                            actor="kernel")

    def _messages_chars(self) -> int:
        """Cheap conversation size proxy (no JSON dump): sum of content
        lengths plus tool-call argument sizes. Gate for the expensive
        token estimate."""
        n = 0
        for m in self.messages:
            c = m.get("content")
            if c:
                n += len(c)
            tcs = m.get("tool_calls")
            if tcs:
                for tc in tcs:
                    fn = tc.get("function") or {}
                    n += len(fn.get("name", "")) + len(
                        fn.get("arguments") or "")
        return n

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
        digest = self._digest_messages(msgs[start:end])
        if digest:
            self._compact_digests.append(digest)
            self._compact_digests = self._compact_digests[-20:]
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
        def _attempt():
            return chat_stream(self.provider, self.model, self.effort,
                               self.messages, schemas,
                               on_token=on_token,
                               on_reasoning=on_reasoning,
                               on_tool_start=lambda n: on_status(f"tool:{n}"),
                               should_cancel=should_cancel,
                               on_overflow=self._overflow_shrink)

        try:
            result = _attempt()
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
                # enterprise failover: provider outage -> switch model once
                self.health["model_errors"][self.cfg.model_id] = \
                    self.health["model_errors"].get(self.cfg.model_id, 0) + 1
                fallback = self._failover_candidate(e)
                if fallback is None:
                    raise
                old_id = self.cfg.model_id
                self.cfg.model_id = fallback
                self._failed_over = True
                self.health["failovers"] += 1
                self.log.append("provider.failover",
                                {"from": old_id, "to": fallback,
                                 "error": str(e)[:140]},
                                actor="kernel")
                label = model_by_id(fallback)
                self._push_status(
                    f"⚠ provider failover → {label.label if label else fallback}")
                result = _attempt()
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

    # -- focus mode (deep work) ---------------------------------------------------

    def focus_continue(self, last_turn: Turn, remaining: int) -> str | None:
        """FOCUS MODE brain: decide whether the deep-work loop should keep
        going, and if so, return the next continuation prompt.

        Returns None when focus should stop. Every decision is sealed in
        the event log (focus.tick / focus.stop) — autonomous work is never
        invisible (A7).

        Stop conditions (mechanical, rung 1):
          * the turn errored, was cancelled, or the budget paused it
          * the goal closed / every clause is proven or waived
          * distance stalled: no improvement for 3 consecutive ticks
          * no goal and the agent produced a final answer with no tool
            work twice in a row (it believes itself done)
        """
        goal = self.goal.status()

        def _stop(reason: str) -> None:
            self.log.append("focus.stop", {"reason": reason},
                            actor="kernel")
            self._focus_history.clear()

        if last_turn.error:
            _stop(f"turn ended with: {last_turn.error[:80]}")
            return None

        if goal.active:
            open_clauses = [c for c in goal.clauses
                            if c.state in ("OPEN", "REGRESSED")]
            if not open_clauses or goal.closed_state in TERMINAL_STATES:
                _stop("goal achieved — every clause proven or waived")
                return None
            distance = goal.distance
            self._focus_history.append(distance)
            recent = self._focus_history[-4:]
            if len(recent) >= 4 and all(
                    recent[i] >= recent[i - 1] - 1e-9
                    for i in range(1, len(recent))):
                _stop(f"stalled — distance stuck at {distance:.2f} for 3 "
                      "ticks; needs a human decision")
                return None
            focus_clause = goal.focus or open_clauses[0].id
            self.log.append("focus.tick",
                            {"distance": distance,
                             "remaining": remaining,
                             "focus": focus_clause},
                            actor="kernel")
            return (f"CONTINUE — deep-work mode ({remaining} turns left). "
                    f"Goal: {goal.statement}. Current focus: clause "
                    f"{focus_clause}. Distance to done: {distance:.2f}. "
                    f"Do NOT repeat completed work and do NOT re-prove "
                    f"PROVEN clauses. Take the next concrete step that "
                    f"moves clause {focus_clause} forward, then verify it.")

        # no active goal — stop when the agent is clearly done
        worked = bool(last_turn.tools)
        self._focus_history.append(1.0 if worked else 0.0)
        tail = self._focus_history[-2:]
        if len(tail) == 2 and tail == [0.0, 0.0]:
            _stop("agent answered without further tool work — done")
            return None
        self.log.append("focus.tick", {"remaining": remaining},
                        actor="kernel")
        return (f"CONTINUE — deep-work mode ({remaining} turns left). "
                f"Review what is done, verify it against reality, and "
                f"complete whatever is still missing. Do not repeat "
                f"completed work.")

    def _push_status(self, text: str) -> None:
        """Stream a live status line into the current turn's UI (if any).
        Used by long-running tools (crew) so subagent work is never
        invisible."""
        cb = self._turn_status
        if cb is None:
            return
        try:
            cb(text)
        except Exception:
            pass

    def _attribute(self, tool_name: str) -> tuple[str | None, str | None]:
        """§38.1 total attribution. Returns (clause_id, orphan_reason).

        With an active goal, every action binds to the focus clause (the
        highest-gravity open clause). If no open clause exists, the action
        is an orphan and is rejected before execution — EXCEPT once the
        contract is closed: a terminal contract (§42) no longer demands
        attribution, or the agent could never run another command after a
        goal completed. Attribution only ever gates an OPEN contract."""
        goal = self.goal.status()
        if not goal.active:
            return None, None  # normal mode: no attribution required
        if goal.closed_state in TERMINAL_STATES:
            return None, None  # the contract is settled — no orphan gate
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

    # -- enterprise: workflows ----------------------------------------------------

    def _workflow_step(self, item: dict) -> dict:
        """Execute ONE workflow step on the Crew (real subagent). Steps
        within a phase arrive sequentially here; the crew runs each in
        the background and we wait for its verdict."""
        try:
            agent = self.crew.spawn(
                item["task"], role=item.get("role", "coder"),
                context=self.scout_context(),
                read_only=self.autonomy <= 1,
                model_id=str(item.get("model", "") or ""))
        except CrewError as e:
            return {"status": "error", "summary": str(e)}
        self.crew.wait([agent.id], timeout=240.0)
        status = ("done" if agent.state == "done"
                  else "blocked" if agent.state == "blocked"
                  else "error")
        summary = agent.summary or agent.error or ""
        try:
            self.crew.close(agent.id)
        except CrewError:
            pass
        return {"status": status, "summary": summary}

    def export_report(self, fmt: str = "md") -> Path:
        """Write the enterprise audit report (md or html) to the cwd."""
        title = f"FullAgent session {self.session_id}"
        if fmt == "html":
            text, suffix = export_html(self.log, title), ".html"
        else:
            text, suffix = export_markdown(self.log, title), ".md"
        path = Path.cwd() / f"fullagent-report-{self.session_id}{suffix}"
        path.write_text(text)
        self.log.append("report.exported",
                        {"path": str(path), "format": fmt},
                        actor="human")
        return path

    def get_forecast(self) -> str:
        return format_forecast(forecast(self.log))

    # -- enterprise: provider health + failover ------------------------------------

    def _failover_candidate(self, err: APIError) -> str | None:
        """Pick the model to fail over to after a provider failure, or
        None if failover is impossible/not allowed right now."""
        if self._failed_over:
            return None  # at most one failover per turn
        status = getattr(err, "status", None)
        if status is not None and status not in _FAILOVER_STATUSES:
            return None  # 4xx (except timeouts) are not provider outages
        explicit = str(self.cfg.extra.get("failover_model", "") or "")
        if explicit and explicit != self.cfg.model_id \
                and model_by_id(explicit) is not None:
            return explicit
        # auto-pick: same provider first, then any capable model
        current = self.model
        from .config import MODELS
        same_provider = [m for m in MODELS
                         if m.provider == current.provider
                         and m.id != current.id
                         and m.supports_tools >= current.supports_tools]
        others = [m for m in MODELS
                  if m.provider != current.provider
                  and m.id != current.id
                  and m.supports_tools >= current.supports_tools]
        for m in same_provider + others:
            return m.id
        return None

    # -- enterprise: turn scorecard ---------------------------------------------------

    def _score_turn(self, turn: Turn) -> dict:
        """Deterministic quality metrics for the finished turn — sealed
        as turn.scorecard. No model judgement, only measured facts."""
        tool_calls = len(turn.tools)
        errors = sum(1 for t in turn.tools
                     if t.status in ("error", "blocked", "denied"))
        writes: dict[str, int] = {}
        for t in turn.tools:
            if t.name in ("write_file", "edit_file") and t.status == "done":
                path = str(t.args.get("path", ""))
                if path:
                    writes[path] = writes.get(path, 0) + 1
        rework = sum(1 for n in writes.values() if n > 1)
        verdicts = [e for e in self.log.events()
                    if e.type == "judge.verdict"
                    and e.seq > self._turn_start_seq]
        verified = sum(1 for v in verdicts if v.data.get("passed"))
        score = max(0, min(100, 100 - 20 * errors - 10 * rework))
        card = {"tool_calls": tool_calls, "errors": errors,
                "rework_files": rework, "verified": verified,
                "verdicts": len(verdicts), "score": score,
                "duration": round(turn.duration, 2)}
        turn.scorecard = card
        self.log.append("turn.scorecard", card, actor="kernel")
        return card

    # -- enterprise: compaction digests -----------------------------------------------

    def _digest_messages(self, msgs: list[dict]) -> str:
        """Deterministic one-line digest of a turn about to be compacted
        away — the knowledge survives even when the tokens don't."""
        tools_used: list[str] = []
        files: list[str] = []
        problems: list[str] = []
        for m in msgs:
            for tc in (m.get("tool_calls") or []):
                name = (tc.get("function") or {}).get("name", "")
                if name and name not in tools_used:
                    tools_used.append(name)
            if m.get("role") == "tool":
                content = str(m.get("content", ""))
                import re as _re
                for match in _re.finditer(
                        r"OK: (?:wrote \d+ chars to|replaced .*? in) (\S+)",
                        content):
                    f = match.group(1)
                    if f not in files:
                        files.append(f)
                if content.startswith("ERROR:"):
                    problems.append(content[6:80])
        parts = []
        if tools_used:
            parts.append("tools: " + ", ".join(tools_used[:6]))
        if files:
            parts.append("wrote: " + ", ".join(files[:5]))
        if problems:
            parts.append("hit: " + problems[0])
        return "; ".join(parts)

    # -- enterprise: notifications ------------------------------------------------------

    def _flush_notifications(self) -> None:
        """Emit any NOTIFY_EVENTS sealed since the last flush."""
        if not self.notifier.sink:
            self._notify_seq = self.log.head()
            return
        for ev in self.log.events():
            if ev.seq <= self._notify_seq:
                continue
            if ev.type in NOTIFY_EVENTS:
                self.notifier.emit(ev.type, ev.data)
        self._notify_seq = self.log.head()

    # -- enterprise: session resume -------------------------------------------------------

    def sessions_catalog(self) -> list[dict]:
        """Every branch that carries a session — for /resume."""
        catalog = []
        for branch in self.log.branches():
            events = self.log.events(branch)
            session_id = ""
            started = 0.0
            for e in events:
                if e.type == "session.start":
                    session_id = str(e.data.get("session_id", ""))
                    started = e.ts
            catalog.append({"branch": branch, "session_id": session_id,
                            "events": len(events),
                            "head": self.log.head(branch),
                            "started": started})
        catalog.sort(key=lambda c: -c["started"])
        return catalog

    def resume_session(self, branch: str) -> int:
        """Checkout a branch and rebuild the conversation from the fold.
        The full history (tool calls, verdicts) stays in the log; the
        model-visible context is rebuilt from user/assistant messages."""
        if branch not in self.log.branches():
            raise ValueError(
                f"unknown branch {branch!r} — known: "
                + ", ".join(self.log.branches()))
        self.log.checkout(branch)
        st = fold(self.log)
        session_id = ""
        for e in self.log.events():
            if e.type == "session.start":
                session_id = str(e.data.get("session_id", "")) or session_id
        self.session_id = session_id or self.session_id
        self.log.session = self.session_id
        self.messages = []
        self._reseat_system_prompt()
        self.messages.extend(st.messages)
        self.turns = []
        self._focus_history.clear()
        self.log.append("session.resumed",
                        {"branch": branch, "session_id": self.session_id,
                         "messages": len(st.messages)},
                        actor="human")
        return len(st.messages)

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

    def _register_crew_tools(self) -> None:
        """Codex-style persistent subagents: spawn / send / wait / close /
        resume. Crew agents run in the BACKGROUND and keep their full
        conversation, so follow-ups never start from zero."""
        crew = self.crew

        def spawn_agent(task: str, role: str = "coder", name: str = "",
                        read_only: bool = False, model: str = "") -> str:
            if not str(task or "").strip():
                return "ERROR: task must be a non-empty string"
            try:
                agent = crew.spawn(
                    task, role=role or "coder", name=name,
                    context=self.scout_context(),
                    read_only=read_only or self.autonomy <= 1,
                    model_id=str(model or ""))
            except CrewError as e:
                return f"ERROR: {e}"
            model_note = (f" on model '{agent.model_id}'"
                          if agent.model_id else "")
            self._push_status(f"⚡ crew · {agent.nickname} ({agent.role}) launched")
            return (f"✓ subagent [{agent.id}] '{agent.nickname}' "
                    f"({agent.role}){model_note} is RUNNING in the "
                    f"background.\n"
                    f"Collect with wait_for_agents, iterate with "
                    f"send_to_agent, retire with close_agent.\n"
                    f"{crew.format_status()}")

        def send_to_agent(id: str, message: str,
                          interrupt: bool = False) -> str:
            try:
                agent = crew.send(id, message, interrupt=interrupt)
            except CrewError as e:
                return f"ERROR: {e}"
            return (f"✓ message delivered to [{agent.id}] "
                    f"'{agent.nickname}' — state: {agent.state}. "
                    f"wait_for_agents collects the reply.")

        def wait_for_agents(ids: list | None = None,
                            timeout: float = 120.0) -> str:
            try:
                timeout = max(1.0, min(float(timeout or 120.0), 600.0))
            except (TypeError, ValueError):
                timeout = 120.0
            clean_ids = None
            if isinstance(ids, list) and ids:
                clean_ids = [str(i) for i in ids if str(i).strip()]
            try:
                targets = ([crew.get(i) for i in clean_ids]
                           if clean_ids else crew.list())
                targets = [a for a in targets if a is not None]
                if not targets:
                    return "ERROR: no matching subagents — spawn one first"
            except CrewError as e:
                return f"ERROR: {e}"
            import time as _time
            deadline = _time.monotonic() + timeout
            while _time.monotonic() < deadline:
                running = sum(1 for a in targets if a.state == "running")
                if running == 0:
                    break
                self._push_status(
                    f"⚡ crew waiting · {running}/{len(targets)} running")
                _time.sleep(0.4)
            states = {a.id: a.state for a in targets}
            still_running = [i for i, s in states.items() if s == "running"]
            lines = [f"crew states: {states}"]
            if still_running:
                lines.append(f"still running after {timeout:.0f}s: "
                             + ", ".join(still_running)
                             + " — wait again or proceed without them")
            lines.append(crew.format([crew.get(i) for i in states
                                      if crew.get(i)]))
            return "\n".join(lines)

        def close_agent(id: str) -> str:
            try:
                agent = crew.close(id)
            except CrewError as e:
                return f"ERROR: {e}"
            return (f"✓ [{agent.id}] '{agent.nickname}' closed. "
                    f"resume_agent brings it back with full context.")

        def resume_agent(id: str) -> str:
            try:
                agent = crew.resume(id)
            except CrewError as e:
                return f"ERROR: {e}"
            return (f"✓ [{agent.id}] '{agent.nickname}' resumed "
                    f"(state: {agent.state}) — send_to_agent works again.")

        def crew_status() -> str:
            return crew.format_status()

        _STR = {"type": "string"}
        self.tools["spawn_agent"] = Tool(
            "spawn_agent",
            "Spawn ONE persistent background subagent (Codex-style). "
            "Returns IMMEDIATELY with the agent id while it runs "
            "serially in the background queue — you stay responsive. "
            "Agents run ONE AT A TIME, never in parallel. Roles: coder, "
            "researcher, tester, reviewer, analyst. The agent keeps its "
            "full conversation: follow up with send_to_agent, collect "
            "with wait_for_agents. Use for independent workstreams you "
            "want to iterate on, not fire-and-forget batches.",
            {"type": "object", "properties": {
                "task": _STR,
                "role": _STR,
                "name": {"type": "string",
                         "description": "optional nickname"},
                "read_only": {"type": "boolean"},
                "model": {"type": "string",
                          "description": "optional model id override for "
                                         "this subagent only"}},
                "required": ["task"]},
            spawn_agent)
        self.tools["send_to_agent"] = Tool(
            "send_to_agent",
            "Send a follow-up message into a living subagent's context "
            "(its full history is preserved). Works on done/blocked/error "
            "agents immediately; queues for running agents. Use to iterate "
            "on a subagent's output instead of re-spawning.",
            {"type": "object", "properties": {
                "id": _STR, "message": _STR,
                "interrupt": {"type": "boolean"}},
                "required": ["id", "message"]},
            send_to_agent)
        self.tools["wait_for_agents"] = Tool(
            "wait_for_agents",
            "Block until the named subagents finish (or all, if no ids) "
            "and return their full reports. Call this when you need the "
            "results of spawned background agents.",
            {"type": "object", "properties": {
                "ids": {"type": "array", "items": _STR,
                        "description": "agent ids; omit for all"},
                "timeout": {"type": "number"}},
                "required": []},
            wait_for_agents)
        self.tools["close_agent"] = Tool(
            "close_agent",
            "Retire a subagent (it keeps its history; resume_agent can "
            "bring it back). Close agents you are done with.",
            {"type": "object", "properties": {"id": _STR},
                "required": ["id"]},
            close_agent)
        self.tools["resume_agent"] = Tool(
            "resume_agent",
            "Bring a closed subagent back so it can receive follow-up "
            "messages again.",
            {"type": "object", "properties": {"id": _STR},
                "required": ["id"]},
            resume_agent)
        self.tools["crew_status"] = Tool(
            "crew_status",
            "Show all crew subagents and their states (running / done / "
            "error / closed).",
            {"type": "object", "properties": {}, "required": []},
            crew_status)

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

    def _run_worker_serial(self, tasks: list[dict],
                           read_only: bool = False) -> list[WorkerReport]:
        """Execute subagent tasks ONE AT A TIME through the Crew (serial).

        The parallel Team fan-out no longer exists; every worker is a
        persistent crew subagent run to completion in order. Returns
        WorkerReports in input order; a failing worker yields an error
        report, never an exception."""
        reports: list[WorkerReport] = []
        ro = read_only or self.autonomy <= 1
        for t in tasks:
            task = str(t.get("task", "") or "").strip()
            role = str(t.get("role", "") or "").strip() or None
            if not task:
                continue
            try:
                agent = self.crew.spawn(
                    task, role=role or "coder",
                    context=self.scout_context(),
                    read_only=ro,
                    model_id=str(t.get("model", "") or ""))
                while agent.state == "running":
                    time.sleep(0.25)
            except Exception as e:
                reports.append(WorkerReport(
                    task=task, role=role or "coder", status="error",
                    error=f"{type(e).__name__}: {e}"))
                continue
            status = agent.state if agent.state in (
                "done", "blocked", "error") else (
                "error" if agent.error else "done")
            reports.append(WorkerReport(
                task=agent.task, role=agent.role, status=status,
                summary=(agent.summary[:MAX_SUMMARY_CHARS]
                         + " …[truncated]" if len(agent.summary)
                         > MAX_SUMMARY_CHARS else agent.summary),
                files_touched=list(agent.files_touched),
                tool_calls=agent.tool_calls,
                tokens_in=agent.tokens_in, tokens_out=agent.tokens_out,
                error=agent.error, elapsed_ms=agent.elapsed_ms))
        return reports

    def _daemon_step(self, task: str) -> str:
        """Run one daemon mission step. A step is executed as a read-only
        researcher probe by default — the daemon advances missions without
        mutating the world unless a write is explicitly part of the task.
        Returns 'ERROR: …' on failure so the daemon can retry/block."""
        try:
            reports = self._run_worker_serial(
                [{"task": task, "role": "researcher"}], read_only=True)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
        if not reports:
            return "ERROR: daemon step produced no report"
        r = reports[0]
        if r.status == "error":
            return f"ERROR: {r.error or 'step failed'}"
        return f"OK: {r.summary[:400]}"

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

    # -- v5 advanced subsystem bridges ---------------------------------------

    def _compile_wave(self, wave: list[dict]) -> list[dict]:
        """Executor for the Intent Compiler: one compiled wave runs as a
        serial batch of crew subagents (one at a time)."""
        tasks = [{"task": it["task"], "role": it["role"]} for it in wave]
        self._push_status(f"⚙ compiled wave · {len(tasks)} item(s)")
        reports = self._run_worker_serial(
            tasks, read_only=self.autonomy <= 1)
        return [r.to_dict() for r in reports]

    def _evolution_mutator(self, role: str, incumbent: str, k: int
                           ) -> list[str]:
        """LLM mutator: k candidate rewrites of a role brief."""
        messages = [
            {"role": "system", "content":
                "You improve agent role briefs. Output ONLY candidate "
                "briefs separated by lines with exactly --- . No prose "
                "around them."},
            {"role": "user", "content":
                f"Current brief for the '{role}' worker:\n{incumbent}\n\n"
                f"Write {k} improved variants. Each must be one "
                f"paragraph, more specific and actionable than the "
                f"original, keeping the same scope."}]
        result = chat_blocking(self.provider, self.model, self.effort,
                               messages, None, timeout=120.0)
        parts = [p.strip() for p in
                 (result.content or "").split("\n---")]
        return [p for p in parts if len(p) > 40][:k]

    def _evolution_evaluator(self, role: str, brief: str
                             ) -> tuple[str, float]:
        """Real evaluation: run the fixed benchmark with the candidate
        brief as the worker's system prompt; score deterministically from
        the reply's structure (STATUS/SUMMARY contract + substance)."""
        from . import systemprompt
        system = systemprompt.WORKER.format(role_brief=brief)
        result = chat_blocking(
            self.provider, self.model, self.effort,
            [{"role": "system", "content": system},
             {"role": "user",
              "content": default_benchmark(role)}],
            None, timeout=180.0)
        content = result.content or ""
        status, summary = parse_worker_final(content)
        score = 0.0
        if status == "done":
            score += 0.5
        if summary and summary != content.strip():
            score += 0.2                       # followed the format
        if len(summary) > 60:
            score += 0.2                       # real substance
        if any(ch.isdigit() for ch in summary) or "/" in summary:
            score += 0.1                       # concrete paths/numbers
        return content, min(1.0, score)

    def _debate_speaker(self, model_id: str, prompt: str) -> str:
        """Route one tournament turn to its participant model."""
        m = model_by_id(model_id) or self.model
        provider = PROVIDERS.get(m.provider, self.provider)
        result = chat_blocking(
            provider, m, self.effort,
            [{"role": "system", "content":
                "You are a participant in an answer tournament. Follow "
                "the instructions exactly and concisely."},
             {"role": "user", "content": prompt}],
            None, timeout=120.0)
        return result.content or ""

    def _market_exec(self, task: str, role: str) -> dict:
        """Market executor: the awarded contract runs as a real serial
        crew worker; its report settles the auction."""
        reports = self._run_worker_serial(
            [{"task": task, "role": role}],
            read_only=self.autonomy <= 1)
        if not reports:
            return {"status": "error", "summary": "no report returned",
                    "tool_calls": 0}
        r = reports[0]
        return {"status": r.status,
                "summary": r.summary or r.error or "",
                "tool_calls": r.tool_calls}

    # -- v6 frontier subsystem bridges --------------------------------------

    def _mcts_evaluator(self, assignment: dict) -> float:
        """Rollout scorer for the strategy tree: mechanical role-task
        fit (write verbs want write-capable roles, run verbs want
        runners) — cheap by design, MCTS supplies the exploration."""
        from .team import ROLES
        if not hasattr(self, "_mcts_items"):
            return 0.0
        fit = 0.0
        for i, item in enumerate(self._mcts_items):
            strat = assignment.get(i)
            spec = ROLES.get(strat)
            if spec is None:
                continue
            low = item.lower()
            writes = any(w in low for w in ("write", "build",
                                            "implement", "fix"))
            runs = any(w in low for w in ("run", "test", "measure"))
            tools = set(spec.get("tools", ()))
            if writes and ("write_file" in tools or "edit_file"
                           in tools):
                fit += 1
            if runs and "run_command" in tools:
                fit += 1
        n = max(1, len(self._mcts_items))
        return min(1.0, fit / (n * 1.5))

    def _mesh_exec(self, task: str, role: str) -> dict:
        """A task delegated over the mesh runs as a real local worker."""
        return self._market_exec(task, role)

    def _role_audition(self, draft) -> float:
        """Run the drafted role's own benchmark with its brief; score
        the reply's structure (same yardstick as evolution)."""
        from . import systemprompt
        system = systemprompt.WORKER.format(role_brief=draft.brief)
        result = chat_blocking(
            self.provider, self.model, self.effort,
            [{"role": "system", "content": system},
             {"role": "user", "content": draft.benchmark}],
            None, timeout=180.0)
        status, summary = parse_worker_final(result.content or "")
        score = 0.5 if status == "done" else 0.0
        if summary and len(summary) > 60:
            score += 0.4
        return min(1.0, score)

    def _ci_runner(self, tests: list[str]) -> tuple[bool, str]:
        """Run the impacted tests through the real shell."""
        from .tools import run_command
        cmd = "python -m pytest -q " + " ".join(tests)
        out = run_command(cmd, timeout=300)
        return ("exit code: 0" in out, out)

    def _dual_fast(self, question: str) -> str:
        """System 1: one cheap direct model call."""
        result = chat_blocking(
            self.provider, self.model, self.effort,
            [{"role": "system", "content":
                "Answer directly and concisely."},
             {"role": "user", "content": question}],
            None, timeout=60.0)
        return result.content or ""

    def _dual_slow(self, question: str) -> str:
        """System 2: the deliberate stack — a full debate tournament."""
        result = self.debate.run(question, rounds=2)
        return (f"[verified via {len(self.debate.models)}-model "
                f"debate, champion {result.champion_model}] "
                f"{result.verdict}")

    def _race_runner(self, strategy: dict, task: str, cancel) -> str:
        """One racing universe: a real serial crew worker under its
        strategy."""
        if cancel.is_set():
            return "CANCELLED"
        reports = self._run_worker_serial(
            [{"task": f"{strategy['instructions']}\n\nTASK: {task}",
              "role": strategy["role"]}],
            read_only=self.autonomy <= 1)
        if not reports:
            return "ERROR: no report"
        r = reports[0]
        return f"STATUS: {r.status.upper()}\nSUMMARY: {r.summary or r.error}"

    def _race_verify(self, task: str, result: str) -> bool:
        return "STATUS: DONE" in result and "ERROR" not in result

    def _repair_reseed_prompts(self) -> bool:
        """Re-seal every vault prompt — cures prompt drift."""
        for name in self.mastermind.vault.names():
            try:
                self.mastermind.vault.resolve(name)
            except Exception:
                pass
        return True

    def _repair_consolidate(self) -> bool:
        """Loop alerts: consolidate the brain (dedupe/decay stale
        patterns that drive repetition)."""
        self.brain.sleep()
        return True

    def _repair_warm_caches(self) -> bool:
        """Latency: rebuild the shared context so speculative caches
        refill."""
        self.scout_context()
        return True

    def _register_advanced_tools(self) -> None:
        """Give the model the v5 powers: intent compilation, debates,
        the task market, cognitive recall, and causal self-inspection."""
        compiler = self.compiler
        debate = self.debate
        market = self.market
        brain = self.brain
        theater = self.theater

        def compile_and_run(goal: str, read_only: bool = False) -> str:
            """Compile the goal into an optimized wave plan, then execute
            it wave by wave, each worker serially."""
            goal = str(goal or "").strip()
            if not goal:
                return "ERROR: goal must be a non-empty string"
            plan = compiler.compile(goal)
            out = [compiler.format(plan)]
            if not plan.waves:
                return "\n".join(out + ["ERROR: compilation produced no "
                                       "work items — rephrase the goal"])
            result = compiler.execute(plan)
            out.append(f"\nEXECUTED: {result['items']} items in "
                       f"{result['waves']} waves · {result['done']} done · "
                       f"{result['blocked']} blocked · {result['error']} "
                       f"error")
            return "\n".join(out)

        def run_debate(question: str, rounds: int = 3) -> str:
            """Run a multi-model debate tournament on a question and
            return the calibrated verdict."""
            question = str(question or "").strip()
            if not question:
                return "ERROR: question must be a non-empty string"
            try:
                rounds = max(1, min(int(rounds), 3))
            except (TypeError, ValueError):
                rounds = 3
            self._push_status(f"⚔ debate · {len(debate.models)} models, "
                              f"{rounds} rounds")
            result = debate.run(question, rounds=rounds)
            return debate.format(result)

        def run_market(tasks: list) -> str:
            """Put tasks on the market: every specialist role bids, the
            best bid wins and executes the contract for real."""
            if not isinstance(tasks, list) or not tasks:
                return "ERROR: tasks must be a non-empty list of strings"
            clean = [str(t).strip() for t in tasks if str(t).strip()]
            if not clean:
                return "ERROR: no valid tasks"
            self._push_status(f"💰 market · {len(clean)} contract(s) "
                              f"up for auction")
            contracts = market.run(clean)
            return market.format(contracts)

        def brain_recall(query: str) -> str:
            """Recall knowledge from the cognitive memory (forgetting
            curve ranks what is still alive)."""
            query = str(query or "").strip()
            if not query:
                return brain.format_stats()
            block = brain.context_block(query, k=5)
            return block or "no live memories match that query"

        def inspect_why(seq: int) -> str:
            """Causal self-inspection: why did the agent do the event at
            this seq? Full evidence chain from the sealed envelope."""
            try:
                seq = int(seq)
            except (TypeError, ValueError):
                return "ERROR: seq must be an integer"
            return theater.why(seq)

        self.tools["compile_and_run"] = Tool(
            "compile_and_run",
            "COMPILE the goal through the Intent Compiler: it is drafted "
            "into typed work items, then deterministic optimizer passes "
            "(dedupe, dead-dependency pruning, topological layering, "
            "write-lock scheduling) build ordered waves, and each wave "
            "executes as REAL serial workers. Best for multi-step goals "
            "where order matters.",
            {"type": "object", "properties": {
                "goal": {"type": "string"}, "read_only": {"type":
                                                          "boolean"}},
                "required": ["goal"]},
            compile_and_run)
        self.tools["run_debate"] = Tool(
            "run_debate",
            "Run a MULTI-MODEL debate tournament on a hard question: "
            "blind proposals, mutual critique, revision, then calibrated "
            "cluster fusion produces the verdict (with dissent). Better "
            "than any single model on contested questions.",
            {"type": "object", "properties": {
                "question": {"type": "string"},
                "rounds": {"type": "integer"}},
                "required": ["question"]},
            run_debate)
        self.tools["run_market"] = Tool(
            "run_market",
            "Put tasks on the AGENT TASK MARKET: every specialist role "
            "bids (capability × trust ÷ pace), the auctioneer awards "
            "each contract to the best bidder, and the winner executes "
            "it as a real worker. Trust updates from every outcome — the "
            "market learns which roles deliver.",
            {"type": "object", "properties": {
                "tasks": {"type": "array", "items": {"type": "string"}}},
                "required": ["tasks"]},
            run_market)
        self.tools["brain_recall"] = Tool(
            "brain_recall",
            "Recall knowledge from the cognitive memory — the four-store "
            "brain with a forgetting curve ranks what is still alive and "
            "relevant. Use before re-researching anything; it may "
            "already be known. With an empty query, returns brain stats.",
            {"type": "object", "properties": {
                "query": {"type": "string"}},
                "required": ["query"]},
            brain_recall)
        self.tools["inspect_why"] = Tool(
            "inspect_why",
            "Causal self-inspection: pass an event seq (shown in tool "
            "results and the event log) and get the full WHY chain — "
            "which user message, goal clause, or prior tool call caused "
            "it. Use it to explain or audit the agent's own behaviour.",
            {"type": "object", "properties": {
                "seq": {"type": "integer"}},
                "required": ["seq"]},
            inspect_why)

        # -- v6 frontier tools ------------------------------------------------

        def verify_plan(goal: str) -> str:
            """Compile a goal into a plan, then formally verify it
            against temporal safety properties before any execution."""
            goal = str(goal or "").strip()
            if not goal:
                return "ERROR: goal must be a non-empty string"
            plan = compiler.compile(goal)
            r = self.formal.verify_plan(plan.waves)
            head = (f"FORMAL VERIFICATION — {'PASS ✓' if r.ok else
                    'REJECTED ✗'} "
                    f"({r.checked} trace(s) checked)")
            return "\n".join([head, compiler.format(plan)]
                             + [f"  ⚠ {v['property']}: {v['why']}"
                                for v in r.violations])

        def mcts_solve(goal: str, strategies: list | None = None
                       ) -> str:
            """Tree-search the best strategy assignment: split the goal
            into work items, search (MCTS) which approach should handle
            each, return the winning assignment."""
            goal = str(goal or "").strip()
            if not goal:
                return "ERROR: goal must be a non-empty string"
            items = [s.strip() for s in goal.split(";") if s.strip()]
            strats = ([str(s) for s in strategies if str(s).strip()]
                      if isinstance(strategies, list) and strategies
                      else ["coder", "architect", "debugger", "tester"])
            self._mcts_items = items
            report = self.mcts.search(items, strats,
                                      iterations=120, deadline_s=20.0)
            lines = [f"MCTS — best score {report.best_score:.2f} in "
                     f"{report.iterations} iterations "
                     f"({report.nodes} nodes)"]
            for i, item in enumerate(items):
                lines.append(f"  [{report.best_assignment.get(i, '?')}] "
                             f"{item[:70]}")
            return "\n".join(lines)

        def synthesize_tool(name: str, description: str,
                            examples: list) -> str:
            """Write a NEW deterministic Python tool: draft from the
            spec, validate the AST, test it against every example, and
            register it as a real callable tool."""
            if not isinstance(examples, list) or not examples:
                return ("ERROR: examples must be a non-empty list of "
                        '{"args": {...}, "want": ...}')
            spec = SynthSpec(name=str(name or ""),
                             description=str(description or ""),
                             examples=examples)
            result = self.synth.synthesize(spec)
            return ("✓ " + result.reason if result.ok
                    else "ERROR: " + result.reason)

        def predict_impact(path: str) -> str:
            """Predict what breaks if a file changes — learned from
            dependencies AND real historical breakage rates."""
            impact = self.world.predict_impact(str(path or ""))
            return impact.format()

        def race_strategies(task: str) -> str:
            """Try 3 strategy universes one at a time on one task; the
            first verified result wins, the rest are skipped."""
            task = str(task or "").strip()
            if not task:
                return "ERROR: task must be a non-empty string"
            self._push_status("⚡ race · 3 universes queued")
            result = self.racer.race(task, timeout=420.0)
            return self.racer.format(result)

        def knowledge_ask(subject: str, predicate: str,
                          at: float | None = None) -> str:
            """Query the bitemporal knowledge graph: the live truth, or
            as of any past moment."""
            hits = self.fabric.query(str(subject), str(predicate), at)
            if not hits:
                return f"no live fact for {subject} {predicate}"
            return "\n".join(f.to_dict().__str__() for f in hits)

        self.tools["verify_plan"] = Tool(
            "verify_plan",
            "FORMALLY VERIFY a plan: compile the goal, then model-check "
            "every execution the plan permits against temporal safety "
            "properties (snapshot-before-write, write serialisation, "
            "verification-after-write). A violated plan is REJECTED "
            "with a counterexample — use before executing risky plans.",
            {"type": "object", "properties": {
                "goal": {"type": "string"}},
                "required": ["goal"]},
            verify_plan)
        self.tools["mcts_solve"] = Tool(
            "mcts_solve",
            "Monte Carlo Tree Search over strategies: split a goal by "
            "semicolons into work items and search which approach "
            "(coder/architect/debugger/tester or custom strategy ids) "
            "should handle each — UCB1 exploration, rollout scoring, "
            "best assignment returned.",
            {"type": "object", "properties": {
                "goal": {"type": "string",
                         "description": "items separated by ;"},
                "strategies": {"type": "array",
                               "items": {"type": "string"}}},
                "required": ["goal"]},
            mcts_solve)
        self.tools["synthesize_tool"] = Tool(
            "synthesize_tool",
            "WRITE A NEW TOOL: given a snake_case name, a description, "
            "and example cases ({'args': {...}, 'want': expected}), a "
            "pure deterministic Python function is drafted, AST-"
            "validated, example-tested, and registered as a REAL "
            "callable tool. Use when no existing tool fits.",
            {"type": "object", "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "examples": {"type": "array",
                             "items": {"type": "object"}}},
                "required": ["name", "description", "examples"]},
            synthesize_tool)
        self.tools["predict_impact"] = Tool(
            "predict_impact",
            "Predict what breaks if a file changes: learned dependency "
            "edges × historical breakage rates, ranked by probability. "
            "Use BEFORE risky edits to see the blast radius.",
            {"type": "object", "properties": {
                "path": {"type": "string"}},
                "required": ["path"]},
            predict_impact)
        self.tools["race_strategies"] = Tool(
            "race_strategies",
            "RACE three strategy universes one at a time on one task "
            "(direct / careful / split approaches); the first "
            "verified result WINS and the rest are skipped. Use for "
            "hard problems where you are unsure which approach works.",
            {"type": "object", "properties": {
                "task": {"type": "string"}},
                "required": ["task"]},
            race_strategies)
        self.tools["knowledge_ask"] = Tool(
            "knowledge_ask",
            "Query the bitemporal knowledge graph — facts with validity "
            "windows. Ask for the live truth or 'as of' any past time "
            "('at' = unix timestamp). Use it to remember project facts "
            "that change over time.",
            {"type": "object", "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "at": {"type": "number"}},
                "required": ["subject", "predicate"]},
            knowledge_ask)

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

    # -- shared subagent context ------------------------------------------------

    def scout_context(self, max_chars: int = 4000) -> str:
        """Shared read-only context handed to every subagent (each gets
        a fresh minimal context — its task plus a little shared state).
        Subagents get little else, so anything they must reason about has
        to be in here: where we are, what the goal is, what is known."""
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
