"""SUPERCOMPUTER — an 8-core parallel mission machine inside FullAgent.

`/on` boots a real machine in the terminal: EIGHT cores that think at the
same time, on the same mission, each owning its own slice — the shape you
want for frontier work (mission software, national-lab tooling, large
research systems), not a toy fan-out.

    ┌ RECON ─────┐  8 cores map the real world + the real repo, in parallel
    ├ RELAY ─────┤  the plan is escalated v1→v8, core by core, WHILE the
    │            │  other cores deep-dive in parallel (no idle silicon)
    ├ DEEPDIVE ──┤  8 cores exhaust GitHub / GitLab / npm / PyPI / wiki /
    │            │  docs / papers / advisories for this exact mission
    ├ FUSE ──────┤  one core reads everything and enriches the plan into
    │            │  the enterprise-grade master plan
    ├ BUILD ─────┤  the plan's 8 workstreams are built in parallel
    ├ VERIFY ────┤  8 adversarial sweeps hunt defects and gaps
    └ REPAIR ────┘  defects are sharded back to 8 cores; verify↔repair
                    loops until the board is clean (end to end)

Engineering rules (the same discipline as the rest of FullAgent):
  * TRULY parallel — one OS thread per core, a bounded pool, real
    wall-clock overlap. Writes still serialise through team._WRITE_LOCK
    (invariant I7), so parallel cores can never corrupt the tree.
  * EVENT-SOURCED — every boot, dispatch, tool, verdict and phase change
    is sealed in the Temporal Kernel as a `super.*` event, so a mission
    is replayable and auditable like everything else.
  * SMALL MEMORY — a 4 GB laptop is a first-class target. Each core gets
    a FRESH bounded context per phase, tool output is clipped, live
    activity is a fixed-size deque, and the blackboard is capped. Peak
    RSS is flat in the number of phases; nothing accumulates.
  * REAL WORLD ONLY — cores are handed read/write/web tools and are
    instructed (systemprompt.SUPER_*) to cite real paths and real URLs.
    No simulated output paths exist in this module.
  * OBSERVABLE — `snapshot()` is a cheap, allocation-light view of every
    core (state, activity, tool, steps, tokens, files). The TUI polls it
    to paint the live board, so you watch the machine like live TV.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import systemprompt
from .config import PROVIDERS, model_by_id
from .kernel import EventLog
from .team import ROLES, _WRITE_LOCK, chat_with_retry
from .tools import Tool, build_registry, parse_tool_arguments

# -- machine constants ------------------------------------------------------

CORES = ("ATLAS", "ORION", "VEGA", "LYRA", "NOVA", "KEPLER", "ARGO", "HELIX")
CORE_COUNT = len(CORES)

# Memory discipline for small machines (4 GB target).
MAX_TOOL_CHARS = 4_000       # per tool result inside a core's context
MAX_CONTEXT_MESSAGES = 26    # per core, per phase (fresh context each phase)
MAX_ACTIVITY_LINES = 5       # live-board lines kept per core (deque)
MAX_BLACKBOARD_CHARS = 9_000  # shared blackboard ceiling
MAX_OUTPUT_CHARS = 6_000     # stored per core output

# Step budgets per phase — the token governor of the machine.
PHASE_STEPS = {"recon": 8, "relay": 10, "deepdive": 10, "fuse": 12,
               "build": 44, "verify": 14, "repair": 26}

# Tool whitelist (role) each core wears per phase.
PHASE_ROLE = {"recon": "researcher", "relay": "planner",
              "deepdive": "researcher", "fuse": "architect",
              "build": "coder", "verify": "reviewer", "repair": "coder"}

PHASES = ("recon", "relay", "deepdive", "fuse", "build", "verify", "repair")

MAX_REPAIR_ROUNDS = 3        # verify↔repair loops before we stop and report

# Recon angles — one per core, so eight different questions are asked at
# the same time instead of eight copies of one question.
RECON_ANGLES = (
    "Decompose the mission into components and boundaries. Inspect the "
    "actual working directory to see what already exists.",
    "Find the prior art: what real projects already solve this, what are "
    "they called, where do they live, what did they get right or wrong.",
    "Determine the hard technical core — the algorithms, data structures "
    "and correctness constraints this mission actually stands on.",
    "Determine the concrete implementation surface: languages, runtimes, "
    "entry points, file layout, packaging.",
    "Determine how this will be tested: what proves it works, what the "
    "failure modes are, what a real test suite looks like.",
    "Determine the performance and resource envelope: memory, latency, "
    "throughput, and what breaks it at scale.",
    "Determine the security, safety and abuse surface, and the licence / "
    "compliance constraints on anything we would reuse.",
    "Determine the delivery surface: install, CLI/UX, docs, CI, and what "
    "'shipped' means for this mission.",
)

# Deep-dive sources — the world's corners, one per core, in parallel.
DEEPDIVE_SOURCES = (
    "GitHub: real repositories, their architecture and their issues",
    "GitLab and other forges: real repositories and CI configurations",
    "Package registries (npm, PyPI, crates.io): the actual libraries, "
    "their versions and their maintenance status",
    "Official documentation and specifications of every technology in play",
    "Wikipedia and academic papers: the underlying theory and terminology",
    "Engineering blogs, RFCs and conference talks: how teams did this in "
    "production",
    "Stack Overflow and issue trackers: the failure modes people actually "
    "hit",
    "Security advisories, benchmarks and licence terms for everything we "
    "would adopt",
)

# Verification sweeps — eight different adversarial lenses, in parallel.
VERIFY_SWEEPS = (
    "Architecture conformance: does what exists on disk match the master "
    "plan's architecture and interfaces?",
    "Correctness: read the real code and find logic errors, wrong edge "
    "cases and broken assumptions.",
    "Completeness: every workstream deliverable and done-criterion in the "
    "plan — is it actually present on disk?",
    "Runtime truth: execute what can be executed (imports, --help, tests) "
    "and report the real exit codes.",
    "Interfaces: do the modules actually fit together — signatures, "
    "imports, data formats, CLI shape?",
    "Resources: memory, complexity and anything that will not survive a "
    "small 4 GB machine.",
    "Security and safety: injection, unsafe shell, secrets, unsafe file "
    "handling, dependency risk.",
    "Delivery: docs, install path, packaging, and whether a new user can "
    "actually run this end to end.",
)

CORE_STATES = ("idle", "thinking", "tool", "done", "blocked", "error",
               "cancelled")


# ---------------------------------------------------------------------------
# Core state — one live CPU on the board
# ---------------------------------------------------------------------------

@dataclass
class Core:
    callsign: str
    index: int
    state: str = "idle"
    phase: str = ""
    activity: str = "cold"
    tool: str = ""
    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    files: list[str] = field(default_factory=list)
    output: str = ""
    error: str = ""
    started: float = 0.0
    finished: float = 0.0
    lines: deque = field(
        default_factory=lambda: deque(maxlen=MAX_ACTIVITY_LINES))

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    def note(self, text: str) -> None:
        self.activity = text[:120]
        self.lines.append(text[:120])

    def view(self) -> dict:
        """Allocation-light snapshot for the live board."""
        return {"callsign": self.callsign, "index": self.index,
                "state": self.state, "phase": self.phase,
                "activity": self.activity, "tool": self.tool,
                "steps": self.steps, "tokens": self.tokens_in
                + self.tokens_out, "files": len(self.files),
                "elapsed": round(self.elapsed, 1)}


@dataclass
class Mission:
    objective: str
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    phase: str = ""
    round: int = 0
    plan: str = ""
    plan_version: int = 0
    findings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    defects: list[str] = field(default_factory=list)
    workstreams: list[dict] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    status: str = "running"     # running | complete | stopped | error
    error: str = ""

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


class SupercomputerError(RuntimeError):
    """Invalid machine operation (boot twice, mission while busy)."""


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------

class Supercomputer:
    """Eight cores, one mission, one event log.

    `chat` is injectable for tests: chat(provider, model, effort, messages,
    schemas, timeout) -> StreamResult. Production uses the rate-limit
    hardened chat_with_retry from team.py.
    """

    def __init__(self, log: EventLog, provider, model, effort,
                 cores: int = CORE_COUNT, chat=None,
                 on_update: Callable[[], None] | None = None,
                 registry: dict[str, Tool] | None = None) -> None:
        self.log = log
        self.provider = provider
        self.model = model
        self.effort = effort
        self.n = max(1, min(int(cores), CORE_COUNT))
        self._chat = chat or chat_with_retry
        self.on_update = on_update
        self.online = False
        self.mission: Mission | None = None
        self.history: list[Mission] = []
        self.cores = [Core(callsign=CORES[i], index=i + 1)
                      for i in range(self.n)]
        self._cancel = threading.Event()
        self._busy = threading.Lock()
        self._blackboard: list[str] = []
        self._bb_lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        registry = registry if registry is not None else build_registry()
        self._registry = registry
        self._toolsets: dict[str, dict[str, Tool]] = {}
        for role, spec in ROLES.items():
            self._toolsets[role] = {n_: registry[n_] for n_ in spec["tools"]
                                    if n_ in registry}

    # -- power ---------------------------------------------------------------

    def boot(self) -> str:
        """Power the machine on. Idempotent; cheap (threads are created
        lazily per mission, so an idle machine costs almost nothing)."""
        if self.online:
            return "supercomputer already ONLINE"
        self.online = True
        self._cancel.clear()
        for c in self.cores:
            c.state = "idle"
            c.phase = ""
            c.note("standby")
        self.log.append("super.boot",
                        {"cores": self.n, "model": self.model.id},
                        actor="sovereign")
        self._ping()
        return (f"SUPERCOMPUTER ONLINE — {self.n} cores standing by "
                f"({', '.join(c.callsign for c in self.cores)})")

    def shutdown(self) -> str:
        """Power down: cancel any running mission and release the pool."""
        self._cancel.set()
        pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:  # older Python without cancel_futures
                pool.shutdown(wait=False)
        self.online = False
        if self.mission and self.mission.status == "running":
            self.mission.status = "stopped"
            self.mission.finished = time.time()
        for c in self.cores:
            if c.state in ("thinking", "tool"):
                c.state = "cancelled"
            c.note("powered down")
        self.log.append("super.shutdown", {}, actor="sovereign")
        self._ping()
        return "SUPERCOMPUTER OFFLINE"

    def stop(self) -> str:
        """Abort the running mission but keep the machine online."""
        if not (self.mission and self.mission.status == "running"):
            return "no mission is running"
        self._cancel.set()
        self.log.append("super.mission.stop",
                        {"objective": self.mission.objective[:200]},
                        actor="sovereign")
        return "stop signalled — cores will land at their next checkpoint"

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    # -- the mission ----------------------------------------------------------

    def run_mission(self, objective: str) -> Mission:
        """Run one end-to-end mission across all cores. Blocking; the TUI
        calls it on a worker thread and polls snapshot() to paint."""
        objective = str(objective or "").strip()
        if not objective:
            raise SupercomputerError("a mission needs an objective")
        if not self.online:
            self.boot()
        if not self._busy.acquire(blocking=False):
            raise SupercomputerError(
                "a mission is already running — /on stop to abort it")
        mission = Mission(objective=objective)
        self.mission = mission
        self._cancel.clear()
        self._blackboard.clear()
        self._pool = ThreadPoolExecutor(max_workers=self.n,
                                        thread_name_prefix="super")
        self.log.append("super.mission.start",
                        {"objective": objective[:400], "cores": self.n},
                        actor="sovereign")
        try:
            self._phase_recon(mission)
            self._phase_relay_and_deepdive(mission)
            self._phase_fuse(mission)
            self._phase_build(mission)
            self._phase_verify_repair(mission)
            mission.status = ("stopped" if self._cancel.is_set()
                              else "complete")
        except Exception as e:  # noqa: BLE001 — a mission never crashes the UI
            mission.status = "error"
            mission.error = f"{type(e).__name__}: {e}"
            self.log.append("super.mission.error",
                            {"error": mission.error}, actor="kernel")
        finally:
            mission.finished = time.time()
            pool, self._pool = self._pool, None
            if pool is not None:
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except TypeError:  # older Python without cancel_futures
                    pool.shutdown(wait=False)
            for c in self.cores:
                if c.state in ("thinking", "tool"):
                    c.state = "cancelled"
            self.history.append(mission)
            del self.history[:-10]     # bounded history (small-RAM target)
            self._busy.release()
            self.log.append("super.mission.done",
                            {"status": mission.status,
                             "elapsed": round(mission.elapsed, 1),
                             "files": len(mission.files),
                             "defects": len(mission.defects),
                             "tokens_in": mission.tokens_in,
                             "tokens_out": mission.tokens_out},
                            actor="sovereign")
            self._ping()
        return mission

    # -- phases -----------------------------------------------------------------

    def _phase_recon(self, m: Mission) -> None:
        self._enter(m, "recon")
        jobs = [(self.cores[i], RECON_ANGLES[i % len(RECON_ANGLES)])
                for i in range(self.n)]
        results = self._fan_out(m, "recon", jobs)
        for core, text in results:
            findings = _section(text, "FINDINGS")
            sources = _section(text, "SOURCES")
            m.findings.extend(f"[{core.callsign}] {f}" for f in findings)
            m.sources.extend(sources)
            if findings:
                self._post(f"RECON/{core.callsign}: "
                           + " | ".join(findings[:3]))
        del m.findings[400:]
        del m.sources[200:]

    def _phase_relay_and_deepdive(self, m: Mission) -> None:
        """The plan RELAY climbs v1→vN — every hop on a DIFFERENT core,
        each one strictly deepening the previous plan — while every core
        not currently holding the baton deep-dives the world in parallel.
        No core is ever idle, and no core is ever driven by two threads.
        """
        self._enter(m, "relay")
        pool = self._pool
        assert pool is not None

        # cores 1..n-1 start deep-diving immediately; core 0 takes the baton
        futures: dict = {}
        for i in range(1, self.n):
            core = self.cores[i]
            futures[pool.submit(
                self._run_core, m, core, "deepdive",
                _deepdive_slice(m, DEEPDIVE_SOURCES[i % len(
                    DEEPDIVE_SOURCES)]))] = core

        def harvest(fut) -> None:
            core = futures.pop(fut, None)
            if core is None:
                return
            try:
                text = fut.result()
            except Exception as e:  # noqa: BLE001 — one lane never stops the relay
                core.state, core.error = "error", str(e)[:200]
                return
            m.findings.extend(f"[{core.callsign}] {f}"
                              for f in _section(text, "FINDINGS"))
            m.findings.extend(f"[{core.callsign}] ADOPT: {a}"
                              for a in _section(text, "ADOPT"))
            m.sources.extend(_section(text, "SOURCES"))

        plan = ""
        for step in range(1, self.n + 1):
            if self._cancel.is_set():
                break
            core = self.cores[step - 1]
            # take the baton only once THIS core's deep dive has landed —
            # a core is never driven by two threads at the same time
            for fut in [f for f, c in list(futures.items()) if c is core]:
                harvest(fut)
            text = self._run_core(m, core, "relay",
                                  _relay_slice(m, plan, step),
                                  version=step, fresh_state=True)
            new_plan = _section_block(text, "PLAN") or plan
            if len(new_plan) >= len(plan) * 0.6:   # never accept a collapse
                plan = new_plan
                m.plan = plan
                m.plan_version = step
            self.log.append("super.plan.version",
                            {"version": step, "core": core.callsign,
                             "chars": len(plan)},
                            actor=f"super:{core.callsign}")
            self._post(f"PLAN v{step} by {core.callsign} "
                       f"({len(plan)} chars)")
            # the core that just handed off the baton goes back to work
            if step < self.n:
                futures[pool.submit(
                    self._run_core, m, core, "deepdive",
                    _deepdive_slice(m, DEEPDIVE_SOURCES[
                        (step + self.n) % len(DEEPDIVE_SOURCES)]))] = core
            self._ping()

        for fut in as_completed(list(futures)):
            harvest(fut)
        del m.findings[600:]
        del m.sources[300:]

    def _phase_fuse(self, m: Mission) -> None:
        """One core reads the whole board — plan vN plus every finding —
        and enriches it into the master plan the build runs from."""
        self._enter(m, "fuse")
        core = self.cores[0]
        digest = "\n".join(m.findings[-90:])[:MAX_BLACKBOARD_CHARS]
        slice_text = (
            f"MISSION: {m.objective}\n\n"
            f"INCOMING PLAN (v{m.plan_version}):\n{m.plan[:12000]}\n\n"
            f"RESEARCH FROM ALL CORES:\n{digest}\n\n"
            "Read every finding above and produce the FINAL master plan. "
            "Fold in everything the research proved we need, delete "
            "nothing that was correct, and make each of the 8 workstreams "
            "independently buildable with machine-checkable done-criteria "
            "and explicit file paths.")
        text = self._run_core(m, core, "relay", slice_text,
                              version=m.plan_version + 1, fresh_state=True)
        plan = _section_block(text, "PLAN")
        if len(plan) > len(m.plan) * 0.6:
            m.plan = plan
            m.plan_version += 1
        m.workstreams = _parse_workstreams(m.plan, self.n)
        self.log.append("super.plan.final",
                        {"version": m.plan_version, "chars": len(m.plan),
                         "workstreams": len(m.workstreams)},
                        actor="sovereign")
        self._post(f"MASTER PLAN v{m.plan_version} sealed — "
                   f"{len(m.workstreams)} workstreams")

    def _phase_build(self, m: Mission) -> None:
        self._enter(m, "build")
        if not m.workstreams:
            m.workstreams = [
                {"id": f"W{i + 1}", "title": f"workstream {i + 1}",
                 "role": "coder", "files": "", "criteria": "",
                 "text": f"Implement part {i + 1} of: {m.objective}"}
                for i in range(self.n)]
        jobs = []
        for i in range(self.n):
            ws = m.workstreams[i % len(m.workstreams)]
            jobs.append((self.cores[i], _build_slice(m, ws)))
        results = self._fan_out(m, "build", jobs)
        for core, _text in results:
            for f in core.files:
                if f not in m.files:
                    m.files.append(f)
        self._post(f"BUILD complete — {len(m.files)} file(s) written")

    def _phase_verify_repair(self, m: Mission) -> None:
        for rnd in range(1, MAX_REPAIR_ROUNDS + 1):
            if self._cancel.is_set():
                return
            m.round = rnd
            self._enter(m, "verify")
            jobs = [(self.cores[i], _verify_slice(m, VERIFY_SWEEPS[
                i % len(VERIFY_SWEEPS)])) for i in range(self.n)]
            results = self._fan_out(m, "verify", jobs)
            defects: list[str] = []
            for core, text in results:
                for d in _section(text, "DEFECTS"):
                    if d.strip().lower() in ("none", "none.", "-"):
                        continue
                    defects.append(f"[{core.callsign}] {d}")
                for g in _section(text, "GAPS"):
                    if g.strip().lower() in ("none", "none.", "-"):
                        continue
                    defects.append(f"[{core.callsign}] GAP: {g}")
            m.defects = defects[:200]
            self.log.append("super.verify.round",
                            {"round": rnd, "defects": len(m.defects)},
                            actor="sovereign")
            self._post(f"VERIFY round {rnd}: {len(m.defects)} defect(s)")
            if not m.defects:
                self._post("board is CLEAN — mission end to end")
                return
            if self._cancel.is_set():
                return
            self._enter(m, "repair")
            shards = _shard(m.defects, self.n)
            jobs = [(self.cores[i], _repair_slice(m, shards[i]))
                    for i in range(self.n) if shards[i]]
            self._fan_out(m, "repair", jobs)
            for core in self.cores:
                for f in core.files:
                    if f not in m.files:
                        m.files.append(f)
            self._post(f"REPAIR round {rnd} complete — re-verifying")
        self._post(f"stopped after {MAX_REPAIR_ROUNDS} repair rounds "
                   f"with {len(m.defects)} open defect(s)")

    # -- the parallel primitive -------------------------------------------------

    def _fan_out(self, m: Mission, phase: str,
                 jobs: list[tuple[Core, str]]) -> list[tuple[Core, str]]:
        """Run one job per core AT THE SAME TIME; collect in submit order."""
        pool = self._pool
        if pool is None or not jobs:
            return []
        futures = {pool.submit(self._run_core, m, core, phase, slice_text):
                   core for core, slice_text in jobs}
        out: list[tuple[Core, str]] = []
        for fut in as_completed(list(futures)):
            core = futures[fut]
            try:
                out.append((core, fut.result()))
            except Exception as e:  # noqa: BLE001 — one core never kills the run
                core.state = "error"
                core.error = f"{type(e).__name__}: {e}"[:200]
                core.note(f"error: {core.error}")
                out.append((core, ""))
            self._ping()
        return out

    def _run_core(self, m: Mission, core: Core, phase: str,
                  slice_text: str, version: int = 1,
                  fresh_state: bool = False) -> str:
        """One core's bounded tool loop for one phase.

        The context is built FRESH here and dropped on return — that is
        what keeps peak memory flat on a small machine.
        """
        if self._cancel.is_set():
            core.state = "cancelled"
            return ""
        role = PHASE_ROLE.get(phase, "coder")
        tools = self._toolsets.get(role, {})
        writes = ROLES.get(role, {}).get("writes", False)
        model = self.model
        provider = PROVIDERS.get(model.provider, self.provider)
        schemas = ([t.openai_schema() for t in tools.values()]
                   if model.supports_tools and tools else None)

        core.state = "thinking"
        core.phase = phase
        core.steps = 0
        core.tool = ""
        core.error = ""
        core.started = time.time()
        core.finished = 0.0
        if fresh_state or phase != "repair":
            core.files = []
        core.note(f"{phase}: starting")
        self.log.append("super.core.start",
                        {"core": core.callsign, "phase": phase,
                         "role": role},
                        actor=f"super:{core.callsign}")
        self._ping()

        prompt = systemprompt.super_core(
            core.callsign, core.index, self.n, phase, slice_text,
            self._board(), version=version)
        messages: list[dict] = [{"role": "system", "content": prompt},
                                {"role": "user",
                                 "content": f"MISSION: {m.objective}\n\n"
                                            f"BEGIN YOUR {phase.upper()} "
                                            f"SLICE NOW."}]
        budget = PHASE_STEPS.get(phase, 12)
        result = None
        try:
            for step in range(budget):
                if self._cancel.is_set():
                    core.state = "cancelled"
                    core.note("cancelled")
                    return ""
                core.state = "thinking"
                core.steps = step + 1
                core.note(f"{phase}: reasoning (step {step + 1}/{budget})")
                self._ping()
                result = self._chat(provider, model, self.effort, messages,
                                    schemas, 150.0)
                if result.usage:
                    ti = int(result.usage.get("prompt_tokens", 0) or 0)
                    to = int(result.usage.get("completion_tokens", 0) or 0)
                    core.tokens_in += ti
                    core.tokens_out += to
                    m.tokens_in += ti
                    m.tokens_out += to
                if not result.tool_calls:
                    break
                from .client import assistant_message
                messages.append(assistant_message(
                    result.content, result.tool_calls,
                    getattr(result, "reasoning", "") or ""))
                for tc in result.tool_calls:
                    if self._cancel.is_set():
                        break
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args = parse_tool_arguments(fn.get("arguments"))
                    m.tool_calls += 1
                    core.state = "tool"
                    core.tool = name
                    core.note(f"{name} {_arg_hint(args)}")
                    self._ping()
                    out = self._invoke(tools, name, args, writes, core)
                    messages.append({"role": "tool",
                                     "tool_call_id": tc.get("id", ""),
                                     "content": out[:MAX_TOOL_CHARS]})
                _trim(messages, MAX_CONTEXT_MESSAGES)
            final = (result.content if result is not None else "") or ""
        except Exception as e:  # noqa: BLE001
            core.state = "error"
            core.error = f"{type(e).__name__}: {e}"[:200]
            core.finished = time.time()
            core.note(f"error: {core.error}")
            self.log.append("super.core.error",
                            {"core": core.callsign, "phase": phase,
                             "error": core.error},
                            actor=f"super:{core.callsign}")
            self._ping()
            return ""

        core.output = final[:MAX_OUTPUT_CHARS]
        core.finished = time.time()
        core.tool = ""
        blocked = re.search(r"STATUS:\s*BLOCKED", final, re.I) is not None
        core.state = "blocked" if blocked else "done"
        core.note(f"{phase}: {core.state} "
                  f"({core.steps} steps, {len(core.files)} file(s))")
        self.log.append("super.core.done",
                        {"core": core.callsign, "phase": phase,
                         "state": core.state, "steps": core.steps,
                         "files": core.files[:10],
                         "tokens_in": core.tokens_in,
                         "tokens_out": core.tokens_out,
                         "ms": int(core.elapsed * 1000)},
                        actor=f"super:{core.callsign}")
        self._ping()
        return final

    def _invoke(self, tools: dict[str, Tool], name: str, args: dict,
                writes: bool, core: Core) -> str:
        tool = tools.get(name)
        if tool is None:
            return (f"ERROR: tool '{name}' is not available in this phase. "
                    f"Available: {', '.join(tools)}")
        mutating = name in ("write_file", "edit_file", "create_directory",
                            "run_command")
        try:
            if writes and mutating:
                # I7 — parallel cores serialise every mutation
                with _WRITE_LOCK:
                    out = tool.handler(**args)
            else:
                out = tool.handler(**args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        if name in ("write_file", "edit_file") and str(out).startswith("OK"):
            p = str(args.get("path", ""))
            if p and p not in core.files:
                core.files.append(p)
        return str(out)

    # -- blackboard + observability -----------------------------------------------

    def _post(self, line: str) -> None:
        """Append to the shared blackboard (bounded)."""
        with self._bb_lock:
            self._blackboard.append(line[:400])
            total = sum(len(x) for x in self._blackboard)
            while total > MAX_BLACKBOARD_CHARS and len(self._blackboard) > 1:
                total -= len(self._blackboard.pop(0))
        self.log.append("super.board", {"line": line[:300]},
                        actor="sovereign")
        self._ping()

    def _board(self) -> str:
        with self._bb_lock:
            return "\n".join(self._blackboard[-40:])

    def _enter(self, m: Mission, phase: str) -> None:
        m.phase = phase
        self.log.append("super.phase", {"phase": phase, "round": m.round},
                        actor="sovereign")
        self._ping()

    def _ping(self) -> None:
        cb = self.on_update
        if cb is None:
            return
        try:
            cb()
        except Exception:  # noqa: BLE001 — the UI never breaks the machine
            pass

    def snapshot(self) -> dict:
        """Cheap live view for the board (called on every UI frame)."""
        m = self.mission
        return {
            "online": self.online,
            "busy": self.busy,
            "cores": [c.view() for c in self.cores],
            "phase": m.phase if m else "",
            "round": m.round if m else 0,
            "objective": (m.objective[:120] if m else ""),
            "status": m.status if m else "idle",
            "elapsed": round(m.elapsed, 1) if m else 0.0,
            "plan_version": m.plan_version if m else 0,
            "findings": len(m.findings) if m else 0,
            "sources": len(m.sources) if m else 0,
            "defects": len(m.defects) if m else 0,
            "files": len(m.files) if m else 0,
            "tokens": (m.tokens_in + m.tokens_out) if m else 0,
            "tool_calls": m.tool_calls if m else 0,
            "board": self._board().splitlines()[-6:],
        }

    def format_status(self) -> str:
        s = self.snapshot()
        head = ("SUPERCOMPUTER — " + ("ONLINE" if s["online"] else "OFFLINE")
                + f" · {self.n} cores")
        if not s["objective"]:
            return head + "\n  no mission yet — type your mission and the " \
                          "machine takes it"
        lines = [head,
                 f"  mission: {s['objective']}",
                 f"  phase: {s['phase'] or '—'} · round {s['round']} · "
                 f"status {s['status']} · {s['elapsed']:.0f}s",
                 f"  plan v{s['plan_version']} · findings {s['findings']} · "
                 f"sources {s['sources']} · files {s['files']} · "
                 f"defects {s['defects']} · tokens {s['tokens']:,}"]
        for c in s["cores"]:
            glyph = {"done": "✓", "error": "✗", "blocked": "◐",
                     "tool": "⚙", "thinking": "…", "idle": "·",
                     "cancelled": "⊘"}.get(c["state"], "?")
            lines.append(f"  {glyph} {c['callsign']:<7} {c['state']:<9} "
                         f"{c['activity'][:60]}")
        return "\n".join(lines)

    def format_report(self) -> str:
        m = self.mission
        if m is None:
            return "no mission has run yet"
        lines = [f"MISSION REPORT — {m.status.upper()} "
                 f"({m.elapsed:.0f}s, {self.n} cores)",
                 f"objective: {m.objective}",
                 f"plan: v{m.plan_version} ({len(m.plan)} chars) · "
                 f"{len(m.workstreams)} workstreams",
                 f"research: {len(m.findings)} findings from "
                 f"{len(m.sources)} sources",
                 f"build: {len(m.files)} file(s) written",
                 f"defects open: {len(m.defects)}",
                 f"tokens: {m.tokens_in:,} in / {m.tokens_out:,} out · "
                 f"{m.tool_calls} tool calls"]
        if m.files:
            lines.append("files: " + ", ".join(m.files[:20]))
        if m.defects:
            lines.append("open defects:")
            lines.extend("  " + d[:160] for d in m.defects[:12])
        if m.error:
            lines.append("error: " + m.error)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Slice builders + parsers (pure functions, easy to test)
# ---------------------------------------------------------------------------

def _relay_slice(m: Mission, plan: str, step: int) -> str:
    research = "\n".join(m.findings[-40:])[:4000]
    if not plan:
        return (f"MISSION: {m.objective}\n\n"
                f"RECON FINDINGS FROM THE OTHER CORES:\n{research}\n\n"
                "You are the FIRST link of the relay. Produce plan v1: a "
                "complete, concrete engineering plan with exactly 8 "
                "parallelisable workstreams.")
    return (f"MISSION: {m.objective}\n\n"
            f"INCOMING PLAN (v{step - 1}):\n{plan[:11000]}\n\n"
            f"RESEARCH SO FAR:\n{research}\n\n"
            f"You are link {step} of the relay. Return plan v{step}: "
            "strictly deeper, more concrete and more complete than the "
            "incoming one. Every workstream must gain something real "
            "(exact files, exact interfaces, exact done-criteria).")


def _deepdive_slice(m: Mission, source: str) -> str:
    return (f"MISSION: {m.objective}\n\n"
            f"YOUR SOURCE LANE: {source}\n\n"
            "Search the real world in that lane and bring back what this "
            "mission must know: real projects, real libraries with real "
            "versions, real techniques, real pitfalls. Every finding needs "
            "a real URL. Do not speculate — fetch and read.")


def _build_slice(m: Mission, ws: dict) -> str:
    return (f"MISSION: {m.objective}\n\n"
            f"MASTER PLAN (extract):\n{m.plan[:9000]}\n\n"
            f"YOUR WORKSTREAM {ws.get('id', '')}: {ws.get('title', '')}\n"
            f"deliverable files: {ws.get('files', '(decide from the plan)')}\n"
            f"done-criteria: {ws.get('criteria', '(from the plan)')}\n"
            f"detail: {ws.get('text', '')}\n\n"
            "Build it for real: read what exists, then write the actual "
            "files with your tools. Match the plan's interfaces exactly so "
            "the other seven workstreams link up with yours.")


def _verify_slice(m: Mission, sweep: str) -> str:
    return (f"MISSION: {m.objective}\n\n"
            f"YOUR SWEEP: {sweep}\n\n"
            f"FILES THE MACHINE WROTE: {', '.join(m.files[:40]) or '(none)'}\n\n"
            f"MASTER PLAN (extract):\n{m.plan[:7000]}\n\n"
            "Read the real files on disk and hunt. Report only defects you "
            "can point at with a path and evidence.")


def _repair_slice(m: Mission, defects: list[str]) -> str:
    return (f"MISSION: {m.objective}\n\n"
            "DEFECTS ASSIGNED TO YOU:\n- " + "\n- ".join(defects)[:5000]
            + "\n\nFix every one of them on disk now. Read the file, make "
              "the minimal correct change, then verify it.")


def _shard(items: list[str], n: int) -> list[list[str]]:
    """Round-robin shard so every core gets a comparable load."""
    out: list[list[str]] = [[] for _ in range(max(1, n))]
    for i, item in enumerate(items):
        out[i % len(out)].append(item)
    return out


def _section(text: str, header: str) -> list[str]:
    """Bullet lines under `HEADER:` up to the next ALL-CAPS header."""
    if not text:
        return []
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if re.match(rf"^{header}\s*:", stripped, re.I):
            inside = True
            tail = stripped.split(":", 1)[1].strip()
            if tail:
                out.append(tail)
            continue
        if inside:
            if re.match(r"^[A-Z][A-Z \-]{2,}:", stripped):
                break
            if stripped.startswith(("-", "*", "•")):
                out.append(stripped.lstrip("-*• ").strip())
            elif stripped:
                out.append(stripped)
    return [o for o in out if o][:60]


def _section_block(text: str, header: str) -> str:
    """Everything under `HEADER:` until the next known top-level header."""
    if not text:
        return ""
    stop = ("PLAN-VERSION:", "UPGRADES:", "STATUS:", "SOURCES:", "RISKS:")
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for raw in lines:
        stripped = raw.strip()
        if not inside and re.match(rf"^{header}\s*:", stripped, re.I):
            inside = True
            tail = stripped.split(":", 1)[1].strip()
            if tail:
                out.append(tail)
            continue
        if inside:
            if any(stripped.upper().startswith(s) for s in stop):
                break
            out.append(raw)
    return "\n".join(out).strip()


_WS_RE = re.compile(
    r"^\s*[-*]?\s*(W\d+)\s*[:.\-]?\s*(.*?)\s*(?:::\s*(.*?))?"
    r"\s*(?:::\s*(.*?))?\s*(?:::\s*(.*))?$")


def _parse_workstreams(plan: str, want: int) -> list[dict]:
    """Pull `- W1 title :: role :: files :: criteria` rows from the plan."""
    out: list[dict] = []
    for raw in (plan or "").splitlines():
        line = raw.strip()
        if not re.match(r"^[-*]?\s*W\d+\b", line):
            continue
        body = line.lstrip("-* ").strip()
        parts = [p.strip() for p in body.split("::")]
        head = parts[0]
        wid = head.split()[0]
        title = head[len(wid):].strip(" :-") or head
        role = parts[1] if len(parts) > 1 else "coder"
        role = role.lower().strip()
        if role not in ROLES:
            role = "coder"
        out.append({"id": wid, "title": title[:200], "role": role,
                    "files": parts[2][:200] if len(parts) > 2 else "",
                    "criteria": parts[3][:300] if len(parts) > 3 else "",
                    "text": body[:600]})
        if len(out) >= max(want, 8):
            break
    return out


def _arg_hint(args: dict) -> str:
    for key in ("path", "query", "command", "url", "pattern", "src"):
        v = args.get(key)
        if v:
            return str(v)[:70]
    return ""


def _trim(messages: list[dict], keep: int) -> None:
    """Bounded context: drop the oldest non-system turns in place."""
    if len(messages) <= keep:
        return
    head = messages[0:1] if messages[0].get("role") == "system" else []
    tail = messages[-(keep - len(head)):]
    # never start the tail with an orphan tool result
    while tail and tail[0].get("role") == "tool":
        tail.pop(0)
    messages[:] = head + tail


# ---------------------------------------------------------------------------
# Self-test — a stub chat drives a whole mission deterministically
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace

    def _self_test() -> None:
        # -- pure parsers -----------------------------------------------
        txt = ("FINDINGS:\n- alpha\n- beta\nSOURCES:\n- http://x\n"
               "STATUS: DONE")
        assert _section(txt, "FINDINGS") == ["alpha", "beta"]
        assert _section(txt, "SOURCES") == ["http://x"]
        plan_txt = ("PLAN:\n# Title\n## Workstreams\n"
                    "- W1 parser :: coder :: src/p.py :: pytest -q\n"
                    "- W2 docs :: documenter :: README.md :: file exists\n"
                    "PLAN-VERSION: v2\nSTATUS: DONE")
        block = _section_block(plan_txt, "PLAN")
        assert "Workstreams" in block and "PLAN-VERSION" not in block
        ws = _parse_workstreams(block, 8)
        assert len(ws) == 2 and ws[0]["id"] == "W1"
        assert ws[0]["role"] == "coder" and ws[1]["role"] == "documenter"
        assert ws[0]["files"] == "src/p.py"
        shards = _shard(["a", "b", "c"], 2)
        assert shards == [["a", "c"], ["b"]]
        msgs = [{"role": "system", "content": "s"}] + \
               [{"role": "user", "content": str(i)} for i in range(30)]
        _trim(msgs, 6)
        assert len(msgs) == 6 and msgs[0]["role"] == "system"

        # -- a full mission on a stub model ------------------------------
        # ignore_cleanup_errors: the EventLog keeps its file handle open,
        # and Windows refuses to unlink an open file during teardown.
        with tempfile.TemporaryDirectory(
                ignore_cleanup_errors=True) as td:
            root = Path(td)
            log = EventLog(root / "super.jsonl")
            provider = SimpleNamespace(key="t", name="T",
                                       base_url="http://t",
                                       api_key="k", color="#fff")
            model = SimpleNamespace(id="stub", provider="t", label="Stub",
                                    supports_tools=True,
                                    supports_reasoning=False,
                                    context_window=8192)
            effort = SimpleNamespace(key="low", label="LOW", color="#fff",
                                     max_tokens=256, temperature=0.0,
                                     reasoning_effort=None)

            seen_parallel = threading.Event()
            live = {"n": 0, "peak": 0}
            live_lock = threading.Lock()
            verify_round = {"n": 0}

            def stub_chat(prov, mdl, eff, messages, schemas, timeout):
                with live_lock:
                    live["n"] += 1
                    live["peak"] = max(live["peak"], live["n"])
                    if live["peak"] >= 2:
                        seen_parallel.set()
                time.sleep(0.03)
                with live_lock:
                    live["n"] -= 1
                system = messages[0]["content"]
                if "Current phase: RECON" in system:
                    body = ("FINDINGS:\n- real fact :: /tmp/x.py\n"
                            "SOURCES:\n- https://example.org\n"
                            "RISKS:\n- none\nSTATUS: DONE")
                elif "Current phase: DEEPDIVE" in system:
                    body = ("FINDINGS:\n- lib exists :: https://pypi.org/x\n"
                            "SOURCES:\n- https://pypi.org/x\n"
                            "ADOPT:\n- use x\nSTATUS: DONE")
                elif "Current phase: RELAY" in system:
                    body = ("PLAN:\n# Mission\n## Workstreams\n"
                            + "\n".join(
                                f"- W{i} part{i} :: coder :: f{i}.py :: "
                                f"exists" for i in range(1, 9))
                            + "\nPLAN-VERSION: v9\nUPGRADES:\n- deeper\n"
                              "STATUS: DONE")
                elif "Current phase: BUILD" in system:
                    body = "STATUS: DONE\nSUMMARY: wrote the file"
                elif "Current phase: VERIFY" in system:
                    verify_round["n"] += 1
                    if verify_round["n"] <= 8:      # first round: defects
                        body = ("DEFECTS:\n- [major] f1.py:2 — missing "
                                "guard\nGAPS:\n- none\nSTATUS: DONE")
                    else:                            # after repair: clean
                        body = "DEFECTS:\n- none\nGAPS:\n- none\nSTATUS: DONE"
                else:                                 # repair
                    body = "STATUS: DONE\nSUMMARY: fixed it"
                return SimpleNamespace(content=body, reasoning="",
                                       tool_calls=[], finish_reason="stop",
                                       usage={"prompt_tokens": 12,
                                              "completion_tokens": 6})

            pings = {"n": 0}
            sc = Supercomputer(log, provider, model, effort, cores=8,
                               chat=stub_chat,
                               on_update=lambda: pings.__setitem__(
                                   "n", pings["n"] + 1))

            assert "ONLINE" in sc.boot()
            assert sc.online and not sc.busy

            m = sc.run_mission("build a terminal AI agent")
            assert m.status == "complete", (m.status, m.error)
            # eight cores really overlapped in wall-clock time
            assert seen_parallel.is_set(), "cores did not run in parallel"
            assert live["peak"] >= 2 and live["peak"] <= 8, live["peak"]
            # the relay climbed and the plan survived into workstreams
            assert m.plan_version >= 8, m.plan_version
            assert len(m.workstreams) == 8, m.workstreams
            # research landed from both recon and the parallel deep dive
            assert any("ADOPT" in f for f in m.findings)
            assert m.sources and m.findings
            # verify found defects, repair ran, re-verify came back clean
            assert m.round >= 2, m.round
            assert m.defects == [], m.defects
            assert m.tokens_in > 0 and m.tokens_out > 0
            assert pings["n"] > 20

            # every phase is sealed in the kernel
            types = [e.type for e in log.events()]
            assert "super.boot" in types
            assert "super.mission.start" in types
            phases = {e.data.get("phase") for e in log.events()
                      if e.type == "super.phase"}
            assert {"recon", "relay", "fuse", "build", "verify",
                    "repair"} <= phases, phases
            assert types.count("super.core.done") >= 40
            assert "super.plan.final" in types
            assert "super.mission.done" in types

            # the live board renders and is bounded
            snap = sc.snapshot()
            assert len(snap["cores"]) == 8
            assert snap["status"] == "complete"
            assert len(snap["board"]) <= 6
            assert "SUPERCOMPUTER" in sc.format_status()
            assert "MISSION REPORT" in sc.format_report()
            assert sum(len(x) for x in sc._blackboard) <= MAX_BLACKBOARD_CHARS

            # a second mission cannot start while one holds the machine
            sc._busy.acquire()
            try:
                sc.run_mission("second")
                raise AssertionError("concurrent mission must be refused")
            except SupercomputerError:
                pass
            finally:
                sc._busy.release()

            assert "OFFLINE" in sc.shutdown()
            assert not sc.online

        print("SUPERCOMPUTER SELF-TEST PASS")

    _self_test()
