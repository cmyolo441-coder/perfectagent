"""CREW — Codex-style persistent subagent lifecycle.

The Team (team.py) is a fire-and-collect batch: spawn N workers, block,
get reports. The Crew is the next level — PERSISTENT, addressable
subagents with a real lifecycle, exactly like a lead engineer managing
a crew of specialists:

    spawn(task, role)   launch a background subagent, returns immediately
    send(id, message)   follow-up message into a living subagent's context
    wait(ids, timeout)  block until the named subagents reach a verdict
    close(id)           retire a subagent (releases its slot)
    resume(id)          bring a closed subagent back with its full context

Each CrewAgent is a REAL agent: its own role brief, its own tool
whitelist, its own multi-step tool loop, its own message history that
SURVIVES follow-up messages — so you can iterate on a subagent instead
of re-spawning from scratch. Up to MAX_AGENTS run concurrently; the
main conversation stays fully responsive while they work (async, like
Codex). Spawning returns at once; progress is polled or waited on.

Hard rules (mechanical, same discipline as the rest of FullAgent):
  * Reads fan out, writes serialise — crew writes pass through the SAME
    global lock as Team writes (invariant I7). Two subagents can never
    mutate the world at once, regardless of which subsystem spawned them.
  * Every lifecycle transition is sealed in the event log: crew.spawn,
    crew.progress, crew.message, crew.done, crew.closed, crew.resumed.
    The crew history is replayable and auditable.
  * A failing subagent never kills the crew; it lands as an error report
    and can be sent a follow-up or closed.
  * Follow-ups reuse the subagent's full conversation — context is the
    dividend of persistence.
"""

from __future__ import annotations

import itertools
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import systemprompt
from .config import PROVIDERS, model_by_id
from .kernel import EventLog, fold
from .team import (ROLES, DEFAULT_ROLE, MAX_WORKER_STEPS, _WRITE_LOCK,
                   chat_with_retry, parse_worker_final)
from .tools import Tool, build_registry, parse_tool_arguments

MAX_AGENTS = 8             # concurrent crew slots
MAX_SEND_STEPS = 40        # tool-loop budget per follow-up message
WAIT_POLL_SECONDS = 0.05   # wait() polling granularity

# Codex-flavoured callsigns for the crew roster.
_CALLSIGNS = ("nova", "atlas", "echo", "lyra", "orion", "vega", "iris",
              "argo", "sable", "kepler", "juno", "helix", "drift", "onyx",
              "piper", "quill")

AGENT_STATES = ("running", "done", "blocked", "error", "closed")

_ROLE_ICON = {"researcher": "🔎", "coder": "👨‍💻", "tester": "🧪",
              "reviewer": "🧐", "analyst": "📊", "architect": "🏛️",
              "debugger": "🐞", "optimizer": "⚡", "refactorer": "🧹",
              "documenter": "📝", "devops": "🛠️", "integrator": "🔗",
              "planner": "🗺️"}


@dataclass
class CrewAgent:
    """One persistent subagent. The message history is the point: it
    survives follow-ups, so iteration never starts from zero."""
    id: str
    nickname: str
    role: str
    task: str
    state: str = "running"      # running | done | blocked | error | closed
    summary: str = ""
    error: str = ""
    messages: list = field(default_factory=list)   # full conversation
    files_touched: list = field(default_factory=list)
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    spawned_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    pending_messages: list = field(default_factory=list)
    model_id: str = ""          # per-agent model override ("" = crew default)
    _thread: threading.Thread | None = None

    @property
    def icon(self) -> str:
        return _ROLE_ICON.get(self.role, "◆")

    @property
    def elapsed_ms(self) -> int:
        end = self.finished_at or time.time()
        return int((end - self.spawned_at) * 1000)

    def to_dict(self) -> dict:
        return {"id": self.id, "nickname": self.nickname, "role": self.role,
                "task": self.task, "state": self.state,
                "model": self.model_id,
                "summary": self.summary[:600], "error": self.error[:300],
                "files_touched": self.files_touched[:12],
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "elapsed_ms": self.elapsed_ms}


class CrewError(RuntimeError):
    """Raised for invalid lifecycle operations (unknown id, spawn at
    capacity, send to a closed agent)."""


class Crew:
    """Persistent, addressable subagents over a shared EventLog.

    `chat` is injectable for tests: chat(provider, model, effort,
    messages, schemas, timeout) -> StreamResult. Production uses the
    rate-limit-hardened chat_with_retry from team.py.
    """

    def __init__(self, log: EventLog, provider, model, effort,
                 mastermind=None, max_agents: int = MAX_AGENTS,
                 chat=None) -> None:
        self.log = log
        self.provider = provider
        self.model = model
        self.effort = effort
        self.mastermind = mastermind
        self.max_agents = max(1, int(max_agents))
        self._chat = chat or chat_with_retry
        self._agents: dict[str, CrewAgent] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()       # protects the roster
        self._names = itertools.cycle(_CALLSIGNS)
        self._counter = 0
        # role tool whitelists carved from the main registry (same as Team)
        registry = build_registry()
        self._toolsets: dict[str, dict[str, Tool]] = {}
        for role, spec in ROLES.items():
            self._toolsets[role] = {n: registry[n] for n in spec["tools"]
                                    if n in registry}

    # -- lifecycle -------------------------------------------------------------

    def spawn(self, task: str, role: str = DEFAULT_ROLE, name: str = "",
              context: str = "", read_only: bool = False,
              model_id: str = "") -> CrewAgent:
        """Launch a subagent in the background; returns IMMEDIATELY.
        The caller keeps working; wait()/poll() collects the verdict.

        model_id optionally overrides the model THIS subagent uses
        (Codex-style per-agent model override) — e.g. a cheap fast model
        for grunt work, the strongest model for the hard piece. Unknown
        ids fall back to the crew default with a sealed note."""
        task = str(task or "").strip()
        if not task:
            raise CrewError("cannot spawn a subagent without a task")
        if role not in ROLES:
            role = DEFAULT_ROLE
        with self._lock:
            live = sum(1 for a in self._agents.values()
                       if a.state == "running")
            if live >= self.max_agents:
                raise CrewError(
                    f"crew is at capacity ({self.max_agents} concurrent "
                    f"subagents) — wait for one to finish or close one")
            self._counter += 1
            agent_id = f"crew-{self._counter}"
            nickname = str(name or "").strip() or next(self._names)
            while any(a.nickname == nickname
                      for a in self._agents.values()):
                nickname = f"{nickname}-{self._counter}"
            agent = CrewAgent(id=agent_id, nickname=nickname, role=role,
                              task=task)
            override = model_by_id(str(model_id or "")) if model_id else None
            if override is not None:
                agent.model_id = override.id
            self._agents[agent_id] = agent
            self._order.append(agent_id)

        user = (f"Shared context:\n{context}\n\nYOUR TASK: {task}"
                if context else f"YOUR TASK: {task}")
        if self.mastermind is not None:
            agent.messages, _ = self.mastermind.gate.dispatch(
                f"worker:{role}", agent.messages)
        else:
            systemprompt.with_system(agent.messages,
                                     systemprompt.worker(role,
                                                         self.max_agents))
        agent.messages.append({"role": "user", "content": user})

        self.log.append("crew.spawn",
                        {"id": agent.id, "nickname": agent.nickname,
                         "role": role, "task": task[:300],
                         "read_only": bool(read_only),
                         "model": agent.model_id or self.model.id},
                        actor="sovereign")
        agent._thread = threading.Thread(
            target=self._run_loop, args=(agent, read_only,
                                         MAX_WORKER_STEPS),
            name=f"crew:{agent.id}", daemon=True)
        agent._thread.start()
        return agent

    def send(self, agent_id: str, message: str,
             interrupt: bool = False) -> CrewAgent:
        """Send a follow-up into a subagent's LIVING context.

        done/blocked/error agents start a new loop iteration with the
        message appended (full history preserved). A running agent gets
        the message queued — it is delivered the moment the current loop
        finishes (interrupt=True clears the agent's pending summary so
        the follow-up takes priority in the next reply)."""
        agent = self._require(agent_id)
        message = str(message or "").strip()
        if not message:
            raise CrewError("cannot send an empty message")
        if agent.state == "closed":
            raise CrewError(
                f"agent {agent_id} is closed — resume it first")
        self.log.append("crew.message",
                        {"id": agent_id, "chars": len(message),
                         "interrupt": bool(interrupt)},
                        actor="sovereign")
        if agent.state == "running":
            agent.pending_messages.append(message)
            return agent
        if interrupt:
            agent.summary = ""
        agent.messages.append({"role": "user",
                               "content": f"FOLLOW-UP: {message}"})
        agent.state = "running"
        agent.error = ""
        agent._thread = threading.Thread(
            target=self._run_loop, args=(agent, False, MAX_SEND_STEPS),
            name=f"crew:{agent.id}:followup", daemon=True)
        agent._thread.start()
        return agent

    def wait(self, ids: list[str] | None = None,
             timeout: float = 30.0) -> dict[str, str]:
        """Block until the named subagents (default: all) leave the
        running state, or the timeout lands. Returns {id: state}."""
        targets = [self._require(i) for i in ids] if ids else list(
            self._agents.values())
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if all(a.state != "running" for a in targets):
                break
            time.sleep(WAIT_POLL_SECONDS)
        return {a.id: a.state for a in targets}

    def close(self, agent_id: str) -> CrewAgent:
        """Retire a subagent. It keeps its history (resume() can bring
        it back) but refuses sends while closed and frees no slot —
        only running agents occupy slots."""
        agent = self._require(agent_id)
        if agent.state == "closed":
            return agent
        prev = agent.state
        agent.state = "closed"
        self.log.append("crew.closed",
                        {"id": agent_id, "prev_state": prev},
                        actor="sovereign")
        return agent

    def resume(self, agent_id: str) -> CrewAgent:
        """Bring a closed subagent back (state 'done', full context),
        so it can receive follow-ups again."""
        agent = self._require(agent_id)
        if agent.state != "closed":
            return agent
        agent.state = "done" if not agent.error else "error"
        self.log.append("crew.resumed", {"id": agent_id},
                        actor="sovereign")
        return agent

    # -- queries ----------------------------------------------------------------

    def get(self, agent_id: str) -> CrewAgent | None:
        return self._agents.get(agent_id)

    def list(self) -> list[CrewAgent]:
        return [self._agents[i] for i in self._order]

    def running(self) -> list[CrewAgent]:
        return [a for a in self.list() if a.state == "running"]

    def _require(self, agent_id: str) -> CrewAgent:
        agent = self._agents.get(agent_id)
        if agent is None:
            known = ", ".join(self._order) or "none"
            raise CrewError(f"unknown subagent {agent_id!r} (known: {known})")
        return agent

    def status(self) -> dict:
        agents = self.list()
        return {"total": len(agents),
                "running": sum(1 for a in agents if a.state == "running"),
                "done": sum(1 for a in agents if a.state == "done"),
                "blocked": sum(1 for a in agents if a.state == "blocked"),
                "error": sum(1 for a in agents if a.state == "error"),
                "closed": sum(1 for a in agents if a.state == "closed"),
                "tool_calls": sum(a.tool_calls for a in agents),
                "tokens_in": sum(a.tokens_in for a in agents),
                "tokens_out": sum(a.tokens_out for a in agents)}

    def format(self, agents: list[CrewAgent] | None = None) -> str:
        """Compact multi-line report — the shape handed back to the LLM."""
        agents = agents if agents is not None else self.list()
        if not agents:
            return "crew is empty — spawn a subagent first"
        lines = []
        for a in agents:
            icon = {"done": "✓", "blocked": "◐", "error": "✗",
                    "closed": "⊘", "running": "…"}.get(a.state, "?")
            model_tag = (f" · {a.model_id}" if a.model_id
                         and a.model_id != self.model.id else "")
            head = (f"{a.icon} [{a.id}] {a.nickname} ({a.role}) {icon} "
                    f"{a.state} · {a.tool_calls} tools{model_tag} · "
                    f"{a.elapsed_ms}ms")
            lines.append(head)
            lines.append(f"  task: {a.task[:200]}")
            if a.files_touched:
                lines.append("  files: " + ", ".join(a.files_touched[:8]))
            if a.error:
                lines.append(f"  error: {a.error[:200]}")
            if a.summary:
                lines.append("  " + a.summary.replace("\n", "\n  ")[:1200])
        return "\n".join(lines)

    def format_status(self) -> str:
        s = self.status()
        lines = [f"CREW — {s['total']} subagent(s): "
                 f"{s['running']} running · {s['done']} done · "
                 f"{s['error']} error · {s['closed']} closed"]
        for a in self.list():
            lines.append(f"  {a.icon} [{a.id}] {a.nickname} ({a.role}) — "
                         f"{a.state}: {a.task[:70]}")
        return "\n".join(lines)

    # -- the worker loop ----------------------------------------------------------

    def _run_loop(self, agent: CrewAgent, read_only: bool,
                  max_steps: int) -> None:
        """One subagent's bounded tool loop. Never raises: every failure
        lands in the agent's report and is sealed as crew.done."""
        spec = ROLES[agent.role]
        tools = dict(self._toolsets[agent.role])
        if read_only:
            tools = {n: t for n, t in tools.items()
                     if n not in ("write_file", "edit_file",
                                  "create_directory", "run_command")}
        # per-agent model override resolves its own provider (schemas must
        # follow the model that will actually serve this agent, not the
        # crew default — tool support differs between models)
        model = (model_by_id(agent.model_id) if agent.model_id
                 else None) or self.model
        provider = PROVIDERS.get(model.provider, self.provider)
        schemas = ([t.openai_schema() for t in tools.values()]
                   if model.supports_tools else None)
        # gentle launch de-sync: parallel spawns never burst the API at t=0
        time.sleep(random.uniform(0.0, 0.4))
        result = None
        try:
            for step in range(max_steps):
                result = self._chat(provider, model, self.effort,
                                    agent.messages, schemas, 120.0)
                if result.usage:
                    agent.tokens_in += int(
                        result.usage.get("prompt_tokens", 0) or 0)
                    agent.tokens_out += int(
                        result.usage.get("completion_tokens", 0) or 0)
                if not result.tool_calls:
                    break
                agent.messages.append({"role": "assistant",
                                       "content": result.content or None,
                                       "tool_calls": result.tool_calls})
                tool_names = []
                for tc in result.tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args = parse_tool_arguments(fn.get("arguments"))
                    agent.tool_calls += 1
                    tool_names.append(name)
                    tool = tools.get(name)
                    if tool is None:
                        out = (f"ERROR: tool '{name}' is not available to "
                               f"a {agent.role} subagent. Available: "
                               + ", ".join(tools))
                    else:
                        # I7 — writes serialise across ALL workers, crew
                        # and team alike.
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
                                if p and p not in agent.files_touched:
                                    agent.files_touched.append(p)
                        except Exception as e:  # noqa: BLE001
                            out = f"ERROR: {type(e).__name__}: {e}"
                    agent.messages.append(
                        {"role": "tool", "tool_call_id": tc.get("id", ""),
                         "content": out[:6000]})
                if step % 2 == 0:
                    self.log.append("crew.progress",
                                    {"id": agent.id, "step": step + 1,
                                     "tools": tool_names[:6]},
                                    actor=f"crew:{agent.id}")
            final = (result.content if result is not None else "") or ""
            state, summary = parse_worker_final(final)
            agent.summary = summary[:1800]
            agent.state = state if state in ("done", "blocked") else "done"
            if not final.strip():
                agent.error = "subagent returned an empty reply"
                agent.state = "error"
        except Exception as e:  # noqa: BLE001 — a failing agent never kills the crew
            agent.state = "error"
            agent.error = f"{type(e).__name__}: {e}"
        finally:
            agent.finished_at = time.time()
            # deliver queued follow-ups, if any arrived mid-loop
            if agent.pending_messages and agent.state != "closed":
                queued = agent.pending_messages.pop(0)
                agent.messages.append({"role": "user",
                                       "content": f"FOLLOW-UP: {queued}"})
                agent.state = "running"
                agent._thread = threading.Thread(
                    target=self._run_loop,
                    args=(agent, read_only, MAX_SEND_STEPS),
                    name=f"crew:{agent.id}:queued", daemon=True)
                agent._thread.start()
                return
            self.log.append("crew.done", agent.to_dict(),
                            actor=f"crew:{agent.id}")


# ---------------------------------------------------------------------------
# Self-test — a stub chat drives the full lifecycle deterministically
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EventLog(Path(td) / "crew-test.jsonl")
            provider = SimpleNamespace(key="t", name="T",
                                       base_url="http://t", api_key="sk-fake",
                                       color="#fff")
            model = SimpleNamespace(id="stub", provider="t", label="Stub",
                                    supports_tools=True, supports_reasoning=False)
            effort = SimpleNamespace(key="low", label="LOW", color="#fff",
                                     max_tokens=100, temperature=0.0,
                                     reasoning_effort=None)

            calls = {"n": 0}

            def stub_chat(provider_, model_, effort_, messages, schemas,
                          timeout):
                calls["n"] += 1
                last_user = next((m["content"] for m in reversed(messages)
                                  if m.get("role") == "user"), "")
                content = ("STATUS: DONE\nSUMMARY: built the thing"
                           if "YOUR TASK" in last_user
                           else "STATUS: DONE\nSUMMARY: follow-up handled")
                return SimpleNamespace(content=content, reasoning="",
                                       tool_calls=[], finish_reason="stop",
                                       usage={"prompt_tokens": 10,
                                              "completion_tokens": 5})

            crew = Crew(log, provider, model, effort, chat=stub_chat)

            # spawn returns immediately; wait collects the verdict
            a1 = crew.spawn("write a parser", role="coder")
            a2 = crew.spawn("research parsers", role="researcher")
            assert a1.id == "crew-1" and a2.id == "crew-2"
            assert a1.state == "running"
            states = crew.wait(timeout=10.0)
            assert states[a1.id] == "done" and states[a2.id] == "done", states
            assert "built the thing" in a1.summary
            assert a1.tokens_in > 0

            # follow-up reuses the full conversation
            crew.send(a1.id, "now add error handling")
            crew.wait([a1.id], timeout=10.0)
            assert a1.state == "done"
            assert "follow-up handled" in a1.summary
            users = [m for m in a1.messages if m.get("role") == "user"]
            assert len(users) == 2  # task + follow-up, history preserved

            # close refuses sends; resume reopens
            crew.close(a2.id)
            assert a2.state == "closed"
            try:
                crew.send(a2.id, "hi")
                raise AssertionError("send to closed agent must fail")
            except CrewError:
                pass
            crew.resume(a2.id)
            assert a2.state == "done"

            # unknown ids raise with the roster listed
            try:
                crew.wait(["crew-99"])
                raise AssertionError("unknown id must raise")
            except CrewError as e:
                assert "crew-1" in str(e)

            # lifecycle events are sealed in the log
            types = [e.type for e in log.events()]
            assert types.count("crew.spawn") == 2
            assert types.count("crew.message") == 1
            assert types.count("crew.done") >= 3
            assert "crew.closed" in types and "crew.resumed" in types

            # report renders
            rep = crew.format()
            assert "crew-1" in rep and "coder" in rep
            assert "CREW" in crew.format_status()

            print("CREW SELF-TEST PASS")

    _self_test()
