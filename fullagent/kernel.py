"""Temporal Kernel — the Mul Bindu.

Every interaction with reality is recorded as an immutable, causally-ordered,
content-addressed event in a single append-only log. Nothing else is state.
State is a pure fold over that log.

Design (pure Python, stdlib only):
  * JSONL is truth. One line per event, append-only, fsync'd.
  * Content addressing: each event's id = sha256 of its canonical encoding.
  * Merkle spine: each event carries its parent's id, so the log is
    tamper-evident and any prefix is independently verifiable.
  * Seqs are global and strictly monotonic — never reused, on any branch.
  * A branch is a head pointer (an event id). Its history is the parent
    chain walked back from the head; events abandoned by a rewind or not
    chosen by a fork simply fall off the chain (fold horizon shrinks).
  * Fork at seq N seeds the new branch's head at event N, so the fork
    inherits the source's history up to the fork point.
  * Rewind moves the head pointer back to the event at seq N and seals a
    'kernel.rewind' marker (parent = event N) so the move survives reload.
    Events are never deleted. (Append-only is sacred.)
  * Reload replays the file in order: every append moved its branch's head
    to the new event, so head pointers rebuild without any second file.
  * Fold: a pure function from a branch's chain to a State projection.
    Rewind, replay, resume, audit, and cost attribution are all folds.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


def _canonical(obj: Any) -> str:
    """Deterministic JSON encoding — the basis of content addressing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Event:
    seq: int
    id: str
    parent: str | None
    branch: str
    ts: float
    type: str
    data: dict
    # §7.1 causal envelope — every event is attributable and traceable
    session: str = ""
    actor: str = "system"            # sovereign | scout:N | human | system | …
    causation_id: str | None = None  # the event that directly caused this one
    correlation_id: str | None = None  # the root goal clause this serves
    provenance: str = "system"       # system | user | tool_output | web | file | model

    @property
    def short(self) -> str:
        return self.id[:10]

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "id": self.id,
            "parent": self.parent,
            "branch": self.branch,
            "ts": self.ts,
            "type": self.type,
            "data": self.data,
            "session": self.session,
            "actor": self.actor,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(seq=d["seq"], id=d["id"], parent=d.get("parent"),
                   branch=d.get("branch", "main"), ts=d.get("ts", 0.0),
                   type=d["type"], data=d.get("data", {}),
                   session=d.get("session", ""),
                   actor=d.get("actor", "system"),
                   causation_id=d.get("causation_id"),
                   correlation_id=d.get("correlation_id"),
                   provenance=d.get("provenance", "system"))

    @staticmethod
    def compute_id(seq: int, parent: str | None, branch: str, ts: float,
                   type_: str, data: dict, session: str = "",
                   actor: str = "system", causation_id: str | None = None,
                   correlation_id: str | None = None,
                   provenance: str = "system") -> str:
        """Content address: hash of everything except the id itself."""
        payload = {"seq": seq, "parent": parent, "branch": branch,
                   "ts": ts, "type": type_, "data": data,
                   "session": session, "actor": actor,
                   "causation_id": causation_id,
                   "correlation_id": correlation_id,
                   "provenance": provenance}
        return _hash(_canonical(payload))


# ---------------------------------------------------------------------------
# State — the projection produced by folding
# ---------------------------------------------------------------------------


@dataclass
class State:
    """A derived, never-authoritative view of the log prefix."""
    messages: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    files_touched: set[str] = field(default_factory=set)
    commands_run: int = 0
    episodes: list[dict] = field(default_factory=list)
    dead_ends: list[dict] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    goal: dict | None = None
    goal_done: list[str] = field(default_factory=list)
    # §42 — the latest goal.closed event AFTER the current goal.set; None
    # while the contract is still open
    goal_closed: dict | None = None
    autonomy: int = 3
    # advanced subsystems (compiler/evolution/brain/merge/theater/debate/
    # market) — one bucket, newest last; each module filters by "type"
    advanced_events: list[dict] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)
    # §8.3 / §9 — snapshot store references, newest last
    snapshots: list[dict] = field(default_factory=list)
    # §13 — plan DAG nodes keyed by node id
    nodes: dict[str, dict] = field(default_factory=dict)
    # §13.5 — budget events (slices, exceeded)
    budget_events: list[dict] = field(default_factory=list)
    # Part VI — clause proof/regression/amendment/focus history
    clause_proven: list[dict] = field(default_factory=list)
    clause_regressed: list[dict] = field(default_factory=list)
    amendments: list[dict] = field(default_factory=list)
    focus_shifts: list[dict] = field(default_factory=list)
    distance_measures: list[dict] = field(default_factory=list)
    # §18 — environment digests (drift detection)
    env_digests: list[dict] = field(default_factory=list)
    # §21 — calibration samples (est vs actual)
    calibration: list[dict] = field(default_factory=list)
    # §13.4 — loop/thrash detector trips
    loop_alerts: list[dict] = field(default_factory=list)
    # Mastermind — prompt coherence ledger
    prompt_sealed: list[dict] = field(default_factory=list)
    prompt_dispatches: list[dict] = field(default_factory=list)
    # v3 advanced subsystems
    router_decisions: list[dict] = field(default_factory=list)
    semantic_index: list[dict] = field(default_factory=list)
    spec_events: list[dict] = field(default_factory=list)
    daemon_events: list[dict] = field(default_factory=list)
    heal_events: list[dict] = field(default_factory=list)
    skill_events: list[dict] = field(default_factory=list)
    council_events: list[dict] = field(default_factory=list)
    # v4 professional subsystems
    lsp_events: list[dict] = field(default_factory=list)
    dap_events: list[dict] = field(default_factory=list)
    analysis_events: list[dict] = field(default_factory=list)
    mutation_events: list[dict] = field(default_factory=list)
    coverage_events: list[dict] = field(default_factory=list)
    fuzz_events: list[dict] = field(default_factory=list)
    graph_events: list[dict] = field(default_factory=list)
    browser_events: list[dict] = field(default_factory=list)
    openapi_events: list[dict] = field(default_factory=list)
    db_events: list[dict] = field(default_factory=list)
    git_events: list[dict] = field(default_factory=list)
    ensemble_events: list[dict] = field(default_factory=list)
    hybrid_events: list[dict] = field(default_factory=list)
    compress_events: list[dict] = field(default_factory=list)
    eval_events: list[dict] = field(default_factory=list)
    sched_events: list[dict] = field(default_factory=list)
    cache_events: list[dict] = field(default_factory=list)
    cost_ledger: list[dict] = field(default_factory=list)
    head_seq: int = -1
    branch: str = "main"

    def cost_summary(self) -> str:
        return f"${self.cost_usd:.4f} · {self.tokens_in}→{self.tokens_out} tok"


# ---------------------------------------------------------------------------
# EventLog — append-only, content-addressed, causally linked
# ---------------------------------------------------------------------------


class EventLog:
    """The single source of truth. Thread-safe for appends.

    Heads are event ids, not seqs: a branch's history is the parent chain
    walked back from its head. Seqs are global and strictly monotonic, so a
    rewind or fork never reuses a seq — abandoned events simply fall off
    the chain and stop contributing to the fold.
    """

    def __init__(self, path: Path, branch: str = "main", session: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._events: list[Event] = []
        self._by_id: dict[str, Event] = {}
        # branch name -> id of its head event (None = empty branch)
        self._heads: dict[str, str | None] = {}
        self._next_seq = 0
        self.branch = branch
        self.session = session
        # SPEED: per-branch chain cache (extended incrementally, O(1) per
        # append) + fold memoisation keyed by (branch, head seq) + one
        # persistent append handle with batched fsync. Event queries used
        # to cost O(n) every time — these caches make steady-state reads
        # O(1) and every append a buffered write.
        self._chains: dict[str, list[Event]] = {}
        self._fold_cache: dict[str, tuple[str | None, object]] = {}
        self._fh = None
        self._writes_since_sync = 0
        self._load()

    # -- persistence -------------------------------------------------------

    # SPEED: writes go through ONE persistent append handle; a full disk
    # sync happens once per batch instead of after every single event.
    # Events still flush to the OS on every write (survives a crash of
    # this process; only a power loss can drop the last few).
    _SYNC_EVERY = 64

    def _load(self) -> None:
        self._heads.setdefault(self.branch, None)
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = Event.from_dict(json.loads(line))
                except (ValueError, KeyError):
                    continue
                self._events.append(ev)
                self._by_id[ev.id] = ev
                self._next_seq = max(self._next_seq, ev.seq + 1)
                # replay: every appended event moved its branch's head to it,
                # so head pointers rebuild exactly (rewinds included, since a
                # rewind seals a marker event that becomes the new head)
                self._heads[ev.branch] = ev.id

    def _persist(self, ev: Event) -> None:
        try:
            self._write_event(ev)
        except FileNotFoundError:
            # the home directory vanished mid-session (deleted externally,
            # fresh mount, etc.) — recreate it and write again. The log
            # must never take the app down over a missing directory.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = None
            self._write_event(ev)
        except ValueError:
            # handle was closed/replaced underneath us — reopen once
            self._fh = None
            self._write_event(ev)

    def _write_event(self, ev: Event) -> None:
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()
        self._writes_since_sync += 1
        if self._writes_since_sync >= self._SYNC_EVERY:
            self._writes_since_sync = 0
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass

    # -- append ------------------------------------------------------------

    def append(self, type_: str, data: dict | None = None,
               branch: str | None = None, *,
               actor: str = "system",
               causation_id: str | None = None,
               correlation_id: str | None = None,
               provenance: str = "system",
               session: str | None = None) -> Event:
        """Append one event to a branch. Returns the sealed Event.

        The causal envelope (§7.1) makes every event attributable:
        causation_id = the event that directly caused this one,
        correlation_id = the goal clause this ultimately serves.
        """
        data = data or {}
        with self._lock:
            br = branch or self.branch
            parent_id = self._heads.get(br)
            seq = self._next_seq
            self._next_seq += 1
            ts = time.time()
            sess = session if session is not None else self.session
            # default causation: the branch's current head caused this event
            caus = causation_id if causation_id is not None else parent_id
            eid = Event.compute_id(seq, parent_id, br, ts, type_, data,
                                   sess, actor, caus, correlation_id,
                                   provenance)
            ev = Event(seq=seq, id=eid, parent=parent_id, branch=br,
                       ts=ts, type=type_, data=data, session=sess,
                       actor=actor, causation_id=caus,
                       correlation_id=correlation_id, provenance=provenance)
            self._events.append(ev)
            self._by_id[eid] = ev
            self._heads[br] = eid
            self._persist(ev)
            return ev

    # -- chain walking -------------------------------------------------------

    def _chain(self, branch: str) -> list[Event]:
        """The branch's history (parent chain from its head), causal order.

        SPEED: cached per branch and extended incrementally — a steady-state
        append costs O(1) here instead of an O(n) walk + reversal."""
        head_id = self._heads.get(branch)
        cached = self._chains.get(branch)
        if cached is not None:
            if not cached and head_id is None:
                return cached
            if cached and head_id == cached[-1].id:
                return cached
            # head moved forward (new appends) — extend from the cached tail
            if cached and head_id is not None:
                tail_id = cached[-1].id
                new: list[Event] = []
                cur = self._by_id.get(head_id)
                seen: set[str] = set()
                while cur is not None and cur.id != tail_id \
                        and cur.id not in seen:
                    seen.add(cur.id)
                    new.append(cur)
                    cur = (self._by_id.get(cur.parent)
                           if cur.parent else None)
                if cur is not None and cur.id == tail_id:
                    new.reverse()
                    self._chains[branch] = cached + new
                    return self._chains[branch]
        # first access, rewind or fork — full rebuild (then cached)
        chain: list[Event] = []
        cur_id = head_id
        seen2: set[str] = set()
        while cur_id and cur_id in self._by_id and cur_id not in seen2:
            seen2.add(cur_id)
            ev = self._by_id[cur_id]
            chain.append(ev)
            cur_id = ev.parent
        chain.reverse()
        self._chains[branch] = chain
        return chain

    def _event_at(self, branch: str, seq: int) -> Event | None:
        """The newest event on the branch's chain with e.seq <= seq."""
        if seq < 0:
            return None
        best: Event | None = None
        for ev in self._chain(branch):
            if ev.seq <= seq and (best is None or ev.seq > best.seq):
                best = ev
        return best

    # -- queries -----------------------------------------------------------

    def events(self, branch: str | None = None,
               upto_seq: int | None = None) -> list[Event]:
        """Events of a branch up to (and including) a seq horizon."""
        br = branch or self.branch
        evs = self._chain(br)
        if upto_seq is None:
            return evs
        return [e for e in evs if e.seq <= upto_seq]

    def head(self, branch: str | None = None) -> int:
        """Seq of the branch's head event (-1 for an empty branch)."""
        head_id = self._heads.get(branch or self.branch)
        ev = self._by_id.get(head_id) if head_id else None
        return ev.seq if ev else -1

    def branches(self) -> list[str]:
        return sorted(self._heads.keys())

    def get(self, event_id: str) -> Event | None:
        return self._by_id.get(event_id)

    def __len__(self) -> int:
        return len(self._events)

    # -- time travel -------------------------------------------------------

    def rewind(self, seq: int, branch: str | None = None) -> int:
        """Move a branch's head back to the event at/below seq.

        Events are NOT deleted; they fall off the chain. The move is sealed
        as a 'kernel.rewind' marker (which becomes the new head) so it
        survives reload. Returns the new head seq (the marker's seq).
        """
        br = branch or self.branch
        with self._lock:
            current = self.head(br)
            target = max(-1, min(seq, current))
            base = self._event_at(br, target)
            self._heads[br] = base.id if base else None
            marker = self.append("kernel.rewind",
                                 {"branch": br, "from": current,
                                  "to": target})
            return marker.seq

    def fork(self, at_seq: int | None = None,
             name: str | None = None) -> str:
        """Create a new branch diverging from at_seq (default: current head).

        The new branch's head starts at the fork-point event, so it inherits
        the source's full history up to that point; a 'kernel.branch' marker
        is sealed on it. Returns the branch name."""
        with self._lock:
            src = self.branch
            at = self.head(src) if at_seq is None else at_seq
            base = self._event_at(src, at)
            new_name = name or f"branch-{len(self.branches()) + 1}"
            self._heads[new_name] = base.id if base else None
            self.append("kernel.branch",
                        {"from": src, "at_seq": base.seq if base else -1,
                         "at_id": base.id if base else None,
                         "name": new_name}, branch=new_name)
            return new_name

    def checkout(self, branch: str) -> None:
        if branch in self._heads:
            self.branch = branch

    # -- integrity ---------------------------------------------------------

    def verify(self, branch: str | None = None) -> tuple[bool, str]:
        """Re-hash every event and check the Merkle spine links."""
        br = branch or self.branch
        evs = self.events(br)
        prev_id: str | None = None
        for ev in evs:
            recomputed = Event.compute_id(ev.seq, ev.parent, ev.branch,
                                          ev.ts, ev.type, ev.data,
                                          ev.session, ev.actor,
                                          ev.causation_id,
                                          ev.correlation_id, ev.provenance)
            if recomputed != ev.id:
                return False, f"seq {ev.seq}: content hash mismatch"
            if ev.parent != prev_id:
                return False, f"seq {ev.seq}: broken spine link"
            prev_id = ev.id
        return True, f"{len(evs)} events verified"

    # -- causality ---------------------------------------------------------

    def why(self, event_id: str, limit: int = 50) -> list[Event]:
        """Walk the causation chain backwards from an event to its root.

        Answers 'why did this happen?' mechanically (§7.1, Appendix A
        `argus why`): each event's causation_id names its direct cause, so
        any file change or dollar spent traces back to the human
        instruction that started it.
        """
        chain: list[Event] = []
        cur = self._by_id.get(event_id)
        seen: set[str] = set()
        while cur and cur.id not in seen and len(chain) < limit:
            seen.add(cur.id)
            chain.append(cur)
            cur = self._by_id.get(cur.causation_id) if cur.causation_id else None
        return chain


# ---------------------------------------------------------------------------
# Fold — state as a pure function of history
# ---------------------------------------------------------------------------

# tool name -> whether it mutates the filesystem
_MUTATING_TOOLS = {"write_file", "edit_file", "create_directory",
                   "copy_path", "move_path", "delete_path"}

# Event types of the advanced subsystems folded into State.advanced_events
ADVANCED_EVENT_TYPES = frozenset({
    "compile.plan", "compile.wave", "compile.done",
    "evolution.generation", "evolution.deployed", "evolution.rollback",
    "brain.remembered", "brain.recalled", "brain.consolidated",
    "brain.forgotten",
    "merge.started", "merge.merged", "merge.conflict",
    "theater.counterfactual",
    "debate.round", "debate.verdict", "debate.calibration",
    "market.announce", "market.bid", "market.award", "market.settle",
    # v6 advanced subsystems
    "verify.plan", "verify.violation", "verify.trace",
    "mcts.search", "mcts.best",
    "causal.edge", "causal.intervention",
    "bandit.pull", "bandit.update",
    "mesh.node", "mesh.task", "mesh.result",
    "meta.role.drafted", "meta.role.sealed", "meta.role.rejected",
    "synth.tool.drafted", "synth.tool.tested", "synth.tool.registered",
    "ci.watch", "ci.run", "ci.streak",
    "tuner.trial", "tuner.best",
    "dual.route", "dual.escalation",
    "world.impact", "world.learn",
    "race.start", "race.winner", "race.cancel",
    "homeo.check", "homeo.repair",
    "attention.auction",
    "fabric.assert", "fabric.retract",
})


def _fold_apply(st: State, ev: Event) -> None:
    """Apply ONE event to a State projection (the reduce step)."""
    st.head_seq = ev.seq
    d = ev.data
    t = ev.type

    if t == "user.message":
        st.messages.append({"role": "user", "content": d.get("text", "")})
    elif t == "assistant.message":
        st.messages.append({"role": "assistant",
                            "content": d.get("text", "")})
    elif t == "tool.call":
        st.tool_calls += 1
        name = d.get("name", "")
        if name == "run_command":
            st.commands_run += 1
        if name in _MUTATING_TOOLS and d.get("args"):
            p = d["args"].get("path") or d["args"].get("dst")
            if p:
                st.files_touched.add(str(p))
    elif t == "tool.result":
        if d.get("status") == "error":
            st.tool_errors += 1
    elif t == "cost.incurred":
        st.cost_usd += float(d.get("usd", 0.0))
        st.tokens_in += int(d.get("tokens_in", 0))
        st.tokens_out += int(d.get("tokens_out", 0))
    elif t == "memory.episode":
        st.episodes.append(d)
    elif t == "deadend.recorded":
        st.dead_ends.append(d)
    elif t == "goal.set":
        st.goal = d
        st.goal_done = []
        st.goal_closed = None  # a new contract reopens the world
    elif t == "goal.clause.done":
        st.goal_done.append(d.get("clause", ""))
    elif t == "autonomy.changed":
        st.autonomy = int(d.get("level", st.autonomy))
    elif t in ADVANCED_EVENT_TYPES:
        st.advanced_events.append({"type": t, **d})
    elif t == "prompt.sealed":
        st.prompt_sealed.append(d)
    elif t == "prompt.dispatch":
        st.prompt_dispatches.append(d)
    elif t == "router.decision":
        st.router_decisions.append(d)
    elif t == "semantic.indexed":
        st.semantic_index.append(d)
    elif t in ("spec.prefetch", "spec.hit", "spec.miss", "spec.evict"):
        st.spec_events.append({"type": t, **d})
    elif t in ("daemon.mission", "daemon.checkpoint", "daemon.tick",
               "daemon.wake", "daemon.done"):
        st.daemon_events.append({"type": t, **d})
    elif t in ("heal.captured", "heal.hypothesis", "heal.patch",
               "heal.retry", "heal.lesson"):
        st.heal_events.append({"type": t, **d})
    elif t in ("skill.authored", "skill.validated", "skill.registered",
               "skill.rejected"):
        st.skill_events.append({"type": t, **d})
    elif t in ("council.convened", "council.position", "council.verdict"):
        st.council_events.append({"type": t, **d})
    elif t in ("lsp.session", "lsp.symbols", "lsp.references",
               "lsp.diagnostics"):
        st.lsp_events.append({"type": t, **d})
    elif t in ("dap.session", "dap.breakpoint", "dap.stopped",
               "dap.variables"):
        st.dap_events.append({"type": t, **d})
    elif t in ("analysis.taint", "analysis.complexity",
               "analysis.cycles"):
        st.analysis_events.append({"type": t, **d})
    elif t in ("mutation.run", "mutation.result"):
        st.mutation_events.append({"type": t, **d})
    elif t in ("coverage.run", "coverage.result"):
        st.coverage_events.append({"type": t, **d})
    elif t in ("fuzz.run", "fuzz.crash", "fuzz.shrunk"):
        st.fuzz_events.append({"type": t, **d})
    elif t in ("graph.entity", "graph.relation", "graph.query"):
        st.graph_events.append({"type": t, **d})
    elif t in ("browser.navigate", "browser.action", "browser.extract"):
        st.browser_events.append({"type": t, **d})
    elif t in ("openapi.compiled", "openapi.call"):
        st.openapi_events.append({"type": t, **d})
    elif t in ("db.query", "db.schema"):
        st.db_events.append({"type": t, **d})
    elif t in ("git.diff", "git.commit", "git.blame"):
        st.git_events.append({"type": t, **d})
    elif t in ("ensemble.run", "ensemble.verdict"):
        st.ensemble_events.append({"type": t, **d})
    elif t in ("hybrid.indexed", "hybrid.query"):
        st.hybrid_events.append({"type": t, **d})
    elif t in ("compress.run",):
        st.compress_events.append({"type": t, **d})
    elif t in ("eval.task", "eval.result"):
        st.eval_events.append({"type": t, **d})
    elif t in ("sched.job", "sched.fired"):
        st.sched_events.append({"type": t, **d})
    elif t in ("cache.metrics",):
        st.cache_events.append({"type": t, **d})
    elif t in ("cost.entry",):
        st.cost_ledger.append(d)
    elif t == "judge.verdict":
        st.verdicts.append(d)
    elif t == "snapshot.taken":
        st.snapshots.append(d)
    elif t == "plan.node":
        node = dict(d)
        st.nodes[node.get("id", "")] = node
    elif t == "plan.node.status":
        node = st.nodes.get(d.get("id", ""))
        if node is not None:
            node["status"] = d.get("status", node.get("status"))
            if "attempts" in d:
                node["attempts"] = d["attempts"]
    elif t == "budget.event":
        st.budget_events.append(d)
    elif t == "clause.proven":
        st.clause_proven.append(d)
    elif t == "clause.regressed":
        st.clause_regressed.append(d)
    elif t == "goal.amendment":
        st.amendments.append(d)
    elif t == "goal.focus":
        st.focus_shifts.append(d)
    elif t == "goal.distance":
        st.distance_measures.append(d)
    elif t == "goal.closed":
        st.goal_closed = d
    elif t == "fact.learned":
        st.facts.append(d)
    elif t == "env.digest":
        st.env_digests.append(d)
    elif t == "calibration.sample":
        st.calibration.append(d)
    elif t == "loop.alert":
        st.loop_alerts.append(d)


def fold(log: EventLog, branch: str | None = None,
         upto_seq: int | None = None,
         from_seq: int = -1) -> State:
    """Pure fold: reduce a log prefix into a State projection.

    from_seq skips every event with seq <= from_seq — used for
    session-scoped projections (budget spend etc.) without mutating
    or copying the log.

    SPEED: full-log folds are cached per branch and extended
    INCREMENTALLY — after the first fold, a steady-state call applies
    only the events appended since the last one (typically 1-5), so
    the dozens of fold() calls inside one turn cost near-zero instead
    of O(n) each. Rewinds naturally miss the cache and rebuild.
    Callers must treat the returned State as read-only."""
    if upto_seq is None and from_seq == -1:
        with log._lock:
            return _fold_cached(log, branch or log.branch)
    br = branch or log.branch
    st = State(branch=br)
    for ev in log.events(br, upto_seq):
        if ev.seq <= from_seq:
            continue
        _fold_apply(st, ev)
    return st


def replay(log: EventLog, branch: str | None = None) -> Iterator[Event]:
    """Yield the branch's events in causal order — the timeline as a film."""
    for ev in log.events(branch):
        yield ev


def _fold_cached(log: EventLog, br: str) -> State:
    """Cached, incremental full-log fold for one branch (caller holds
    log._lock)."""
    head_id = log._heads.get(br)
    cached = log._fold_cache.get(br)
    if cached is not None:
        cached_id, st = cached
        if cached_id == head_id:
            return st
        # extend: walk the new events back from the head to the cached
        # head, then apply them in causal order onto the cached state
        new_evs: list[Event] = []
        cur = head_id
        seen: set[str] = set()
        chained = False
        while cur and cur != cached_id and cur not in seen:
            seen.add(cur)
            ev = log._by_id.get(cur)
            if ev is None:
                break
            new_evs.append(ev)
            cur = ev.parent
        else:
            if cur == cached_id:
                chained = True
        if chained:
            for ev in reversed(new_evs):
                _fold_apply(st, ev)
            log._fold_cache[br] = (head_id, st)
            return st
    # first fold on this branch, or the chain diverged (rewind) — rebuild
    st = State(branch=br)
    for ev in log._chain(br):
        _fold_apply(st, ev)
    log._fold_cache[br] = (head_id, st)
    return st


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "kernel-selftest.jsonl"
        log = EventLog(path)

        # basic append: monotonic seqs, intact spine
        for i in range(5):
            log.append("user.message", {"text": f"msg{i}"})
        assert log.head() == 4
        assert [e.seq for e in log.events()] == [0, 1, 2, 3, 4]
        ok, msg = log.verify()
        assert ok, msg

        # rewind: fold horizon shrinks, seqs are NEVER reused afterwards
        log.rewind(2)
        evs = log.events()
        texts = [e.data.get("text") for e in evs if e.type == "user.message"]
        assert texts == ["msg0", "msg1", "msg2"], texts
        assert log.head() == 5  # the kernel.rewind marker is the new head
        log.append("user.message", {"text": "after-rewind"})
        evs = log.events()
        seqs = [e.seq for e in evs]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs
        texts = [e.data.get("text") for e in evs if e.type == "user.message"]
        assert texts == ["msg0", "msg1", "msg2", "after-rewind"], texts
        ok, msg = log.verify()
        assert ok, msg

        # fold sees only the live chain
        st = fold(log)
        assert [m["content"] for m in st.messages] == \
            ["msg0", "msg1", "msg2", "after-rewind"]

        # fork: new branch inherits history up to the fork point
        fork_at = log.head()
        branch = log.fork(at_seq=fork_at, name="alt")
        log.checkout(branch)
        assert log.branch == "alt"
        st = fold(log, branch="alt")
        assert [m["content"] for m in st.messages] == \
            ["msg0", "msg1", "msg2", "after-rewind"]
        log.append("user.message", {"text": "alt-only"})
        # main is untouched by writes on alt
        st_main = fold(log, branch="main")
        assert "alt-only" not in [m["content"] for m in st_main.messages]
        ok, msg = log.verify(branch="alt")
        assert ok, msg

        # reload from disk: heads rebuild, chains and verify survive
        log2 = EventLog(path)
        assert set(log2.branches()) == {"main", "alt"}
        assert log2.head("main") == log.head("main")
        assert log2.head("alt") == log.head("alt")
        st = fold(log2, branch="alt")
        assert [m["content"] for m in st.messages][-1] == "alt-only"
        ok, msg = log2.verify(branch="main")
        assert ok, msg
        ok, msg = log2.verify(branch="alt")
        assert ok, msg

        # rewind survives reload (the marker event is the persisted head)
        log2.checkout("main")
        marker_seq = log2.rewind(1)
        log3 = EventLog(path)
        texts = [e.data.get("text") for e in log3.events("main")
                 if e.type == "user.message"]
        assert texts == ["msg0", "msg1"], texts
        ok, msg = log3.verify("main")
        assert ok, msg

        # content addressing: tampering is detected
        ev = log3.events("main")[1]
        ev.data["text"] = "tampered"
        ok, msg = log3.verify("main")
        assert not ok and "content hash" in msg, (ok, msg)

    # causal envelope: causation chains + why() (§7.1, Appendix A)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "causal.jsonl"
        log = EventLog(path, session="s1")
        root = log.append("user.message", {"text": "fix the bug"},
                          actor="human", provenance="user")
        mid = log.append("tool.call", {"name": "edit_file"},
                         actor="sovereign", causation_id=root.id,
                         correlation_id="C1", provenance="model")
        leaf = log.append("tool.result", {"status": "done"},
                          actor="system", causation_id=mid.id,
                          correlation_id="C1", provenance="tool_output")
        chain = log.why(leaf.id)
        assert [e.type for e in chain] == ["tool.result", "tool.call",
                                           "user.message"], chain
        assert chain[-1].actor == "human"
        assert leaf.correlation_id == "C1"
        ok, msg = log.verify()
        assert ok, msg
        # envelope survives reload
        log2 = EventLog(path, session="s1")
        ev = log2.events()[-1]
        assert ev.causation_id == mid.id and ev.provenance == "tool_output"
        assert [e.type for e in log2.why(ev.id)] == \
            ["tool.result", "tool.call", "user.message"]

    print("KERNEL SELF-TEST PASS")
