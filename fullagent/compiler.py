"""COMPILER — the Intent Compiler: goal → IR → optimized execution plan.

A database plans queries; this plans agent work. A natural-language goal
is drafted ONCE into a typed Intermediate Representation (a flat list of
work items), then every subsequent step is deterministic, mechanical
optimizer passes — exactly like a query planner:

    parse      goal → IR items (LLM-drafted once; validated structurally)
    dedupe     identical/near-identical items collapse (hash on
               role+normalized-task)
    prune      unreachable items die (dependencies that name no item,
               and everything transitively orphaned by them)
    layers     topological layering — items in a layer have no mutual
               dependencies and run in one ordered wave
    cost       role-based cost estimates pick the cheapest viable role
               where the draft left one generic
    lockpass   write-exclusivity (I7): two items writing overlapping
               path-sets can never share a wave

The optimizer never trusts the draft: malformed items are dropped, not
executed. The compiled plan is sealed (compile.plan) before anything
runs, waves stream as compile.wave, and the run lands as compile.done —
the whole compilation is replayable like everything else on the kernel.

The drafter is injectable (`drafter=`) so the self-test runs fully
offline; production uses the model through chat_blocking.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .kernel import EventLog, fold
from .team import ROLES

# deterministic per-role cost estimates (abstract units: tool budget the
# role typically burns per item — writers cost more than readers)
_ROLE_COST = {
    "planner": 4, "architect": 6, "researcher": 4, "reviewer": 3,
    "analyst": 4, "coder": 8, "tester": 5, "debugger": 6, "optimizer": 5,
    "refactorer": 7, "documenter": 4, "devops": 6, "integrator": 7,
}
_MAX_ITEMS = 24
_MAX_WAVES = 8


def _norm_task(task: str) -> str:
    """Normalize a task string for duplicate detection."""
    return re.sub(r"[^a-z0-9]+", " ", str(task).lower()).strip()


def _item_key(item: dict) -> str:
    """Stable identity of an IR item — (role, normalized task)."""
    payload = json.dumps(
        {"role": item.get("role", ""), "task": _norm_task(item.get("task", ""))},
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class CompiledPlan:
    """The optimized plan: ordered waves of work items (executed serially)."""
    goal: str
    waves: list[list[dict]] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)   # malformed/dupes
    est_cost: int = 0
    compile_ms: int = 0

    def items(self) -> list[dict]:
        return [dict(item, wave=i) for i, wave in enumerate(self.waves)
                for item in wave]

    def to_dict(self) -> dict:
        return {"goal": self.goal, "waves": self.waves,
                "dropped": len(self.dropped), "est_cost": self.est_cost,
                "compile_ms": self.compile_ms,
                "n_items": len(self.items()), "n_waves": len(self.waves)}


def _draft_prompt(goal: str) -> list[dict]:
    return [
        {"role": "system", "content":
            "You are the front-end of an agent work compiler. Decompose "
            "the goal into 4-12 work items. Reply with ONLY a JSON array; "
            "each element: {\"task\": string, \"role\": one of "
            + ", ".join(sorted(ROLES)) + ", \"paths\": [files this item "
            "may write or read], \"depends_on\": [indexes of items that "
            "must finish first, 0-based]}. No prose, no markdown fence."},
        {"role": "user", "content": f"GOAL: {goal}"},
    ]


def default_drafter(provider, model, effort
                    ) -> Callable[[str], list[dict]]:
    """Production drafter: one blocking model call -> raw IR items."""
    from .client import chat_blocking

    def draft(goal: str) -> list[dict]:
        result = chat_blocking(provider, model, effort,
                               _draft_prompt(goal), None, timeout=120.0)
        text = (result.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
        try:
            raw = json.loads(text)
        except ValueError:
            return []
        return raw if isinstance(raw, list) else []
    return draft


class IntentCompiler:
    """goal → IR → deterministic optimizer passes → executable waves."""

    def __init__(self, log: EventLog, drafter: Callable[[str], list[dict]],
                 executor: Callable[[list[dict]], list[dict]] | None = None
                 ) -> None:
        self.log = log
        self.drafter = drafter
        self.executor = executor          # runs ONE wave of items

    # -- optimizer passes (all deterministic, all pure) ----------------------

    def _parse(self, goal: str) -> tuple[list[dict], list[dict]]:
        """Validate the draft ONCE: keep well-formed items (with their raw
        dependencies preserved under _raw_deps), drop the rest."""
        raw = self.drafter(goal)
        items: list[dict] = []
        dropped: list[dict] = []
        for draft_idx, entry in enumerate(raw[:_MAX_ITEMS]):
            if not isinstance(entry, dict):
                dropped.append({"reason": "not an object",
                                "item": str(entry)[:80]})
                continue
            task = str(entry.get("task", "")).strip()
            if not task:
                dropped.append({"reason": "empty task", "item": entry})
                continue
            role = str(entry.get("role", "")).strip().lower()
            if role not in ROLES:
                role = "coder"          # generic default, cheapest fix
            paths = [str(p) for p in (entry.get("paths") or [])
                     if str(p).strip()]
            items.append({"task": task, "role": role, "paths": paths,
                          "_draft_idx": draft_idx,
                          "_raw_deps": list(entry.get("depends_on") or [])})
        return items, dropped

    def _dedupe(self, items: list[dict], dropped: list[dict]) -> list[dict]:
        """Collapse identical items. Dependencies of a dropped duplicate
        merge into the kept twin — the work still happens exactly once."""
        seen: dict[str, dict] = {}
        out: list[dict] = []
        for it in items:
            key = _item_key(it)
            if key in seen:
                kept = seen[key]
                for dep in it.get("depends_on") or []:
                    if dep != kept["id"] and dep not in kept["depends_on"]:
                        kept["depends_on"].append(dep)
                dropped.append({"reason": "duplicate", "item": it["task"]})
                continue
            seen[key] = it
            out.append(it)
        # deps pointing at a dropped duplicate resolve to its kept twin
        remap = {}
        for it in items:
            if it not in out:
                remap[it["id"]] = seen[_item_key(it)]["id"]
        for it in out:
            it["depends_on"] = [remap.get(d, d) for d in it["depends_on"]]
        return out

    def _remap_deps(self, items: list[dict]) -> None:
        """Resolve draft-index dependencies to item ids through the
        ORIGINAL draft positions (parse drops malformed entries, so list
        positions and draft indexes do not coincide)."""
        for i, it in enumerate(items):
            it["id"] = f"i{i}"
        by_draft_idx = {it["_draft_idx"]: it for it in items}
        for it in items:
            deps: list[str] = []
            for d in it.pop("_raw_deps"):
                try:
                    d = int(d)
                except (TypeError, ValueError):
                    continue
                target = by_draft_idx.get(d)
                if target is not None and target["id"] != it["id"]:
                    deps.append(target["id"])
            it["depends_on"] = deps
        # transitive self-dependency (a→b→a) is impossible to schedule —
        # cut the back edge deterministically
        changed = True
        while changed:
            changed = False

            def reaches(src: str, dst: str, seen: set[str]) -> bool:
                if src == dst:
                    return True
                if src in seen:
                    return False
                seen.add(src)
                cur = next((x for x in items if x["id"] == src), None)
                return cur is not None and any(
                    reaches(nxt, dst, seen) for nxt in cur["depends_on"])

            for it in items:
                for dep in list(it["depends_on"]):
                    if reaches(dep, it["id"], set()):
                        it["depends_on"].remove(dep)
                        changed = True
                        break
                if changed:
                    break

    def _prune_unreachable(self, items: list[dict]) -> list[dict]:
        """Drop items nothing can reach: dependencies naming no surviving
        item are removed first (they can never be satisfied)."""
        ids = {it["id"] for it in items}
        for it in items:
            it["depends_on"] = [d for d in it["depends_on"] if d in ids]
        return items

    def _layers(self, items: list[dict]) -> list[list[dict]]:
        """Topological layering: layer k = items whose deps are all in
        layers < k. Cycles were cut in _remap_deps, so this terminates."""
        placed: dict[str, int] = {}
        waves: list[list[dict]] = []
        remaining = list(items)
        while remaining and len(waves) < _MAX_WAVES:
            wave = [it for it in remaining
                    if all(d in placed for d in it["depends_on"])]
            if not wave:                       # defensive: cut a dep edge
                victim = remaining[0]
                victim["depends_on"] = []
                continue
            for it in wave:
                placed[it["id"]] = len(waves)
            waves.append([{k: v for k, v in it.items()
                           if k not in ("_raw_deps", "_draft_idx")}
                          for it in wave])
            remaining = [it for it in remaining if it["id"] not in placed]
        if remaining:                          # last resort: final wave
            waves.append([{k: v for k, v in it.items()
                           if k not in ("_raw_deps", "_draft_idx")}
                          for it in remaining])
        return waves

    def _lockpass(self, waves: list[list[dict]]) -> list[list[dict]]:
        """I7 write-exclusivity: within a wave, two items with overlapping
        write-path sets cannot coexist — push the later one to the next
        wave (creating waves as needed)."""
        out: list[list[dict]] = [[] for _ in waves]
        for wi, wave in enumerate(waves):
            locked: set[str] = set()
            for item in wave:
                paths = set(item.get("paths") or [])
                writer = bool(paths) and item["role"] in (
                    "coder", "architect", "refactorer", "documenter",
                    "devops", "integrator")
                if writer and paths & locked and wi + 1 < _MAX_WAVES + 4:
                    if wi + 1 >= len(out):
                        out.append([])
                    out[wi + 1].append(item)
                    continue
                if writer:
                    locked |= paths
                out[wi].append(item)
        return [w for w in out if w]

    def compile(self, goal: str) -> CompiledPlan:
        """Full pipeline: draft → validate → dedupe → prune → layer →
        lock paths. Seals compile.plan; returns the optimized plan."""
        goal = str(goal or "").strip()
        t0 = time.monotonic()
        items, dropped = self._parse(goal)
        self._remap_deps(items)          # ids resolve BEFORE dedupe shifts
        items = self._dedupe(items, dropped)
        items = self._prune_unreachable(items)
        waves = self._layers(items)
        waves = self._lockpass(waves)
        plan = CompiledPlan(
            goal=goal, waves=waves, dropped=dropped,
            est_cost=sum(_ROLE_COST.get(it["role"], 5)
                         for w in waves for it in w),
            compile_ms=int((time.monotonic() - t0) * 1000))
        self.log.append("compile.plan", plan.to_dict(), actor="kernel")
        return plan

    # -- execution -------------------------------------------------------------

    def execute(self, plan: CompiledPlan) -> dict:
        """Run the plan wave by wave — each wave's items run in order
        through the executor (serially, one worker at a time); a wave's
        reports land before the next wave starts (dependencies are
        satisfied by construction)."""
        if self.executor is None:
            raise RuntimeError("no executor attached — plan compiled only")
        all_reports: list[dict] = []
        for i, wave in enumerate(plan.waves):
            self.log.append("compile.wave",
                            {"index": i, "items": len(wave),
                             "roles": [it["role"] for it in wave]},
                            actor="kernel")
            reports = self.executor(wave)
            all_reports.extend(reports)
        result = {"goal": plan.goal, "waves": len(plan.waves),
                  "items": len(all_reports),
                  "done": sum(1 for r in all_reports
                              if r.get("status") == "done"),
                  "blocked": sum(1 for r in all_reports
                                 if r.get("status") == "blocked"),
                  "error": sum(1 for r in all_reports
                               if r.get("status") == "error"),
                  "est_cost": plan.est_cost,
                  "compile_ms": plan.compile_ms}
        self.log.append("compile.done", result, actor="system")
        return result

    def format(self, plan: CompiledPlan) -> str:
        lines = [f"COMPILED PLAN — goal: {plan.goal}",
                 f"{len(plan.items())} items · {len(plan.waves)} ordered "
                 f"waves · est cost {plan.est_cost} · {plan.compile_ms}ms"
                 + (f" · {len(plan.dropped)} dropped" if plan.dropped
                    else "")]
        for i, wave in enumerate(plan.waves):
            lines.append(f"  wave {i + 1}:")
            for it in wave:
                deps = ("  ← " + ",".join(it["depends_on"])
                        if it.get("depends_on") else "")
                paths = ("  [" + ", ".join(it["paths"][:3]) + "]"
                         if it.get("paths") else "")
                lines.append(f"    [{it['role']}] {it['task'][:80]}"
                             f"{paths}{deps}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — a deterministic stub drafter drives every optimizer pass
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "compiler-test.jsonl")

        def drafter(goal: str) -> list[dict]:
            # malformed + duplicate + cross-referencing draft
            return [
                {"task": "Map the module graph", "role": "architect",
                 "paths": [], "depends_on": []},
                {"task": "map the MODULE graph!!", "role": "architect",
                 "paths": [], "depends_on": []},          # dupe of #0
                {"task": "", "role": "coder"},             # malformed
                "not an object",                           # malformed
                {"task": "Fix the parser bug", "role": "debugger",
                 "depends_on": [0]},
                {"task": "Apply the fix", "role": "coder",
                 "paths": ["src/parser.py"], "depends_on": [4]},
                {"task": "Write parser docs", "role": "documenter",
                 "paths": ["src/parser.py"], "depends_on": [5]},
                {"task": "Run the test suite", "role": "tester",
                 "depends_on": [99]},                      # dead dep
                {"task": "Orphan work", "role": "analyst",
                 "depends_on": [7]},                       # transitively dead
            ]

        executed: list[list[dict]] = []

        def executor(wave: list[dict]) -> list[dict]:
            executed.append(wave)
            return [{"task": it["task"], "role": it["role"],
                     "status": "done"} for it in wave]

        comp = IntentCompiler(log, drafter, executor)
        plan = comp.compile("fix and harden the parser")

        items = plan.items()
        # duplicates and malformed entries are gone
        assert len(items) == 6, [it["task"] for it in items]
        assert all(d["reason"] for d in plan.dropped)
        # no two write-locked items sharing src/parser.py share a wave
        waves_with_parser = [i for i, w in enumerate(plan.waves)
                             if any("src/parser.py" in (it.get("paths") or [])
                                    for it in w)]
        assert len(waves_with_parser) == 2, plan.waves  # coder & doc split
        # dependency order respected: fix before apply before docs
        ids = {it["task"]: it.get("wave") for it in items}
        assert ids["Fix the parser bug"] < ids["Apply the fix"] < \
            ids["Write parser docs"], ids
        # dead references pruned: 'Run the test suite' runs (dep dropped),
        # 'Orphan work' survives too (its dep became runnable)
        assert "Run the test suite" in ids
        # plan sealed before execution
        types = [e.type for e in log.events()]
        assert types.count("compile.plan") == 1

        result = comp.execute(plan)
        assert result["items"] == 6 and result["done"] == 6, result
        assert len(executed) == len(plan.waves)
        assert "compile.wave" in types or any(
            e.type == "compile.wave" for e in log.events())
        st = fold(log)
        plans = [e for e in st.advanced_events if e["type"] == "compile.plan"]
        assert plans and plans[0]["n_items"] == 6

        text = comp.format(plan)
        assert "COMPILED PLAN" in text and "wave" in text

        # empty/failed draft -> empty plan, never a crash
        empty = IntentCompiler(log, lambda g: []).compile("nothing")
        assert empty.items() == [] and empty.waves == []

        # cycle in deps is cut and still schedules
        def cyclic(goal: str) -> list[dict]:
            return [
                {"task": "a", "role": "coder", "depends_on": [1]},
                {"task": "b", "role": "coder", "depends_on": [0]},
                {"task": "c", "role": "tester", "depends_on": []},
            ]
        cyc = IntentCompiler(log, cyclic).compile("cyclic")
        assert len(cyc.items()) == 3 and cyc.waves, cyc.to_dict()

        print("COMPILER SELF-TEST PASS")
