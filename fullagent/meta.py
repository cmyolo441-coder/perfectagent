"""META — the agent that creates agents.

The roster is not fixed. When work demands a specialist that does not
exist, the RoleForge drafts one:

    draft       the model proposes a new role: name, brief, tool
                whitelist, benchmark (the drafter is injectable — the
                self-test runs offline with a scripted draft)
    validate    mechanical gates, never prompt advice: the name must be
    new (no shadowing an existing role), the brief must be substantial,
    every tool must exist in the registry, the whitelist must respect
    read-only vs write separation
    audition    the drafted role runs its own benchmark with its brief
    (injectable evaluator) and must clear the pass bar — a role that
    cannot do its own job is never sealed
    seal        the role goes LIVE: ROLES, ROLE_BRIEFS, the worker
    prompt registry, the vault, and every team/crew toolset — the
    roster grows at runtime

Every transition is sealed (meta.role.drafted / sealed / rejected), so
the roster's evolution is replayable — and a role can always be checked
against its audition record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import systemprompt
from .kernel import EventLog
from .team import ROLES

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,20}$")
_MIN_BRIEF_CHARS = 80
_PASS_BAR = 0.6


@dataclass
class RoleDraft:
    name: str
    brief: str
    tools: list[str]
    benchmark: str
    writes: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "brief": self.brief[:400],
                "tools": self.tools, "benchmark": self.benchmark[:200],
                "writes": self.writes}


def default_drafter(provider, model, effort):
    """Production drafter: one blocking model call -> a role draft."""
    from .client import chat_blocking
    import json

    def draft(mission: str) -> dict:
        result = chat_blocking(
            provider, model, effort,
            [{"role": "system", "content":
                "You design agent specialists. Reply ONLY with a JSON "
                "object: {\"name\": snake_case_id, \"brief\": one strong "
                "paragraph (>=100 words) telling this specialist exactly "
                "how to work, \"tools\": subset of the allowed list, "
                "\"benchmark\": one task proving the role works}. No "
                "prose around the JSON."},
             {"role": "user",
              "content": f"MISSION: {mission}\nALLOWED TOOLS: "
                         + ", ".join(sorted(_all_tool_names()))}],
            None, timeout=120.0)
        text = (result.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
        try:
            return json.loads(text)
        except ValueError:
            return {}
    return draft


def _all_tool_names() -> list[str]:
    """Every tool name the role whitelists may draw from (the base
    registry's read/write shell tools)."""
    from .tools import build_registry
    return sorted(build_registry())


class RoleForge:
    """Draft → validate → audition → seal new specialist roles."""

    def __init__(self, log: EventLog, drafter, evaluator,
                 pass_bar: float = _PASS_BAR) -> None:
        """evaluator(draft) -> float in [0, 1] — the audition score."""
        self.log = log
        self.drafter = drafter
        self.evaluator = evaluator
        self.pass_bar = pass_bar

    # -- the pipeline --------------------------------------------------------

    def forge(self, mission: str) -> tuple[str, str]:
        """Create one new role for the mission. Returns (status, msg)
        with status in {"sealed", "rejected"}."""
        mission = str(mission or "").strip()
        if not mission:
            return "rejected", "empty mission"
        raw = self.drafter(mission)
        draft, problem = self._validate(raw)
        self.log.append("meta.role.drafted",
                        {"mission": mission[:200],
                         "draft": draft.to_dict() if draft else None,
                         "problem": problem}, actor="sovereign")
        if draft is None:
            return "rejected", problem

        score = self.evaluator(draft)
        if score < self.pass_bar:
            self.log.append("meta.role.rejected",
                            {"name": draft.name,
                             "reason": f"audition {score:.2f} < "
                                       f"{self.pass_bar}",
                             "score": score}, actor="kernel")
            return "rejected", (f"role '{draft.name}' failed its "
                                f"audition ({score:.2f} < "
                                f"{self.pass_bar:.2f}) — nothing sealed")
        self._seal(draft)
        self.log.append("meta.role.sealed",
                        {"name": draft.name, "score": round(score, 3),
                         **draft.to_dict()}, actor="kernel")
        return "sealed", (f"role '{draft.name}' is LIVE — sealed at "
                          f"audition {score:.2f}, available to team/"
                          f"crew/squad and sealed in the vault")

    # -- mechanical gates --------------------------------------------------------

    def _validate(self, raw: dict) -> tuple["RoleDraft | None", str]:
        if not isinstance(raw, dict):
            return None, "draft is not an object"
        name = str(raw.get("name", "")).strip().lower()
        if not _NAME_RE.match(name):
            return None, f"bad role name {name!r} (snake_case, 3-21 chars)"
        if name in ROLES:
            return None, f"role '{name}' already exists — refusing to " \
                         "shadow it"
        brief = str(raw.get("brief", "")).strip()
        if len(brief) < _MIN_BRIEF_CHARS:
            return None, (f"brief too thin ({len(brief)} chars < "
                          f"{_MIN_BRIEF_CHARS}) — a specialist needs "
                          "real instructions")
        known = set(_all_tool_names())
        tools = [str(t).strip() for t in (raw.get("tools") or [])
                 if str(t).strip() in known]
        if not tools:
            return None, "no valid tools in the whitelist"
        benchmark = str(raw.get("benchmark", "")).strip()
        if not benchmark:
            return None, "no benchmark — a role must prove itself"
        writes = bool({"write_file", "edit_file", "create_directory",
                       "delete_path", "move_path", "copy_path",
                       "run_command"} & set(tools))
        return RoleDraft(name=name, brief=brief, tools=sorted(tools),
                         benchmark=benchmark, writes=writes), ""

    # -- going live -----------------------------------------------------------------

    def _seal(self, draft: RoleDraft) -> None:
        """Install the role everywhere a role must be."""
        ROLES[draft.name] = {"tools": tuple(draft.tools),
                             "writes": draft.writes}
        systemprompt.ROLE_BRIEFS[draft.name] = draft.brief
        systemprompt.register(f"worker:{draft.name}",
                              systemprompt.worker(draft.name, 8))

    def roster(self) -> list[str]:
        return sorted(ROLES)


# ---------------------------------------------------------------------------
# Self-test — scripted drafter/evaluator drive the full forge, offline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "meta.jsonl")

        def good_drafter(mission: str) -> dict:
            return {"name": "sql_surgeon",
                    "brief": "You are a database surgery specialist. "
                             "Diagnose schema pain, write precise "
                             "migrations, verify with real queries, and "
                             "never destroy data without a rollback "
                             "path. You read the schema first, form a "
                             "migration plan, apply it, and prove it "
                             "with a round-trip query.",
                    "tools": ["read_file", "write_file", "run_command"],
                    "benchmark": "add an index to users.email and prove "
                                 "the query plan uses it"}

        forge = RoleForge(log, good_drafter, lambda d: 0.9)
        from .team import ROLES as _ROLES
        from . import systemprompt as _sp
        before = set(_ROLES)

        status, msg = forge.forge("we need a database specialist")
        assert status == "sealed", msg
        assert "sql_surgeon" in _ROLES and "sql_surgeon" in _sp.ROLE_BRIEFS
        assert _ROLES["sql_surgeon"]["writes"] is True   # has write tools
        assert f"worker:sql_surgeon" in _sp.PROMPTS      # registered live
        assert set(_ROLES["sql_surgeon"]["tools"]) <= set(
            _all_tool_names())

        # a Team built AFTER the seal carries the new toolset
        from .team import Team
        from .config import Effort, Model, Provider
        team = Team(log, Provider(key="t", name="T",
                                  base_url="http://t", api_key="x",
                                  color="#fff"),
                    Model(id="s", provider="t", label="S"),
                    Effort(key="low", label="L", color="#f",
                           max_tokens=8, temperature=0.0,
                           reasoning_effort=None, description="t"))
        assert "sql_surgeon" in team._toolsets
        assert "run_command" in team._toolsets["sql_surgeon"]

        # duplicate names never shadow
        dup = RoleForge(log, good_drafter, lambda d: 0.99)
        status2, msg2 = dup.forge("another db specialist")
        assert status2 == "rejected" and "already exists" in msg2

        # thin briefs, bad names, ghost tools are rejected mechanically
        cases = [
            ({"name": "Bad Name!", "brief": "x" * 200, "tools":
              ["read_file"], "benchmark": "b"}, "bad role name"),
            ({"name": "ghost", "brief": "x" * 200, "tools":
              ["not_a_tool"], "benchmark": "b"}, "no valid tools"),
            ({"name": "thin", "brief": "too short", "tools":
              ["read_file"], "benchmark": "b"}, "brief too thin"),
            ({"name": "nobench", "brief": "x" * 200, "tools":
              ["read_file"], "benchmark": ""}, "no benchmark"),
            ("not a dict", "not an object"),
        ]
        for raw, why in cases:
            forge2 = RoleForge(log, lambda m: raw, lambda d: 1.0)
            st, m = forge2.forge("x")
            assert st == "rejected" and why in m, (raw, m)

        # a failed audition seals nothing (fresh name so the audition,
        # not the duplicate gate, is what rejects it)
        def fresh_drafter(mission: str) -> dict:
            d = good_drafter(mission)
            d["name"] = "data_wrangler"
            return d

        fail_aud = RoleForge(log, fresh_drafter, lambda d: 0.2)
        st3, msg3 = fail_aud.forge("db help")
        assert st3 == "rejected" and "audition" in msg3, msg3
        assert "data_wrangler" not in _ROLES

        # empty mission is a clean rejection
        assert forge.forge("")[0] == "rejected"

        # lineage sealed
        kinds = [e.type for e in log.events()]
        assert "meta.role.drafted" in kinds and \
            "meta.role.sealed" in kinds and "meta.role.rejected" in kinds

        print("META SELF-TEST PASS")
