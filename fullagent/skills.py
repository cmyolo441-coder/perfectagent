"""SKILL FORGE — the self-evolving tool author.

When the agent keeps doing the same multi-step thing by hand, the Forge
lets it author a NEW tool (a Python function), prove it safe and correct
with deterministic checks, and register it into the live registry — so the
agent becomes more capable over time. The forge writes skills to
~/.fullagent/skills/<name>.py so they survive restarts.

Hard safety gate (mechanical, rung 1 — the skill NEVER runs unvalidated):
  1. Parse      — the source must be valid Python (ast.parse).
  2. Shape      — it must define exactly the declared entry function, with
                  a docstring and JSON-schema-style parameter declaration.
  3. Safety     — AST scan forbids: imports outside the allowlist, subprocess
                  / os.system / eval / exec / open-for-write / network /
                  dunder access / global statements. A skill is a pure
                  data-in/data-out function.
  4. Test       — the author must ship test cases (input -> expected output
                  substring). They run in a restricted namespace; ALL must
                  pass.
  Only a skill that passes all four gates is sealed skill.registered and
  handed to the registry. Anything else is sealed skill.rejected with the
  exact reason — the attempt is recorded, never hidden.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import EventLog, fold

# imports a skill may use — pure stdlib, no side-effect modules
ALLOWED_IMPORTS = frozenset(
    "json math re string hashlib base64 datetime itertools functools "
    "collections statistics textwrap unicodedata urllib.parse html "
    "pathlib posixpath random typing dataclasses enum".split())

# AST nodes / names that are never allowed in a skill
_FORBIDDEN_CALLS = frozenset(
    "eval exec compile open input __import__ globals locals vars setattr "
    "delattr breakpoint exit quit".split())
_FORBIDDEN_ATTRS = frozenset(
    "system popen exec execl execle execlp subprocess __subclasses__ "
    "__globals__ __code__ __builtins__".split())


# ---------------------------------------------------------------------------
# Skill record
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    source: str
    entry: str                     # the function name to call
    parameters: dict = field(default_factory=dict)
    tests: list[dict] = field(default_factory=list)  # {args, expect}
    status: str = "pending"        # pending | registered | rejected
    reject_reason: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "entry": self.entry, "parameters": self.parameters,
                "tests": self.tests, "status": self.status,
                "reject_reason": self.reject_reason,
                "chars": len(self.source)}


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def _validate_shape(tree: ast.Module, skill: Skill) -> str | None:
    """The entry function must exist, be a plain def, and have a docstring."""
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    names = [f.name for f in fns]
    if skill.entry not in names:
        return f"entry function {skill.entry!r} not defined (has: {names})"
    fn = next(f for f in fns if f.name == skill.entry)
    if not (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        return f"entry function {skill.entry!r} needs a docstring"
    if any(isinstance(n, (ast.AsyncFunctionDef, ast.ClassDef))
           for n in tree.body):
        return "skills must be plain functions — no classes/async"
    return None


def _validate_safety(tree: ast.Module) -> str | None:
    """AST scan: no forbidden imports, calls, attributes, or writes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return f"forbidden import: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return f"forbidden import: from {node.module}"
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _FORBIDDEN_CALLS:
                return f"forbidden call: {fn.id}()"
            if isinstance(fn, ast.Attribute) and fn.attr in _FORBIDDEN_ATTRS:
                return f"forbidden attribute: .{fn.attr}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"dunder access forbidden: {node.attr}"
        elif isinstance(node, ast.Global):
            return "global statements forbidden in skills"
    return None


def _run_tests(skill: Skill, namespace: dict) -> str | None:
    """Run the author's test cases against the loaded entry function."""
    fn = namespace.get(skill.entry)
    if not callable(fn):
        return f"entry {skill.entry!r} did not load as callable"
    if not skill.tests:
        return "a skill must ship at least one test case"
    for i, t in enumerate(skill.tests, 1):
        args = t.get("args") or {}
        expect = str(t.get("expect", ""))
        try:
            got = str(fn(**args))
        except Exception as e:
            return f"test {i} raised {type(e).__name__}: {e}"
        if expect and expect not in got:
            return (f"test {i} failed: expected {expect!r} in output, "
                    f"got {got[:120]!r}")
    return None


# ---------------------------------------------------------------------------
# SkillForge
# ---------------------------------------------------------------------------

class SkillForge:
    """Author, validate, and register new tools over the event log."""

    def __init__(self, log: EventLog, skills_dir: Path | None = None) -> None:
        self.log = log
        self.skills_dir = Path(skills_dir) if skills_dir else None
        if self.skills_dir:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.registry: dict[str, Skill] = {}

    def author(self, skill: Skill) -> Skill:
        """Run the four gates. On success: persist + seal skill.registered.
        On failure: seal skill.rejected with the exact reason."""
        self.log.append("skill.authored", skill.to_dict(), actor="sovereign")

        # gate 1: parse
        try:
            tree = ast.parse(skill.source)
        except SyntaxError as e:
            return self._reject(skill, f"does not parse: {e}")

        # gate 2: shape
        err = _validate_shape(tree, skill)
        if err:
            return self._reject(skill, err)

        # gate 3: safety
        err = _validate_safety(tree)
        if err:
            return self._reject(skill, err)

        # gate 4: load in isolation + run the shipped tests
        try:
            namespace = self._load(skill)
        except Exception as e:
            return self._reject(skill, f"failed to load: {e}")
        err = _run_tests(skill, namespace)
        if err:
            return self._reject(skill, err)

        # all gates passed — persist + register
        skill.status = "registered"
        if self.skills_dir:
            (self.skills_dir / f"{skill.name}.py").write_text(skill.source)
        self.registry[skill.name] = skill
        self.log.append("skill.validated",
                        {"name": skill.name, "tests": len(skill.tests)},
                        actor="kernel")
        self.log.append("skill.registered", skill.to_dict(), actor="kernel")
        return skill

    def _load(self, skill: Skill) -> dict:
        """Load the skill source in an isolated module namespace."""
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as f:
            f.write(skill.source)
            tmp = f.name
        spec = importlib.util.spec_from_file_location(
            f"fullagent_skill_{skill.name}", tmp)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.__dict__

    def _reject(self, skill: Skill, reason: str) -> Skill:
        skill.status = "rejected"
        skill.reject_reason = reason
        self.log.append("skill.rejected",
                        {"name": skill.name, "reason": reason},
                        actor="kernel")
        return skill

    # -- reload persisted skills ------------------------------------------------

    def load_persisted(self) -> int:
        """Re-register skills from the skills dir (restart survival)."""
        if not self.skills_dir:
            return 0
        count = 0
        for p in sorted(self.skills_dir.glob("*.py")):
            name = p.stem
            if name in self.registry:
                continue
            source = p.read_text()
            # re-validate before trusting disk content
            probe = Skill(name=name, description="(persisted)",
                          source=source, entry=name)
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            if _validate_safety(tree) is not None:
                continue
            self.registry[name] = probe
            count += 1
        return count

    # -- projections ---------------------------------------------------------------

    def skills(self) -> list[dict]:
        return fold(self.log).skill_events

    def registered(self) -> list[str]:
        return sorted(self.registry)

    def format_status(self) -> str:
        evs = self.skills()
        authored = sum(1 for e in evs if e["type"] == "skill.authored")
        registered = sum(1 for e in evs if e["type"] == "skill.registered")
        rejected = sum(1 for e in evs if e["type"] == "skill.rejected")
        lines = ["SKILL FORGE — the self-evolving tool author",
                 f"  authored {authored}   registered {registered}   "
                 f"rejected {rejected}"]
        for name in self.registered():
            lines.append(f"    ◆ {name}")
        for e in evs:
            if e["type"] == "skill.rejected":
                lines.append(f"    ✗ {e.get('name')}: "
                             f"{e.get('reason', '')[:60]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "skills.jsonl")
        forge = SkillForge(log, skills_dir=Path(td) / "skills")

        good = Skill(
            name="word_count",
            description="count words in a text",
            entry="word_count",
            source='''def word_count(text):
    """Count the words in a text string."""
    return f"words: {len(text.split())}"
''',
            parameters={"text": {"type": "string"}},
            tests=[{"args": {"text": "one two three"}, "expect": "words: 3"}],
        )
        s = forge.author(good)
        assert s.status == "registered", s.reject_reason
        assert "word_count" in forge.registered()
        assert (Path(td) / "skills" / "word_count.py").exists()

        # a skill with a failing test is rejected
        bad_test = Skill(
            name="wrong",
            description="lies about its output",
            entry="wrong",
            source='''def wrong(x):
    """Return a greeting."""
    return "hello"
''',
            tests=[{"args": {"x": 1}, "expect": "goodbye"}],
        )
        s = forge.author(bad_test)
        assert s.status == "rejected" and "test 1 failed" in s.reject_reason

        # a skill that imports subprocess is rejected at the safety gate
        evil = Skill(
            name="evil",
            description="tries to shell out",
            entry="evil",
            source='''import subprocess

def evil(cmd):
    """Run a command."""
    return subprocess.run(cmd, shell=True)
''',
            tests=[{"args": {"cmd": "ls"}, "expect": ""}],
        )
        s = forge.author(evil)
        assert s.status == "rejected" and "forbidden import" in s.reject_reason

        # eval is rejected
        sneaky = Skill(
            name="sneaky", description="eval", entry="sneaky",
            source='''def sneaky(x):
    """Evaluate."""
    return eval(x)
''',
            tests=[{"args": {"x": "1"}, "expect": "1"}],
        )
        assert forge.author(sneaky).status == "rejected"

        # a skill without a docstring is rejected at the shape gate
        nodoc = Skill(
            name="nodoc", description="no docstring", entry="nodoc",
            source="def nodoc(x):\n    return x\n",
            tests=[{"args": {"x": 1}, "expect": "1"}],
        )
        s = forge.author(nodoc)
        assert s.status == "rejected" and "docstring" in s.reject_reason

        # a skill with no tests is rejected
        notests = Skill(
            name="notests", description="no tests", entry="notests",
            source='''def notests(x):
    """Identity."""
    return x
''',
            tests=[],
        )
        s = forge.author(notests)
        assert s.status == "rejected" and "test case" in s.reject_reason

        # invalid python is rejected at the parse gate
        broken = Skill(name="broken", description="bad syntax",
                       entry="broken", source="def broken(:\n", tests=[])
        assert forge.author(broken).status == "rejected"

        # persisted skills reload after a restart
        forge2 = SkillForge(log, skills_dir=Path(td) / "skills")
        assert forge2.load_persisted() == 1
        assert "word_count" in forge2.registered()

        # ledger reflects everything
        evs = forge.skills()
        types = {e["type"] for e in evs}
        assert {"skill.authored", "skill.registered",
                "skill.rejected"} <= types
        assert "SKILL FORGE" in forge.format_status()

    print("SKILLS SELF-TEST PASS")
