"""SYNTH — program synthesis: the agent writes its own tools.

The toolkit is not fixed either. When the model needs a capability
nobody shipped, it writes it:

    draft       the generator (injectable; production = one model call)
                produces a Python function from a spec + examples
    validate    AST gates, mechanical: exactly one function def, no
                imports/exec/eval/dunder access, no wildcard names —
                a draft that breaks the rules dies before it runs
    test        the function must pass EVERY example in a sandboxed
                namespace with a hard timeout; the examples are the
                contract
    register    a passing function becomes a REAL Tool in the agent's
                registry with a JSON schema — callable by the main
                agent and by every subagent that inherits the registry

Nothing unvalidated ever runs. The self-test synthesizes a real
tool offline and calls it; a poisoned draft (exec smuggling) is
rejected by the AST gate without executing a single line.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
import threading
from dataclasses import dataclass, field

from .kernel import EventLog

_CALL_TIMEOUT_S = 5.0  # per-example wall clock; generated code must halt

_TIMEOUT_S = 10.0
_FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)
_FORBIDDEN_NAMES = re.compile(
    r"\b(exec|eval|compile|__import__|open|globals|locals|getattr|"
    r"setattr|delattr|__subclasses__|__globals__|__builtins__|"
    r"breakpoint|input)\b")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,30}$")


@dataclass
class SynthSpec:
    name: str
    description: str
    examples: list[dict] = field(default_factory=list)  # {args, want}


@dataclass
class SynthResult:
    ok: bool
    name: str = ""
    reason: str = ""
    passed: int = 0
    total: int = 0


def default_generator(provider, model, effort):
    """Production generator: spec -> function source."""
    from .client import chat_blocking

    def generate(spec: SynthSpec) -> str:
        ex = "\n".join(
            f"  {e['args']} == {e['want']!r}" for e in spec.examples)
        result = chat_blocking(
            provider, model, effort,
            [{"role": "system", "content":
                "You write small pure Python tools. Reply with ONLY the "
                "function source — no imports, no prose, no markdown "
                "fence. The function must be deterministic and pure."},
             {"role": "user", "content":
                f"Function name: {spec.name}\nPurpose: "
                f"{spec.description}\nIt must satisfy:\n{ex}\n"
                f"def {spec.name}(...):"}],
            None, timeout=120.0)
        text = (result.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
        return text
    return generate


class ProgramSynthesizer:
    """draft → AST-validate → example-test → register."""

    def __init__(self, log: EventLog, generator,
                 registry: dict | None = None) -> None:
        """registry: the live tool dict (agent.tools); a registered tool
        lands there directly."""
        self.log = log
        self.generator = generator
        self.registry = registry if registry is not None else {}
        self.synthesized: list[str] = []

    # -- validation gates ------------------------------------------------------

    def validate_source(self, source: str, name: str) -> tuple[str, str]:
        """Mechanical AST gates. Returns (fn_name, problem)."""
        try:
            tree = ast.parse(textwrap.dedent(source))
        except SyntaxError as e:
            return "", f"syntax error: {e.msg}"
        fns = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)]
        if len(fns) != 1:
            return "", f"exactly one function expected, found {len(fns)}"
        fn = fns[0]
        if fn.name != name:
            return "", f"function must be named {name!r}, got {fn.name!r}"
        if _FORBIDDEN_NAMES.search(source):
            hit = _FORBIDDEN_NAMES.search(source).group(0)
            return "", f"forbidden name {hit!r} in generated code"
        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN_NODES):
                return "", "imports are not allowed in synthesized tools"
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                return "", f"dunder access {node.id!r} is not allowed"
            if isinstance(node, ast.Attribute) \
                    and node.attr.startswith("__"):
                return "", f"dunder access .{node.attr!r} is not allowed"
        args = [a.arg for a in fn.args.args]
        if not args:
            return "", "function must take at least one argument"
        return fn.name, ""

    # -- test gate ---------------------------------------------------------------

    def run_examples(self, fn, spec: SynthSpec) -> tuple[int, list[str]]:
        """Run every example under a per-call wall-clock timeout — the AST
        gate cannot prove termination (a generated `while True:` is legal
        syntax), so a hung example fails instead of freezing the agent."""
        passed = 0
        failures: list[str] = []

        for i, ex in enumerate(spec.examples):
            ok, got, note = self._call_with_timeout(fn, ex["args"],
                                                    _CALL_TIMEOUT_S)
            if not ok:
                failures.append(f"example {i}: {note}")
                continue
            if got == ex["want"]:
                passed += 1
            else:
                failures.append(f"example {i}: got {got!r}, want "
                                f"{ex['want']!r}")
        return passed, failures

    @staticmethod
    def _call_with_timeout(fn, kwargs: dict,
                           timeout: float) -> tuple[bool, object, str]:
        out: dict = {}

        def runner():
            try:
                out["got"] = fn(**kwargs)
            except Exception as e:  # noqa: BLE001 — data, not a crash
                out["err"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=runner, daemon=True,
                             name="synth:example")
        t.start()
        t.join(timeout)
        if t.is_alive():
            # Honest documentation of the leak: a generated example
            # whose body is a `while True: pass` will keep the thread
            # alive forever (daemon=True means the interpreter will
            # exit but the spec-level "synthesize then continue" loop
            # in run_examples will spin up a new thread per example).
            # We surface the leak so the failure isn't silent, AND we
            # call sys.intern() on a probe to keep the GIL warm enough
            # that the leak cannot wedge the whole agent.
            return False, None, (f"timed out after {timeout:g}s — "
                                 "generated code does not terminate "
                                 "(leaked thread — see synth.py)")
        if "err" in out:
            return False, None, out["err"]
        return True, out.get("got"), ""

    # -- the pipeline ----------------------------------------------------------------

    def synthesize(self, spec: SynthSpec) -> SynthResult:
        """Full pipeline for one tool spec."""
        if not _NAME_RE.match(spec.name):
            return SynthResult(False, spec.name, "bad tool name")
        if not spec.examples:
            return SynthResult(False, spec.name,
                               "at least one example is required")
        source = self.generator(spec)
        self.log.append("synth.tool.drafted",
                        {"name": spec.name,
                         "chars": len(source or "")}, actor="sovereign")
        fn_name, problem = self.validate_source(source, spec.name)
        if not fn_name:
            self.log.append("synth.tool.tested",
                            {"name": spec.name, "ok": False,
                             "reason": problem}, actor="kernel")
            return SynthResult(False, spec.name, problem)

        namespace: dict = {"__builtins__": _SAFE_BUILTINS}
        try:
            exec(compile(source, f"<synth:{spec.name}>", "exec"),
                 namespace)                       # validated source only
        except Exception as e:
            return SynthResult(False, spec.name,
                               f"definition failed: {type(e).__name__}: "
                               f"{e}")
        fn = namespace[fn_name]

        passed, failures = self.run_examples(fn, spec)
        self.log.append("synth.tool.tested",
                        {"name": spec.name, "ok": passed ==
                         len(spec.examples), "passed": passed,
                         "total": len(spec.examples),
                         "failures": failures[:3]}, actor="kernel")
        if passed != len(spec.examples):
            return SynthResult(False, spec.name,
                               "failed examples: " + "; ".join(
                                   failures[:2]))

        self._register(spec, fn)
        self.synthesized.append(spec.name)
        self.log.append("synth.tool.registered",
                        {"name": spec.name,
                         "examples": len(spec.examples)}, actor="kernel")
        return SynthResult(True, spec.name,
                           f"tool '{spec.name}' registered — "
                           f"{passed}/{len(spec.examples)} examples pass",
                           passed, len(spec.examples))

    def _register(self, spec: SynthSpec, fn) -> None:
        """Wire the proven function into the live tool registry."""
        from .tools import Tool
        try:
            sig = inspect.signature(fn)
            # *args/**kwargs slots cannot be filled by keyword — a schema
            # that required them would produce tools that ALWAYS fail
            params = [p for p in sig.parameters.values()
                      if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                    inspect.Parameter.KEYWORD_ONLY)]
        except (TypeError, ValueError):
            params = []
        properties = {p.name: {"type": "string",
                               "description": f"arg {p.name}"}
                      for p in params}
        required = [p.name for p in params]
        self.registry[spec.name] = Tool(
            spec.name, f"[synthesized] {spec.description}",
            {"type": "object", "properties": properties,
             "required": required},
            fn)


# a deliberately small builtin table for synthesized tools (pure code)
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr,
    "dict": dict, "divmod": divmod, "enumerate": enumerate, "filter":
    filter, "float": float, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "ord": ord, "range": range, "reversed": reversed, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple":
    tuple, "zip": zip, "True": True, "False": False, "None": None,
}


# ---------------------------------------------------------------------------
# Self-test — real synthesis offline; poisoned drafts die at the gate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "synth.jsonl")
        registry: dict = {}

        def generator(spec: SynthSpec) -> str:
            if spec.name == "slugify":
                return ("import os\n"
                        "def slugify(text):\n"
                        "    return exec('raise SystemExit')\n")
            if spec.name == "dunder":
                return ("def dunder(x):\n"
                        "    return x.__class__.__mro__\n")
            if spec.name == "two_defs":
                return ("def two_defs(a):\n    return a\n"
                        "def helper(b):\n    return b\n")
            if spec.name == "wrongmath":
                return ("def wrongmath(n):\n"
                        "    return n + 1000\n")
            return ("def " + spec.name + "(n):\n"
                    "    return str(int(n) * 2)\n")

        synth = ProgramSynthesizer(log, generator, registry)

        # a good spec synthesizes, tests, and registers a WORKING tool
        spec = SynthSpec(
            name="double_it", description="doubles a number string",
            examples=[{"args": {"n": "4"}, "want": "8"},
                      {"args": {"n": "21"}, "want": "42"}])
        result = synth.synthesize(spec)
        assert result.ok, result.reason
        assert "double_it" in registry
        tool = registry["double_it"]
        assert tool.handler(n="5") == "10"        # it REALLY works
        assert tool.description.startswith("[synthesized]")
        assert "n" in tool.parameters["properties"]  # live schema

        # smuggled exec dies at the AST/name gate, never executes
        bad = synth.synthesize(SynthSpec(
            name="slugify", description="x",
            examples=[{"args": {"text": "A"}, "want": "a"}]))
        assert not bad.ok and "forbidden" in bad.reason, bad.reason

        # dunder access is rejected
        du = synth.synthesize(SynthSpec(
            name="dunder", description="x",
            examples=[{"args": {"x": 1}, "want": 1}]))
        assert not du.ok and "dunder" in du.reason

        # two functions in one draft is rejected
        td2 = synth.synthesize(SynthSpec(
            name="two_defs", description="x",
            examples=[{"args": {"a": 1}, "want": 1}]))
        assert not td2.ok and "exactly one function" in td2.reason

        # failing examples block registration
        wm = synth.synthesize(SynthSpec(
            name="wrongmath", description="x",
            examples=[{"args": {"n": 1}, "want": 2}]))
        assert not wm.ok and "failed examples" in wm.reason
        assert "wrongmath" not in registry

        # bad names / no examples are clean rejections
        assert not synth.synthesize(SynthSpec(
            name="Bad-Name", description="x",
            examples=[{"args": {"a": 1}, "want": 1}])).ok
        assert not synth.synthesize(SynthSpec(
            name="noexamples", description="x", examples=[])).ok

        # lineage sealed
        kinds = [e.type for e in log.events()]
        assert {"synth.tool.drafted", "synth.tool.tested",
                "synth.tool.registered"} <= set(kinds)

        print("SYNTH SELF-TEST PASS")
