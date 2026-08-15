"""NEXUS — the code property graph (§15).

What separates "an LLM with grep" from something that understands a
codebase. This implementation is the rung-2 AST layer (pure Python, stdlib
only): symbols, call graph, import graph, and the killer query — impact
analysis (§15.2). LSP/DAP integration degrades gracefully to this layer
when no language server is present (§46 risk register).

Everything is deterministic and free:
  * index(root) parses every .py file with ast, keyed by content hash —
    unchanged files are never reparsed (§15.5).
  * impact(symbol) answers "if I change this signature, what breaks?" with
    a graph traversal: direct callers, transitive callers, test coverage
    heuristics, public-API exposure, and a risk score.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             ".tox", "dist", "build", ".eggs"}


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class Symbol:
    name: str
    kind: str                 # def | class | method
    path: str
    lineno: int
    params: list[str] = field(default_factory=list)
    docstring: str = ""

    @property
    def key(self) -> str:
        return f"{self.path}:{self.name}"


@dataclass
class GraphIndex:
    """The parsed view of a repo: symbols + edges."""
    symbols: dict[str, Symbol] = field(default_factory=dict)   # key -> Symbol
    calls: dict[str, set[str]] = field(default_factory=dict)   # caller_key -> {callee names}
    imports: dict[str, set[str]] = field(default_factory=dict) # path -> {module names}
    file_hashes: dict[str, str] = field(default_factory=dict)  # path -> content hash
    errors: dict[str, str] = field(default_factory=dict)       # path -> parse error


class Nexus:
    """AST code graph with incremental maintenance."""

    def __init__(self) -> None:
        self.idx = GraphIndex()

    # -- indexing --------------------------------------------------------------

    def index(self, root: str | Path, max_files: int = 5000) -> GraphIndex:
        """Parse every .py file under root. Content-hash keyed: a second
        call on an unchanged repo reparses nothing (§15.5)."""
        root = Path(root).expanduser().resolve()
        files = [p for p in root.rglob("*.py")
                 if not any(part in SKIP_DIRS for part in p.parts)]
        for p in files[:max_files]:
            self.index_file(p)
        return self.idx

    def index_file(self, path: str | Path) -> bool:
        """(Re)parse one file. Returns True if it was reparsed."""
        p = Path(path).expanduser().resolve()
        try:
            text = p.read_text(errors="replace")
        except OSError as e:
            self.idx.errors[str(p)] = str(e)
            return False
        h = _content_hash(text)
        if self.idx.file_hashes.get(str(p)) == h:
            return False  # unchanged — never reparsed
        self._drop_file(str(p))
        self.idx.file_hashes[str(p)] = h
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            self.idx.errors[str(p)] = f"syntax error line {e.lineno}"
            return True
        self._extract(p, tree)
        return True

    def _drop_file(self, path: str) -> None:
        for key in [k for k in self.idx.symbols if k.startswith(path + ":")]:
            del self.idx.symbols[key]
        self.idx.calls.pop(path, None)
        self.idx.imports.pop(path, None)
        self.idx.errors.pop(path, None)

    def _extract(self, path: Path, tree: ast.Module) -> None:
        pstr = str(path)
        self.idx.calls.setdefault(pstr, set())
        self.idx.imports.setdefault(pstr, set())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if node.col_offset > 0 else "def"
                sym = Symbol(name=node.name, kind=kind, path=pstr,
                             lineno=node.lineno,
                             params=[a.arg for a in node.args.args],
                             docstring=(ast.get_docstring(node) or "")[:200])
                self.idx.symbols[sym.key] = sym
            elif isinstance(node, ast.ClassDef):
                sym = Symbol(name=node.name, kind="class", path=pstr,
                             lineno=node.lineno,
                             docstring=(ast.get_docstring(node) or "")[:200])
                self.idx.symbols[sym.key] = sym
            elif isinstance(node, ast.Call):
                name = self._call_name(node)
                if name:
                    self.idx.calls[pstr].add(name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.idx.imports[pstr].add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.idx.imports[pstr].add(node.module.split(".")[0])

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        fn = node.func
        if isinstance(fn, ast.Name):
            return fn.id
        if isinstance(fn, ast.Attribute):
            return fn.attr
        return ""

    # -- queries -----------------------------------------------------------------

    def find_symbol(self, name: str) -> list[Symbol]:
        """All definitions named `name` — the 'which of six process()
        definitions' answer."""
        return [s for s in self.idx.symbols.values() if s.name == name]

    def callers(self, name: str) -> list[tuple[str, int]]:
        """Direct call sites of `name`: (path, count)."""
        out: list[tuple[str, int]] = []
        for path, names in self.idx.calls.items():
            n = sum(1 for x in names if x == name)
            if n:
                out.append((path, n))
        return sorted(out)

    def transitive_callers(self, name: str, max_depth: int = 3) -> set[str]:
        """Symbols that call `name` directly or transitively (via symbols
        defined in the calling files)."""
        direct = {p for p, _ in self.callers(name)}
        frontier = set(direct)
        seen = set(direct)
        for _ in range(max_depth - 1):
            next_frontier: set[str] = set()
            for path in frontier:
                # symbols defined in this file that call into the frontier
                for sym in self.idx.symbols.values():
                    if sym.path == path:
                        for caller_path, _ in self.callers(sym.name):
                            if caller_path not in seen:
                                seen.add(caller_path)
                                next_frontier.add(caller_path)
            if not next_frontier:
                break
            frontier = next_frontier
        return seen

    def impact(self, name: str) -> dict:
        """The killer query (§15.2): blast radius of changing `name`.
        Deterministic graph traversal — the planner knows the blast radius
        BEFORE it writes a line."""
        defs = self.find_symbol(name)
        direct = self.callers(name)
        transitive = self.transitive_callers(name)
        tests = [p for p in transitive
                 if "test" in Path(p).name.lower()
                 or "/tests/" in p or "\\tests\\" in p]
        public = any(self._is_exported(s) for s in defs)
        n_direct = sum(n for _, n in direct)
        coverage = len(tests) / max(1, len(transitive))
        risk = self._risk_score(public, len(transitive), coverage, defs)
        return {
            "symbol": name,
            "definitions": [{"path": s.path, "line": s.lineno,
                             "kind": s.kind, "params": s.params}
                            for s in defs],
            "direct_callers": n_direct,
            "direct_files": [p for p, _ in direct],
            "transitive_files": sorted(transitive),
            "tests_covering": tests,
            "public_api": public,
            "coverage_ratio": round(coverage, 2),
            "risk": risk,
        }

    def _is_exported(self, sym: Symbol) -> bool:
        """Heuristic: exported via an __init__.py in the same package."""
        pkg_init = Path(sym.path).parent / "__init__.py"
        if not pkg_init.exists():
            return False
        try:
            text = pkg_init.read_text(errors="replace")
        except OSError:
            return False
        return sym.name in text

    @staticmethod
    def _risk_score(public: bool, n_transitive: int, coverage: float,
                    defs: list) -> str:
        score = 0
        if public:
            score += 2
        if n_transitive >= 10:
            score += 2
        elif n_transitive >= 3:
            score += 1
        if coverage < 0.3:
            score += 2
        elif coverage < 0.6:
            score += 1
        if len(defs) > 1:
            score += 1  # ambiguous: multiple same-named definitions
        return {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "MEDIUM",
                4: "HIGH", 5: "HIGH"}.get(min(score, 5), "HIGH")

    def format_impact(self, name: str) -> str:
        """Human-readable impact report for the TUI."""
        imp = self.impact(name)
        lines = [f"impact({name}) — risk {imp['risk']}",
                 f"  definitions     : {len(imp['definitions'])}"]
        for d in imp["definitions"]:
            lines.append(f"    {d['path']}:{d['line']} ({d['kind']})")
        lines += [
            f"  direct callers  : {imp['direct_callers']} sites across "
            f"{len(imp['direct_files'])} files",
            f"  transitive      : {len(imp['transitive_files'])} files",
            f"  tests covering  : {len(imp['tests_covering'])}",
            f"  public API      : {'yes' if imp['public_api'] else 'no'}",
            f"  coverage ratio  : {imp['coverage_ratio']}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pkg").mkdir()
        (root / "pkg" / "__init__.py").write_text("from .auth import verify_token\n")
        (root / "pkg" / "auth.py").write_text(
            "def verify_token(token):\n"
            "    return bool(token)\n\n"
            "def login(token):\n"
            "    return verify_token(token)\n")
        (root / "pkg" / "api.py").write_text(
            "from .auth import login\n\n"
            "def handle(req):\n"
            "    return login(req)\n")
        (root / "tests").mkdir()
        (root / "tests" / "test_auth.py").write_text(
            "from pkg.auth import verify_token\n\n"
            "def test_verify():\n"
            "    assert verify_token('x')\n")

        nx = Nexus()
        idx = nx.index(root)
        assert len(idx.symbols) >= 4, idx.symbols.keys()
        assert not idx.errors, idx.errors

        # incremental: unchanged file is not reparsed
        auth = root / "pkg" / "auth.py"
        assert nx.index_file(auth) is False  # already indexed, same hash
        auth.write_text(auth.read_text() + "\n# touched\n")
        assert nx.index_file(auth) is True   # hash changed -> reparsed

        # find_symbol / callers
        defs = nx.find_symbol("verify_token")
        assert len(defs) == 1 and defs[0].kind == "def"
        callers = nx.callers("verify_token")
        paths = {p for p, _ in callers}
        assert any("api.py" in p or "auth.py" in p for p in paths), paths

        # impact: the killer query
        imp = nx.impact("verify_token")
        assert imp["direct_callers"] >= 1
        assert imp["public_api"] is True  # exported in __init__.py
        assert imp["risk"] in ("LOW", "MEDIUM", "HIGH")
        assert any("test" in t for t in imp["tests_covering"]), imp

        text = nx.format_impact("verify_token")
        assert "impact(verify_token)" in text and "risk" in text

        # unknown symbol: empty but well-formed report
        imp2 = nx.impact("no_such_fn")
        assert imp2["direct_callers"] == 0 and imp2["definitions"] == []

    print("NEXUS SELF-TEST PASS")
