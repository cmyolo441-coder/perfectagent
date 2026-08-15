"""Tool registry: everything the agent can do — files, shell, search, web."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import config

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

RISK_SAFE = "safe"          # runs without asking
RISK_CONFIRM = "confirm"    # needs user approval (unless auto-approve)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]
    risk: str = RISK_SAFE

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _clip(text: str, limit: int = config.MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 4:]
    return f"{head}\n… [{len(text) - len(head) - len(tail)} chars truncated] …\n{tail}"


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _line_numbered(text: str, start: int = 1) -> str:
    lines = text.splitlines()
    width = len(str(start + len(lines) - 1))
    out = []
    for i, line in enumerate(lines):
        n = start + i
        if n == start or n == start + len(lines) - 1 or n % 10 == 0:
            out.append(f"{n:>{width}}→{line}")
        else:
            out.append(f"{'':>{width}} {line}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

def read_file(path: str, offset: int = 1, limit: int = 1000) -> str:
    """Read a text file with line numbers."""
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: file not found: {p}"
    if p.is_dir():
        return f"ERROR: {p} is a directory (use list_dir)"
    if p.stat().st_size > 2_000_000:
        return f"ERROR: file is very large ({p.stat().st_size} bytes); use offset/limit"
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        return f"ERROR: {e}"
    lines = text.splitlines()
    offset = max(1, offset)
    chunk = lines[offset - 1: offset - 1 + limit]
    header = f"[{p} — {len(lines)} lines total, showing {offset}..{offset + len(chunk) - 1}]"
    return f"{header}\n{_line_numbered(chr(10).join(chunk), offset)}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file (parents created automatically)."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"OK: wrote {len(content)} chars to {p}"


def edit_file(path: str, old_string: str, new_string: str,
              replace_all: bool = False) -> str:
    """Replace an exact string in a file. old_string must match exactly once
    (or set replace_all=true to replace every occurrence)."""
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: file not found: {p}"
    text = p.read_text(errors="replace")
    count = text.count(old_string)
    if count == 0:
        return "ERROR: old_string not found in file (it must match exactly, " \
               "including indentation)"
    if count > 1 and not replace_all:
        return f"ERROR: old_string matches {count} places; add more context " \
               "to make it unique or set replace_all=true"
    if replace_all:
        new_text = text.replace(old_string, new_string)
    else:
        new_text = text.replace(old_string, new_string, 1)
    p.write_text(new_text)
    return f"OK: replaced {count if replace_all else 1} occurrence(s) in {p}"


def list_dir(path: str = ".") -> str:
    """List a directory's contents (one level)."""
    p = _resolve(path)
    if not p.is_dir():
        return f"ERROR: not a directory: {p}"
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    lines = [f"[{p}]"]
    for e in entries[:300]:
        if e.is_dir():
            lines.append(f"  {e.name}/")
        else:
            size = e.stat().st_size
            lines.append(f"  {e.name}  ({size} bytes)")
    if len(entries) > 300:
        lines.append(f"  … and {len(entries) - 300} more")
    return "\n".join(lines) if len(lines) > 1 else f"[{p}] (empty)"


def file_info(path: str) -> str:
    """Show metadata about a file or directory."""
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: not found: {p}"
    st = p.stat()
    kind = "directory" if p.is_dir() else "file"
    return (f"{p}\n  type: {kind}\n  size: {st.st_size} bytes\n"
            f"  modified: {st.st_mtime}")


def create_directory(path: str) -> str:
    p = _resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"OK: created directory {p}"


def copy_path(src: str, dst: str) -> str:
    s, d = _resolve(src), _resolve(dst)
    if not s.exists():
        return f"ERROR: source not found: {s}"
    if s.is_dir():
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    return f"OK: copied {s} -> {d}"


def move_path(src: str, dst: str) -> str:
    s, d = _resolve(src), _resolve(dst)
    if not s.exists():
        return f"ERROR: source not found: {s}"
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return f"OK: moved {s} -> {d}"


def delete_path(path: str) -> str:
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: not found: {p}"
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return f"OK: deleted {p}"


# ---------------------------------------------------------------------------
# Search tools
# ---------------------------------------------------------------------------

def search_files(pattern: str, path: str = ".", glob_filter: str = "*",
                 max_results: int = 100) -> str:
    """Regex search through file contents (ripgrep-style), respecting common
    ignore dirs. Returns matching lines as path:line:content."""
    root = _resolve(path)
    if root.is_file():
        files = [root]
        root = root.parent
    else:
        files = []
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                     "dist", "build", ".tox", ".mypy_cache", ".ruff_cache"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                if fnmatch.fnmatch(fn, glob_filter):
                    files.append(Path(dirpath) / fn)
            if len(files) > 5000:
                break
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    hits: list[str] = []
    for f in files:
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f}:{i}:{line.strip()[:200]}")
                if len(hits) >= max_results:
                    return "\n".join(hits) + f"\n… (stopped at {max_results} results)"
    return "\n".join(hits) if hits else "no matches"


def glob_files(pattern: str, path: str = ".") -> str:
    """Find files by glob pattern (e.g. '**/*.py')."""
    root = _resolve(path)
    matches = sorted(str(p) for p in root.glob(pattern) if p.is_file())[:300]
    return "\n".join(matches) if matches else "no matches"


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def run_command(command: str, timeout: int = 120) -> str:
    """Run a shell command via bash and return exit code + output."""
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    except OSError as e:
        return f"ERROR: {e}"
    out = []
    out.append(f"exit code: {proc.returncode}")
    if proc.stdout:
        out.append("--- stdout ---\n" + proc.stdout)
    if proc.stderr:
        out.append("--- stderr ---\n" + proc.stderr)
    return _clip("\n".join(out))


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------

def web_fetch(url: str) -> str:
    """Fetch a URL and return its text content."""
    import requests
    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) FullAgent/1.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"ERROR: {e}"
    ctype = resp.headers.get("content-type", "")
    text = resp.text
    if "html" in ctype:
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>",
                      "", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
    return _clip(text.strip(), 16_000)


def _ddg_search(query: str) -> list[tuple[str, str, str]]:
    """DuckDuckGo HTML search -> [(title, url, snippet)]."""
    import requests
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query}, timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) FullAgent/1.0"})
    resp.raise_for_status()
    results = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>',
        resp.text)
    snippets = re.findall(
        r'class="result__snippet"[^>]*>([\s\S]*?)</a>', resp.text)
    out = []
    for i, (href, title) in enumerate(results):
        title = re.sub(r"<[^>]+>", "", title).strip()
        snip = re.sub(r"<[^>]+>", "", snippets[i]).strip() \
            if i < len(snippets) else ""
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            import urllib.parse
            href = urllib.parse.unquote(m.group(1))
        out.append((title, href, snip))
    return out


def _bing_search(query: str) -> list[tuple[str, str, str]]:
    """Bing HTML search fallback -> [(title, url, snippet)]."""
    import requests
    resp = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "count": "10"}, timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
                               "Gecko/20100101 Firefox/128.0"})
    resp.raise_for_status()
    out = []
    for block in re.findall(r'<li class="b_algo"[\s\S]*?</li>', resp.text):
        m = re.search(r'<h2><a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>', block)
        if not m:
            continue
        url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        sm = re.search(r'<p[^>]*>([\s\S]*?)</p>', block)
        snip = re.sub(r"<[^>]+>", "", sm.group(1)).strip() if sm else ""
        out.append((title, url, snip))
    return out


def web_search(query: str) -> str:
    """Real-time web search. Tries DuckDuckGo, then Bing; returns the top
    results with titles, URLs and snippets, stamped with the retrieval
    time so the data's freshness is explicit."""
    from datetime import datetime
    errors = []
    results: list[tuple[str, str, str]] = []
    for engine, fn in (("DuckDuckGo", _ddg_search), ("Bing", _bing_search)):
        try:
            results = fn(query)
            if results:
                break
        except Exception as e:
            errors.append(f"{engine}: {e}")
    if not results:
        return ("ERROR: all search engines failed — "
                + "; ".join(errors) if errors else "no results")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"web search: {query!r}  (retrieved {stamp}, live results)"]
    for i, (title, url, snip) in enumerate(results[:8], 1):
        lines.append(f"{i}. {title}\n   {url}\n   {snip[:220]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STR = {"type": "string"}


def build_registry() -> dict[str, Tool]:
    tools: list[Tool] = [
        Tool("read_file",
             "Read a text file with line numbers. Use offset/limit for large files.",
             {"type": "object", "properties": {
                 "path": _STR,
                 "offset": {"type": "integer", "description": "first line (1-based)"},
                 "limit": {"type": "integer", "description": "max lines (default 1000)"}},
              "required": ["path"]},
             read_file),
        Tool("write_file",
             "Create or overwrite a file with the given content. Parent dirs are created.",
             {"type": "object", "properties": {
                 "path": _STR, "content": _STR},
              "required": ["path", "content"]},
             write_file, risk=RISK_CONFIRM),
        Tool("edit_file",
             "Replace an exact string in a file. old_string must match exactly "
             "(including indentation) and uniquely, unless replace_all=true.",
             {"type": "object", "properties": {
                 "path": _STR, "old_string": _STR, "new_string": _STR,
                 "replace_all": {"type": "boolean"}},
              "required": ["path", "old_string", "new_string"]},
             edit_file, risk=RISK_CONFIRM),
        Tool("list_dir", "List a directory's contents (one level).",
             {"type": "object", "properties": {"path": _STR}},
             list_dir),
        Tool("file_info", "Show metadata (size, mtime, type) for a path.",
             {"type": "object", "properties": {"path": _STR}, "required": ["path"]},
             file_info),
        Tool("create_directory", "Create a directory (parents included).",
             {"type": "object", "properties": {"path": _STR}, "required": ["path"]},
             create_directory),
        Tool("copy_path", "Copy a file or directory.",
             {"type": "object", "properties": {"src": _STR, "dst": _STR},
              "required": ["src", "dst"]},
             copy_path, risk=RISK_CONFIRM),
        Tool("move_path", "Move/rename a file or directory.",
             {"type": "object", "properties": {"src": _STR, "dst": _STR},
              "required": ["src", "dst"]},
             move_path, risk=RISK_CONFIRM),
        Tool("delete_path", "Delete a file or directory permanently.",
             {"type": "object", "properties": {"path": _STR}, "required": ["path"]},
             delete_path, risk=RISK_CONFIRM),
        Tool("search_files",
             "Regex search through file contents (ripgrep-style). "
             "Returns path:line:content for matches.",
             {"type": "object", "properties": {
                 "pattern": {"type": "string", "description": "regex pattern"},
                 "path": {"type": "string", "description": "dir or file to search"},
                 "glob_filter": {"type": "string", "description": "filename glob, e.g. '*.py'"}},
              "required": ["pattern"]},
             search_files),
        Tool("glob_files", "Find files by glob pattern, e.g. '**/*.py'.",
             {"type": "object", "properties": {
                 "pattern": _STR, "path": _STR}, "required": ["pattern"]},
             glob_files),
        Tool("run_command",
             "Run a shell command via bash and return exit code, stdout, stderr. "
             "Use for builds, tests, git, installs, running programs.",
             {"type": "object", "properties": {
                 "command": _STR,
                 "timeout": {"type": "integer", "description": "seconds, default 120"}},
              "required": ["command"]},
             run_command, risk=RISK_CONFIRM),
        Tool("web_fetch", "Fetch a URL and return its text content.",
             {"type": "object", "properties": {"url": _STR}, "required": ["url"]},
             web_fetch),
        Tool("web_search", "Search the web (DuckDuckGo) and return top results.",
             {"type": "object", "properties": {"query": _STR}, "required": ["query"]},
             web_search),
    ]
    return {t.name: t for t in tools}


def parse_tool_arguments(raw: Any) -> dict:
    """Tool-call arguments arrive as a JSON string (native) or dict."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (ValueError, TypeError):
        return {"_raw": str(raw)}
