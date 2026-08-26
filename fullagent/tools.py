"""Tool registry: everything the agent can do — files, shell, search, web."""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
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


def _diff_summary(old: str, new: str) -> tuple[int, int]:
    """(additions, removals) line counts between two versions."""
    adds = removes = 0
    for line in difflib.unified_diff(
            old.splitlines(), new.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            adds += 1
        elif line.startswith("-") and not line.startswith("---"):
            removes += 1
    return adds, removes


def _edit_report(path: Path, old: str, new: str,
                 extra: str = "") -> str:
    """The live-coding style receipt: 'Updated X with N additions and
    M removals' — so the caller sees exactly what changed at a glance."""
    adds, removes = _diff_summary(old, new)
    head = f"Updated {path} with {adds} addition(s) and {removes} removal(s)"
    return f"{head}{extra}"


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

def _atomic_write_text(p: Path, text: str) -> None:
    """Write UTF-8 atomically: a crash mid-write can never leave a
    truncated/corrupt file behind."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


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
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"ERROR: {e}"
    lines = text.splitlines()
    offset = max(1, offset)
    if offset > len(lines):
        return (f"[{p} — {len(lines)} lines total; offset {offset} is past the "
                f"end of file]")
    chunk = lines[offset - 1: offset - 1 + limit]
    end = offset + len(chunk) - 1
    header = f"[{p} — {len(lines)} lines total, showing {offset}..{end}]"
    return f"{header}\n{_line_numbered(chr(10).join(chunk), offset)}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file (parents created automatically)."""
    p = _resolve(path)
    old = ""
    if p.exists():
        try:
            old = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            old = ""
    try:
        _atomic_write_text(p, content)
    except OSError as e:
        return f"ERROR: {e}"
    if not p.exists() or old == "":
        # brand-new file — a pure-addition report reads better
        adds = len(content.splitlines())
        return f"OK: created {p} with {adds} line(s) ({len(content)} chars)"
    adds, removes = _diff_summary(old, content)
    return f"Updated {p} with {adds} addition(s) and {removes} removal(s)"


def edit_file(path: str, old_string: str, new_string: str,
              replace_all: bool = False) -> str:
    """Replace an exact string in a file. old_string must match exactly once
    (or set replace_all=true to replace every occurrence)."""
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: file not found: {p}"
    if not old_string:
        # str.count("") counts len+1 positions and str.replace("") would
        # interleave new_string between EVERY character — destroy the file
        return "ERROR: old_string must be a non-empty string"
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ("ERROR: file is not valid UTF-8 text; refusing to edit "
                "(lossy rewrite would corrupt unrelated bytes)")
    except OSError as e:
        return f"ERROR: {e}"
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
    try:
        _atomic_write_text(p, new_text)
    except OSError as e:
        return f"ERROR: {e}"
    adds, removes = _diff_summary(text, new_text)
    n = count if replace_all else 1
    return (f"Updated {p} — {n} occurrence(s) replaced, "
            f"{adds} addition(s), {removes} removal(s)")


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
            try:
                size = e.stat().st_size
                lines.append(f"  {e.name}  ({size} bytes)")
            except OSError:
                # broken symlink or vanished entry — don't crash the listing
                lines.append(f"  {e.name}  (unreadable)")
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
    if pattern.startswith("/"):
        # Path.glob() rejects absolute patterns outright
        return "ERROR: pattern must be relative to the search path"
    try:
        matches = sorted(str(p) for p in root.glob(pattern)
                         if p.is_file())[:300]
    except (NotImplementedError, ValueError) as e:
        return f"ERROR: bad glob pattern: {e}"
    return "\n".join(matches) if matches else "no matches"


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def _pump_process(proc: "subprocess.Popen", timeout: float,
                  on_output: "Callable[[str, str], None] | None" = None,
                  ) -> tuple[list, list] | None:
    """Read stdout/stderr of `proc` line-by-line until it exits or the
    timeout elapses. Returns (stdout_lines, stderr_lines), or None on
    timeout (the process is killed). Every line is relayed to
    on_output(line, "out"|"err") the moment it is produced — this is what
    lets the TUI stream shell output live, like watching a real terminal."""
    q: "queue.Queue" = queue.Queue()

    def pump(stream, tag: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                q.put((tag, line))
        finally:
            q.put((tag, None))  # sentinel: this stream is done

    threading.Thread(target=pump, args=(proc.stdout, "out"),
                     daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, "err"),
                     daemon=True).start()

    out_lines: list = []
    err_lines: list = []
    open_streams = 2
    deadline = time.monotonic() + timeout
    while open_streams > 0:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.kill()
            proc.wait()
            return None
        try:
            tag, line = q.get(timeout=min(0.25, max(0.01, remaining)))
        except queue.Empty:
            if proc.poll() is not None:
                # process exited; drain whatever is left briefly
                continue
            continue
        if line is None:
            open_streams -= 1
            continue
        (out_lines if tag == "out" else err_lines).append(line)
        if on_output is not None:
            try:
                on_output(line.rstrip("\n"), tag)
            except Exception:  # noqa: BLE001 — never break the tool
                pass
    proc.wait()
    return out_lines, err_lines


def run_command(command: str, timeout: int = 120,
                on_output: "Callable[[str, str], None] | None" = None) -> str:
    """Run a shell command via bash and return exit code + output.

    The shell is resolved once (judge.resolve_shell): on Windows,
    System32\\bash.exe is the WSL stub and fails when no distro is
    installed, so Git Bash is probed and preferred. If `on_output` is
    provided, each output line is streamed to it live as it appears."""
    from .judge import resolve_shell
    argv = resolve_shell()
    if argv is None:
        return ("ERROR: no POSIX shell available — install Git Bash "
                "(windows) or bash (posix)")
    try:
        proc = subprocess.Popen(
            argv + [command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            cwd=os.getcwd(),
        )
    except OSError as e:
        return f"ERROR: {e}"
    pumped = _pump_process(proc, timeout, on_output)
    if pumped is None:
        return f"ERROR: command timed out after {timeout}s"
    out_lines, err_lines = pumped
    stdout = "".join(out_lines)
    stderr = "".join(err_lines)
    out = []
    out.append(f"exit code: {proc.returncode}")
    if stdout:
        out.append("--- stdout ---\n" + stdout)
    if stderr:
        out.append("--- stderr ---\n" + stderr)
    return _clip("\n".join(out))


# ---------------------------------------------------------------------------
# Live shell — a PERSISTENT bash session (cd/env/exports survive between
# calls), plus a unified-diff applier for live code editing.
# ---------------------------------------------------------------------------

_SHELL_LOCK = threading.Lock()      # one command at a time on the session
_SHELL_STATE = {"cwd": None,        # sticky working directory
                "env": None}        # sticky environment (persistent exports)


def live_shell(command: str, timeout: int = 120,
               on_output: "Callable[[str, str], None] | None" = None) -> str:
    """Run a command inside a persistent bash session.

    Unlike `run_command` (which spawns a fresh shell per call), this one
    keeps state across calls: a `cd src` in one call is still in effect in
    the next, and exported variables persist. Use it for live workflows:
    cd -> build -> test -> inspect -> fix.

    State is shared process-wide and guarded by a lock so two callers can
    never interleave their commands into the same session. If `on_output`
    is provided, each output line is streamed to it live as it appears."""
    from .judge import resolve_shell
    argv = resolve_shell()
    if argv is None:
        return ("ERROR: no POSIX shell available — install Git Bash "
                "(windows) or bash (posix)")
    cwd = _SHELL_STATE["cwd"] or os.getcwd()
    # Wrap so the command's own exit code survives, and the FINAL cwd +
    # environment are reported back on marker lines we strip before
    # returning. Replaying the env on the next call is what makes
    # `export FOO=...` stick across calls despite fresh processes.
    wrapped = (
        command + "\n"
        "__fa_rc=$?\n"
        'printf "\\n__FA_CWD__%s" "$PWD"\n'
        'printf "\\n__FA_ENV__"\n'
        "env -0\n"
        "exit $__fa_rc\n")
    extra_env = _SHELL_STATE["env"]

    # Live-stream filter: the __FA_CWD__ marker and everything after it
    # (the env blob) are bookkeeping, not real output — never show them.
    # The wrapper's leading "\n" can arrive as one last blank line, so
    # trailing blanks are held back until the next real line proves they
    # are genuine output.
    cutoff = {"hit": False}
    pending_blanks = {"n": 0}

    def relay(line: str, stream: str) -> None:
        if on_output is None:
            return
        if stream == "out":
            if "__FA_CWD__" in line:
                pre, _, _ = line.partition("__FA_CWD__")
                cutoff["hit"] = True
                pending_blanks["n"] = 0
                if not pre:
                    return
                line = pre  # stream the real output before the marker
            if cutoff["hit"]:
                return
            if line == "":
                pending_blanks["n"] += 1
                return
            while pending_blanks["n"] > 0:
                pending_blanks["n"] -= 1
                try:
                    on_output("", "out")
                except Exception:  # noqa: BLE001
                    pass
        try:
            on_output(line, stream)
        except Exception:  # noqa: BLE001
            pass

    try:
        proc = subprocess.Popen(
            argv + [wrapped],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            cwd=cwd, env=extra_env,
        )
    except OSError as e:
        return f"ERROR: {e}"
    pumped = _pump_process(proc, timeout, relay)
    if pumped is None:
        return f"ERROR: command timed out after {timeout}s (cwd={cwd})"
    out_lines, err_lines = pumped
    stdout = "".join(out_lines)
    stderr = "".join(err_lines)
    new_cwd, new_env = cwd, extra_env
    idx = stdout.rfind("__FA_ENV__")
    if idx >= 0:
        blob = stdout[idx + len("__FA_ENV__"):]
        env_map: dict[str, str] = {}
        for pair in blob.split("\0"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                env_map[k] = v
        if env_map:
            new_env = env_map
        stdout = stdout[:idx].rstrip("\n")
    j = stdout.rfind("__FA_CWD__")
    if j >= 0:
        new_cwd = stdout[j + len("__FA_CWD__"):].strip() or cwd
        stdout = stdout[:j].rstrip("\n")
    with _SHELL_LOCK:
        if os.path.isdir(new_cwd):
            _SHELL_STATE["cwd"] = new_cwd
        if new_env:
            _SHELL_STATE["env"] = new_env
    parts = [f"cwd: {_SHELL_STATE['cwd']}", f"exit code: {proc.returncode}"]
    if stdout:
        parts.append("--- stdout ---\n" + stdout)
    if stderr:
        parts.append("--- stderr ---\n" + stderr)
    return _clip("\n".join(parts))


def live_shell_reset() -> str:
    """Reset the persistent shell session back to the process cwd/env."""
    prev = _SHELL_STATE["cwd"]
    _SHELL_STATE["cwd"] = None
    _SHELL_STATE["env"] = None
    return f"OK: session reset ({prev} -> {os.getcwd()})"


def apply_patch(patch: str) -> str:
    """Apply a unified diff to the working tree (live multi-file edit).

    Accepts standard `diff -u` / `git diff` output. File paths are read
    from ---/+++ headers (a/ b/ prefixes stripped). Each hunk is applied
    with its own context tolerance; a hunk that no longer matches its
    context lines fails loudly instead of silently corrupting the file.

    Returns a per-file report: 'Updated X with N addition(s) and M
    removal(s)', matching the style of write_file/edit_file.
    Relative paths resolve against the persistent live_shell cwd if a
    session is active (so `live_shell("cd src")` followed by a patch on
    `a/main.py` does the intuitive thing)."""
    if not (patch or "").strip():
        return "ERROR: empty patch"
    base = _SHELL_STATE["cwd"] or os.getcwd()
    # -- parse the patch into per-file hunks ------------------------------
    files: dict[str, dict] = {}
    cur: dict | None = None
    for line in patch.splitlines():
        if line.startswith(("--- ", "+++ ")) \
                and not line.startswith(("--- \t", "+++ \t")):
            name = line[4:].split("\t")[0].strip()
            if name.startswith("a/") or name.startswith("b/"):
                name = name[2:]
            if name == "/dev/null":
                continue
            side_is_new = line.startswith("+++ ")
            key = name
            if cur is None or cur.get("_pending") != key or not side_is_new:
                pass
            entry = files.setdefault(key, {"old": "", "new": "",
                                           "_seen_new": False})
            if side_is_new:
                entry["_seen_new"] = True
                cur = entry
            continue
        if line.startswith("@@"):
            if cur is not None:
                cur.setdefault("hunks", []).append(
                    {"lines": [], "old_start": 1, "new_start": 1})
                h = cur["hunks"][-1]
                m = re.match(
                    r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if m:
                    h["old_start"] = int(m.group(1))
                    h["new_start"] = int(m.group(3))
            continue
        if cur is not None and cur.get("hunks"):
            h = cur["hunks"][-1]
            if line.startswith("+") or line.startswith("-") \
                    or line.startswith(" ") or line == "":
                h["lines"].append(line if line else " ")
    if not files or not any(f.get("hunks") for f in files.values()):
        return ("ERROR: no hunks found — expected '@@' headers "
                "(unified diff format)")
    # -- load old contents -------------------------------------------------
    for name, entry in files.items():
        p = Path(name).expanduser()
        if not p.is_absolute():
            p = base / p
        if p.exists():
            try:
                entry["old"] = p.read_text(encoding="utf-8",
                                           errors="replace")
            except OSError as e:
                return f"ERROR: cannot read {p}: {e}"
    # -- apply each file's hunks -------------------------------------------
    reports: list[str] = []
    for name, entry in files.items():
        if not entry.get("hunks"):
            continue
        p = Path(name).expanduser()
        if not p.is_absolute():
            p = base / p
        old_lines = entry["old"].splitlines()
        new_lines = list(old_lines)
        # apply hunks bottom-up so earlier offsets stay valid
        for hunk in sorted(entry["hunks"],
                           key=lambda h: h["old_start"], reverse=True):
            ctx: list[tuple[str, str]] = []   # (tag, text)
            for raw in hunk["lines"]:
                tag, txt = (raw[0], raw[1:]) if len(raw) > 1 else (" ", "")
                ctx.append((tag, txt))
            # find where the hunk's old-side lines start in the file
            old_side = [(t, x) for t, x in ctx if t in (" ", "-")]
            pos = hunk["old_start"] - 1
            matched = False
            for delta in range(0, max(len(old_lines), 1) + 1):
                for off in (delta, -delta):
                    cand = pos + off
                    if cand < 0 or cand + len(old_side) > len(old_lines):
                        continue
                    if all(cand + i < len(old_lines)
                           and old_lines[cand + i] == old_side[i][1]
                           for i in range(len(old_side))):
                        pos = cand
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                return (f"ERROR: hunk context mismatch in {name} near "
                        f"line {hunk['old_start']} — file changed since "
                        "the diff was made; regenerate the diff and retry")
            # rebuild: keep everything before, splice the new-side lines,
            # keep everything after
            new_side = [x for t, x in ctx if t in (" ", "+")]
            new_lines = (new_lines[:pos] + new_side
                         + new_lines[pos + len(old_side):])
        new_text = "\n".join(new_lines) + ("\n" if entry["old"].endswith(
            "\n") or new_lines else "")
        adds, removes = _diff_summary(entry["old"], new_text)
        try:
            _atomic_write_text(p, new_text)
        except OSError as e:
            return f"ERROR: writing {p}: {e}"
        reports.append(_edit_report(p, entry["old"], new_text))
    return "\n".join(reports)


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
                + ("; ".join(errors) if errors else "no results"))
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
        Tool("live_shell",
             "Run a command in a PERSISTENT bash session: cd, exports and "
             "background jobs survive between calls. Use for live workflows "
             "(cd src && build, then run tests, then inspect, then fix).",
             {"type": "object", "properties": {
                 "command": _STR,
                 "timeout": {"type": "integer",
                             "description": "seconds, default 120"}},
              "required": ["command"]},
             live_shell, risk=RISK_CONFIRM),
        Tool("live_shell_reset",
             "Reset the persistent live-shell session back to the process "
             "working directory.",
             {"type": "object", "properties": {}},
             live_shell_reset),
        Tool("apply_patch",
             "Apply a unified diff (git diff / diff -u format) to the working "
             "tree. Multi-file edits in one call; each hunk is context-checked "
             "and fails loudly on mismatch instead of corrupting files. "
             "Returns a per-file 'N additions / M removals' report.",
             {"type": "object", "properties": {"patch": _STR},
              "required": ["patch"]},
             apply_patch, risk=RISK_CONFIRM),
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


# ---------------------------------------------------------------------------
# Self-test (offline: no API key needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    reg = build_registry()
    assert "live_shell" in reg and "live_shell_reset" in reg \
        and "apply_patch" in reg, "new tools missing from registry"

    live_shell_reset()
    # persistent cwd across calls
    live_shell("cd /tmp")
    r = live_shell("pwd")
    assert "cwd: /tmp" in r, f"cwd did not persist:\n{r}"
    # persistent exports across calls (env replay)
    r = live_shell("export FA_TEST_VAR=ok7")
    assert "exit code: 0" in r, r
    r = live_shell('echo "$FA_TEST_VAR"')
    assert "ok7" in r, f"export did not persist:\n{r}"
    # multi-line compound command
    r = live_shell("for i in 1 2; do echo n=$i; done")
    assert "n=1" in r and "n=2" in r, r
    # failure exit code propagates; failed cd leaves state intact
    live_shell("cd /tmp")
    r = live_shell("cd /definitely_missing_dir_xyz; true")
    assert "cwd: /tmp" in r.splitlines()[0], r
    live_shell_reset()

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "st.py"
        msg = write_file(str(fp), "a\nb\nc\n")
        assert msg.startswith("OK: created"), msg
        patch = (
            f"--- a/{td}/st.py\n+++ b/{td}/st.py\n"
            "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n")
        out = apply_patch(patch)
        assert out.startswith("Updated") and "1 addition(s)" in out, out
        assert fp.read_text() == "a\nB\nc\n", fp.read_text()
        bad = patch.replace(" c\n", " WRONG CONTEXT\n")
        err = apply_patch(bad)
        assert err.startswith("ERROR: hunk context mismatch"), err
        assert fp.read_text() == "a\nB\nc\n", "failed patch mutated the file!"

    print("TOOLS SELF-TEST PASS")
