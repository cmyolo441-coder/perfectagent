"""Terminal UI: persistent double-line prompt box, live streaming, overlays.

Architecture: ONE prompt_toolkit Application runs for the whole session.
The double-line box stays pinned at the bottom at all times; all output
(user echo, streamed tokens, tool lines, errors) scrolls above it through
prompt_toolkit's patch_stdout. The box border carries live state:

    ╭─ FullAgent ── model: MiMo v2.5 FREE ── effort: HIGH ── session a1b2c3d4 ─╮
    │ ❯ user types here…                                                       │
    ╰─ ⠹ thinking…  ·  Ctrl+C cancel ──────────────────────────────────────────╯

Overlays (model / effort / help / history) render directly above the box and
are navigated with ↑↓, PgUp/PgDn, Tab, Home/End.
"""

from __future__ import annotations

import difflib
import os
import shutil
import re
import sys
import threading
import time
from html import escape as html_escape
from pathlib import Path
from typing import Callable

from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.formatted_text import (
    HTML,
    fragment_list_to_text,
    to_formatted_text,
)
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    BufferControl,
    CompletionsMenu,
    ConditionalContainer,
    Dimension,
    Float,
    FloatContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.search import start_search
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import SearchToolbar
from rich.console import Console
from rich.markdown import CodeBlock, Heading, Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from . import config
from .agent import Agent, ToolEvent
from .config import (
    APP_NAME,
    EFFORTS,
    MODELS,
    PROVIDERS,
    Config,
    Effort,
    Model,
    effort_by_key,
    model_by_id,
)
from .tools import Tool


class SafeFileHistory(FileHistory):
    """FileHistory that can never take the UI down.

    prompt_toolkit calls store_string() from inside the event loop when the
    user presses Enter; if the history file's directory is missing (fresh
    install, home dir deleted mid-session, fresh mount) the base class
    raises FileNotFoundError straight into the event loop. This subclass
    recreates the directory and swallows any persistence error — input
    history is a convenience, never a crash path."""

    def load_history_strings(self):
        try:
            yield from super().load_history_strings()
        except OSError:
            return

    def store_string(self, string: str) -> None:
        try:
            super().store_string(string)
        except FileNotFoundError:
            try:
                Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
                super().store_string(string)
            except OSError:
                pass
        except OSError:
            pass

# ---------------------------------------------------------------------------
# Palette (dracula-flavoured)
# ---------------------------------------------------------------------------

C = {
    "border": "#6272a4",
    "accent": "#bd93f9",
    "cyan": "#8be9fd",
    "green": "#50fa7b",
    "yellow": "#f1fa8c",
    "orange": "#ffb86c",
    "red": "#ff5555",
    "pink": "#ff79c6",
    "fg": "#f8f8f2",
    "dim": "#6272a4",
}

STYLE = Style.from_dict({
    "box": C["border"],
    "box.title": f"bold {C['accent']}",
    "box.model": f"bold {C['cyan']}",
    "box.tag": f"bold {C['green']}",
    "box.effort": "bold",
    "box.session": C["dim"],
    "box.hint": C["dim"],
    "box.status": f"bold {C['cyan']}",
    "box.spinner": f"bold {C['accent']}",
    "box.flash": f"bold {C['yellow']}",
    "box.approve": f"bold {C['yellow']}",
    "arrow": f"bold {C['green']}",
    "cont": C["dim"],
    "stream.preview": C["fg"],
    "overlay.border": C["accent"],
    "overlay.item": C["fg"],
    "overlay.item.selected": f"bg:#44475a bold {C['fg']}",
    "overlay.footer": C["dim"],
    "completion-menu.completion": "bg:#282a36 #f8f8f2",
    "completion-menu.completion.current": "bg:#44475a #ffffff bold",
    "completion-menu.meta.completion": "bg:#282a36 #6272a4",
    "completion-menu.meta.completion.current": "bg:#44475a #8be9fd",
    "completion-menu.multi-column-meta": "bg:#282a36 #6272a4",
    "scrollbar.background": "bg:#282a36",
    "scrollbar.button": "bg:#6272a4",
})

EFFORT_COLORS = {e.key: e.color for e in EFFORTS}
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ---------------------------------------------------------------------------
# Rich markdown rendering tweaks (used for non-streamed output)
# ---------------------------------------------------------------------------


class _Heading(Heading):
    def __rich_console__(self, console, options):
        yield Text(f"◆ {self.plain}", style=f"bold {C['accent']}")


class _CodeBlock(CodeBlock):
    def __rich_console__(self, console, options):
        code = str(self.text).rstrip("\n")
        yield Syntax(code, self.lexer_name or "text", theme="monokai",
                     background_color="default", word_wrap=True, padding=(0, 1))
        yield Text()


Markdown.elements["fence"] = _CodeBlock
Markdown.elements["code_block"] = _CodeBlock
for _h in ("h1", "h2", "h3", "h4", "h5", "h6"):
    Markdown.elements[_h] = _Heading


class _StdoutProxy:
    """Write to whatever sys.stdout currently is.

    During app.run(), prompt_toolkit's patch_stdout swaps sys.stdout for a
    proxy that knows how to interleave output with the pinned box render.
    If rich wrote to the raw stdout instead, prompt_toolkit's cursor math
    would break and redraws would overwrite already-streamed text.
    """

    @property
    def encoding(self):
        return getattr(sys.stdout, "encoding", None) or "utf-8"

    @property
    def closed(self):
        return getattr(sys.stdout, "closed", False)

    def write(self, s: str):
        return sys.stdout.write(s)

    def flush(self):
        sys.stdout.flush()

    def isatty(self) -> bool:
        try:
            return sys.stdout.isatty()
        except Exception:
            return True


def make_console() -> Console:
    # lock in terminal-ness now; patch_stdout later swaps sys.stdout for a
    # proxy that would otherwise make rich think it lost its terminal
    is_tty = True
    try:
        is_tty = sys.stdout.isatty()
    except Exception:
        pass
    return Console(file=_StdoutProxy(), highlight=False, soft_wrap=False,
                   force_terminal=is_tty)


# ---------------------------------------------------------------------------
# Slash-command completion
# ---------------------------------------------------------------------------

SLASH_COMMANDS = [
    ("/model", "select model — PgUp/PgDn/Tab to navigate"),
    ("/effort", "low · medium · high · extrahigh · ultrahigh"),
    ("/goal", "goal contract — set · prove · close · status · waive · clear"),
    ("/autonomy", "autonomy level 0-5 (observer → autonomous)"),
    ("/focus", "deep-work mode — /focus <1-20> auto-continues until done"),
    ("/render", "toggle rendered-markdown replies — /render [on|off]"),
    ("/workflow", "saved pipelines — /workflow [list|run <name>|delete <name>]"),
    ("/export", "enterprise audit report — /export [md|html]"),
    ("/forecast", "projection from measured velocity + usage"),
    ("/health", "provider health — model errors + failovers"),
    ("/notify", "event notifications — /notify <url|file:path|off>"),
    ("/resume", "resume a previous session — /resume [branch]"),
    ("/state", "live projection of the event log (cost, goal, dead-ends)"),
    ("/rewind", "rewind timeline + files to a seq — /rewind <seq>"),
    ("/revert", "revert FILES only to a seq (agent keeps memory)"),
    ("/fork", "fork the timeline into a new branch — /fork [name]"),
    ("/why", "causal chain for an event — /why <seq>"),
    ("/impact", "code impact analysis — /impact <symbol> [path]"),
    ("/forge", "environment digest + drift — /forge [probe|drift]"),
    ("/oracle", "post-run analysis, calibration, facts"),
    ("/budget", "budget governor status/extend — /budget steps N|usd X|reset"),
    ("/constitution", "standing rules — /constitution [show|edit]"),
    ("/replay", "replay the session log as a film (text)"),
    ("/memory", "recent episodes + dead-end ledger"),
    ("/judge", "deterministic check — /judge <type> <json-or-args>"),
    ("/compile", "intent compiler: goal → optimized ordered waves"),
    ("/evolve", "evolve a role brief — /evolve [role|rollback <role>]"),
    ("/brain", "cognitive memory — /brain <query>|sleep|stats"),
    ("/merge", "semantic timeline merge — /merge <branchA> <branchB>"),
    ("/theater", "time-travel debugger — /theater <seq|why N|cf N|diff A B>"),
    ("/debate", "multi-model debate tournament — /debate <question>"),
    ("/market", "task market auction — /market <t1> | <t2>"),
    ("/tower", "web control tower dashboard — /tower [port]"),
    ("/verify", "formal LTL verification — /verify <goal>|log"),
    ("/mcts", "tree-of-agents strategy search — /mcts <i1>; <i2>; …"),
    ("/causal", "causal analysis — /causal | /causal do <feature>"),
    ("/bandit", "thompson router — /bandit | /bandit <task>"),
    ("/mesh", "agent-to-agent network — /mesh serve|discover|delegate"),
    ("/roleforge", "create a NEW specialist role — /roleforge <mission>"),
    ("/synth", "synthesize a new tool — /synth {json spec}"),
    ("/ci", "CI pilot — /ci start|stop|status"),
    ("/tune", "auto-tune knobs (TPE) — /tune [trials]"),
    ("/dual", "system 1/2 routing — /dual <q>|stats"),
    ("/predict", "predict change impact — /predict <path>"),
    ("/race", "racing strategy universes — /race <task>"),
    ("/vitals", "homeostasis check + self-repair"),
    ("/attention", "last context token auction"),
    ("/fabric", "bitemporal knowledge — /fabric ask|assert|history"),
    ("/crew", "persistent subagents — /crew [spawn|send|wait|close|resume|status]"),
    ("/auto", "autopilot self-routing — /auto [on|off|status]"),
    ("/prompt", "system prompt — /prompt [main|master|list]"),
    ("/mastermind", "prompt coherence ledger — sealed prompts, gate, lineage"),
    ("/dashboard", "live observability — cost, goal, agents, router, spec"),
    ("/router", "smart model routing — decisions + savings"),
    ("/spec", "speculative execution — prefetch stats + hit-rate"),
    ("/recall", "semantic memory — /recall <question>"),
    ("/mission", "daemon missions — /mission [start|tick|list|abandon]"),
    ("/heal", "self-healing ledger — root causes captured + healed"),
    ("/skills", "skill forge — self-authored tools"),
    ("/council", "adversarial debate — /council <proposition>"),
    ("/analyze", "static analysis — /analyze <path> (taint, complexity, cycles)"),
    ("/graph", "knowledge graph — /graph [index <path>|query <name>|impact <name>]"),
    ("/coverage", "line-coverage ledger — last measured runs"),
    ("/fuzz", "fuzzing ledger — runs, crashes, shrunk reproducers"),
    ("/mutate", "mutation testing — /mutate <file> <suite-command>"),
    ("/help", "commands and key bindings"),
    ("/clear", "clear the screen"),
    ("/new", "fresh conversation"),
    ("/history", "browse previous turns"),
    ("/save", "save session to disk"),
    ("/approve", "toggle auto-approve for tools"),
    ("/reasoning", "toggle showing model reasoning"),
    ("/usage", "token usage for this session"),
    ("/about", "about FullAgent"),
    ("/exit", "quit FullAgent"),
]


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        word = text.lower()
        if word.startswith("/effort ") or word == "/effort":
            arg = word[len("/effort"):].strip()
            for e in EFFORTS:
                if e.key.startswith(arg):
                    yield Completion(f"/effort {e.key}",
                                     start_position=-len(text),
                                     display_meta=e.description)
            return
        if word.startswith("/goal ") or word == "/goal":
            arg = word[len("/goal"):].strip()
            for sub, meta in (
                    ("set", "set <statement> | <clause> | <clause> …"),
                    ("prove", "prove <clause-id> — run its own predicate"),
                    ("prove-all", "prove every clause from scratch"),
                    ("close", "closure ritual — compute the terminal state"),
                    ("status", "the Goal Compass: distance, velocity, focus"),
                    ("waive", "waive <clause-id> --reason '…' (human waiver)"),
                    ("clear", "deactivate the goal")):
                if sub.startswith(arg):
                    yield Completion(f"/goal {sub}",
                                     start_position=-len(text),
                                     display_meta=meta)
            return
        if word.startswith("/autonomy ") or word == "/autonomy":
            arg = word[len("/autonomy"):].strip()
            from .agent import AUTONOMY_LEVELS
            for level, desc in AUTONOMY_LEVELS.items():
                if str(level).startswith(arg):
                    yield Completion(f"/autonomy {level}",
                                     start_position=-len(text),
                                     display_meta=desc)
            return
        if word.startswith("/judge ") or word == "/judge":
            arg = word[len("/judge"):].strip()
            for ptype, meta in (
                    ("exit_code", '{"command": "pytest -q", "expect": 0}'),
                    ("file_exists", '{"path": "src/x.py"}'),
                    ("file_contains", '{"path": "src/x.py", "text": "def foo"}'),
                    ("file_matches", '{"path": "src/x.py", "pattern": "…"}'),
                    ("command_output_contains",
                     '{"command": "python -V", "text": "Python"}')):
                if ptype.startswith(arg):
                    yield Completion(f"/judge {ptype} ",
                                     start_position=-len(text),
                                     display_meta=meta)
            return
        for name, meta in SLASH_COMMANDS:
            if name.startswith(word):
                yield Completion(name, start_position=-len(text),
                                 display_meta=meta)


# ---------------------------------------------------------------------------
# Live-write tracker — true live file writing
# ---------------------------------------------------------------------------


class _LiveWrite:
    """Incrementally pulls a growing string field out of a streaming
    tool-call's JSON arguments, so a file change can be shown happening
    line-by-line WHILE the model generates it (a true live write/edit,
    not a post-hoc replay).

    `key` selects which JSON string field to stream: "content" for
    write_file, "new_string" for edit_file.

    Robust by design: if the JSON is shaped unexpectedly, `feed` simply
    returns no lines and the caller falls back to the normal cascade.
    Nothing here can break the turn."""

    _PATH_KEY = re.compile(r'"path"\s*:\s*"')

    def __init__(self, key: str = "content"):
        self._CONTENT_KEY = re.compile(
            r'"' + re.escape(key) + r'"\s*:\s*"')
        self.buf = ""            # accumulated argument JSON so far
        self.content_start = -1  # index into buf of first content char
        self.raw_pos = 0         # content raw chars already consumed
        self.pending = ""        # unescaped text not yet emitted as lines
        self.lines = 0           # full lines emitted so far
        self.done = False        # saw the closing quote of the content

    def path(self) -> str | None:
        return self._field(self.buf, "path")

    @classmethod
    def _field(cls, buf: str, key: str) -> str | None:
        """Return the value of a JSON string field if it is COMPLETE
        (closing quote seen), else None. Used for path / old_string."""
        m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', buf)
        if not m:
            return None
        val, _, done = cls._unescape(buf[m.end():])
        if not done:
            return None  # value still streaming — wait for closing quote
        return val

    def feed(self, chunk: str) -> list:
        """Feed a new argument chunk; returns the complete content lines that
        became available as a result."""
        self.buf += chunk
        if self.done:
            return []  # content fully captured; ignore the rest of the JSON
        if self.content_start < 0:
            m = self._CONTENT_KEY.search(self.buf)
            if not m:
                return []
            self.content_start = m.end()
        raw = self.buf[self.content_start:]
        text, consumed, done = self._unescape(raw[self.raw_pos:])
        self.raw_pos += consumed
        self.done = done
        self.pending += text
        out = []
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.lines += 1
            out.append(line)
        return out

    def flush(self) -> str | None:
        """Return the final partial line (if any) once generation ends."""
        if self.pending:
            line, self.pending = self.pending, ""
            self.lines += 1
            return line
        return None

    @staticmethod
    def _unescape(s: str) -> tuple:
        """Unescape a JSON string fragment. Returns (text, consumed, done).
        Stops before a trailing incomplete escape; `done` is True when the
        unescaped closing quote of the value was reached."""
        out = []
        i = 0
        n = len(s)
        done = False
        while i < n:
            c = s[i]
            if c == '"':
                done = True
                i += 1
                break
            if c != '\\':
                out.append(c)
                i += 1
                continue
            if i + 1 >= n:
                break  # trailing backslash — wait for the next chunk
            e = s[i + 1]
            if e == 'u':
                if i + 6 > n:
                    break  # incomplete \\uXXXX — wait for more
                hexs = s[i + 2:i + 6]
                try:
                    out.append(chr(int(hexs, 16)))
                except ValueError:
                    out.append('\\u' + hexs)
                i += 6
            else:
                mapping = {'n': '\n', 't': '\t', 'r': '\r', 'b': '\b',
                           'f': '\f', '"': '"', '\\': '\\', '/': '/'}
                out.append(mapping.get(e, '\\' + e))
                i += 2
        return ''.join(out), i, done


# ---------------------------------------------------------------------------
# Overlay list (model / effort / help / history pickers)
# ---------------------------------------------------------------------------


class OverlayList:
    """A modal selection list rendered directly above the prompt box."""

    PAGE = 5
    WINDOW = 10

    def __init__(self, title: str, items: list[tuple[str, str]],
                 selected_index: int = 0,
                 on_select: Callable[[int], None] | None = None,
                 footer: str = ""):
        self.title = title
        self.items = items              # list of (html, meta)
        self.index = max(0, min(selected_index, len(items) - 1))
        self.on_select = on_select
        self.footer = footer
        self.visible = False
        self._top = max(0, self.index - self.WINDOW // 2)

    def open(self) -> None:
        self.visible = True

    def close(self) -> None:
        self.visible = False

    def move(self, delta: int) -> None:
        if not self.items:
            return  # empty list — no-op instead of ZeroDivisionError
        self.index = (self.index + delta) % len(self.items)
        if self.index < self._top:
            self._top = self.index
        elif self.index >= self._top + self.WINDOW:
            self._top = self.index - self.WINDOW + 1

    def page(self, delta: int) -> None:
        self.move(delta * self.PAGE)

    def select(self) -> None:
        idx = self.index
        self.close()
        if self.on_select:
            self.on_select(idx)

    def fragments(self, width: int) -> list:
        inner = max(30, width - 2)
        frags: list = []
        title = f" {self.title} "
        frags.append(("class:overlay.border",
                      f"╔{title}{'═' * max(0, inner - len(title))}╗\n"))
        shown = self.items[self._top:self._top + self.WINDOW]
        for i, (html, meta) in enumerate(shown):
            real_i = self._top + i
            selected = real_i == self.index
            marker = "▶ " if selected else "  "
            text = fragment_list_to_text(to_formatted_text(HTML(html)))
            line = f"{marker}{text}"
            if len(line) > inner - 2:
                line = line[:inner - 3] + "…"
            style = ("class:overlay.item.selected" if selected
                     else "class:overlay.item")
            frags.append((style, f"║{line}{' ' * max(0, inner - len(line))}║\n"))
        foot = (self.footer or
                "↑↓ PgUp PgDn Tab move · Enter select · Esc close")
        if len(foot) > inner:
            foot = foot[:inner]
        frags.append(("class:overlay.footer",
                      f"╚{foot}{'═' * max(0, inner - len(foot))}╝"))
        return frags


# ---------------------------------------------------------------------------
# The UI — one persistent application for the whole session
# ---------------------------------------------------------------------------


class UI:
    def __init__(self, cfg: Config, agent: Agent):
        self.cfg = cfg
        self.agent = agent
        self.console = make_console()
        self.overlay: OverlayList | None = None

        # turn state
        self._busy = False
        self._status_text = ""
        self._spinner_i = 0
        self._spinner_on = False
        self._cancel_flag = threading.Event()
        self._last_flush = 0.0
        # SPEED: cached terminal size (queried at most twice a second)
        self._width_cache: int | None = None
        self._width_cache_ts = 0.0

        # approval state
        self._approve_request: tuple[Tool, dict, threading.Event] | None = None
        self._approve_result = False

        # flash message in the bottom border
        self._flash: tuple[str, str] | None = None
        self._flash_timer: threading.Timer | None = None

        # goal-status cache for the border (fold is not free per frame)
        self._goal_cache = None
        self._goal_cache_ts = 0.0

        # focus mode (deep work): remaining auto-continuation turns
        self._focus_remaining = 0

        # live context-usage cache for the border (est. tokens / window)
        self._ctx_cache: int | None = None
        self._ctx_cache_ts = 0.0

        self._build()

    # -- small helpers ---------------------------------------------------------

    def _model(self) -> Model:
        m = model_by_id(self.cfg.model_id)
        assert m is not None
        return m

    def _effort(self) -> Effort:
        e = effort_by_key(self.cfg.effort)
        assert e is not None
        return e

    def _width(self) -> int:
        # SPEED: terminal size changes rarely; the query runs at most twice
        # a second (the streaming preview asks for it on every token chunk)
        now = time.time()
        if self._width_cache is not None and now - self._width_cache_ts < 0.5:
            return self._width_cache
        try:
            cols = get_app().output.get_size().columns
        except Exception:
            cols = shutil.get_terminal_size((100, 24)).columns
        self._width_cache = max(60, cols)
        self._width_cache_ts = now
        return self._width_cache

    def _invalidate(self) -> None:
        try:
            get_app().invalidate()
        except Exception:
            pass

    def _set_flash(self, text: str, color: str = C["yellow"]) -> None:
        self._flash = (text, color)
        if self._flash_timer:
            self._flash_timer.cancel()
        self._flash_timer = threading.Timer(4.0, self._clear_flash)
        self._flash_timer.daemon = True
        self._flash_timer.start()
        self._invalidate()

    def _clear_flash(self) -> None:
        self._flash = None
        self._invalidate()

    def _set_status(self, text: str) -> None:
        if text == self._status_text:
            return
        self._status_text = text
        self._invalidate()

    # -- box border fragments -----------------------------------------------------

    def _top_fragments(self) -> list:
        width = self._width()
        model = self._model()
        effort = self._effort()
        effort_color = EFFORT_COLORS.get(effort.key, C["fg"])

        segs: list[tuple[str, str]] = []
        segs.append((f" {APP_NAME} ", "class:box.title"))
        segs.append((f" model: {model.label} ", "class:box.model"))
        if model.tag:
            segs.append((f"{model.tag} ", "class:box.tag"))
        segs.append((f" effort: {effort.label.lower()} ",
                     f"class:box.effort {effort_color}"))
        segs.append((f" L{self.agent.autonomy} ", "class:box.tag"))
        # live goal distance — always on screen when a goal is active (§24).
        # cached for 1s: the border re-renders every frame, and status()
        # folds the whole log
        now = time.time()
        if self._goal_cache is None or now - self._goal_cache_ts > 1.0:
            self._goal_cache = self.agent.goal.status()
            self._goal_cache_ts = now
        goal = self._goal_cache
        if goal.active:
            segs.append((f" goal: {(1 - goal.distance) * 100:.0f}% ",
                         "class:box.status"))
        if self._focus_remaining > 0:
            segs.append((f" 🎯 focus×{self._focus_remaining} ",
                         "class:box.flash"))
        # live context usage — cached 1s (the border redraws every frame)
        if self._ctx_cache is None or now - self._ctx_cache_ts > 1.0:
            try:
                from .client import estimate_tokens
                used = estimate_tokens(self.agent.messages,
                                       self.agent.model.id)
                window = max(1, self.agent.model.context_window)
                self._ctx_cache = min(100, int(used * 100 / window))
            except Exception:
                self._ctx_cache = 0
            self._ctx_cache_ts = now
        ctx_color = (C["green"] if self._ctx_cache < 60
                     else C["yellow"] if self._ctx_cache < 85
                     else C["red"])
        segs.append((f" ctx {self._ctx_cache}% ",
                     f"bold {ctx_color}"))
        segs.append((f" session: {self.agent.session_id} ", "class:box.session"))

        # fixed = corners (2) + first dash (1) + "──" before each later seg
        def fixed_len(segs: list) -> int:
            return sum(len(t) for t, _ in segs) + 3 + 2 * (len(segs) - 1)

        # drop trailing segments (session, effort, …) if the terminal is
        # too narrow — never let the border exceed the terminal width or the
        # closing "╮" wraps to the next line and corrupts the render
        while fixed_len(segs) > width and len(segs) > 2:
            segs.pop()

        fill = max(0, width - fixed_len(segs))
        frags: list = [("class:box", "╭")]
        for i, (text, style) in enumerate(segs):
            frags.append(("class:box", "──" if i else "─"))
            frags.append((style, text))
        frags.append(("class:box", "─" * fill + "╮"))
        return frags

    def _bottom_fragments(self) -> list:
        width = self._width()
        inner = width - 2

        if self._approve_request is not None:
            tool = self._approve_request[0]
            bar = f" ⚠ approve {tool.name}?  [y]es  [n]o  [a]lways "[:max(1, inner)]
            return ([("class:box", "╰"), ("class:box.approve", bar),
                     ("class:box", "─" * max(0, inner - len(bar)) + "╯")]
                    + self._approve_args_line(tool, self._approve_request[1]))

        if self._busy:
            frame = SPINNER_FRAMES[self._spinner_i]
            max_status = max(0, inner - len(" ⠹  ·  Esc/Ctrl+C cancel ") - 4)
            status = self._status_text[:max_status]
            bar = f" {frame} {status}  ·  Esc/Ctrl+C cancel "
            return [("class:box", "╰"),
                    ("class:box.spinner", f" {frame} "),
                    ("class:box.status", status),
                    ("class:box.hint", "  ·  Esc/Ctrl+C cancel "),
                    ("class:box",
                     "─" * max(0, inner - len(bar)) + "╯")]

        if self._flash:
            text, color = self._flash
            hint = f" {text} "[:max(1, inner)]
            return [("class:box", "╰"), (f"bold {color}", hint),
                    ("class:box", "─" * max(0, inner - len(hint)) + "╯")]

        hint = (" Enter send · Esc+Enter newline · / commands · "
                "Ctrl+T models · Ctrl+E effort · Ctrl+C cancel ")
        if len(hint) > inner:
            hint = " Enter send · / commands · Ctrl+T models · Ctrl+C cancel "
        if len(hint) > inner:
            hint = " Enter send "
        return [("class:box", "╰"), ("class:box.hint", hint[:inner]),
                ("class:box", "─" * max(0, inner - len(hint)) + "╯")]

    def _approve_args_line(self, tool: Tool, args: dict) -> list:
        return []

    def _overlay_fragments(self) -> list:
        if self.overlay and self.overlay.visible:
            return self.overlay.fragments(self._width())
        return []

    # -- construction ----------------------------------------------------------------

    def _build(self) -> None:
        config.ensure_dirs()
        self.search_toolbar = SearchToolbar()
        self.buffer = Buffer(
            name=DEFAULT_BUFFER,
            multiline=True,
            completer=SlashCompleter(),
            complete_while_typing=True,
            enable_history_search=True,
            auto_suggest=AutoSuggestFromHistory(),
            history=SafeFileHistory(str(config.HISTORY_FILE)),
            accept_handler=self._accept_handler,
        )

        def get_line_prefix(line: int, wrap_count: int):
            if line == 0 and wrap_count == 0:
                return [("class:arrow", "❯ ")]
            return [("class:cont", "  ")]

        self.input_control = BufferControl(
            buffer=self.buffer,
            search_buffer_control=self.search_toolbar.control,
            preview_search=True,
        )
        input_window = Window(
            self.input_control,
            height=Dimension(min=1, max=10),
            wrap_lines=True,
            get_line_prefix=get_line_prefix,
        )
        self.input_window = input_window

        top_window = Window(FormattedTextControl(self._top_fragments),
                            height=1, dont_extend_height=True)
        bottom_window = Window(FormattedTextControl(self._bottom_fragments),
                               height=1, dont_extend_height=True)
        left_window = Window(FormattedTextControl(
            lambda: [("class:box", "│ ")]), width=2, dont_extend_width=True)
        right_window = Window(FormattedTextControl(
            lambda: [("class:box", " │")]), width=2, dont_extend_width=True)
        overlay_window = Window(
            FormattedTextControl(self._overlay_fragments),
            dont_extend_height=True)

        root = FloatContainer(
            HSplit([
                ConditionalContainer(
                    overlay_window,
                    filter=Condition(self._overlay_open)),
                top_window,
                VSplit([left_window, input_window, right_window], padding=0),
                bottom_window,
                ConditionalContainer(self.search_toolbar,
                                     filter=Condition(self._search_visible)),
            ]),
            [
                Float(xcursor=True, ycursor=True,
                      content=CompletionsMenu(
                          max_height=10, scroll_offset=1,
                          extra_filter=has_focus(self.buffer))),
            ],
        )

        self.app = Application(
            layout=Layout(root, focused_element=input_window),
            style=STYLE,
            key_bindings=self._build_key_bindings(),
            full_screen=False,
            mouse_support=False,
        )

    def _search_visible(self) -> bool:
        try:
            return get_app().layout.current_control == self.search_toolbar.control
        except Exception:
            return False

    def _accept_handler(self, buff: Buffer) -> bool:
        text = buff.text.strip()
        if text:
            self._dispatch(text)
        return False  # clear the buffer

    # -- state filters ------------------------------------------------------------------

    def _overlay_open(self) -> bool:
        return self.overlay is not None and self.overlay.visible

    def _approving(self) -> bool:
        return self._approve_request is not None

    def _completion_active(self) -> bool:
        return self.buffer.complete_state is not None

    def _on_first_line(self) -> bool:
        return self.buffer.document.cursor_position_row == 0

    def _on_last_line(self) -> bool:
        doc = self.buffer.document
        return doc.cursor_position_row == len(doc.lines) - 1

    def _buffer_empty(self) -> bool:
        return not self.buffer.text

    # -- key bindings ----------------------------------------------------------------------

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        focused = has_focus(self.buffer)
        ov = Condition(self._overlay_open)
        approving = Condition(self._approving)
        idle = ~approving  # normal input only when not at the approve bar
        # submitting a new turn while one is already running would spawn
        # concurrent agent.run_turn threads over the same message list
        can_submit = focused & ~ov & idle & Condition(lambda: not self._busy)

        # --- approval bar: y / n / a ---
        # register the catch-all FIRST so it has the lowest priority; the
        # specific y/n/a/Enter/Ctrl+C handlers below override it.
        @kb.add(Keys.Any, filter=approving)
        def _appr_swallow(event):
            pass  # ignore everything else while approving

        @kb.add("y", filter=approving)
        def _appr_y(event):
            self._answer_approve("y")

        @kb.add("n", filter=approving)
        def _appr_n(event):
            self._answer_approve("n")

        @kb.add("a", filter=approving)
        def _appr_a(event):
            self._answer_approve("a")

        @kb.add(Keys.Enter, filter=approving)
        def _appr_enter(event):
            self._answer_approve("n")

        @kb.add("c-c", filter=approving)
        def _appr_cancel(event):
            self._answer_approve("n")

        # --- overlay navigation ---
        @kb.add(Keys.Enter, filter=ov)
        def _ov_enter(event):
            self.overlay.select()

        @kb.add(Keys.Escape, filter=ov)
        @kb.add("c-c", filter=ov)
        def _ov_close(event):
            self.overlay.close()

        @kb.add(Keys.Up, filter=ov)
        @kb.add("c-p", filter=ov)
        @kb.add(Keys.BackTab, filter=ov)
        def _ov_up(event):
            self.overlay.move(-1)

        @kb.add(Keys.Down, filter=ov)
        @kb.add("c-n", filter=ov)
        @kb.add(Keys.Tab, filter=ov)
        def _ov_down(event):
            self.overlay.move(1)

        @kb.add(Keys.PageUp, filter=ov)
        def _ov_pgup(event):
            self.overlay.page(-1)

        @kb.add(Keys.PageDown, filter=ov)
        def _ov_pgdn(event):
            self.overlay.page(1)

        @kb.add(Keys.Home, filter=ov)
        def _ov_home(event):
            self.overlay.index = 0
            self.overlay._top = 0

        @kb.add(Keys.End, filter=ov)
        def _ov_end(event):
            self.overlay.index = len(self.overlay.items) - 1
            self.overlay._top = max(0, self.overlay.index - OverlayList.WINDOW + 1)

        # --- completion menu: PgUp/PgDn also navigate it ---
        @kb.add(Keys.PageDown, filter=focused & ~ov & idle &
                Condition(self._completion_active))
        def _cm_pgdn(event):
            self.buffer.complete_next(5)

        @kb.add(Keys.PageUp, filter=focused & ~ov & idle &
                Condition(self._completion_active))
        def _cm_pgup(event):
            self.buffer.complete_previous(5)

        # --- input ---
        @kb.add(Keys.Enter, filter=can_submit)
        def _enter(event):
            b = self.buffer
            if b.complete_state is not None:
                # the highlighted completion is already previewed in the
                # buffer — close the menu and submit the text as-is
                b.complete_state = None
            if not b.text.strip():
                return
            b.validate_and_handle()

        @kb.add(Keys.Escape, Keys.Enter, filter=focused & ~ov & idle)
        def _newline(event):
            self.buffer.insert_text("\n")

        @kb.add(Keys.Up, filter=focused & ~ov & idle &
                Condition(self._on_first_line))
        def _hist_prev(event):
            self.buffer.history_backward()

        @kb.add(Keys.Down, filter=focused & ~ov & idle &
                Condition(self._on_last_line))
        def _hist_next(event):
            self.buffer.history_forward()

        @kb.add("c-c", filter=focused & ~ov & idle)
        def _ctrl_c(event):
            if self._busy:
                self._cancel_flag.set()
                self._set_status("cancelling…")
            elif self.buffer.text:
                self.buffer.reset()
                self._set_flash("input cleared — Ctrl+C again or /exit to quit",
                                C["red"])
            else:
                self._set_flash("type /exit or Ctrl+D to quit", C["dim"])

        # single Esc while a turn is running = interrupt, exactly like
        # Ctrl+C (Esc+Enter stays the multi-line newline binding above)
        @kb.add(Keys.Escape, filter=focused & ~ov &
                Condition(lambda: self._busy))
        def _esc_cancel(event):
            self._cancel_flag.set()
            self._set_status("cancelling…")

        # Esc at the approve bar = "no", same as Ctrl+C there
        @kb.add(Keys.Escape, filter=approving)
        def _esc_deny(event):
            self._answer_approve("n")

        @kb.add("c-d", filter=focused & ~ov & idle &
                Condition(self._buffer_empty))
        def _ctrl_d(event):
            event.app.exit()

        @kb.add("c-l", filter=~ov & idle)
        def _ctrl_l(event):
            event.app.renderer.clear()
            self._invalidate()

        @kb.add("c-t", filter=focused & ~ov & idle)
        def _ctrl_t(event):
            self.open_model_selector()

        @kb.add("c-e", filter=focused & ~ov & idle)
        def _ctrl_e(event):
            self.open_effort_selector()

        @kb.add("c-r", filter=focused & ~ov & idle)
        def _ctrl_r(event):
            start_search(self.input_control)

        @kb.add("c-x", "c-e", filter=focused & ~ov & idle)
        def _external_editor(event):
            self.buffer.open_in_editor(validate_and_handle=True)

        return kb

    def _bg(self, fn) -> None:
        """Run fn on a daemon thread with exceptions surfaced to the UI —
        a bare thread's default excepthook prints to stderr, invisible
        under patch_stdout, so failures would look like silent no-ops."""
        def guarded():
            try:
                fn()
            except Exception as e:  # noqa: BLE001 — report, never crash
                self.print_error(f"{type(e).__name__}: {e}")
        threading.Thread(target=guarded, daemon=True).start()

    # -- dispatch: slash commands + turns ----------------------------------------------------

    def _dispatch(self, text: str) -> None:
        if text.startswith("/"):
            self._handle_slash(text)
            return
        if self._busy:
            # a turn is already running — never overlap two agent loops
            self._set_flash("busy — wait for the current turn (Ctrl+C to "
                            "cancel)", C["yellow"])
            return
        self._emit_user(text)
        threading.Thread(target=self._run_turn_thread, args=(text,),
                         daemon=True).start()

    def _handle_slash(self, text: str) -> None:
        parts = text.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        # one typo or a transient OSError inside a command must never
        # unwind through prompt_toolkit and kill the whole session —
        # same policy as the guarded turn thread below
        try:
            self._route_slash(cmd, arg)
        except Exception as e:  # noqa: BLE001 — the UI must survive
            self.print_error(f"{type(e).__name__}: {e}")

    def _route_slash(self, cmd: str, arg: str) -> None:
        if cmd in ("/exit", "/quit", "/q"):
            self.print_info(f"bye — session {self.agent.session_id} saved",
                            C["dim"])
            self.agent.save_session()
            self.app.exit()
        elif cmd == "/new":
            self.agent.reset()
            self.print_info(
                f"✓ new conversation started (session {self.agent.session_id})",
                C["green"])
        elif cmd == "/clear":
            try:
                self.app.renderer.clear()
            except Exception:
                pass
            self._invalidate()
        elif cmd == "/model":
            arg = arg.split()[0] if arg else ""
            if arg:
                m = model_by_id(arg)
                if m is None:
                    self.print_error(f"unknown model: {arg} — use /model to browse")
                else:
                    self.cfg.model_id = m.id
                    self.cfg.save()
                    self.print_info(f"✓ model → {m.label} ({m.id})", C["green"])
            else:
                self.open_model_selector()
        elif cmd == "/effort":
            arg = arg.split()[0] if arg else ""
            if arg:
                e = effort_by_key(arg.lower())
                if e is None:
                    self.print_error("effort levels: " +
                                     " · ".join(x.key for x in EFFORTS))
                else:
                    self.cfg.effort = e.key
                    self.cfg.save()
                    self.print_info(f"✓ effort → {e.label}", e.color)
            else:
                self.open_effort_selector()
        elif cmd == "/help":
            self.open_help()
        elif cmd == "/history":
            self.open_history()
        elif cmd == "/save":
            path = self.agent.save_session()
            if path:
                self.print_info(f"✓ session saved → {path}", C["green"])
            else:
                self.print_error("could not save session")
        elif cmd == "/approve":
            self.cfg.auto_approve = not self.cfg.auto_approve
            self.cfg.save()
            state = "ON" if self.cfg.auto_approve else "OFF"
            self.print_info(f"✓ auto-approve: {state}",
                            C["yellow"] if self.cfg.auto_approve else C["dim"])
        elif cmd == "/reasoning":
            self.cfg.show_reasoning = not self.cfg.show_reasoning
            self.cfg.save()
            state = "ON" if self.cfg.show_reasoning else "OFF"
            self.print_info(f"✓ reasoning display: {state}", C["pink"])
        elif cmd == "/usage":
            self.print_usage(self.agent.turns)
        elif cmd == "/goal":
            self._cmd_goal(arg)
        elif cmd == "/autonomy":
            self._cmd_autonomy(arg)
        elif cmd == "/focus":
            self._cmd_focus(arg)
        elif cmd == "/render":
            self._cmd_render(arg)
        elif cmd == "/workflow":
            self._cmd_workflow(arg)
        elif cmd == "/export":
            self._cmd_export(arg)
        elif cmd == "/forecast":
            self.print_info(self.agent.get_forecast(), C["cyan"])
        elif cmd == "/health":
            self._cmd_health()
        elif cmd == "/notify":
            self._cmd_notify(arg)
        elif cmd == "/resume":
            self._cmd_resume(arg)
        elif cmd == "/state":
            self._cmd_state()
        elif cmd == "/rewind":
            self._cmd_rewind(arg)
        elif cmd == "/revert":
            self._cmd_revert(arg)
        elif cmd == "/fork":
            self._cmd_fork(arg)
        elif cmd == "/why":
            self._cmd_why(arg)
        elif cmd == "/impact":
            self._cmd_impact(arg)
        elif cmd == "/forge":
            self._cmd_forge(arg)
        elif cmd == "/oracle":
            self.print_info(self.agent.oracle.format_report(), C["cyan"])
        elif cmd == "/budget":
            self._cmd_budget(arg)
        elif cmd == "/constitution":
            self._cmd_constitution(arg)
        elif cmd == "/replay":
            self._cmd_replay()
        elif cmd == "/memory":
            self._cmd_memory()
        elif cmd == "/judge":
            self._cmd_judge(arg)
        elif cmd == "/compile":
            self._cmd_compile(arg)
        elif cmd == "/evolve":
            self._cmd_evolve(arg)
        elif cmd == "/brain":
            self._cmd_brain(arg)
        elif cmd == "/merge":
            self._cmd_merge(arg)
        elif cmd == "/theater":
            self._cmd_theater(arg)
        elif cmd == "/debate":
            self._cmd_debate(arg)
        elif cmd == "/market":
            self._cmd_market(arg)
        elif cmd == "/tower":
            self._cmd_tower(arg)
        elif cmd == "/verify":
            self._cmd_verify(arg)
        elif cmd == "/mcts":
            self._cmd_mcts(arg)
        elif cmd == "/causal":
            self._cmd_causal(arg)
        elif cmd == "/bandit":
            self._cmd_bandit(arg)
        elif cmd == "/mesh":
            self._cmd_mesh(arg)
        elif cmd == "/roleforge":
            self._cmd_roleforge(arg)
        elif cmd == "/synth":
            self._cmd_synth(arg)
        elif cmd == "/ci":
            self._cmd_ci(arg)
        elif cmd == "/tune":
            self._cmd_tune(arg)
        elif cmd == "/dual":
            self._cmd_dual(arg)
        elif cmd == "/predict":
            self._cmd_predict_impact(arg)
        elif cmd == "/race":
            self._cmd_race(arg)
        elif cmd == "/vitals":
            self.print_info(self.agent.homeo.check_and_repair()
                            .format(), C["cyan"])
        elif cmd == "/attention":
            self.print_info(self.agent.attention.format_last(), C["cyan"])
        elif cmd == "/fabric":
            self._cmd_fabric(arg)
        elif cmd == "/crew":
            self._cmd_crew(arg)
        elif cmd == "/auto":
            self._cmd_auto(arg)
        elif cmd == "/prompt":
            self._cmd_prompt(arg)
        elif cmd == "/mastermind":
            self.print_info(self.agent.mastermind.format_status(), C["pink"])
        elif cmd == "/dashboard":
            self.print_info(self.agent.dashboard.render(), C["cyan"])
        elif cmd == "/router":
            self.print_info(self.agent.router.format_status(), C["green"])
        elif cmd == "/spec":
            self.print_info(self.agent.speculator.format_status(), C["cyan"])
        elif cmd == "/recall":
            self._cmd_recall(arg)
        elif cmd == "/mission":
            self._cmd_mission(arg)
        elif cmd == "/heal":
            self.print_info(self.agent.healer.format_status(), C["yellow"])
        elif cmd == "/skills":
            self.print_info(self.agent.skill_forge.format_status(), C["pink"])
        elif cmd == "/council":
            self._cmd_council(arg)
        elif cmd == "/analyze":
            self._cmd_analyze(arg)
        elif cmd == "/graph":
            self._cmd_graph(arg)
        elif cmd == "/coverage":
            self.print_info(self.agent.coverage.format_status(), C["cyan"])
        elif cmd == "/fuzz":
            self.print_info(self.agent.fuzzer.format_status(), C["yellow"])
        elif cmd == "/mutate":
            self._cmd_mutate(arg)
        elif cmd == "/about":
            from . import __version__
            self.print_info(f"{APP_NAME} v{__version__} — advanced terminal "
                            "AI agent", C["accent"])
            self.print_info("  python + prompt_toolkit + rich · "
                            "OpenCode Zen & TokenRouter providers", C["dim"])
        else:
            self.print_error(f"unknown command: {cmd} — try /help")

    # -- event-log commands ------------------------------------------------------

    def _cmd_goal(self, arg: str) -> None:
        parts = arg.split(None, 1)
        sub = parts[0].lower() if parts else "status"
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub in ("", "status", "show"):
            self.print_info(self.agent.goal.format(), C["cyan"])
        elif sub == "set":
            self._goal_set(rest)
        elif sub == "prove":
            if not rest:
                self.print_error("usage: /goal prove <clause-id>  "
                                 "(or /goal prove-all)")
                return
            ok, detail = self.agent.goal.prove_by_predicate(rest)
            self.print_info(f"{'✓' if ok else '✗'} {rest}: {detail}",
                            C["green"] if ok else C["red"])
            self.print_info(self.agent.goal.format(), C["dim"])
        elif sub == "prove-all":
            st = self.agent.goal.status()
            for c in st.clauses:
                if c.proof and not c.advisory:
                    ok, detail = self.agent.goal.prove_by_predicate(c.id)
                    self.print_info(f"{'✓' if ok else '✗'} {c.id}: {detail}",
                                    C["green"] if ok else C["red"])
            self.print_info(self.agent.goal.format(), C["dim"])
        elif sub == "close":
            result = self.agent.goal.close(fresh=True)
            color = C["green"] if result["state"] == "ACHIEVED" else C["yellow"]
            self.print_info(f"GOAL CLOSED: {result['state']}", color)
            for r in result["reasons"]:
                self.print_info(f"  - {r}", C["dim"])
            self.print_info(result["bundle"], C["cyan"])
        elif sub == "waive":
            wparts = rest.split("--reason", 1)
            cid = wparts[0].strip()
            reason = wparts[1].strip() if len(wparts) > 1 else "human waiver"
            if not cid:
                self.print_error("usage: /goal waive <clause-id> --reason '…'")
                return
            if self.agent.goal.waive(cid, reason):
                self.print_info(f"✓ clause {cid} waived (recorded as an event)",
                                C["yellow"])
            else:
                self.print_error(f"no such clause: {cid}")
        elif sub == "clear":
            self.agent.goal.clear()
            self.print_info("✓ goal cleared", C["dim"])
        else:
            self.print_error("goal subcommands: set · prove · prove-all · "
                             "close · status · waive · clear")

    def _goal_set(self, rest: str) -> None:
        """Parse the TUI goal grammar into a contract:

        /goal set <statement> | <clause> @ <proof-type>:<arg> | …
        Prefix a clause with ! for an anti-clause, ~ for an invariant.
        A clause piece may also be raw JSON for full control.

        Proof types: exit_code, file_exists, file_contains, file_matches,
        command_output_contains, ast_assert, diff_assert, file_unchanged,
        tool_delta.
        """
        import json as _json
        from .goal import GoalContractError
        if not rest:
            self.print_error(
                "usage: /goal set <statement> | <clause> @ <type>:<arg> | …\n"
                "  e.g. /goal set ship it | tests pass @ exit_code:pytest -q"
                " | docs exist @ file_exists:docs.md\n"
                "  ! prefix = anti-clause, ~ prefix = invariant")
            return
        pieces = [p.strip() for p in rest.split("|") if p.strip()]
        if not pieces:
            self.print_error("goal needs a statement: "
                             "/goal set <statement> | <clause> @ <type>:<arg>")
            return
        statement = pieces[0]
        clauses: list[dict] = []
        anti: list[dict] = []
        invariants: list[dict] = []
        for i, piece in enumerate(pieces[1:], 1):
            try:
                if piece.startswith("{"):
                    c = _json.loads(piece)
                    clauses.append(c)
                    continue
                is_anti = piece.startswith("!")
                is_inv = piece.startswith("~")
                text = piece.lstrip("!~ ").strip()
                if "@" not in text:
                    self.print_error(
                        f"clause {i} needs a machine-checkable proof: "
                        f"'{text}' @ <type>:<arg>  (or mark it advisory "
                        "with raw JSON)")
                    return
                ctext, proof_str = [x.strip() for x in text.split("@", 1)]
                proof = self._parse_proof(proof_str)
                if proof is None:
                    return
                if is_anti:
                    anti.append({"id": f"A{len(anti) + 1}", "text": ctext,
                                 "check": proof})
                elif is_inv:
                    invariants.append({"id": f"I{len(invariants) + 1}",
                                       "text": ctext, "check": proof})
                else:
                    clauses.append({"id": f"C{len(clauses) + 1}",
                                    "text": ctext, "weight": 1.0,
                                    "proof": proof})
            except _json.JSONDecodeError as e:
                self.print_error(f"clause {i}: invalid JSON — {e}")
                return
        if not clauses:
            self.print_error("a goal needs at least one clause with a proof")
            return
        try:
            self.agent.goal.set_goal(statement, clauses, anti, invariants)
        except GoalContractError as e:
            self.print_error(f"contract rejected: {e}")
            return
        self.print_info(f"✓ goal contract frozen — {len(clauses)} clause(s)"
                        + (f", {len(anti)} anti" if anti else "")
                        + (f", {len(invariants)} invariant" if invariants
                           else ""), C["green"])
        self.print_info(self.agent.goal.format(), C["dim"])

    def _parse_proof(self, s: str) -> dict | None:
        """'<type>:<arg>' -> predicate dict (None + error print on failure)."""
        if ":" not in s:
            self.print_error(f"proof needs '<type>:<arg>', got: {s!r}")
            return None
        ptype, arg = s.split(":", 1)
        ptype, arg = ptype.strip(), arg.strip()
        if ptype in ("exit_code", "tool_delta"):
            return {"type": ptype, "command": arg, "expect": 0}
        if ptype == "file_exists":
            return {"type": ptype, "path": arg}
        if ptype in ("file_contains", "file_matches",
                     "command_output_contains"):
            if ":" not in arg:
                self.print_error(f"{ptype} needs '<path-or-cmd>:<text>'")
                return None
            a, b = arg.split(":", 1)
            key = "path" if ptype.startswith("file") else "command"
            field2 = "text" if ptype != "file_matches" else "pattern"
            return {"type": ptype, key: a.strip(), field2: b.strip()}
        if ptype == "ast_assert":
            if ":" not in arg:
                self.print_error("ast_assert needs '<path>:<symbol>'")
                return None
            a, b = arg.split(":", 1)
            return {"type": ptype, "path": a.strip(), "symbol": b.strip()}
        if ptype == "diff_assert":
            return {"type": ptype, "path": arg, "forbid": []}
        if ptype == "file_unchanged":
            if ":" not in arg:
                self.print_error("file_unchanged needs '<path>:<sha256>'")
                return None
            a, b = arg.split(":", 1)
            return {"type": ptype, "path": a.strip(),
                    "baseline_hash": b.strip()}
        self.print_error(f"unknown proof type: {ptype!r}")
        return None

    def _cmd_autonomy(self, arg: str) -> None:
        from .agent import AUTONOMY_LEVELS
        if not arg:
            lines = [f"current: L{self.agent.autonomy} — "
                     f"{AUTONOMY_LEVELS[self.agent.autonomy]}"]
            for level, desc in AUTONOMY_LEVELS.items():
                lines.append(f"  L{level}  {desc}")
            self.print_info("\n".join(lines), C["cyan"])
            return
        try:
            level = int(arg)
        except ValueError:
            self.print_error("usage: /autonomy <0-5>")
            return
        desc = self.agent.set_autonomy(level)
        self.print_info(f"✓ autonomy → L{self.agent.autonomy} — {desc}",
                        C["green"])

    def _cmd_focus(self, arg: str) -> None:
        """FOCUS MODE (deep work): arms auto-continuation. After each turn
        the kernel decides — mechanically — whether work remains, and the
        UI submits the next continuation turn automatically, until the
        goal closes, the agent stalls, the budget pauses, or N turns land.
        """
        arg = arg.strip().lower()
        if arg in ("", "status"):
            state = (f"ARMED — {self._focus_remaining} auto-turn(s) left"
                     if self._focus_remaining > 0 else "off")
            self.print_info(f"🎯 focus mode: {state}\n"
                            "  /focus <1-20>  arm, then send your task\n"
                            "  /focus off     disarm", C["cyan"])
            return
        if arg == "off":
            self._focus_remaining = 0
            self.print_info("🎯 focus disarmed", C["yellow"])
            return
        try:
            n = max(1, min(int(arg), 20))
        except ValueError:
            self.print_error("usage: /focus <1-20> | /focus off")
            return
        self._focus_remaining = n
        self.agent._focus_history.clear()
        goal = self.agent.goal.status()
        hint = ("an active goal drives continuation — focus stops when "
                "every clause is proven or progress stalls"
                if goal.active else
                "no active goal — focus stops when the agent answers "
                "without further tool work")
        self.print_info(f"🎯 focus armed: up to {n} auto-turns. {hint}. "
                        f"Send your task now.", C["green"])

    def _cmd_render(self, arg: str) -> None:
        """Toggle rendered-markdown replies: streaming stays in the border
        preview, the finished reply prints as rich Markdown."""
        arg = arg.strip().lower()
        current = bool(self.cfg.extra.get("render_markdown", False))
        if arg in ("on", "off"):
            new_state = arg == "on"
        elif arg == "":
            new_state = not current
        else:
            self.print_error("usage: /render [on|off]")
            return
        self.cfg.extra["render_markdown"] = new_state
        self.cfg.save()
        self.print_info(f"✓ markdown rendering → "
                        f"{'ON (replies render as rich Markdown)' if new_state else 'OFF (raw streaming)'}",
                        C["green"])

    def _cmd_workflow(self, arg: str) -> None:
        """Saved enterprise pipelines: /workflow [list|run|delete]."""
        from .workflows import WorkflowError
        wf = self.agent.workflows
        parts = arg.split(None, 1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""
        if sub in ("", "list"):
            self.print_info(wf.format_list(), C["cyan"])
            return
        if sub == "run":
            if not rest:
                self.print_error("usage: /workflow run <name>")
                return
            self.print_info(f"⚙ running workflow {rest!r} …", C["dim"])
            try:
                report = wf.run(rest)
            except WorkflowError as e:
                self.print_error(str(e))
                return
            color = C["green"] if report["state"] == "DONE" else C["red"]
            self.print_info(wf.format_report(report), color)
            return
        if sub == "delete":
            if wf.delete(rest):
                self.print_info(f"✓ workflow {rest!r} deleted", C["yellow"])
            else:
                self.print_error(f"no such workflow: {rest!r}")
            return
        self.print_error("workflow subcommands: list · run <name> · "
                         "delete <name>")

    def _cmd_export(self, arg: str) -> None:
        fmt = arg.strip().lower()
        if fmt not in ("", "md", "markdown", "html"):
            self.print_error("usage: /export [md|html]")
            return
        fmt = "html" if fmt == "html" else "md"
        try:
            path = self.agent.export_report(fmt)
        except OSError as e:
            self.print_error(f"cannot write report: {e}")
            return
        self.print_info(f"✓ audit report exported → {path}", C["green"])

    def _cmd_health(self) -> None:
        h = self.agent.health
        lines = ["PROVIDER HEALTH"]
        lines.append(f"  failovers this session : {h['failovers']}")
        if h["model_errors"]:
            for m, n in sorted(h["model_errors"].items()):
                lines.append(f"  model errors           : {m} × {n}")
        else:
            lines.append("  model errors           : none")
        fb = str(self.cfg.extra.get("failover_model", "") or "")
        lines.append(f"  failover target        : "
                     f"{fb or 'auto (same provider first)'}")
        lines.append("  set explicit target    : edit config.json -> "
                     "\"failover_model\"")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_notify(self, arg: str) -> None:
        try:
            state = self.agent.notifier.configure(arg)
        except ValueError as e:
            self.print_error(str(e))
            return
        self.print_info(f"✓ notifications → {state}", C["green"])

    def _cmd_resume(self, arg: str) -> None:
        branch = arg.strip()
        catalog = self.agent.sessions_catalog()
        if not branch:
            if not catalog:
                self.print_info("no sessions found", C["dim"])
                return
            lines = ["SESSIONS — /resume <branch> to continue one:"]
            import time as _time
            for c in catalog[:12]:
                stamp = (_time.strftime("%m-%d %H:%M",
                                        _time.localtime(c["started"]))
                         if c["started"] else "?")
                lines.append(f"  ◆ {c['branch']:<14} session "
                             f"{c['session_id'] or '?'} · "
                             f"{c['events']} events · {stamp}")
            self.print_info("\n".join(lines), C["cyan"])
            return
        try:
            n = self.agent.resume_session(branch)
        except ValueError as e:
            self.print_error(str(e))
            return
        self.print_info(f"✓ resumed branch {branch!r} — {n} message(s) "
                        f"restored from the event log", C["green"])

    def _cmd_state(self) -> None:
        st = self.agent.state()
        goal = self.agent.goal.status()
        lines = [
            f"branch: {st.branch}   head seq: {st.head_seq}   "
            f"events: {len(self.agent.log)}",
            f"cost: {st.cost_summary()}",
            f"tool calls: {st.tool_calls}   errors: {st.tool_errors}   "
            f"commands: {st.commands_run}",
            f"autonomy: L{st.autonomy}",
        ]
        if st.files_touched:
            lines.append("files touched: " + ", ".join(sorted(st.files_touched)))
        if goal.active:
            proven = sum(1 for c in goal.clauses if c.state == "PROVEN")
            lines.append(f"goal: {goal.statement} — "
                         f"{proven}/{len(goal.clauses)} clauses "
                         f"({(1 - goal.distance) * 100:.0f}% done)")
        if st.dead_ends:
            lines.append(f"dead ends: {len(st.dead_ends)}")
        if st.verdicts:
            passed = sum(1 for v in st.verdicts if v.get("passed"))
            lines.append(f"judge verdicts: {passed}/{len(st.verdicts)} passed")
        if st.episodes:
            lines.append(f"memory episodes: {len(st.episodes)}")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_rewind(self, arg: str) -> None:
        if not arg:
            evs = self.agent.log.events()[-12:]
            if not evs:
                self.print_info("log is empty", C["dim"])
                return
            lines = ["recent events (pick a seq, then /rewind <seq>):"]
            for ev in evs:
                preview = ""
                if ev.type in ("user.message", "assistant.message"):
                    preview = str(ev.data.get("text", ""))[:50]
                elif ev.type in ("tool.call", "tool.result"):
                    preview = str(ev.data.get("name", ""))
                lines.append(f"  {ev.seq:>4}  {ev.type:<20} {preview}")
            self.print_info("\n".join(lines), C["dim"])
            return
        try:
            seq = int(arg)
        except ValueError:
            self.print_error("usage: /rewind <seq>  (see /rewind for seqs)")
            return
        new_head, kept = self.agent.rewind_to(seq)
        self.print_info(f"✓ rewound to seq {new_head} — {kept} message(s) kept "
                        "(tool-call context is dropped)", C["green"])

    def _cmd_fork(self, arg: str) -> None:
        branch = self.agent.fork_timeline(name=arg or None)
        self.print_info(f"✓ forked timeline → branch '{branch}' "
                        "(now continuing on it)", C["green"])

    def _cmd_revert(self, arg: str) -> None:
        """REVERT (§9.1): files only return to seq N — the agent KEEPS its
        memory. This is what feeds the Dead-End Ledger."""
        if not arg:
            self.print_error("usage: /revert <seq>  (see /rewind for seqs)")
            return
        try:
            seq = int(arg)
        except ValueError:
            self.print_error("usage: /revert <seq>")
            return
        result = self.agent.revert_files_to(seq)
        if "error" in result:
            self.print_error(result["error"])
            return
        self.print_info(f"✓ files reverted to seq {seq} — "
                        f"{result['restored']} restored, "
                        f"{result['removed']} removed "
                        "(agent memory kept)", C["green"])

    def _cmd_why(self, arg: str) -> None:
        """Appendix A `argus why`: walk the causation chain from an event
        back to the human instruction that caused it."""
        if not arg:
            self.print_error("usage: /why <seq>  (see /rewind for seqs)")
            return
        try:
            seq = int(arg)
        except ValueError:
            self.print_error("usage: /why <seq>")
            return
        evs = self.agent.log.events()
        target = next((e for e in evs if e.seq == seq), None)
        if target is None:
            self.print_error(f"no event at seq {seq}")
            return
        chain = self.agent.log.why(target.id)
        lines = [f"causal chain for seq {seq} ({target.type}):"]
        for i, ev in enumerate(chain):
            indent = "  " * i
            preview = ""
            if ev.type in ("user.message", "assistant.message"):
                preview = str(ev.data.get("text", ""))[:40]
            elif ev.type in ("tool.call", "tool.result"):
                preview = str(ev.data.get("name", ""))
            clause = f"  [clause {ev.correlation_id}]" \
                if ev.correlation_id else ""
            lines.append(f"{indent}← seq {ev.seq} {ev.type} "
                         f"({ev.actor}) {preview}{clause}")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_impact(self, arg: str) -> None:
        """§15.2 the killer query: blast radius of changing a symbol."""
        parts = arg.split(None, 1)
        if not parts:
            self.print_error("usage: /impact <symbol> [path]")
            return
        symbol = parts[0]
        path = parts[1].strip() if len(parts) > 1 else "."
        self.print_info(f"⠹ indexing {path}…", C["dim"])

        def run():
            self.agent.nexus.index(path)
            self.console.print(Text(self.agent.nexus.format_impact(symbol),
                                    style=C["fg"]))

        self._bg(run)

    def _cmd_forge(self, arg: str) -> None:
        sub = arg.strip().lower()
        if sub == "drift":
            delta = self.agent.forge.drift()
            if delta is None:
                self.print_info("✓ no environment drift detected", C["green"])
            else:
                self.print_info(f"⚠ environment drift: {delta['changed']}",
                                C["yellow"])
        else:
            d = self.agent.forge.probe()
            lines = [f"environment digest: {d['digest']}",
                     f"  os {d['os']} {d['arch']}   python {d['python']}",
                     f"  cwd {d['cwd']}",
                     f"  lockfile {d['lockfile_hash'] or 'none'}"]
            if d["tools"]:
                lines.append("  tools: " + ", ".join(
                    f"{k}={v.split()[0] if v else '?'}"
                    for k, v in d["tools"].items()))
            self.print_info("\n".join(lines), C["cyan"])

    def _cmd_budget(self, arg: str = "") -> None:
        gov = self.agent.budget_gov
        parts = arg.split()
        if parts:
            sub = parts[0].lower()
            try:
                if sub == "reset":
                    gov.reset()
                    self.print_info("budget spend reset — the governor "
                                    "counts from now", C["green"])
                    return
                if sub in ("steps", "usd", "tokens", "files") \
                        and len(parts) >= 2:
                    self.print_info(gov.set_limit(sub, parts[1]), C["green"])
                    return
                if sub in ("extend", "set") and len(parts) >= 3 \
                        and parts[1].lower() in ("steps", "usd", "tokens",
                                                 "files"):
                    self.print_info(
                        gov.set_limit(parts[1].lower(), parts[2]), C["green"])
                    return
            except ValueError as e:
                self.print_error(f"bad value: {e}")
                return
            self.print_error("usage: /budget [reset | steps N | usd X | "
                             "tokens N | files N]")
            return
        s = gov.spend()
        b = gov.budget
        ok, reason = gov.check()

        def _fmt(v) -> str:
            return "∞" if v == float("inf") else str(v)

        lines = [f"budget {'OK' if ok else 'BREACHED'}",
                 f"  usd    ${s['usd']:.4f} / ${_fmt(b.max_usd)}",
                 f"  steps  {s['steps']} / {_fmt(b.max_steps)}",
                 f"  tokens {s['tokens']} / {_fmt(b.max_tokens)}",
                 f"  files  {s['files']} / {_fmt(b.max_files)}",
                 "  spend is UNLIMITED by default — set a cap with "
                 "/budget steps N · /budget usd X · /budget reset"]
        if not ok:
            lines.append(f"  ⚠ {reason}")
        self.print_info("\n".join(lines),
                        C["green"] if ok else C["red"])

    def _cmd_constitution(self, arg: str) -> None:
        sub = arg.strip().lower()
        path = self.agent.oracle.constitution_path()
        if sub == "edit":
            if path:
                self.print_info(f"constitution file: {path}", C["dim"])
                self.print_info("edit it directly — it is human-owned and "
                                "never auto-modified. It is always present "
                                "in the agent's context.", C["cyan"])
            else:
                self.print_error("no memory dir configured")
        else:
            text = self.agent.oracle.read_constitution()
            if text.strip():
                self.print_info("CONSTITUTION (standing rules):\n" + text,
                                C["cyan"])
            else:
                self.print_info(f"constitution is empty — create {path} "
                                "with your standing rules", C["dim"])

    def _cmd_replay(self) -> None:
        """Replay the session log as a text film (§26)."""
        from .kernel import replay
        lines = ["REPLAY — the session as a film:"]
        for ev in replay(self.agent.log):
            t = ev.type
            d = ev.data
            if t == "user.message":
                lines.append(f"  [{ev.seq}] ❯ {str(d.get('text', ''))[:60]}")
            elif t == "assistant.message":
                lines.append(f"  [{ev.seq}] ◆ {str(d.get('text', ''))[:60]}")
            elif t == "tool.call":
                lines.append(f"  [{ev.seq}] ⚙ {d.get('name', '')}")
            elif t == "tool.result":
                icon = "✓" if d.get("status") == "done" else "✗"
                lines.append(f"  [{ev.seq}] {icon} {d.get('name', '')} "
                             f"({d.get('status', '')})")
            elif t == "snapshot.taken":
                lines.append(f"  [{ev.seq}] 📸 snapshot {str(d.get('tree', ''))[:10]}")
            elif t == "clause.proven":
                lines.append(f"  [{ev.seq}] ★ clause {d.get('clause', '')} PROVEN")
            elif t == "goal.closed":
                lines.append(f"  [{ev.seq}] ■ GOAL {d.get('state', '')}")
            elif t == "cost.incurred":
                lines.append(f"  [{ev.seq}] $ cost "
                             f"{d.get('tokens_in', 0)}→{d.get('tokens_out', 0)} tok")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_memory(self) -> None:
        block = self.agent.memory.context_block(max_episodes=5)
        self.print_info(block, C["cyan"])

    def _cmd_judge(self, arg: str) -> None:
        import json as _json
        predicate: dict | None = None
        arg = arg.strip()
        if arg.startswith("{"):
            try:
                predicate = _json.loads(arg)
            except ValueError as e:
                self.print_error(f"invalid JSON predicate: {e}")
                return
        elif arg:
            # shorthand: /judge <type> <value>  → predicate with default key
            tokens = arg.split(None, 1)
            ptype = tokens[0]
            value = tokens[1] if len(tokens) > 1 else ""
            key = {"exit_code": "command",
                   "file_exists": "path",
                   "file_contains": "path",
                   "file_matches": "path",
                   "command_output_contains": "command"}.get(ptype)
            if key is None:
                self.print_error("predicate types: exit_code · file_exists · "
                                 "file_contains · file_matches · "
                                 "command_output_contains")
                return
            predicate = {"type": ptype, key: value}
        else:
            self.print_error("usage: /judge <type> <arg>  or  "
                             '/judge {"type": "file_exists", "path": "…"}')
            return

        def run():
            verdict = self.agent.judge.check(predicate)
            icon = "✓" if verdict.passed else "✗"
            color = C["green"] if verdict.passed else C["red"]
            self.print_info(f"{icon} [{verdict.kind}] {verdict.detail}", color)
            if verdict.evidence:
                self.print_info(f"  evidence: {verdict.evidence[:200]}",
                                C["dim"])

        self._bg(run)

    # -- v5 advanced subsystem commands ----------------------------------

    def _cmd_compile(self, arg: str) -> None:
        """Intent Compiler: goal → optimized ordered waves → execute."""
        goal = arg.strip()
        if not goal:
            self.print_error("usage: /compile <goal>  — the compiler "
                             "plans + executes it in ordered waves")
            return
        self.print_info("⚙ compiling…", C["pink"])

        def run():
            plan = self.agent.compiler.compile(goal)
            self.console.print(Text(self.agent.compiler.format(plan),
                                    style=C["fg"]))
            if not plan.waves:
                return
            self.print_info(f"⚙ executing {len(plan.waves)} wave(s)…",
                            C["pink"])
            result = self.agent.compiler.execute(plan)
            self.print_info(
                f"✓ {result['items']} items · {result['done']} done · "
                f"{result['blocked']} blocked · {result['error']} error",
                C["green"] if result["error"] == 0 else C["yellow"])

        self._bg(run)

    def _cmd_evolve(self, arg: str) -> None:
        """Evolution Engine: mutate → evaluate → deploy one role brief."""
        parts = arg.split()
        if parts and parts[0].lower() == "rollback":
            if len(parts) < 2:
                self.print_error("usage: /evolve rollback <role>")
                return
            self.print_info(self.agent.evolution.rollback(parts[1]),
                            C["cyan"])
            return
        role = parts[0] if parts else ""

        def run():
            self.print_info("🧬 evolving — benchmark runs are real "
                            "worker calls…", C["pink"])
            gen = self.agent.evolution.evolve(role or None)
            color = C["green"] if gen.deployed else C["yellow"]
            self.print_info(self.agent.evolution.format(gen), color)

        self._bg(run)

    def _cmd_brain(self, arg: str) -> None:
        """Cognitive memory: query it, put it to sleep, or read stats."""
        sub = arg.strip().lower()
        if sub == "sleep":
            stats = self.agent.brain.sleep()
            self.print_info(f"🧠 slept — merged {stats['merged']} · "
                            f"distilled {stats['distilled']} · promoted "
                            f"{stats['promoted']} · forgotten "
                            f"{stats['forgotten']}", C["pink"])
            return
        if sub in ("", "stats"):
            self.print_info(self.agent.brain.format_stats(), C["cyan"])
            return
        self.print_info(self.agent.brain.context_block(arg, k=5)
                        or "no live memories match that query", C["fg"])

    def _cmd_merge(self, arg: str) -> None:
        """Semantic timeline merge of two branches."""
        parts = arg.split()
        if len(parts) < 2:
            known = ", ".join(self.agent.log.branches())
            self.print_error(f"usage: /merge <branchA> <branchB>  "
                             f"(known: {known})")
            return
        try:
            result = self.agent.merger.merge(parts[0], parts[1])
        except ValueError as e:
            self.print_error(str(e))
            return
        self.print_info(self.agent.merger.format(result),
                        C["yellow"] if result.conflicts else C["green"])

    def _cmd_theater(self, arg: str) -> None:
        """Time-travel debugger: frames, whys, diffs, counterfactuals."""
        parts = arg.split()
        th = self.agent.theater
        if not parts:
            frames = th.frames()[-30:]
            lines = ["THEATER — last frames (scrub with /theater <seq>):"]
            for f in frames:
                lines.append(f"  seq {f['seq']:>4} {f['type']:<18} "
                             f"{f['summary'][:60]}")
            self.print_info("\n".join(lines), C["cyan"])
            return
        if parts[0].lower() == "why" and len(parts) > 1:
            try:
                self.print_info(th.why(int(parts[1])), C["cyan"])
            except ValueError:
                self.print_error("seq must be an integer")
            return
        if parts[0].lower() in ("cf", "counterfactual") and len(parts) > 1:
            try:
                report = th.counterfactual(int(parts[1]))
            except ValueError as e:
                self.print_error(str(e))
                return
            self.print_info(th.format_cf(report), C["pink"])
            return
        if parts[0].lower() == "diff" and len(parts) > 2:
            try:
                self.print_info(th.diff(int(parts[1]), int(parts[2])),
                                C["cyan"])
            except ValueError:
                self.print_error("seqs must be integers")
            return
        try:
            frame = th.frame(int(parts[0]))
        except ValueError:
            self.print_error("usage: /theater [seq | why N | diff A B | "
                             "cf N]")
            return
        if frame is None:
            self.print_error("no event at that seq")
            return
        self.print_info(frame.format(), C["cyan"])

    def _cmd_debate(self, arg: str) -> None:
        """Multi-model debate tournament."""
        parts = arg.split(maxsplit=1)
        if parts and parts[0].lower() in ("confirm", "refute"):
            if len(parts) < 2:
                self.print_error("usage: /debate confirm|refute <model-id>")
                return
            fn = (self.agent.debate.confirm if parts[0].lower() == "confirm"
                  else self.agent.debate.refute)
            trust = fn(parts[1].strip())
            self.print_info("calibration: " + ", ".join(
                f"{m}={t:.2f}" for m, t in sorted(trust.items())), C["cyan"])
            return
        question = arg.strip()
        if not question:
            self.print_error("usage: /debate <question>")
            return

        def run():
            self.print_info(f"⚔ tournament — "
                            f"{', '.join(self.agent.debate.models)}",
                            C["pink"])
            result = self.agent.debate.run(question)
            self.console.print(Text(self.agent.debate.format(result),
                                    style=C["fg"]))

        self._bg(run)

    def _cmd_market(self, arg: str) -> None:
        """Task market: auctions the given tasks to bidding specialists."""
        tasks = [t.strip() for t in arg.split("|") if t.strip()]
        if not tasks:
            self.print_error("usage: /market <task1> | <task2> | …")
            return

        def run():
            self.print_info(f"💰 {len(tasks)} contract(s) up for "
                            "auction…", C["pink"])
            contracts = self.agent.market.run(tasks)
            self.console.print(Text(self.agent.market.format(contracts),
                                    style=C["fg"]))

        self._bg(run)

    def _cmd_tower(self, arg: str) -> None:
        """Web Control Tower: mission-control dashboard in the browser."""
        try:
            port = int(arg.strip()) if arg.strip() else 7860
        except ValueError:
            port = 7860
        if self.agent.tower.server is not None:
            self.print_info(f"control tower already live at "
                            f"{self.agent.tower.url}", C["cyan"])
            return
        url = self.agent.tower.start(port=port)
        self.print_info(f"🖥 control tower LIVE at {url} — event river, "
                        "timeline scrubber, crew status, live command "
                        "box", C["green"])

    # -- v6 frontier commands ------------------------------------------------

    def _cmd_verify(self, arg: str) -> None:
        """Formal verification: compile a plan and model-check it, or
        audit the real history."""
        ag = self.agent
        if arg.strip().lower() == "log":
            r = ag.formal.audit_log()
            self.print_info(("HISTORY AUDIT — CLEAN ✓" if r.ok else
                             "HISTORY AUDIT — VIOLATIONS FOUND ✗")
                            + ("\n".join("" if not r.violations else
                                         "\n" + "\n".join(
                                             f"  ⚠ {v['property']}: "
                                             f"{v['why']}"
                                             for v in r.violations))),
                            C["green"] if r.ok else C["red"])
            return
        goal = arg.strip()
        if not goal:
            self.print_error("usage: /verify <goal> | /verify log")
            return

        def run():
            plan = ag.compiler.compile(goal)
            r = ag.formal.verify_plan(plan.waves)
            color = C["green"] if r.ok else C["red"]
            body = "\n".join([f"FORMAL — {'PASS' if r.ok else 'REJECTED'}"
                              f" ({r.checked} traces)"]
                             + [f"  ⚠ {v['property']}: {v['why']}"
                                for v in r.violations])
            self.print_info(body, color)

        self._bg(run)

    def _cmd_mcts(self, arg: str) -> None:
        goal = arg.strip()
        if not goal:
            self.print_error("usage: /mcts <item1>; <item2>; …")
            return
        items = [s.strip() for s in goal.split(";") if s.strip()]
        self.agent._mcts_items = items
        report = self.agent.mcts.search(
            items, ["coder", "architect", "debugger", "tester",
                    "documenter"], iterations=120, deadline_s=20.0)
        lines = [f"🌳 MCTS — score {report.best_score:.2f} · "
                 f"{report.iterations} iterations · {report.nodes} "
                 f"nodes"]
        for i, item in enumerate(items):
            lines.append(f"  [{report.best_assignment.get(i, '?')}] "
                         f"{item[:70]}")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_causal(self, arg: str) -> None:
        sub = arg.strip().lower()
        if sub.startswith("do "):
            toks = arg.strip().split()
            # trailing "on"/"off" token is the switch — never a substring
            # match (a feature literally named "soft_off" must work)
            if toks[-1] in ("on", "off") and len(toks) > 2:
                enable = toks[-1] == "on"
                cause = " ".join(toks[1:-1])
            else:
                enable = True
                cause = " ".join(toks[1:])
            report = self.agent.causal.do(cause, enable)
            verdict = ("trustworthy" if report["trustworthy"]
                       else "NOT ENOUGH DATA")
            self.print_info(
                f"do({cause}) → outcome change "
                f"{report['estimated_outcome_change']:+.3f} "
                f"({verdict})", C["cyan"])
            return
        edges = self.agent.causal.discover()
        self.print_info(self.agent.causal.format(edges), C["cyan"])

    def _cmd_bandit(self, arg: str) -> None:
        if not arg.strip():
            self.print_info(self.agent.bandit.format(), C["cyan"])
            return
        rec = self.agent.bandit.recommend(arg)
        self.print_info(f"bandit recommends [{rec.arm}] for this "
                        f"{rec.context} task", C["green"])

    def _cmd_mesh(self, arg: str) -> None:
        ag = self.agent
        parts = arg.split()
        if not parts or parts[0].lower() == "serve":
            port_arg = parts[1] if len(parts) > 1 else "0"
            try:
                want = int(port_arg)
            except ValueError:
                self.print_error(f"bad port: {port_arg!r} — usage: /mesh "
                                 f"serve [port]")
                return
            port = ag.mesh.serve(want)
            self.print_info(f"📡 mesh node '{ag.mesh.node_id}' serving "
                            f"on port {port}", C["green"])
            return
        if parts[0].lower() == "discover" and len(parts) > 1:
            host, _, port = parts[1].partition(":")
            try:
                port_num = int(port or 7861)
            except ValueError:
                self.print_error(f"bad port: {port!r} — usage: /mesh "
                                 f"discover host:port")
                return
            peer = ag.mesh.discover(host, port_num)
            if peer:
                self.print_info(f"discovered peer {peer.capabilities}",
                                C["green"])
            else:
                self.print_error(f"no FullAgent mesh at {parts[1]}")
            return
        if parts[0].lower() == "delegate" and len(parts) > 1:
            reply = ag.mesh.delegate(arg.split(None, 1)[1])
            self.print_info(str(reply), C["cyan"])
            return
        if parts[0].lower() == "status":
            self.print_info(f"peers: {list(ag.mesh.peers) or 'none'} · "
                            f"handled {ag.mesh.handled} task(s)",
                            C["cyan"])
            return
        self.print_error("usage: /mesh [serve [port] | discover "
                         "host:port | delegate <task> | status]")

    def _cmd_roleforge(self, arg: str) -> None:
        mission = arg.strip()
        if not mission:
            self.print_error("usage: /roleforge <what specialist do you "
                             "need and why>")
            return

        def run():
            status, msg = self.agent.roleforge.forge(mission)
            self.print_info(msg, C["green"] if status == "sealed"
                            else C["yellow"])

        self._bg(run)

    def _cmd_synth(self, arg: str) -> None:
        """Synthesize a tool from a JSON spec: name, description,
        examples ([{"args": {...}, "want": ...}])."""
        import json as _json
        try:
            spec = _json.loads(arg.strip())
        except ValueError:
            self.print_error('usage: /synth {"name": "f", "description":'
                             ' "...", "examples": [{"args": {"x": 1}, '
                             '"want": 2}]}')
            return
        from fullagent.synth import SynthSpec
        result = self.agent.synth.synthesize(SynthSpec(
            name=str(spec.get("name", "")),
            description=str(spec.get("description", "")),
            examples=spec.get("examples", [])))
        self.print_info(("✓ " if result.ok else "ERROR: ") +
                        result.reason,
                        C["green"] if result.ok else C["red"])

    def _cmd_ci(self, arg: str) -> None:
        sub = arg.strip().lower()
        ci = self.agent.ci
        if sub == "start":
            ci.start()
            self.print_info("🌊 CI pilot watching "
                            f"{ci.root} (every {ci.poll:.0f}s)",
                            C["green"])
        elif sub == "stop":
            ci.stop()
            self.print_info("CI pilot stopped", C["yellow"])
        else:
            self.print_info(ci.status(), C["cyan"])

    def _cmd_tune(self, arg: str) -> None:
        try:
            n = max(4, min(int(arg.strip() or 12), 40))
        except ValueError:
            n = 12
        tuner = self.agent.tuner

        def score(cfg: dict) -> float:
            self.agent.cfg.effort = cfg["effort"]
            return {"low": 0.45, "medium": 0.7, "high": 0.85}.get(
                cfg["effort"], 0.5) * (
                {"low": 0.9, "medium": 1.0, "high": 0.8}.get(
                    cfg["worker_steps"], 0.9))

        tuner.objective = score
        report = tuner.run(n=n)
        best = tuner.best()
        if best:
            self.agent.cfg.effort = best.config["effort"]
        self.print_info(f"🎛 tuned {report.trials} trials — best "
                        f"{best.score:.2f} with {best.config} (applied)",
                        C["green"])

    def _cmd_dual(self, arg: str) -> None:
        if not arg.strip() or arg.strip().lower() == "stats":
            self.print_info(self.agent.dual.format_stats(), C["cyan"])
            return

        def run():
            r = self.agent.dual.ask(arg.strip())
            self.print_info(f"[system {r.system} · conf {r.confidence:.2f}"
                            f" · {r.elapsed_ms}ms]\n{r.answer}",
                            C["cyan"])

        self._bg(run)

    def _cmd_predict_impact(self, arg: str) -> None:
        path = arg.strip()
        if not path:
            self.print_error("usage: /predict <file path>")
            return
        impact = self.agent.world.predict_impact(path)
        self.print_info(impact.format(), C["cyan"])

    def _cmd_race(self, arg: str) -> None:
        task = arg.strip()
        if not task:
            self.print_error("usage: /race <task>")
            return

        def run():
            result = self.agent.racer.race(task, timeout=420.0)
            self.print_info(self.agent.racer.format(result),
                            C["green"] if result.winner else C["yellow"])

        self._bg(run)

    def _cmd_fabric(self, arg: str) -> None:
        """Bitemporal knowledge graph: ask, assert, or see history."""
        parts = arg.split()
        fab = self.agent.fabric
        if not parts:
            self.print_info("usage: /fabric ask <s> <p> | assert <s> "
                            "<p> <o> | history <s> <p>", C["dim"])
            return
        sub = parts[0].lower()
        if sub == "ask" and len(parts) >= 3:
            answer = fab.ask(parts[1], parts[2])
            self.print_info(f"{parts[1]} {parts[2]} = "
                            + (answer or "unknown"), C["cyan"])
        elif sub == "assert" and len(parts) >= 4:
            fab.assert_fact(parts[1], parts[2], " ".join(parts[3:]))
            self.print_info("asserted", C["green"])
        elif sub == "history" and len(parts) >= 3:
            self.print_info(fab.history(parts[1], parts[2]), C["cyan"])
        else:
            self.print_error("usage: /fabric ask|assert|history …")

    def _cmd_auto(self, arg: str) -> None:
        """Toggle or inspect the AutoPilot self-routing brain."""
        sub = arg.strip().lower()
        ap = self.agent.autopilot
        if sub == "on":
            ap.enabled = True
            self.print_info("✓ autopilot ON — the agent will auto-enable "
                            "goal mode / real-time web as "
                            "each turn needs them", C["green"])
        elif sub == "off":
            ap.enabled = False
            self.print_info("autopilot OFF — manual control only", C["dim"])
        elif sub in ("", "status"):
            state = "ON" if ap.enabled else "OFF"
            self.print_info(f"autopilot: {state}\n"
                            "  auto-enables, per turn:\n"
                            "  ⚡ goal mode      — verifiable mission "
                            "detected (auto-drafted contract)\n"
                            "  ⚡ real-time web  — live-data question "
                            "detected\n"
                            "  toggle: /auto on · /auto off",
                            C["cyan"])
        else:
            self.print_error("usage: /auto [on|off|status]")

    def _cmd_prompt(self, arg: str) -> None:
        """Select which system prompt the model gets (systemprompt.py is
        the single source). 'main' is the compact prompt; 'master' is the
        extended 130k+ specification prompt."""
        from . import systemprompt
        sub = arg.strip().lower()
        if sub in ("", "list", "status"):
            current = self.cfg.prompt
            lines = [f"system prompt: {current}  (source: systemprompt.py)"]
            for name in systemprompt.names():
                mark = "●" if name == current else "○"
                size = len(systemprompt.get(name))
                lines.append(f"  {mark} {name:<8} {size:>8,} chars")
            lines.append("switch: /prompt main · /prompt master")
            self.print_info("\n".join(lines), C["cyan"])
            return
        if sub not in systemprompt.PROMPTS:
            self.print_error(f"unknown prompt {sub!r} — available: "
                             + ", ".join(systemprompt.names()))
            return
        self.cfg.prompt = sub
        self.cfg.save()
        # re-seat the live conversation's system prompt through the gate
        self.agent._reseat_system_prompt()
        size = len(self.agent._base_prompt())
        self.print_info(f"✓ system prompt → {sub} ({size:,} chars) — "
                        "applies from the next model call", C["green"])

    # -- v3 advanced subsystem commands ----------------------------------------

    def _cmd_recall(self, arg: str) -> None:
        """Semantic (meaning-based) recall over the episodic corpus."""
        query = arg.strip()
        if not query:
            s = self.agent.semantic.stats()
            self.print_info(f"semantic memory: {s['items']} items indexed "
                            f"{s['kinds']} — usage: /recall <question>",
                            C["cyan"])
            return
        hits = self.agent.semantic.recall(query, k=5)
        if not hits:
            self.print_info(f"no memories similar to: {query}", C["dim"])
            return
        lines = [f"SEMANTIC RECALL — {query}"]
        for h in hits:
            lines.append(f"  [{h['kind']} {h['similarity']:.2f}] {h['text']}")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_mission(self, arg: str) -> None:
        """Daemon mission control: /mission start <stmt> | task1 | task2,
        /mission tick <id>, /mission list, /mission abandon <id>."""
        parts = arg.split(None, 1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""
        d = self.agent.daemon
        if sub in ("", "list", "status"):
            self.print_info(d.format_status(), C["cyan"])
            return
        if sub == "start":
            if not rest:
                self.print_error("usage: /mission start <statement> | "
                                 "task1 | task2 | …")
                return
            segs = [s.strip() for s in rest.split("|") if s.strip()]
            if not segs:
                self.print_error("usage: /mission start <statement> | "
                                 "task1 | task2 | …")
                return
            statement = segs[0]
            tasks = segs[1:] if len(segs) > 1 else [statement]
            m = d.start(statement, tasks)
            self.print_info(f"✓ mission {m.mission_id} started — "
                            f"{len(m.steps)} step(s). Advance with "
                            f"/mission tick {m.mission_id}", C["green"])
            return
        if sub == "tick":
            mid = rest.strip()
            if not mid:
                self.print_error("usage: /mission tick <mission_id>")
                return
            r = d.tick(mid)
            if r.get("error"):
                self.print_error(f"{r.get('state', '')} "
                                 f"{r['error']}".strip())
                return
            self.print_info(f"mission {mid}: step {r.get('step', '?')} → "
                            f"{r.get('state', '?')}  "
                            f"progress {r.get('progress', 0):.0%}\n"
                            f"  {str(r.get('result', ''))[:200]}", C["cyan"])
            return
        if sub == "abandon":
            mid = rest.strip()
            if d.abandon(mid, "abandoned by user"):
                self.print_info(f"✓ mission {mid} abandoned", C["yellow"])
            else:
                self.print_error(f"cannot abandon mission {mid!r}")
            return
        self.print_error("usage: /mission [start|tick|list|abandon] …")

    def _cmd_crew(self, arg: str) -> None:
        """Persistent Codex-style subagents:
        /crew                             roster + states
        /crew spawn <role> <task>         launch a background subagent
        /crew send <id> <message>         follow-up into its context
        /crew wait [id,…]                 collect results (blocking)
        /crew close <id> · /crew resume <id>"""
        from .crew import CrewError
        crew = self.agent.crew
        parts = arg.split(None, 1)
        sub = parts[0].lower() if parts else "status"
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub in ("", "status", "list"):
            self.print_info(crew.format_status(), C["cyan"])
            return
        if sub == "spawn":
            rparts = rest.split(None, 1)
            if len(rparts) < 2:
                self.print_error("usage: /crew spawn <role> <task>  "
                                 "(roles: coder researcher tester "
                                 "reviewer analyst)")
                return
            role, task = rparts[0], rparts[1]
            try:
                agent = crew.spawn(task, role=role,
                                   context=self.agent.scout_context(),
                                   read_only=self.agent.autonomy <= 1)
            except CrewError as e:
                self.print_error(str(e))
                return
            self.print_info(f"⚡ subagent [{agent.id}] '{agent.nickname}' "
                            f"({agent.role}) launched in background — "
                            f"/crew wait collects it", C["green"])
            return
        if sub == "send":
            sparts = rest.split(None, 1)
            if len(sparts) < 2:
                self.print_error("usage: /crew send <id> <message>")
                return
            try:
                agent = crew.send(sparts[0], sparts[1])
            except CrewError as e:
                self.print_error(str(e))
                return
            self.print_info(f"✓ message → [{agent.id}] '{agent.nickname}' "
                            f"(state: {agent.state})", C["green"])
            return
        if sub == "wait":
            ids = [s.strip() for s in rest.split(",") if s.strip()] or None
            self.print_info("⏳ waiting for subagents…", C["dim"])
            try:
                states = crew.wait(ids, timeout=300.0)
            except CrewError as e:
                self.print_error(str(e))
                return
            self.print_info(f"states: {states}", C["cyan"])
            self.console.print(self._crew_panel(
                [crew.get(i) for i in states if crew.get(i)]))
            return
        if sub == "close":
            try:
                agent = crew.close(rest)
            except CrewError as e:
                self.print_error(str(e))
                return
            self.print_info(f"✓ [{agent.id}] '{agent.nickname}' closed",
                            C["yellow"])
            return
        if sub == "resume":
            try:
                agent = crew.resume(rest)
            except CrewError as e:
                self.print_error(str(e))
                return
            self.print_info(f"✓ [{agent.id}] '{agent.nickname}' resumed "
                            f"({agent.state})", C["green"])
            return
        self.print_error("crew subcommands: spawn · send · wait · close · "
                         "resume · status")

    def _crew_panel(self, agents) -> Panel:
        """A rich panel rendering the crew's reports — role icons, status
        glyphs, files touched, summaries."""
        body = Text()
        icons = {"done": ("✓", C["green"]), "blocked": ("◐", C["yellow"]),
                 "error": ("✗", C["red"]), "closed": ("⊘", C["dim"]),
                 "running": ("…", C["cyan"])}
        role_icons = {"researcher": "🔎", "coder": "👨‍💻", "tester": "🧪",
                      "reviewer": "🧐", "analyst": "📊"}
        for a in agents:
            glyph, color = icons.get(a.state, ("?", C["dim"]))
            body.append(f"{role_icons.get(a.role, '◆')} ", style=color)
            body.append(f"[{a.id}] {a.nickname}", style=f"bold {C['fg']}")
            body.append(f" ({a.role}) ", style=C["dim"])
            body.append(f"{glyph} {a.state}", style=f"bold {color}")
            body.append(f"  ·  {a.tool_calls} tools · "
                        f"{a.elapsed_ms / 1000:.1f}s\n", style=C["dim"])
            body.append(f"  task: {a.task[:160]}\n", style=C["dim"])
            if a.files_touched:
                body.append("  files: ", style=C["dim"])
                body.append(", ".join(a.files_touched[:8]) + "\n",
                            style=C["cyan"])
            if a.error:
                body.append(f"  error: {a.error[:200]}\n", style=C["red"])
            if a.summary:
                body.append("  " + a.summary.replace("\n", "\n  ")[:900]
                            + "\n", style=C["fg"])
            body.append("\n")
        return Panel(body, title=f"⚡ CREW — {len(agents)} subagent(s)",
                     border_style=C["accent"], expand=False,
                     padding=(0, 1))

    def _cmd_council(self, arg: str) -> None:
        """Convene an adversarial debate: /council <proposition>."""
        question = arg.strip()
        if not question:
            self.print_info(self.agent.council.format_status(), C["pink"])
            return
        self.print_info(f"⚖ convening council on: {question} …", C["dim"])
        v = self.agent.council.convene(question)
        if not v.ok:
            self.print_error(f"council failed: {v.error}")
            return
        lines = [f"COUNCIL VERDICT — winner: {v.winner.upper()}  "
                 f"(confidence {v.confidence:.0f}%)",
                 f"  reason: {v.reason}"]
        for role, text in v.positions.items():
            lines.append(f"  [{role}] {text[:200]}")
        self.print_info("\n".join(lines), C["pink"])

    # -- v4 professional subsystem commands ------------------------------------

    def _cmd_analyze(self, arg: str) -> None:
        """Static analysis: /analyze <path> — taint, complexity, cycles."""
        path = arg.strip() or "."
        p = Path(path).expanduser()
        if p.is_file():
            result = self.agent.static.analyze_file(str(p))
        else:
            result = self.agent.static.analyze_tree(str(p))
        self.print_info(self.agent.static.format_report(result), C["cyan"])

    def _cmd_graph(self, arg: str) -> None:
        """Knowledge graph: /graph [index <path>|query <name>|impact <name>]."""
        parts = arg.split(None, 1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        kg = self.agent.kgraph
        if sub in ("", "status"):
            self.print_info(kg.format_status(), C["cyan"])
            return
        if sub == "index":
            root = Path(rest or ".").expanduser()
            files = ([root] if root.is_file()
                     else sorted(root.glob("**/*.py"))[:200])
            sources = {}
            for f in files:
                if f.is_file():
                    try:
                        sources[f.stem] = f.read_text(errors="replace")
                    except OSError:
                        continue
            n = kg.index_code(sources)
            kg.index_log()
            self.print_info(f"✓ indexed {len(sources)} module(s) → "
                            f"{n} entities\n{kg.format_status()}", C["green"])
            return
        if sub == "query":
            if not rest:
                self.print_error("usage: /graph query <name>")
                return
            hits = kg.find(rest)
            if not hits:
                self.print_info(f"no entity matching {rest!r} — "
                                f"/graph index first", C["dim"])
                return
            lines = []
            for e in hits[:20]:
                lines.append(f"{e.kind} {e.id}  ({e.name})")
                for r in kg.out_edges(e.id)[:8]:
                    lines.append(f"    --{r.rel}--> {r.dst}")
                for r in kg.in_edges(e.id)[:8]:
                    lines.append(f"    <--{r.rel}-- {r.src}")
            self.print_info("\n".join(lines), C["cyan"])
            return
        if sub == "impact":
            if not rest:
                self.print_error("usage: /graph impact <name>")
                return
            hits = kg.find(rest)
            if not hits:
                self.print_info(f"no entity matching {rest!r} — "
                                f"/graph index first", C["dim"])
                return
            lines = []
            for e in hits[:5]:
                dep = kg.impact(e.id)
                lines.append(f"{e.id}: {len(dep)} dependent(s)")
                lines.extend(f"    {d}" for d in dep[:20])
            self.print_info("\n".join(lines), C["yellow"])
            return
        self.print_error("usage: /graph [index <path>|query <name>|"
                         "impact <name>]")

    def _cmd_mutate(self, arg: str) -> None:
        """Mutation testing: /mutate <file> <suite-command>. Runs the suite
        against AST-generated mutants; survivors are holes in the tests."""
        parts = arg.split(None, 1)
        if len(parts) < 2:
            if self.agent.mutator is not None:
                self.print_info(self.agent.mutator.format_status(),
                                C["yellow"])
            else:
                self.print_error("usage: /mutate <file> <suite-command>  "
                                 "(e.g. /mutate src/x.py 'python -m pytest "
                                 "-q tests/')")
            return
        from .mutate import MutationTester
        path, suite = parts[0], parts[1]
        if not Path(path).expanduser().is_file():
            self.print_error(f"not a file: {path}")
            return
        self.print_info(f"⚙ mutating {path} — suite: {suite} …", C["dim"])
        tester = MutationTester(self.agent.log, suite)
        report = tester.run(path)
        self.agent.mutator = tester
        lines = [f"MUTATION — {report.path}",
                 f"  score {report.score:.0%}   killed {report.killed}   "
                 f"survived {report.survived}   errors {report.errors}   "
                 f"total {report.total}"]
        for r in report.results:
            if r.status == "survived":
                lines.append(f"    ⚠ SURVIVED [{r.kind}] {r.description}")
        self.print_info("\n".join(lines), C["yellow"])

    # -- turn execution (worker thread) ---------------------------------------------------------

    def _run_turn_thread(self, text: str) -> None:
        self._busy = True
        self._cancel_flag.clear()
        self._set_status("thinking…")
        self._start_spinner()
        self._last_preview_ts = 0.0
        self._preview_pending = False
        streamed = {"n": 0}
        # SPEED: single tail string — appending a token is O(1), never a
        # re-join of the whole stream (that was quadratic on long replies).
        # "blank" tracks blank-line runs so the model's "\n\n\n" spam
        # between tool calls collapses to a single blank line.
        stream_tail = {"t": "", "blank": False}
        render_md = bool(self.cfg.extra.get("render_markdown", False))
        md_buf: list[str] = []
        md_len = {"n": 0}

        def on_token(piece: str):
            streamed["n"] += len(piece)
            now = time.time()
            if now - self._last_preview_ts < 0.03:
                # SPEED: coalesce border redraws to ~33fps — a long reply
                # streams thousands of chunks; redrawing on every one
                # wastes CPU and makes the box stutter
                if render_md:
                    md_buf.append(piece)
                    md_len["n"] += len(piece)
                else:
                    stream_tail["t"] += piece
                self._preview_pending = True
                return
            self._last_preview_ts = now
            self._preview_pending = False
            if render_md:
                # markdown mode: nothing raw hits the console — the live
                # preview runs in the box border, the finished reply is
                # printed once, rendered as rich Markdown.
                md_buf.append(piece)
                md_len["n"] += len(piece)
                maxw = self._width() - 26
                if md_len["n"] <= maxw:
                    preview = "".join(md_buf).strip("\n")
                else:
                    preview = ("".join(md_buf))[-maxw:].strip("\n")
                self._set_status(preview if preview else "writing…")
                return
            # patch_stdout can only interleave output safely when every
            # write ends in a newline, so emit complete lines here and keep
            # the partial line as a live preview inside the box border.
            tail = stream_tail["t"] + piece
            if "\n" in tail:
                before, _, rem = tail.rpartition("\n")
                # collapse blank-line runs (the model loves "\n\n\n" between
                # tool calls) — keep at most one blank in a row
                out: list[str] = []
                for ln in before.split("\n"):
                    if not ln.strip():
                        if stream_tail["blank"]:
                            continue
                        stream_tail["blank"] = True
                    else:
                        stream_tail["blank"] = False
                    out.append(ln)
                if out:
                    self.console.print(Text("\n".join(out)), soft_wrap=True)
                tail = rem
            stream_tail["t"] = tail
            maxw = self._width() - 26
            preview = tail.strip("\n")
            if len(preview) > maxw:
                preview = preview[-maxw:]
            self._set_status(preview if preview else "writing…")

        def on_reasoning(piece: str):
            self._set_status("reasoning…")

        # live shell streaming state — one tool runs at a time, so a plain
        # dict is enough: on_tool_call arms it, on_tool_output streams the
        # lines, on_tool_update closes the block with the exit-code footer
        shell_live = {"active": False, "streamed": False, "lines": 0}

        def on_tool_output(line: str, stream: str):
            if not shell_live["active"]:
                return
            n = shell_live["lines"]
            if n == 500:
                # flood guard — the full output stays in the tool result
                self.console.print(Text(
                    " │ … live view truncated (full output kept in the "
                    "tool result)", style=C["dim"]), soft_wrap=True)
                shell_live["lines"] = n + 1
                return
            if n > 500:
                return
            shell_live["lines"] = n + 1
            shell_live["streamed"] = True
            style = C["orange"] if stream == "err" else C["fg"]
            self.console.print(Text(" │ " + line, style=style),
                               soft_wrap=True)

        # live file-change streaming — write_file / edit_file / apply_patch
        # are shown happening WHILE the model generates them (true live, not
        # a replay). on_tool_args feeds the growing JSON; a tracker pulls the
        # relevant string field; on_tool_update closes the block.
        live_f = {"kind": None, "tracker": None, "header": False,
                  "count": 0, "held": [], "path": "", "old_emitted": False}

        def _lf_write_line(ln: str):
            live_f["count"] += 1
            t = Text()
            t.append(" │ ", style=C["dim"])
            t.append(f"{live_f['count']:>4} ", style=C["dim"])
            t.append("+  ", style=C["green"])
            t.append(ln, style=C["fg"])
            self.console.print(t, soft_wrap=True)

        def _lf_edit_line(ln: str, plus: bool):
            live_f["count"] += 1
            t = Text()
            t.append(" │ ", style=C["dim"])
            if plus:
                t.append("+  ", style=C["green"])
                t.append(ln, style=C["fg"])
            else:
                t.append("-  ", style=C["red"])
                t.append(ln, style=C["red"])
            self.console.print(t, soft_wrap=True)

        def _lf_patch_line(ln: str):
            live_f["count"] += 1
            t = Text()
            t.append(" │ ", style=C["dim"])
            if ln.startswith("@@"):
                col = C["cyan"]
            elif ln.startswith(("+++ ", "--- ")):
                col = f"bold {C['dim']}"
            elif ln.startswith("+"):
                col = C["green"]
            elif ln.startswith("-"):
                col = C["red"]
            else:
                col = C["fg"]
            t.append(ln, style=col)
            self.console.print(t, soft_wrap=True)

        def _lf_emit(ln: str):
            k = live_f["kind"]
            if k == "write":
                _lf_write_line(ln)
            elif k == "edit":
                _lf_edit_line(ln, plus=True)
            else:
                _lf_patch_line(ln)

        def on_tool_args(name: str, chunk: str):
            if name not in ("write_file", "edit_file", "apply_patch"):
                return
            if not self.cfg.extra.get("live_stream_edits", True):
                return  # animations disabled — fall back to instant render
            kind = {"write_file": "write", "edit_file": "edit",
                    "apply_patch": "patch"}[name]
            if live_f["tracker"] is None:
                key = {"write": "content", "edit": "new_string",
                       "patch": "patch"}[kind]
                live_f["tracker"] = _LiveWrite(key)
                live_f["kind"] = kind
            tr = live_f["tracker"]
            try:
                new_lines = tr.feed(chunk)
            except Exception:  # noqa: BLE001 — never break the turn
                return
            # header once the path is known (patch has no single path)
            if not live_f["header"]:
                if kind == "patch":
                    p = True  # no path arg — header can go immediately
                else:
                    p = tr.path()
                if p:
                    if kind != "patch":
                        live_f["path"] = p
                    verb = {"write": "Wrote", "edit": "Edited",
                            "patch": "Applied patch"}[kind]
                    label = (verb if kind == "patch"
                             else f"{verb} {self._rel_path(p)}")
                    t = Text()
                    t.append(" ⏺ ", style=f"bold {C['accent']}")
                    t.append(label, style=f"bold {C['fg']}")
                    self.console.print(t, soft_wrap=True)
                    live_f["header"] = True
            # edit: show the removed (old) lines once old_string is complete,
            # before any added lines stream
            if kind == "edit" and live_f["header"] \
                    and not live_f["old_emitted"]:
                old = tr._field(tr.buf, "old_string")
                if old is not None:
                    for ol in old.splitlines():
                        _lf_edit_line(ol, plus=False)
                    live_f["old_emitted"] = True
                    for held in live_f["held"]:
                        _lf_emit(held)
                    live_f["held"] = []
            # stream the new lines once header (and old lines) are done
            ready = live_f["header"] and \
                (kind != "edit" or live_f["old_emitted"])
            if ready:
                for ln in new_lines:
                    _lf_emit(ln)
                if new_lines:
                    verb = {"write": "writing", "edit": "editing",
                            "patch": "patching"}[kind]
                    if kind != "patch":
                        self._set_status(f"{verb} "
                                         f"{self._rel_path(live_f['path'])} · "
                                         f"{live_f['count']} lines…")
                    else:
                        self._set_status(f"{verb} · "
                                         f"{live_f['count']} lines…")
            else:
                live_f["held"].extend(new_lines)

        def on_tool_call(ev: ToolEvent):
            rem = stream_tail["t"].strip("\n")
            stream_tail["t"] = ""
            stream_tail["blank"] = False
            if rem:
                self.console.print(Text(rem), soft_wrap=True)
            shell_live["active"] = ev.name in ("run_command", "live_shell")
            shell_live["streamed"] = False
            shell_live["lines"] = 0
            if ev.name not in ("write_file", "edit_file", "apply_patch"):
                # a non file-change tool starting clears stale live state
                live_f["kind"] = None
                live_f["tracker"] = None
                live_f["header"] = False
                live_f["count"] = 0
                live_f["held"] = []
                live_f["old_emitted"] = False
            if ev.name in ("run_command", "live_shell", "apply_patch",
                           "write_file", "edit_file", "read_file"):
                # live-action block header instead of the generic gear line.
                # For file-change tools, skip the header if it was already
                # printed during live streaming.
                if ev.name in ("write_file", "edit_file", "apply_patch") \
                        and live_f["header"]:
                    pass
                else:
                    self.console.print(self._live_block(ev))
            else:
                self.console.print(self._tool_call_line(ev))
            self._set_status(f"running {ev.name}…")

        def on_tool_update(ev: ToolEvent):
            if ev.name == "wait_for_agents" and ev.status == "done":
                # subagent reports deserve a real panel, not one line
                self.console.print(self._subagent_panel(ev))
            elif ev.name in ("write_file", "edit_file", "apply_patch") \
                    and ev.status in ("done", "error") \
                    and live_f["count"] > 0:
                # change already streamed live — flush the last partial line
                # and close the block; no replay cascade
                tr = live_f["tracker"]
                if tr is not None:
                    last = tr.flush()
                    if last is not None:
                        _lf_emit(last)
                self.console.print(self._write_footer(ev, live_f["count"]))
            elif ev.name in ("write_file", "edit_file", "apply_patch") \
                    and ev.status in ("done", "error"):
                # live-write animation — the file changes cascade down the
                # terminal line by line, like model tokens streaming
                self._stream_live_block(ev)
            elif ev.name in ("run_command", "live_shell") \
                    and ev.status in ("done", "error"):
                if shell_live["streamed"]:
                    # output already streamed live — just close the block
                    self.console.print(self._shell_footer(ev))
                else:
                    self.console.print(self._live_block(ev))
            elif ev.name == "read_file" and ev.status in ("done", "error"):
                self.console.print(self._live_block(ev))
            else:
                self.console.print(self._tool_result_line(ev))
            shell_live["active"] = False
            live_f["kind"] = None
            live_f["tracker"] = None
            live_f["header"] = False
            live_f["count"] = 0
            live_f["held"] = []
            live_f["old_emitted"] = False
            self._set_status("thinking…")

        def on_status(s: str):
            if s == "thinking":
                self._set_status("thinking…")
            elif s.startswith("tool:"):
                self._set_status(f"calling {s[5:]}…")
            elif s.startswith("running:"):
                self._set_status(f"running {s[8:]}…")
            else:
                self._set_status(s)

        def on_route(route):
            # the AutoPilot's decision, shown live — nothing hidden (A7)
            self.print_info("AUTOPILOT  " + route.summary(), C["pink"])
            for r in route.reasons:
                self.print_info(f"  ↳ {r}", C["dim"])

        turn = None
        try:
            turn = self.agent.run_turn(
                text, on_token, on_reasoning, on_tool_call, on_tool_update,
                on_status, self._approve_blocking,
                should_cancel=self._cancel_flag.is_set,
                on_route=on_route, on_tool_output=on_tool_output,
                on_tool_args=on_tool_args)
        except Exception as e:  # noqa: BLE001 — never kill the UI thread
            self.print_error(f"{type(e).__name__}: {e}")
        finally:
            rem = stream_tail["t"].strip("\n")
            stream_tail["t"] = ""
            if rem:
                self.console.print(Text(rem), soft_wrap=True)
            else:
                self.console.print()
            self._stop_spinner()
            self._busy = False
            self._set_status("")
            self._invalidate()

        if turn is not None:
            if render_md and turn.assistant_text.strip() and not turn.error:
                # the finished reply, rendered once as rich Markdown
                self.console.print()
                self.console.print(Markdown(turn.assistant_text),
                                   soft_wrap=True)
            if turn.reasoning and self.cfg.show_reasoning:
                self.print_reasoning(turn.reasoning)
            if turn.error:
                if turn.error == "cancelled":
                    self.print_info("⊘ cancelled", C["yellow"])
                else:
                    self.print_error(turn.error)
            self._print_turn_stats(turn)
            self.agent.save_session()

            # FOCUS MODE — the kernel decides whether work remains; the
            # UI drives the next continuation turn automatically.
            if (self._focus_remaining > 0
                    and not self._cancel_flag.is_set()):
                cont = self.agent.focus_continue(turn,
                                                 self._focus_remaining)
                if cont is None:
                    stops = [e for e in self.agent.log.events()
                             if e.type == "focus.stop"]
                    reason = (stops[-1].data.get("reason", "complete")
                              if stops else "complete")
                    self._focus_remaining = 0
                    self._invalidate()
                    self.print_info(f"🎯 focus ended — {reason}",
                                    C["yellow"])
                else:
                    self._focus_remaining -= 1
                    self._invalidate()
                    self.print_info(
                        f"🎯 focus · auto-continuing "
                        f"({self._focus_remaining} turn(s) left)…",
                        C["pink"])
                    self._emit_user("CONTINUE (focus mode)")
                    self._run_turn_thread(cont)
                    return

    def _print_turn_stats(self, turn) -> None:
        parts = [f"{turn.duration:.1f}s"]
        if turn.usage:
            tin = turn.usage.get("prompt_tokens", 0) or 0
            tout = turn.usage.get("completion_tokens", 0) or 0
            if tin or tout:
                parts.append(f"{tin}→{tout} tokens")
        t = Text("  ·  ".join(parts), style=C["dim"])
        self.console.print(t)

    def _start_spinner(self) -> None:
        # generation token: back-to-back turns used to leak tick threads
        # (old thread wakes from sleep after the flag flips and keeps
        # looping) — a stale generation exits instead
        self._spinner_gen = getattr(self, "_spinner_gen", 0) + 1
        gen = self._spinner_gen
        self._spinner_on = True

        def tick():
            while self._spinner_on and gen == self._spinner_gen:
                self._spinner_i = (self._spinner_i + 1) % len(SPINNER_FRAMES)
                self._invalidate()
                time.sleep(0.09)

        threading.Thread(target=tick, daemon=True).start()

    def _stop_spinner(self) -> None:
        self._spinner_gen = getattr(self, "_spinner_gen", 0) + 1
        self._spinner_on = False

    # -- approval (in-app) ------------------------------------------------------------------------

    def _approve_blocking(self, tool: Tool, args: dict) -> bool:
        if self.cfg.auto_approve:
            return True
        done = threading.Event()
        self._approve_result = False
        self._approve_request = (tool, args, done)
        self._invalidate()
        # show what is being approved above the box — with a real diff
        self.console.print(self._approval_line(tool, args))
        preview = self._diff_preview(tool, args)
        if preview is not None:
            self.console.print(preview)
        done.wait()
        self._approve_request = None
        self._invalidate()
        return self._approve_result

    def _answer_approve(self, answer: str) -> None:
        if self._approve_request is None:
            return
        if answer == "a":
            self.cfg.auto_approve = True
            self.cfg.save()
            self.print_info("  auto-approve enabled", C["yellow"])
            answer = "y"
        self._approve_result = answer == "y"
        self._approve_request[2].set()

    def _diff_preview(self, tool: Tool, args: dict):
        """Real preview of what the mutation will do — unified diff for
        edit_file, overwrite warning for write_file. None = no preview."""
        if tool.name == "write_file":
            p = Path(str(args.get("path", ""))).expanduser()
            if p.exists() and p.is_file():
                return Text(f"  ⚠ overwrites existing file "
                            f"({p.stat().st_size:,} bytes)",
                            style=C["yellow"])
            return None
        if tool.name != "edit_file":
            return None
        path = Path(str(args.get("path", ""))).expanduser()
        old_s = str(args.get("old_string", ""))
        new_s = str(args.get("new_string", ""))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return Text(f"  ⚠ file not readable: {path}", style=C["red"])
        if old_s and old_s not in text:
            return Text("  ⚠ old_string NOT FOUND in file — this edit "
                        "will fail", style=C["red"])
        import difflib
        t = Text()
        lines = list(difflib.unified_diff(
            old_s.splitlines(), new_s.splitlines(),
            fromfile="before", tofile="after", lineterm="", n=1))
        for line in lines[:40]:
            style = (C["green"] if line.startswith("+")
                     else C["red"] if line.startswith("-")
                     else C["dim"])
            t.append(line + "\n", style=style)
        if len(lines) > 40:
            t.append(f"  … {len(lines) - 40} more diff line(s)\n",
                     style=C["dim"])
        return t if t.plain.strip() else None

    def _approval_line(self, tool: Tool, args: dict) -> Text:
        import json as _json
        try:
            arg_str = _json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            arg_str = str(args)
        if len(arg_str) > 200:
            arg_str = arg_str[:197] + "…"
        t = Text()
        t.append("  ⚠ ", style=f"bold {C['yellow']}")
        t.append(tool.name, style=f"bold {C['yellow']}")
        t.append(f"  {arg_str}", style=C["fg"])
        return t

    # -- overlays ------------------------------------------------------------------------------------

    def open_model_selector(self) -> None:
        items = []
        current = 0
        for i, m in enumerate(MODELS):
            provider = PROVIDERS[m.provider]
            tag = f' <style color="{m.tag_color}">[{m.tag}]</style>' if m.tag else ""
            html = (f'<style color="{C["cyan"]}">{m.label}</style>{tag}'
                    f' <style color="{C["dim"]}">{m.id} · {provider.name}</style>')
            items.append((html, m.id))
            if m.id == self.cfg.model_id:
                current = i

        def on_select(idx: int):
            m = MODELS[idx]
            self.cfg.model_id = m.id
            self.cfg.save()
            self._set_flash(f"model → {m.label} ({m.id})", C["green"])

        self.overlay = OverlayList("SELECT MODEL", items, current, on_select)
        self.overlay.open()
        self._invalidate()

    def open_effort_selector(self) -> None:
        items = []
        current = 0
        for i, e in enumerate(EFFORTS):
            html = (f'<style color="{e.color}"><b>{e.label}</b></style>'
                    f' <style color="{C["dim"]}">{e.description}</style>')
            items.append((html, e.key))
            if e.key == self.cfg.effort:
                current = i

        def on_select(idx: int):
            e = EFFORTS[idx]
            self.cfg.effort = e.key
            self.cfg.save()
            self._set_flash(f"effort → {e.label}", e.color)

        self.overlay = OverlayList("SELECT EFFORT", items, current, on_select)
        self.overlay.open()
        self._invalidate()

    def open_help(self) -> None:
        def row(cmd: str, desc: str) -> tuple[str, str]:
            return (f'<style color="{C["cyan"]}">{cmd}</style>'
                    f' <style color="{C["dim"]}">{desc}</style>', "")

        items = [
            row("/model", "select model (PgUp/PgDn/Tab to navigate)"),
            row("/effort", "low · medium · high · extrahigh · ultrahigh"),
            row("/goal", "set · prove · close · status · waive · clear"),
            row("/autonomy", "0-5 observer → autonomous"),
            row("/state", "live projection of the event log"),
            row("/rewind", "rewind timeline+files · /revert files only"),
            row("/fork", "branch the timeline · /verify Merkle spine"),
            row("/why", "causal chain for an event seq"),
            row("/impact", "code blast-radius analysis (Nexus)"),
            row("/forge", "environment digest + drift"),
            row("/oracle", "post-run analysis + calibration"),
            row("/budget [steps N|usd X|reset]", "budget governor status/extend"),
            row("/constitution", "standing rules (human-owned)"),
            row("/replay", "replay the session log as a film"),
            row("/memory", "episodes + dead-end ledger"),
            row("/judge", "deterministic check (exit_code, file_exists, …)"),
            row("/new", "fresh conversation"),
            row("/history", "browse previous turns"),
            row("/save", "save session to disk"),
            row("/approve", "toggle auto-approve for tools"),
            row("/reasoning", "toggle reasoning display"),
            row("/usage", "token usage"),
            row("/clear", "clear screen"),
            row("/exit", "quit"),
            row("Enter", "send    Esc+Enter: newline    Ctrl+R: search history"),
            row("Ctrl+T", "models    Ctrl+E: effort    Ctrl+L: clear"),
        ]
        self.overlay = OverlayList("HELP", items, 0, None, footer="Esc close")
        self.overlay.open()
        self._invalidate()

    def open_history(self) -> None:
        turns = self.agent.turns
        if not turns:
            self._set_flash("no history yet", C["dim"])
            return
        items = []
        for t in turns[-15:]:
            # prompt_toolkit parses HTML() with an XML parser — raw user
            # text containing '&' or '<' would crash the RENDER loop and
            # kill the whole session; escape everything user-controlled
            preview = html_escape(t.user_text.replace("\n", " ")[:70])
            stamp = html_escape(str(t.timestamp))
            html = (f'<style color="{C["dim"]}">{stamp}</style> '
                    f'<style color="{C["fg"]}">{preview}</style>')
            items.append((html, ""))
        self.overlay = OverlayList("HISTORY (recent turns)", items,
                                   len(items) - 1, None)
        self.overlay.open()
        self._invalidate()

    # -- output -----------------------------------------------------------------------------------------

    def run(self) -> None:
        """Run the persistent app for the whole session."""
        # raw=True: let rich's ANSI escape sequences pass through untouched;
        # with the default (raw=False) prompt_toolkit replaces ESC with "?"
        # and all colors show up as literal "[1;38;2..." text.
        with patch_stdout(raw=True):
            try:
                self.app.run()
            except KeyboardInterrupt:
                pass
            except EOFError:
                # stdin closed (piped/non-interactive) — exit cleanly
                self.console.print()
                self.print_info("⊘ no interactive terminal — run fullagent "
                                "in a real TTY, or use headless commands "
                                "(python main.py --help)", C["yellow"])
        self.agent.save_session()

    def print_banner(self) -> None:
        from . import __version__
        term_w = shutil.get_terminal_size((100, 24)).columns
        width = min(term_w - 2, 84)
        model = self._model()
        effort = self._effort()

        # ── FullAgent ASCII banner (6-line block, dracula gradient) ──
        # Compact width ~69-72 fits safely in 80-col consoles (inner 76).
        # On narrow terminals (<78) fall back to single-line logo to avoid wrap.
        ascii_lines = [
            "███████╗██╗   ██╗██╗    ██╗    █████╗  ██████╗ ███████╗███╗ ██╗████████╗",
            "██╔════╝██║   ██║██║    ██║   ██╔══██╗██╔════╝ ██╔════╝████╗██║╚══██╔══╝",
            "█████╗  ██║   ██║██║    ██║   ███████║██║  ███╗█████╗  ██╔██╗██║  ██║   ",
            "██╔══╝  ██║   ██║██║    ██║   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║  ██║   ",
            "██║     ╚██████╔╝██████╗█████╗██║  ██║╚██████╔╝███████╗██║ ╚████║  ██║   ",
            "╚═╝      ╚═════╝ ╚═════╝╚════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝  ╚═╝   ",
        ]
        accent_styles = [C["accent"], C["accent"], C["pink"], C["cyan"], C["cyan"], C["pink"]]

        # effective console width may be smaller than term_w (Panel caps to console width)
        # so be conservative: need inner width >= max ascii length + 4
        max_ascii = max(len(s) for s in ascii_lines)
        need_width = max_ascii + 4 + 2  # borders + padding
        if width >= max(78, need_width):
            banner = Text()
            for i, line in enumerate(ascii_lines):
                banner.append(line.rstrip(), style=f"bold {accent_styles[i % len(accent_styles)]}")
                banner.append("\n")
            # tagline + feature stripe under the ascii (kept ≤74 chars to avoid Panel wrap)
            banner.append("  ◆ FullAgent ", style=f"bold {C['accent']}")
            banner.append(f"v{__version__}", style=f"bold {C['pink']}")
            banner.append("  ·  Event-Sourced Kernel  ·  Goal Contracts  ·  Crew", style=C["dim"])
            banner.append("\n")
            banner.append("  ⚡ 40+ Commands · 16 Tools · 5 Providers · Real-time Web · Self-Healing", style=C["dim"])
            self.console.print(Panel(banner, width=width,
                                     border_style=C["border"], padding=(0, 1),
                                     title=f"[bold {C['accent']}]FullAgent[/]",
                                     subtitle=f"[dim]session {self.agent.session_id}[/]"))
        else:
            # narrow fallback — original compact logo (never wraps)
            logo = Text()
            logo.append("◆ ", style=f"bold {C['accent']}")
            logo.append(APP_NAME, style=f"bold {C['accent']}")
            logo.append(f" v{__version__}", style=f"bold {C['pink']}")
            logo.append("  ·  advanced terminal AI agent", style=C["dim"])
            logo.append("\n")
            logo.append("event-sourced kernel · goal contracts · persistent crew · "
                        "self-healing", style=C["dim"])
            self.console.print(Panel(logo, width=width,
                                     border_style=C["border"], padding=(0, 1)))

        line = Text()
        line.append(" ❯ model  ", style=C["dim"])
        line.append(model.label, style=f"bold {C['cyan']}")
        if model.tag:
            line.append(f" {model.tag} ", style=f"bold {C['green']}")
        line.append("   effort  ", style=C["dim"])
        line.append(effort.label.lower(),
                    style=f"bold {EFFORT_COLORS[effort.key]}")
        line.append("   autonomy  ", style=C["dim"])
        line.append(f"L{self.agent.autonomy}", style=f"bold {C['yellow']}")
        line.append("   session  ", style=C["dim"])
        line.append(self.agent.session_id, style=C["fg"])
        self.console.print(line)

        hints = Text()
        hints.append("   ", style=C["dim"])
        hints.append("/", style=f"bold {C['green']}")
        hints.append(" commands · ", style=C["dim"])
        hints.append("Ctrl+T", style=f"bold {C['cyan']}")
        hints.append(" models · ", style=C["dim"])
        hints.append("Ctrl+E", style=f"bold {C['cyan']}")
        hints.append(" effort · ", style=C["dim"])
        hints.append("/crew", style=f"bold {C['pink']}")
        hints.append(" background subagents", style=C["dim"])
        self.console.print(hints)
        self.console.print()

    def _emit_user(self, text: str) -> None:
        t = Text()
        t.append("❯ ", style=f"bold {C['green']}")
        t.append(text, style=f"bold {C['fg']}")
        self.console.print(t)

    # Devin-style titles + bodies for every tool that is not a live-action
    # block (glob / search / list / web / file-ops). One ⏺ header on call,
    # one │/└ body on result — no ⚙ gear lines, no ✓ lines.
    _DEVIN_VERB = {
        "glob_files": "Globbed",
        "search_files": "Searched",
        "list_dir": "Listed",
        "file_info": "Inspected",
        "create_directory": "Created directory",
        "copy_path": "Copied",
        "move_path": "Moved",
        "delete_path": "Deleted",
        "web_fetch": "Fetched",
        "web_search": "Searched the web",
        "live_shell_reset": "Reset the shell session",
    }
    _DEVIN_ARG = {
        "glob_files": "pattern", "search_files": "pattern",
        "list_dir": "path", "file_info": "path",
        "create_directory": "path", "delete_path": "path",
        "copy_path": "src", "move_path": "src",
        "web_fetch": "url", "web_search": "query",
    }
    _DEVIN_PATH_ARG = {"list_dir", "file_info", "create_directory",
                       "delete_path", "copy_path", "move_path"}

    def _devin_title(self, ev: ToolEvent) -> str:
        verb = self._DEVIN_VERB.get(ev.name,
                                    ev.name.replace("_", " ").strip())
        key = self._DEVIN_ARG.get(ev.name)
        arg = ""
        if key and ev.args.get(key) is not None:
            raw = str(ev.args[key])
            arg = self._rel_path(raw) if ev.name in self._DEVIN_PATH_ARG \
                else raw
            if len(arg) > 60:
                arg = arg[:57] + "…"
        if ev.name in ("copy_path", "move_path"):
            dst = str(ev.args.get("dst", ""))
            arg = f"{arg} → {dst}" if arg else dst
        return f"{verb} {arg}".strip()

    def _tool_call_line(self, ev: ToolEvent) -> Text:
        t = Text()
        t.append(" ⏺ ", style=f"bold {C['accent']}")
        t.append(self._devin_title(ev), style=f"bold {C['fg']}")
        return t

    def _generic_rows(self, ev: ToolEvent) -> list:
        """Body rows for a finished non-live tool (same segment format as
        _live_rows). Always returns at least one row (the └ footer)."""
        name = ev.name
        res = (ev.result or "").rstrip()
        if ev.status != "done" or res.startswith("ERROR"):
            msg = res.splitlines()[0] if res else ev.status
            return [[(msg[:200], C["red"])]]
        lines = res.splitlines()

        def capped(items, cap):
            rows = [[(it, C["fg"])] for it in items[:cap]]
            if len(items) > cap:
                rows.append([(f"… +{len(items) - cap} more", C["dim"])])
            return rows

        if name == "glob_files":
            if res.strip() == "no matches":
                return [[("no matches", C["dim"])]]
            files = [l for l in lines if l.strip()]
            rows = capped([self._rel_path(f) for f in files], 5)
            rows.append([(f"{len(files)} file(s)", C["green"])])
            return rows
        if name == "search_files":
            if res.strip() == "no matches":
                return [[("no matches", C["dim"])]]
            hits = [l for l in lines if l.strip()]
            rows = capped([h[:160] for h in hits], 5)
            rows.append([(f"{len(hits)} match(es)", C["green"])])
            return rows
        if name == "list_dir":
            entries = [l.strip() for l in lines[1:] if l.strip()]
            if not entries:
                return [[("(empty)", C["dim"])]]
            rows = capped(entries, 8)
            rows.append([(f"{len(entries)} entr"
                          f"{'y' if len(entries) == 1 else 'ies'}",
                          C["green"])])
            return rows
        if name == "web_search":
            n = len(re.findall(r"(?m)^\d+\.", res))
            body = [l for l in lines if l.strip()
                    and not l.startswith("web search:")]
            rows = capped([l[:160] for l in body], 6)
            rows.append([(f"{n} result(s)", C["green"])])
            return rows
        if name == "web_fetch":
            return [[(f"{len(res)} characters fetched", C["green"])]]
        if name in ("create_directory", "copy_path", "move_path",
                    "delete_path"):
            return [[("done", C["green"])]]
        if name == "file_info":
            rows = [[(l.strip()[:160], C["fg"])] for l in lines if l.strip()]
            return rows or [[("done", C["green"])]]
        first = lines[0] if lines else "done"
        return [[(first[:200], C["green"])]]

    def _tool_result_line(self, ev: ToolEvent) -> Text:
        rows = self._generic_rows(ev)
        t = Text()
        n = len(rows)
        for idx, row in enumerate(rows):
            t.append(" └ " if idx == n - 1 else " │ ", style=C["dim"])
            for seg, style in row:
                t.append(seg, style=style)
            if idx < n - 1:
                t.append("\n")
        return t

    # ------------------------------------------------------------------
    # Live-action blocks, Devin-CLI style:
    #
    #  ⏺ Ran command               ⏺ Wrote ./webhack/example.md
    #  │ $ ls -la                  │  1 +  # AI ke baare mein
    #  │ total 40                  │  2 +  ## Kya hai AI?
    #  └ Exited with code 0        └ …
    #
    #  ⏺ Read ./file.md            ⏺ Edited ./file.md
    #  └ 57 lines                  │ 24    ## AI ke applications
    #                              │ 26 -  - **Healthcare**: diagnosis
    #                              │ 26 +  - **Healthcare**: diagnosis, imaging
    #                              └ 33 +  - **Cybersecurity**: threats
    # ------------------------------------------------------------------

    def _rel_path(self, path: str) -> str:
        """Devin-style path display: relative to cwd with a ./ prefix."""
        if not path:
            return ""
        try:
            rel = os.path.relpath(str(path))
        except ValueError:
            return str(path)
        if rel.startswith(".."):
            return str(path)
        return rel if rel.startswith(".") else "./" + rel

    @staticmethod
    def _parse_shell_result(result: str) -> tuple:
        """Split a run_command/live_shell receipt into (exit, stdout, stderr)."""
        exit_code = None
        m = re.search(r"^exit code: (-?\d+)", result, flags=re.M)
        if m:
            exit_code = int(m.group(1))
        stdout = stderr = ""
        sm = re.search(
            r"--- stdout ---\n(.*?)(?:\n--- stderr ---\n(.*))?$",
            result, flags=re.S)
        if sm:
            stdout = (sm.group(1) or "").rstrip("\n")
            stderr = (sm.group(2) or "").rstrip("\n")
        else:
            em = re.search(r"--- stderr ---\n(.*)$", result, flags=re.S)
            if em:
                stderr = em.group(1).rstrip("\n")
            elif exit_code is None:
                stdout = result.rstrip("\n")
        return exit_code, stdout, stderr

    @staticmethod
    def _find_hunk_line(path: str, new: str) -> int:
        """1-based line number where the edited hunk now sits, so the diff
        view can show real file line numbers instead of hunk-relative ones."""
        try:
            text = Path(path).expanduser().read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return 1
        if not new:
            return 1
        idx = text.find(new)
        if idx < 0:
            return 1
        return text.count("\n", 0, idx) + 1

    @staticmethod
    def _diff_rows(old: str, new: str, start: int) -> list:
        """Rows of (line_no, marker, color, text) for the edit diff view.
        Old-side and new-side numbers both start at the hunk's file line,
        exactly like Devin's edit blocks."""
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        sm = difflib.SequenceMatcher(None, old_lines, new_lines,
                                     autojunk=False)
        rows = []
        i = j = start
        changed = False
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                # context before any change keeps old-file numbers; context
                # after a change switches to new-file numbers (Devin style)
                base = j if changed else i
                for k in range(i1, i2):
                    rows.append((base + (k - i1), "   ", C["fg"],
                                 old_lines[k]))
                i += i2 - i1
                j += j2 - j1
            elif tag == "delete":
                changed = True
                for k in range(i1, i2):
                    rows.append((i + (k - i1), "-  ", C["red"], old_lines[k]))
                i += i2 - i1
            elif tag == "insert":
                changed = True
                for k in range(j1, j2):
                    rows.append((j + (k - j1), "+  ", C["green"],
                                 new_lines[k]))
                j += j2 - j1
            else:  # replace — removals first, then additions
                changed = True
                for k in range(i1, i2):
                    rows.append((i + (k - i1), "-  ", C["red"], old_lines[k]))
                i += i2 - i1
                for k in range(j1, j2):
                    rows.append((j + (k - j1), "+  ", C["green"],
                                 new_lines[k]))
                j += j2 - j1
        return rows

    def _live_rows(self, ev: ToolEvent, w: int) -> list:
        """Body rows for a finished live block; each row is a list of
        (text, style) segments. The caller adds the │ / └ tree glyphs."""
        rows: list = []
        failed = ev.status != "done" or ev.result.startswith("ERROR")

        def add(text: str, style: str):
            rows.append([(text[:w], style)])

        if ev.name in ("run_command", "live_shell"):
            add(f"$ {ev.args.get('command', '')}", f"bold {C['fg']}")
            if failed:
                add(ev.result.strip() or ev.status, C["red"])
                return rows
            code, out, err = self._parse_shell_result(ev.result)
            lines = out.splitlines()
            for ln in lines[:30]:
                add(ln if ln else " ", C["fg"])
            if len(lines) > 30:
                add(f"… +{len(lines) - 30} more lines", C["dim"])
            for ln in err.splitlines()[:10]:
                add(ln, C["orange"])
            if code is not None:
                add(f"Exited with code {code}",
                    C["green"] if code == 0 else C["red"])
        elif ev.name == "write_file":
            if failed:
                add(ev.result.strip() or ev.status, C["red"])
                return rows
            content = str(ev.args.get("content", ""))
            lines = content.splitlines()
            width = len(str(max(len(lines), 1)))
            for n, ln in enumerate(lines[:300], 1):
                rows.append([(f"{n:>{width}} ", C["dim"]),
                             ("+  ", C["green"]),
                             (ln[:w], C["fg"])])
            if len(lines) > 300:
                add(f"… +{len(lines) - 300} more lines", C["dim"])
        elif ev.name == "edit_file":
            if failed:
                add(ev.result.strip() or ev.status, C["red"])
                return rows
            old = str(ev.args.get("old_string", ""))
            new = str(ev.args.get("new_string", ""))
            start = self._find_hunk_line(str(ev.args.get("path", "")), new)
            drows = self._diff_rows(old, new, start)
            width = len(str(max((r[0] for r in drows), default=1)))
            for num, mark, color, text in drows[:100]:
                rows.append([(f"{num:>{width}} ", C["dim"]),
                             (mark, C["dim"] if mark == "   " else color),
                             (text[:w], color)])
            if len(drows) > 100:
                add(f"… +{len(drows) - 100} more lines", C["dim"])
        elif ev.name == "read_file":
            m = re.search(r"(\d+) lines total, showing (\d+)\.\.(\d+)",
                          ev.result)
            if m and m.group(2) == "1" and m.group(3) == m.group(1):
                add(f"{m.group(1)} lines", C["dim"])
            elif m:
                add(f"lines {m.group(2)}..{m.group(3)} of {m.group(1)}",
                    C["dim"])
            else:
                add((ev.result.splitlines() or [""])[0], C["dim"])
        else:  # apply_patch
            lines = str(ev.args.get("patch", "")).splitlines()
            for ln in lines[:60]:
                if ln.startswith("@@"):
                    add(ln, C["cyan"])
                elif ln.startswith(("+++ ", "--- ")):
                    add(ln, f"bold {C['dim']}")
                elif ln.startswith("+"):
                    add(ln, C["green"])
                elif ln.startswith("-"):
                    add(ln, C["red"])
                else:
                    add(ln, C["fg"])
            if len(lines) > 60:
                add(f"… +{len(lines) - 60} more lines", C["dim"])
            if failed:
                add(ev.result.strip() or ev.status, C["red"])
        if not rows:
            add(ev.result.splitlines()[0] if ev.result else ev.status,
                C["dim"])
        return rows

    def _live_body_lines(self, ev: ToolEvent) -> list:
        """The finished body of a live block as one Text per line (tree
        glyphs included) — printable all at once or streamed line by line."""
        w = max(20, self._width() - 8)
        rows = self._live_rows(ev, w)
        n = len(rows)
        out = []
        for idx, row in enumerate(rows):
            t = Text()
            t.append(" └ " if idx == n - 1 else " │ ", style=C["dim"])
            for seg, style in row:
                t.append(seg, style=style)
            out.append(t)
        return out

    def _stream_delay(self, n_lines: int) -> float:
        """Per-line delay for the live-write animation. Tuned so the cascade
        is unmistakably 'live': ~24ms/line, the whole block capped at ~3.5s,
        and even tiny edits get >= ~0.5s so they don't just flash by. A fast
        ~8ms/line dump reads as an instant append on a scrolling terminal —
        that's what made writes look non-live before."""
        if not self.cfg.extra.get("live_stream_edits", True):
            return 0.0
        if not self.console.is_terminal:
            return 0.0  # piped/redirected output — never animate
        if n_lines <= 0:
            return 0.0
        per = min(0.024, 3.5 / n_lines)   # keep the whole block under ~3.5s
        per = max(per, 0.5 / n_lines)     # keep small blocks >= ~0.5s total
        return per

    def _stream_live_block(self, ev: ToolEvent) -> None:
        """Stream a finished write/edit/patch block line by line — the same
        live feel as model token streaming, but for file changes. Runs on
        the turn's worker thread, so sleeping between lines never blocks
        the prompt_toolkit event loop."""
        lines = self._live_body_lines(ev)
        n = len(lines)
        if n == 0:
            return
        delay = self._stream_delay(n)
        rel = self._rel_path(str(ev.args.get("path", "")))
        verb = {"write_file": "writing", "edit_file": "editing",
                "apply_patch": "patching"}.get(ev.name, "writing")
        last_status = 0.0
        for idx, line in enumerate(lines):
            if self._cancel_flag.is_set():
                delay = 0.0  # cancelled — flush the rest instantly
            self.console.print(line, soft_wrap=True)
            if delay and idx < n - 1:
                now = time.time()
                if now - last_status >= 0.08:
                    self._set_status(f"{verb} {rel} · "
                                     f"{idx + 1}/{n} lines…")
                    last_status = now
                time.sleep(delay)

    def _live_block(self, ev: ToolEvent) -> Text:
        """Devin-CLI style live-action block (see banner above). The header
        is printed when the tool starts; the │/└ body when it finishes."""
        rel = self._rel_path(str(ev.args.get("path", "")))
        t = Text()
        if ev.status == "running" or not ev.result:
            t.append(" ⏺ ", style=f"bold {C['accent']}")
            if ev.name in ("run_command", "live_shell"):
                t.append("Ran command", style=f"bold {C['fg']}")
            elif ev.name == "write_file":
                t.append(f"Wrote {rel}", style=f"bold {C['fg']}")
            elif ev.name == "edit_file":
                t.append(f"Edited {rel}", style=f"bold {C['fg']}")
            elif ev.name == "read_file":
                t.append(f"Read {rel}", style=f"bold {C['fg']}")
            else:
                t.append("Applied patch", style=f"bold {C['fg']}")
            return t
        body = self._live_body_lines(ev)
        for idx, line in enumerate(body):
            t.append_text(line)
            if idx < len(body) - 1:
                t.append("\n")
        return t

    def _shell_footer(self, ev: ToolEvent) -> Text:
        """Closing line of a live-streamed shell block: the exit code (or
        the error), printed once the command finishes."""
        t = Text()
        t.append(" └ ", style=C["dim"])
        if ev.status != "done" or ev.result.startswith("ERROR"):
            msg = ev.result.splitlines()[0] if ev.result else ev.status
            t.append(msg[:200], style=C["red"])
            return t
        code, _, _ = self._parse_shell_result(ev.result)
        if code is not None:
            t.append(f"Exited with code {code}",
                     style=C["green"] if code == 0 else C["red"])
        else:
            t.append("done", style=C["green"])
        return t

    def _write_footer(self, ev: ToolEvent, nlines: int) -> Text:
        """Closing line of a live-streamed write block: the tool's receipt
        (e.g. 'OK: created … 286 line(s)'), printed once writing finishes."""
        t = Text()
        t.append(" └ ", style=C["dim"])
        if ev.status != "done" or ev.result.startswith("ERROR"):
            msg = ev.result.splitlines()[0] if ev.result else ev.status
            t.append(msg[:200], style=C["red"])
            return t
        first = ev.result.splitlines()[0] if ev.result else ""
        t.append(first[:200] if first else f"{nlines} line(s) written",
                 style=C["green"])
        return t

    def _subagent_panel(self, ev: ToolEvent) -> Panel:
        """Render a subagent tool result as a bordered panel."""
        title = {"wait_for_agents": "⚡ CREW RESULTS"}.get(ev.name,
                                                          "⚡ SUBAGENTS")
        color = C["green"] if ev.status == "done" else C["red"]
        body = Text(ev.result[:6000], style=C["fg"])
        return Panel(body, title=title, border_style=color, expand=False,
                     padding=(0, 1))

    def print_reasoning(self, text: str) -> None:
        preview = " ".join(text.strip().split())
        if len(preview) > 300:
            preview = preview[:297] + "…"
        t = Text()
        t.append("  💭 ", style=C["pink"])
        t.append(preview, style=f"italic {C['dim']}")
        self.console.print(t)

    def print_error(self, text: str) -> None:
        self.console.print(Panel(Text(text, style=C["red"]),
                                 border_style=C["red"], title="error",
                                 expand=False))

    def print_info(self, text: str, color: str | None = None) -> None:
        self.console.print(Text(text, style=color or C["cyan"]))

    def print_usage(self, turns) -> None:
        total_in = total_out = 0
        for t in turns:
            if t.usage:
                total_in += t.usage.get("prompt_tokens", 0) or 0
                total_out += t.usage.get("completion_tokens", 0) or 0
        t = Text()
        t.append(f" turns: {len(turns)}", style=C["fg"])
        t.append(f"   prompt tokens: {total_in}", style=C["dim"])
        t.append(f"   completion tokens: {total_out}", style=C["dim"])
        self.console.print(t)
