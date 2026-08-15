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

import shutil
import sys
import threading
import time
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
    ("/state", "live projection of the event log (cost, goal, dead-ends)"),
    ("/rewind", "rewind timeline + files to a seq — /rewind <seq>"),
    ("/revert", "revert FILES only to a seq (agent keeps memory)"),
    ("/fork", "fork the timeline into a new branch — /fork [name]"),
    ("/verify", "verify the event log's Merkle spine"),
    ("/why", "causal chain for an event — /why <seq>"),
    ("/impact", "code impact analysis — /impact <symbol> [path]"),
    ("/forge", "environment digest + drift — /forge [probe|drift]"),
    ("/oracle", "post-run analysis, calibration, facts"),
    ("/budget", "budget governor status"),
    ("/constitution", "standing rules — /constitution [show|edit]"),
    ("/replay", "replay the session log as a film (text)"),
    ("/memory", "recent episodes + dead-end ledger"),
    ("/judge", "deterministic check — /judge <type> <json-or-args>"),
    ("/scout", "fan out read-only scouts — /scout q1 | q2 | q3"),
    ("/team", "parallel worker team (up to 8) — /team task1 | task2 | …"),
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
        self._stream_lock = threading.Lock()
        self._last_flush = 0.0

        # approval state
        self._approve_request: tuple[Tool, dict, threading.Event] | None = None
        self._approve_result = False

        # flash message in the bottom border
        self._flash: tuple[str, str] | None = None
        self._flash_timer: threading.Timer | None = None

        # goal-status cache for the border (fold is not free per frame)
        self._goal_cache = None
        self._goal_cache_ts = 0.0

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
        try:
            cols = get_app().output.get_size().columns
        except Exception:
            cols = shutil.get_terminal_size((100, 24)).columns
        return max(60, cols)

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
            max_status = max(0, inner - len(" ⠹  ·  Ctrl+C cancel ") - 4)
            status = self._status_text[:max_status]
            bar = f" {frame} {status}  ·  Ctrl+C cancel "
            return [("class:box", "╰"),
                    ("class:box.spinner", f" {frame} "),
                    ("class:box.status", status),
                    ("class:box.hint", "  ·  Ctrl+C cancel "),
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
            history=FileHistory(str(config.HISTORY_FILE)),
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
        @kb.add(Keys.Enter, filter=focused & ~ov & idle)
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

    # -- dispatch: slash commands + turns ----------------------------------------------------

    def _dispatch(self, text: str) -> None:
        if text.startswith("/"):
            self._handle_slash(text)
            return
        self._emit_user(text)
        threading.Thread(target=self._run_turn_thread, args=(text,),
                         daemon=True).start()

    def _handle_slash(self, text: str) -> None:
        parts = text.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

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
        elif cmd == "/state":
            self._cmd_state()
        elif cmd == "/rewind":
            self._cmd_rewind(arg)
        elif cmd == "/revert":
            self._cmd_revert(arg)
        elif cmd == "/fork":
            self._cmd_fork(arg)
        elif cmd == "/verify":
            ok, msg = self.agent.log.verify()
            self.print_info(f"{'✓' if ok else '✗'} {msg}",
                            C["green"] if ok else C["red"])
        elif cmd == "/why":
            self._cmd_why(arg)
        elif cmd == "/impact":
            self._cmd_impact(arg)
        elif cmd == "/forge":
            self._cmd_forge(arg)
        elif cmd == "/oracle":
            self.print_info(self.agent.oracle.format_report(), C["cyan"])
        elif cmd == "/budget":
            self._cmd_budget()
        elif cmd == "/constitution":
            self._cmd_constitution(arg)
        elif cmd == "/replay":
            self._cmd_replay()
        elif cmd == "/memory":
            self._cmd_memory()
        elif cmd == "/judge":
            self._cmd_judge(arg)
        elif cmd == "/scout":
            self._cmd_scout(arg)
        elif cmd == "/team":
            self._cmd_team(arg)
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
        if st.swarm_reports:
            lines.append(f"scout reports: {len(st.swarm_reports)}")
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

        threading.Thread(target=run, daemon=True).start()

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

    def _cmd_budget(self) -> None:
        gov = self.agent.budget_gov
        s = gov.spend()
        b = gov.budget
        ok, reason = gov.check()
        lines = [f"budget {'OK' if ok else 'BREACHED'}",
                 f"  usd    ${s['usd']:.4f} / ${b.max_usd}",
                 f"  steps  {s['steps']} / {b.max_steps}",
                 f"  tokens {s['tokens']} / {b.max_tokens}",
                 f"  files  {s['files']} / {b.max_files}"]
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

        threading.Thread(target=run, daemon=True).start()

    def _cmd_scout(self, arg: str) -> None:
        questions = [q.strip() for q in arg.split("|") if q.strip()]
        if not questions:
            self.print_error("usage: /scout <question1> | <question2> | …")
            return
        self.print_info(f"⠹ scouting {len(questions)} question(s)…", C["dim"])

        def run():
            reports = self.agent.swarm.scout(
                questions, context=self.agent.scout_context())
            self.console.print(Text(self.agent.swarm.format(reports),
                                    style=C["fg"]))

        threading.Thread(target=run, daemon=True).start()

    def _cmd_team(self, arg: str) -> None:
        """Dispatch a parallel worker team (up to 8) on explicit tasks.

        /team <task1> | <task2> | …   — each task gets its own worker.
        Prefix a task with 'role:' to pin a role, e.g.
        'researcher: find the latest flask version'."""
        from .team import MAX_WORKERS, ROLES
        tasks: list[dict] = []
        for piece in (p.strip() for p in arg.split("|")):
            if not piece:
                continue
            role = None
            for r in ROLES:
                if piece.lower().startswith(r + ":"):
                    role = r
                    piece = piece[len(r) + 1:].strip()
                    break
            if piece:
                tasks.append({"task": piece, "role": role})
        if not tasks:
            self.print_error("usage: /team <task1> | <task2> | …  "
                             "(up to 8, optional 'role:' prefix)")
            return
        tasks = tasks[:MAX_WORKERS]
        self.print_info(f"⚡ dispatching {len(tasks)} worker(s) in parallel…",
                        C["pink"])

        def run():
            reports = self.agent.team.run(
                tasks, context=self.agent.scout_context(),
                read_only=self.agent.autonomy <= 1)
            self.console.print(Text(self.agent.team.format(reports),
                                    style=C["fg"]))

        threading.Thread(target=run, daemon=True).start()

    def _cmd_auto(self, arg: str) -> None:
        """Toggle or inspect the AutoPilot self-routing brain."""
        sub = arg.strip().lower()
        ap = self.agent.autopilot
        if sub == "on":
            ap.enabled = True
            self.print_info("✓ autopilot ON — the agent will auto-enable "
                            "parallel team / goal mode / real-time web as "
                            "each turn needs them", C["green"])
        elif sub == "off":
            ap.enabled = False
            self.print_info("autopilot OFF — manual control only", C["dim"])
        elif sub in ("", "status"):
            state = "ON" if ap.enabled else "OFF"
            self.print_info(f"autopilot: {state}\n"
                            "  auto-enables, per turn:\n"
                            "  ⚡ parallel team  — independent subtasks "
                            "detected (up to 8 workers)\n"
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
        streamed = {"n": 0}
        stream_buf: list[str] = []

        def on_token(piece: str):
            streamed["n"] += len(piece)
            # patch_stdout can only interleave output safely when every
            # write ends in a newline, so emit complete lines here and keep
            # the partial line as a live preview inside the box border.
            with self._stream_lock:
                stream_buf.append(piece)
                joined = "".join(stream_buf)
                if "\n" in joined:
                    before, _, rem = joined.rpartition("\n")
                    self.console.print(Text(before), soft_wrap=True)
                    stream_buf[:] = [rem]
                    joined = rem
                maxw = self._width() - 26
                preview = joined.strip("\n")
                if len(preview) > maxw:
                    preview = preview[-maxw:]
                self._set_status(preview if preview else "writing…")

        def on_reasoning(piece: str):
            self._set_status("reasoning…")

        def on_tool_call(ev: ToolEvent):
            with self._stream_lock:
                rem = "".join(stream_buf).strip("\n")
                stream_buf.clear()
                if rem:
                    self.console.print(Text(rem), soft_wrap=True)
            self.console.print(self._tool_call_line(ev))
            self._set_status(f"running {ev.name}…")

        def on_tool_update(ev: ToolEvent):
            self.console.print(self._tool_result_line(ev))
            self._set_status("thinking…")

        def on_status(s: str):
            if s == "thinking":
                self._set_status("thinking…")
            elif s.startswith("tool:"):
                self._set_status(f"calling {s[5:]}…")
            elif s.startswith("running:"):
                self._set_status(f"running {s[8:]}…")
            elif s.startswith("team:"):
                self._set_status(f"⚡ {s[5:]} workers running in parallel…")
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
                on_route=on_route)
        except Exception as e:  # noqa: BLE001 — never kill the UI thread
            self.print_error(f"{type(e).__name__}: {e}")
        finally:
            with self._stream_lock:
                rem = "".join(stream_buf).strip("\n")
                stream_buf.clear()
                if rem:
                    self.console.print(Text(rem), soft_wrap=True)
                else:
                    self.console.print()
            self._stop_spinner()
            self._busy = False
            self._set_status("")
            self._invalidate()

        if turn is not None:
            if turn.reasoning and self.cfg.show_reasoning:
                self.print_reasoning(turn.reasoning)
            if turn.error:
                if turn.error == "cancelled":
                    self.print_info("⊘ cancelled", C["yellow"])
                else:
                    self.print_error(turn.error)
            self._print_turn_stats(turn)
            self.agent.save_session()

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
        self._spinner_on = True

        def tick():
            while self._spinner_on:
                self._spinner_i = (self._spinner_i + 1) % len(SPINNER_FRAMES)
                self._invalidate()
                time.sleep(0.09)

        threading.Thread(target=tick, daemon=True).start()

    def _stop_spinner(self) -> None:
        self._spinner_on = False

    # -- approval (in-app) ------------------------------------------------------------------------

    def _approve_blocking(self, tool: Tool, args: dict) -> bool:
        if self.cfg.auto_approve:
            return True
        done = threading.Event()
        self._approve_result = False
        self._approve_request = (tool, args, done)
        self._invalidate()
        # show what is being approved above the box
        self.console.print(self._approval_line(tool, args))
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
            row("/budget", "budget governor status"),
            row("/constitution", "standing rules (human-owned)"),
            row("/replay", "replay the session log as a film"),
            row("/memory", "episodes + dead-end ledger"),
            row("/judge", "deterministic check (exit_code, file_exists, …)"),
            row("/scout", "parallel read-only scouts — /scout q1 | q2"),
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
            preview = t.user_text.replace("\n", " ")[:70]
            html = (f'<style color="{C["dim"]}">{t.timestamp}</style> '
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
        self.agent.save_session()

    def print_banner(self) -> None:
        width = min(shutil.get_terminal_size((100, 24)).columns - 2, 78)
        logo = Text()
        logo.append("◆ ", style=f"bold {C['accent']}")
        logo.append(APP_NAME, style=f"bold {C['accent']}")
        logo.append("  ·  advanced terminal AI agent", style=C["dim"])
        self.console.print(Panel(logo, width=width, border_style=C["border"],
                                 padding=(0, 1)))
        model = self._model()
        effort = self._effort()
        line = Text()
        line.append(" model ", style=C["dim"])
        line.append(model.label, style=f"bold {C['cyan']}")
        line.append(f" ({model.id})", style=C["dim"])
        line.append("   effort ", style=C["dim"])
        line.append(effort.label, style=f"bold {EFFORT_COLORS[effort.key]}")
        line.append(f"   session {self.agent.session_id}", style=C["dim"])
        self.console.print(line)
        self.console.print(Text(
            " type / for commands · Ctrl+T models · Ctrl+E effort",
            style=C["dim"]))
        self.console.print()

    def _emit_user(self, text: str) -> None:
        t = Text()
        t.append("❯ ", style=f"bold {C['green']}")
        t.append(text, style=f"bold {C['fg']}")
        self.console.print(t)

    def _tool_call_line(self, ev: ToolEvent) -> Text:
        shown = {}
        for k, v in ev.args.items():
            s = str(v)
            shown[k] = s if len(s) <= 80 else s[:77] + "…"
        arg_str = "  ".join(f"{k}={v!r}" for k, v in shown.items())
        if len(arg_str) > 120:
            arg_str = arg_str[:117] + "…"
        t = Text()
        t.append("  ⚙ ", style=C["orange"])
        t.append(ev.name, style=f"bold {C['orange']}")
        if arg_str:
            t.append(f"  {arg_str}", style=C["dim"])
        return t

    def _tool_result_line(self, ev: ToolEvent) -> Text:
        icon = {"done": "✓", "error": "✗", "denied": "⊘",
                "blocked": "⛔"}.get(ev.status, "·")
        color = {"done": C["green"], "error": C["red"],
                 "denied": C["yellow"], "blocked": C["red"]}.get(
                     ev.status, C["dim"])
        first_line = ev.result.splitlines()[0] if ev.result else ""
        if len(first_line) > 100:
            first_line = first_line[:97] + "…"
        t = Text()
        t.append(f"  {icon} ", style=color)
        t.append(first_line, style=C["dim"])
        t.append(f"  ({ev.duration:.1f}s)", style=C["dim"])
        return t

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
