"""Configuration: providers, models, effort levels, paths."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "FullAgent"


def _pick_app_dir() -> Path:
    """Choose a WRITABLE home for app state — never a crash path.

    Order: $FULLAGENT_HOME, ~/.fullagent, <tmp>/fullagent-<uid>.
    Each candidate is probed with a real write; the first one that
    actually works wins, so a read-only home dir degrades gracefully
    instead of killing the app at startup (OSError on event log)."""
    import tempfile
    candidates: list[Path] = []
    env = os.environ.get("FULLAGENT_HOME")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.home() / ".fullagent")
    uid = str(os.getuid()) if hasattr(os, "getuid") else "user"
    candidates.append(Path(tempfile.gettempdir()) / f"fullagent-{uid}")
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".write-probe"
            probe.write_text("ok")
            probe.unlink()
            return c
        except OSError:
            continue
    # last resort: per-user tmp subdir so two users on the same box don't
    # clobber each other's config / event log / sessions
    fallback = Path(tempfile.gettempdir()) / f"fullagent-{uid}"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except OSError:
        return Path(tempfile.gettempdir())


APP_DIR = _pick_app_dir()
CONFIG_FILE = APP_DIR / "config.json"
HISTORY_FILE = APP_DIR / "history"
SESSIONS_DIR = APP_DIR / "sessions"
EVENT_LOG_FILE = APP_DIR / "eventlog.jsonl"

DEFAULT_TIMEOUT = 300.0
MAX_TOOL_ITERATIONS = 80
MAX_TOOL_OUTPUT_CHARS = 24_000
# One output ceiling for every effort level: 200k tokens.
MAX_TOKENS = 200_000
# Backends reject a request when input + max_tokens exceeds the model's
# context window. Every request's max_tokens is clamped to fit (client.py).
DEFAULT_CONTEXT_WINDOW = 262_144


@dataclass(frozen=True)
class Provider:
    key: str
    name: str
    base_url: str
    api_key: str
    color: str


@dataclass(frozen=True)
class Model:
    id: str
    provider: str
    label: str
    tag: str = ""
    tag_color: str = "grey62"
    supports_tools: bool = True
    supports_reasoning: bool = False
    # Total context window (input + output tokens). Used to clamp max_tokens
    # at send time so a request is never rejected for exceeding the window.
    context_window: int = DEFAULT_CONTEXT_WINDOW


PROVIDERS: dict[str, Provider] = {
    "zen": Provider(
        key="zen",
        name="OpenCode Zen",
        base_url="https://opencode.ai/zen/v1",
        api_key=os.environ.get(
            "OPENCODE_API_KEY",
            "sk-h11yU0O2sQxGL9CC0Y5bHQxtdWQSqXAi1mRUG7TSLpA7EvFAzYBpyAJ7NQ6xhDvm",
        ),
        color="#8be9fd",
    ),
    "opencode": Provider(
        key="opencode",
        name="OpenCode",
        base_url="https://opencode.ai/zen/v1",
        api_key=os.environ.get(
            "OPENCODE_API_KEY",
            "sk-h11yU0O2sQxGL9CC0Y5bHQxtdWQSqXAi1mRUG7TSLpA7EvFAzYBpyAJ7NQ6xhDvm",
        ),
        color="#bd93f9",
    ),
    "tokenrouter": Provider(
        key="tokenrouter",
        name="TokenRouter",
        base_url="https://api.tokenrouter.com/v1",
        api_key=os.environ.get(
            "TOKENROUTER_API_KEY",
            "sk-cTiHfKWRCDK6EO64AuloBS09hQGu06careTB2oQ9OETBe2wK",
        ),
        color="#ffb86c",
    ),
    "agnes": Provider(
        key="agnes",
        name="Agnes",
        base_url="https://apihub.agnes-ai.com/v1",
        api_key=os.environ.get(
            "AGNES_API_KEY",
            "sk-fKLLAlhfkYdwCMrznXi1rKlh3ZQXgNtucHrpPatC7MQCHYVi",
        ),
        color="#50fa7b",
    ),
    "zenmux": Provider(
        key="zenmux",
        name="ZenMux",
        base_url="https://zenmux.ai/api/v1",
        api_key=os.environ.get(
            "ZENMUX_API_KEY",
            "sk-ai-v1-9424a61af5fea4355a34de00530e189d1972da4d4f8324815be47b8d5a6280eb",
        ),
        color="#f1fa8c",
    ),
}

MODELS: list[Model] = [
    Model("mimo-v2.5-free", "zen", "MiMo v2.5", tag="FREE", tag_color="green",
          supports_tools=False),
    Model("big-pickle", "zen", "Big Pickle", tag="FREE", tag_color="green"),
    Model("grok-code-fast-1", "zen", "Grok Code Fast", tag="FAST", tag_color="cyan"),
    Model("claude-sonnet-4-5", "zen", "Claude Sonnet 4.5",
          context_window=200_000),
    Model("claude-opus-4-6", "zen", "Claude Opus 4.6",
          context_window=200_000),
    Model("gemini-3.1-pro", "zen", "Gemini 3.1 Pro",
          context_window=1_048_576),
    Model("gpt-5.2", "zen", "GPT-5.2", context_window=400_000),
    Model("muse-spark-1.2-contributor-free", "opencode",
          "Muse Spark 1.2", tag="FREE", tag_color="green",
          supports_tools=True, supports_reasoning=True),
    Model("opencode/muse-spark-1.2-contributor-free", "opencode",
          "Muse Spark 1.2", tag="FREE", tag_color="green",
          supports_tools=True, supports_reasoning=True),
    # supports_reasoning=True means the backend understands a reasoning
    # switch — the client uses it to send an EXPLICIT "none" (thinking
    # off globally), not to turn thinking on.
    Model("qwen/qwen3.8-max-free", "tokenrouter", "Qwen3.8 Max", tag="FREE",
          tag_color="green", supports_reasoning=True),
    Model("deepseek-ai/DeepSeek-V3.2", "tokenrouter", "DeepSeek V3.2",
          supports_reasoning=False, context_window=131_072),
    Model("deepseek/deepseek-v4-pro-0813-free", "tokenrouter",
          "DeepSeek V4 Pro 0813", tag="FREE", tag_color="green",
          supports_tools=True, supports_reasoning=True,
          context_window=131_072),
    Model("moonshotai/Kimi-K2-Instruct", "tokenrouter", "Kimi K2",
          context_window=131_072),
    Model("agnes-2.5-flash", "agnes", "Agnes 2.5 Flash", tag="FAST",
          tag_color="green", supports_tools=True, supports_reasoning=True),
    Model("dots-studio/dots3-note-prev", "zenmux", "Dots.OCR Note Prev",
          tag="NEW", tag_color="yellow", supports_tools=True,
          supports_reasoning=True),
]

DEFAULT_MODEL_ID = "mimo-v2.5-free"


@dataclass(frozen=True)
class Effort:
    key: str
    label: str
    color: str
    max_tokens: int | None
    temperature: float
    reasoning_effort: str | None
    description: str


EFFORTS: list[Effort] = [
    Effort("low", "LOW", "#6272a4", MAX_TOKENS, 0.2, None,
           "short answers, minimal tokens"),
    Effort("medium", "MEDIUM", "#8be9fd", MAX_TOKENS, 0.4, None,
           "balanced length and speed"),
    Effort("high", "HIGH", "#50fa7b", MAX_TOKENS, 0.6, None,
           "thorough, detailed answers"),
    Effort("extrahigh", "EXTRA HIGH", "#ffb86c", MAX_TOKENS, 0.7, None,
           "deep work, long outputs"),
    Effort("ultrahigh", "ULTRA HIGH", "#ff5555", MAX_TOKENS, 0.8, None,
           "maximum depth, exhaustive work"),
]

DEFAULT_EFFORT = "high"


def model_by_id(model_id: str) -> Model | None:
    for m in MODELS:
        if m.id == model_id:
            return m
    return None


def effort_by_key(key: str) -> Effort | None:
    for e in EFFORTS:
        if e.key == key:
            return e
    return None


@dataclass
class Config:
    model_id: str = DEFAULT_MODEL_ID
    effort: str = DEFAULT_EFFORT
    auto_approve: bool = False
    show_reasoning: bool = False
    theme: str = "dracula"
    # which system prompt to send: "main" (compact) or "master" (130k+)
    prompt: str = "main"
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        try:
            data = json.loads(CONFIG_FILE.read_text())
            if not isinstance(data, dict):
                data = {}
            for k in ("model_id", "effort", "auto_approve", "show_reasoning",
                      "theme", "prompt"):
                if k not in data:
                    continue
                if k in ("auto_approve", "show_reasoning"):
                    # safety gates must be real booleans — a drifted
                    # config with "auto_approve": "false" (truthy string)
                    # would silently disable the approval prompt
                    if isinstance(data[k], bool):
                        setattr(cfg, k, data[k])
                else:
                    setattr(cfg, k, data[k])
            cfg.extra = {k: v for k, v in data.items()
                         if k not in ("model_id", "effort", "auto_approve",
                                      "show_reasoning", "theme", "prompt")}
        except (OSError, ValueError):
            pass
        if model_by_id(cfg.model_id) is None:
            cfg.model_id = DEFAULT_MODEL_ID
        if effort_by_key(cfg.effort) is None:
            cfg.effort = DEFAULT_EFFORT
        if not isinstance(cfg.prompt, str) or not cfg.prompt:
            cfg.prompt = "main"
        return cfg

    def save(self) -> None:
        try:
            ensure_dirs()
            data = {
                "model_id": self.model_id,
                "effort": self.effort,
                "auto_approve": self.auto_approve,
                "show_reasoning": self.show_reasoning,
                "theme": self.theme,
                "prompt": self.prompt,
            }
            data.update(self.extra)
            # atomic write: a crash mid-write must never leave truncated
            # JSON that would reset the whole config on next load
            tmp = CONFIG_FILE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, CONFIG_FILE)
        except OSError:
            pass  # config persistence is a convenience, never a crash path


def ensure_dirs() -> None:
    """Create every directory the app writes into. Called at startup AND
    before individual writes, so a deleted home dir heals itself."""
    for d in (APP_DIR, SESSIONS_DIR, APP_DIR / "memory",
              APP_DIR / "skills", APP_DIR / "store"):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
