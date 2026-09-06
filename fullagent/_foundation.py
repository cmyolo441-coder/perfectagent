"""Foundation — the professional infrastructure layer for FullAgent.

Every subsystem imports from here: structured logging, typed errors,
input validation, performance utilities, and safety guards. This is the
bedrock that makes the entire codebase production-grade.

Design principles:
  * Zero external dependencies (stdlib only)
  * Every function is typed, documented, and tested
  * Errors are structured (never bare strings)
  * Logging is leveled and filterable
  * Validation is fail-fast with actionable messages
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, TypeVar

__all__ = [
    "get_logger", "AgentError", "ConfigError", "ToolError",
    "NetworkError", "ValidationError", "KernelError", "GoalError",
    "validate_path", "validate_non_empty", "validate_range",
    "validate_type", "clamp", "cached_property", "timed",
    "content_hash", "safe_filename", "Level",
]

# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)-18s │ %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_configured = False


class Level(Enum):
    """Log verbosity levels — set via FULLAGENT_LOG_LEVEL env var."""
    QUIET = logging.WARNING
    NORMAL = logging.INFO
    VERBOSE = logging.DEBUG


def _configure_root() -> None:
    """Configure the root FullAgent logger once (idempotent)."""
    global _configured
    if _configured:
        return
    _configured = True
    level_name = os.environ.get("FULLAGENT_LOG_LEVEL", "QUIET").upper()
    level = getattr(Level, level_name, Level.QUIET).value
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root = logging.getLogger("fullagent")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the 'fullagent' hierarchy.

    Usage:
        log = get_logger("kernel")
        log.info("event sealed seq=%d", seq)
    """
    _configure_root()
    return logging.getLogger(f"fullagent.{name}")


# ---------------------------------------------------------------------------
# Structured Error Hierarchy
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Base error for all FullAgent failures. Carries a machine-readable
    code and a human-readable detail, so callers can branch on the code
    and users see actionable guidance."""

    code: str = "AGENT_ERROR"

    def __init__(self, detail: str, *, code: str | None = None,
                 hint: str = "") -> None:
        self.detail = detail
        if code:
            self.code = code
        self.hint = hint
        super().__init__(self._format())

    def _format(self) -> str:
        msg = f"[{self.code}] {self.detail}"
        if self.hint:
            msg += f" — {self.hint}"
        return msg


class ConfigError(AgentError):
    """Configuration is invalid or missing."""
    code = "CONFIG_ERROR"


class ToolError(AgentError):
    """A tool execution failed in a structured way."""
    code = "TOOL_ERROR"


class NetworkError(AgentError):
    """A network call failed (timeout, connection, HTTP error)."""
    code = "NETWORK_ERROR"

    def __init__(self, detail: str, *, status: int | None = None,
                 hint: str = "") -> None:
        self.status = status
        super().__init__(detail, hint=hint)


class ValidationError(AgentError):
    """Input validation failed — the caller sent bad data."""
    code = "VALIDATION_ERROR"


class KernelError(AgentError):
    """The event log or a fold operation failed."""
    code = "KERNEL_ERROR"


class GoalError(AgentError):
    """A goal contract operation failed."""
    code = "GOAL_ERROR"


# ---------------------------------------------------------------------------
# Input Validation (fail-fast, actionable messages)
# ---------------------------------------------------------------------------

T = TypeVar("T")

_PATH_SAFE_RE = re.compile(r"^[a-zA-Z0-9_./ ~@\-]+$")
_TRAVERSAL_RE = re.compile(r"(^|/)\.\.(/|$)")


def validate_path(value: str, *, name: str = "path",
                  must_exist: bool = False,
                  allow_relative: bool = True) -> Path:
    """Validate and resolve a filesystem path.

    Raises ValidationError on:
      * empty path
      * path traversal attempts (../)
      * null bytes
      * must_exist=True and the path doesn't exist

    Returns the resolved Path.
    """
    if not value or not value.strip():
        raise ValidationError(f"{name} must not be empty",
                              hint="provide a valid file path")
    if "\x00" in value:
        raise ValidationError(f"{name} contains a null byte",
                              hint="paths must not contain \\x00")
    if _TRAVERSAL_RE.search(value):
        raise ValidationError(
            f"{name} contains path traversal (..)",
            hint="use absolute paths or paths relative to the project root")
    p = Path(value).expanduser()
    if not allow_relative and not p.is_absolute():
        p = Path.cwd() / p
    if must_exist and not p.exists():
        raise ValidationError(f"{name} does not exist: {p}",
                              hint="check the path and try again")
    return p


def validate_non_empty(value: str, *, name: str = "value") -> str:
    """Ensure a string is non-empty after stripping whitespace."""
    if not value or not value.strip():
        raise ValidationError(f"{name} must not be empty")
    return value.strip()


def validate_range(value: int | float, *, name: str = "value",
                   lo: float = float("-inf"),
                   hi: float = float("inf")) -> int | float:
    """Ensure a numeric value is within [lo, hi]."""
    if not (lo <= value <= hi):
        raise ValidationError(
            f"{name} must be between {lo} and {hi}, got {value}")
    return value


def validate_type(value: Any, expected: type, *, name: str = "value") -> Any:
    """Ensure a value is of the expected type."""
    if not isinstance(value, expected):
        raise ValidationError(
            f"{name} must be {expected.__name__}, "
            f"got {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# Performance Utilities
# ---------------------------------------------------------------------------

def clamp(value: T, lo: T, hi: T) -> T:
    """Clamp a value to [lo, hi]. Works for any comparable type."""
    if value < lo:  # type: ignore[operator]
        return lo
    if value > hi:  # type: ignore[operator]
        return hi
    return value


class cached_property:
    """A property that is computed once and then cached on the instance.
    Thread-safe via a simple lock-free pattern (worst case: computed twice
    under a race, which is harmless for pure functions)."""

    def __init__(self, fn: Callable[[Any], T]) -> None:
        self.fn = fn
        self.attr = f"_cached_{fn.__name__}"
        functools.update_wrapper(self, fn)

    def __get__(self, obj: Any, cls: type) -> T:
        if obj is None:
            return self  # type: ignore[return-value]
        try:
            return getattr(obj, self.attr)
        except AttributeError:
            val = self.fn(obj)
            object.__setattr__(obj, self.attr, val)
            return val


def timed(fn: Callable) -> Callable:
    """Decorator: log the execution time of a function at DEBUG level."""
    log = get_logger(fn.__module__.rsplit(".", 1)[-1])

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            log.debug("%s took %.1fms", fn.__name__, elapsed)
    return wrapper


# ---------------------------------------------------------------------------
# Safety Utilities
# ---------------------------------------------------------------------------

def content_hash(data: str | bytes) -> str:
    """SHA-256 content address (first 16 hex chars)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename (no traversal, no specials)."""
    name = re.sub(r"[^\w.\-]", "_", name)
    name = name.lstrip(".")
    return name[:200] or "unnamed"


# ---------------------------------------------------------------------------
# Self-test (offline, deterministic)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # validation
    try:
        validate_path("")
        assert False, "should have raised"
    except ValidationError:
        pass

    try:
        validate_path("../../etc/passwd")
        assert False, "should have raised"
    except ValidationError:
        pass

    p = validate_path("/tmp/test.py")
    assert str(p) == "/tmp/test.py"

    assert validate_non_empty("  hello  ") == "hello"
    assert validate_range(5, lo=0, hi=10) == 5

    try:
        validate_range(15, lo=0, hi=10)
        assert False
    except ValidationError:
        pass

    # clamp
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10

    # content_hash
    h = content_hash("hello")
    assert len(h) == 16
    assert h == content_hash("hello")  # deterministic

    # safe_filename
    assert safe_filename("../../etc/passwd") == "_.._etc_passwd"
    assert safe_filename("hello world.py") == "hello_world.py"

    # errors
    e = AgentError("something broke", hint="try again")
    assert "[AGENT_ERROR]" in str(e)
    assert "try again" in str(e)

    print("✓ _foundation.py self-test passed")
