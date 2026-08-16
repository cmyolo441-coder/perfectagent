"""OpenAI-compatible streaming chat client (requests + manual SSE parsing)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator

import requests

from . import config
from .config import Effort, Model, Provider

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 3

# TokenRouter's free tier admits a request only when its prompt cache is
# warm; a cold or overloaded one is rejected with "cache-only admission
# rejected". This is transient — backing off and retrying succeeds once the
# cache warms up (the CLIs that "just work" retry implicitly). We give cold
# rejections their own, longer retry budget, separate from MAX_RETRIES.
_COLD_ADMISSION_RE = re.compile(
    r"cache-only admission|cold or overloaded", re.I)
MAX_COLD_RETRIES = 4
COLD_RETRY_WAIT = 8.0  # seconds; multiplied by the retry number


def is_cold_admission(message: str) -> bool:
    return bool(_COLD_ADMISSION_RE.search(message or ""))


class APIError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class TurnCancelled(Exception):
    """Raised inside the stream loop when the user cancels (Ctrl+C)."""


@dataclass
class ToolCallDelta:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict | None = None
    model: str = ""


def build_payload(model: Model, effort: Effort, messages: list[dict],
                  tools: list[dict] | None, stream: bool = True) -> dict:
    payload: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": stream,
        "temperature": effort.temperature,
    }
    if effort.max_tokens:
        # The request must fit in the window: input + max_tokens <= window,
        # otherwise the backend rejects it wholesale.
        windowed = _window_max_tokens(model, effort, messages, tools)
        payload["max_tokens"] = _clamp_max_tokens(model.provider, windowed)
        # HARD INVARIANT — the last mechanical gate before the wire. No
        # request may leave with estimated input + max_tokens over the
        # window. If any earlier layer drifted, clamp again here rather
        # than send a doomed request.
        window = effective_window(model)
        est_input = estimate_tokens(messages, model.id)
        if tools and model.supports_tools:
            est_input += estimate_tokens(tools, model.id)
        if est_input + payload["max_tokens"] > window:
            payload["max_tokens"] = max(
                _MIN_COMPLETION_TOKENS, window - est_input - CONTEXT_MARGIN)
    if tools and model.supports_tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if model.supports_reasoning and effort.reasoning_effort:
        payload["reasoning_effort"] = _normalize_reasoning_effort(
            model.provider, effort.reasoning_effort)
    return payload


def estimate_tokens(obj: Any, model_id: str = "") -> int:
    """Deterministic token estimate for any JSON-serialisable object.

    Starts from a conservative ~3.2 chars/token baseline, then corrects
    itself with the REAL chars/token ratio learned from every backend
    response (usage.prompt_tokens vs the prompt we actually sent). The
    backend's own tokenizer is the ground truth — once a few responses
    have landed, this estimate tracks reality instead of a fixed guess,
    which is what keeps long sessions from ever overflowing the window.

    Calibration is PER MODEL (tokenizers differ between models) and is
    persisted to disk, so the very first request of a fresh process on a
    big project already uses the learned ratio."""
    try:
        payload = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = str(obj)
    return max(1, int(len(payload) / _chars_per_token(model_id)))


# -- learned tokenizer calibration ------------------------------------------
# The backend rejects a request when input + max_tokens exceeds the model's
# context window, and it counts input with ITS OWN tokenizer. A fixed
# chars/token guess always drifts from reality (code, unicode, tool schemas
# all tokenize differently), so we learn the real ratio from every response
# and keep a safety margin on top.
#
# Enterprise hardening:
#   * calibration is keyed by model id — switching models mid-session can
#     never poison the estimate (each model has its own tokenizer);
#   * calibration AND learned context windows persist to disk, so a restart
#     on a huge project starts already calibrated;
#   * the context window itself is learned: when a backend error reports a
#     smaller window than configured, we remember it for that model.

_BASELINE_CHARS_PER_TOKEN = 3.2   # conservative start (real code is denser)
_MIN_CHARS_PER_TOKEN = 2.0        # never assume text is cheaper than this
_RATIO_SAMPLES_MAX = 8            # rolling window of recent measurements
_ratio_samples: dict[str, list[float]] = {}   # model id -> measured ratios
_learned_windows: dict[str, int] = {}         # model id -> real window
_calibration_loaded = False
_CALIBRATION_FILE = config.APP_DIR / "calibration.json"


def _cal_key(model_id: str) -> str:
    """Calibration bucket key. Unknown/empty model ids share one bucket."""
    return model_id or "_default"


def _load_calibration() -> None:
    """Restore persisted calibration once per process (best effort)."""
    global _calibration_loaded
    if _calibration_loaded:
        return
    _calibration_loaded = True
    try:
        data = json.loads(_CALIBRATION_FILE.read_text())
    except (OSError, ValueError):
        return
    ratios = data.get("ratios") or {}
    for key, samples in ratios.items():
        if isinstance(samples, list):
            clean = [float(s) for s in samples
                     if isinstance(s, (int, float))
                     and _MIN_CHARS_PER_TOKEN * 0.5 <= float(s) <= 16.0]
            if clean:
                _ratio_samples[key] = clean[-_RATIO_SAMPLES_MAX:]
    windows = data.get("windows") or {}
    for key, win in windows.items():
        if isinstance(win, int) and win > 0:
            _learned_windows[key] = win


def _save_calibration() -> None:
    """Persist calibration so the next process starts already tuned."""
    try:
        config.ensure_dirs()
        _CALIBRATION_FILE.write_text(json.dumps({
            "ratios": _ratio_samples,
            "windows": _learned_windows,
        }))
    except OSError:
        pass  # persistence is an optimisation, never a failure path


def _chars_per_token(model_id: str = "") -> float:
    """Current best chars/token for this model: the mean of its recent
    real measurements when we have any, else the conservative baseline."""
    _load_calibration()
    samples = _ratio_samples.get(_cal_key(model_id)) \
        or _ratio_samples.get("_default")
    if not samples:
        return _BASELINE_CHARS_PER_TOKEN
    return sum(samples) / len(samples)


def learn_token_ratio(sent_chars: int, actual_tokens: int,
                      model_id: str = "") -> None:
    """Record one real (chars, tokens) measurement from a backend response.
    Called after every completion whose usage reports prompt_tokens."""
    if sent_chars <= 0 or actual_tokens <= 0:
        return
    ratio = sent_chars / actual_tokens
    # ignore implausible outliers (broken usage reporting)
    if not (_MIN_CHARS_PER_TOKEN * 0.5 <= ratio <= 16.0):
        return
    key = _cal_key(model_id)
    samples = _ratio_samples.setdefault(key, [])
    samples.append(ratio)
    if len(samples) > _RATIO_SAMPLES_MAX:
        del samples[: len(samples) - _RATIO_SAMPLES_MAX]
    _save_calibration()


def learn_context_window(model_id: str, window: int) -> None:
    """Remember the real context window a backend reported for a model.
    We only ever shrink the configured window — a backend that reports a
    smaller window is authoritative for that deployment."""
    if window <= 0 or not model_id:
        return
    key = _cal_key(model_id)
    known = _learned_windows.get(key)
    if known is None or window < known:
        _learned_windows[key] = window
        _save_calibration()


def effective_window(model: Model) -> int:
    """The context window to plan against: the configured value, capped by
    anything the backend has actually reported for this model."""
    _load_calibration()
    learned = _learned_windows.get(_cal_key(model.id))
    if learned is None:
        return model.context_window
    return min(model.context_window, learned)


def _prompt_chars(messages: list[dict], tools: list[dict] | None) -> int:
    """Character count of exactly what we send as the prompt."""
    try:
        payload = json.dumps({"messages": messages, "tools": tools or []},
                             ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = str(messages) + str(tools or [])
    return len(payload)


# Safety headroom between the estimated input and the window. Scales with
# input size: the bigger the prompt, the bigger the absolute tokenizer
# error can be, so the margin grows instead of staying a fixed constant.
CONTEXT_MARGIN = 8_192
_MIN_COMPLETION_TOKENS = 1_024    # never clamp max_tokens below this


def _window_max_tokens(model: Model, effort: Effort, messages: list[dict],
                       tools: list[dict] | None) -> int:
    """Clamp the requested max_tokens so that input + max_tokens fits in
    the model's context window. Backends reject the whole request when
    the sum exceeds the window (e.g. 'maximum context length of 262144
    tokens'). The input size is estimated with the learned per-model
    chars/token ratio (see estimate_tokens) plus a size-scaled margin, so
    the clamp stays correct even as the conversation grows for hours. If
    the input alone overflows the window, raise a clear, recoverable
    error instead of sending a doomed request."""
    requested = effort.max_tokens or 0
    if not requested:
        return 0
    window = effective_window(model)
    input_tokens = estimate_tokens(messages, model.id)
    if tools and model.supports_tools:
        input_tokens += estimate_tokens(tools, model.id)
    # margin grows with the prompt: ~1 extra token of headroom per 32
    # estimated input tokens, on top of the fixed floor
    margin = CONTEXT_MARGIN + input_tokens // 32
    if input_tokens + margin + _MIN_COMPLETION_TOKENS > window:
        raise APIError(
            f"conversation is too large for {model.label} "
            f"(~{input_tokens:,} input tokens vs {window:,} "
            f"context window) — start a new session (/new), rewind "
            f"(/rewind), or switch to a larger-context model (Ctrl+T)")
    headroom = window - input_tokens - margin
    return max(_MIN_COMPLETION_TOKENS, min(requested, headroom))


# FullAgent asks for 200k output tokens everywhere, but each backend has its
# own hard ceiling. Clamp at send time so the request is never rejected for
# an oversized max_tokens; providers without a known cap pass through.
_MAX_TOKENS_CAP: dict[str, int] = {
    "agnes": 65_536,   # sglang backend: "max_tokens exceeds the limit of 65536"
}


def _clamp_max_tokens(provider_key: str, value: int) -> int:
    cap = _MAX_TOKENS_CAP.get(provider_key)
    if cap is None:
        return value
    return min(value, cap)


# Providers accept different reasoning-effort vocabularies. Map our internal
# levels onto what each backend actually accepts; unknown providers pass
# through unchanged.
_REASONING_EFFORT_MAP: dict[str, dict[str, str]] = {
    # sglang-backed Agnes: none | low | medium | high | max
    "agnes": {"low": "low", "medium": "medium", "xhigh": "max"},
}


def _normalize_reasoning_effort(provider_key: str, value: str) -> str:
    mapping = _REASONING_EFFORT_MAP.get(provider_key)
    if mapping is None:
        return value
    return mapping.get(value, value)


def _iter_sse_events(resp: requests.Response) -> Iterator[dict]:
    """Yield parsed JSON data objects from an SSE stream."""
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if raw_line.startswith(":"):  # comment / keepalive
            continue
        if not raw_line.startswith("data:"):
            continue
        data = raw_line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


def _extract_error_message(body: str) -> str:
    try:
        obj = json.loads(body)
        err = obj.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if isinstance(err, str):
            return err
        return body[:500]
    except ValueError:
        return body[:500]


# -- context-overflow detection + self-healing ------------------------------
# Backends reject a request outright when input + max_tokens exceeds the
# context window, and the error message carries the REAL token counts
# (e.g. "maximum context length of 262144 tokens ... 67440 tokens from the
# input messages and 195680 tokens for the completion"). We parse those
# numbers, learn the true chars/token ratio from them, re-clamp max_tokens
# to the actual headroom, and retry — so a long session heals itself
# instead of dying with the error.

_OVERFLOW_MARKERS = ("context length", "context_length", "context window",
                     "too many tokens", "maximum context",
                     "reduce the number of tokens")


def is_context_overflow(message: str) -> bool:
    """True when an API error message is a context-window overflow."""
    low = message.lower()
    return any(m in low for m in _OVERFLOW_MARKERS)


def _parse_overflow(message: str) -> dict | None:
    """Extract real token counts from a backend overflow error. Returns a
    dict with any of: window, input_tokens, completion_tokens, total."""
    if not is_context_overflow(message):
        return None
    low = message.lower()
    info: dict[str, int] = {}

    def _num(pattern: str) -> int | None:
        m = re.search(pattern, low)
        return int(m.group(1).replace(",", "")) if m else None

    window = (_num(r"maximum context length (?:of|is)\s+([\d,]+)\s+tokens")
              or _num(r"context (?:length|window) (?:of|is)\s+([\d,]+)\s+tokens"))
    if window:
        info["window"] = window
    inp = (_num(r"([\d,]+)\s+tokens?\s+from the input")
           or _num(r"([\d,]+)\s+tokens?\s+(?:in|from)\s+(?:the\s+)?(?:input|prompt|messages)")
           or _num(r"(?:input|prompt|messages)\s+(?:resulted in|is|are)\s+([\d,]+)\s+tokens"))
    if inp:
        info["input_tokens"] = inp
    comp = _num(r"([\d,]+)\s+tokens?\s+for (?:the )?completion")
    if comp:
        info["completion_tokens"] = comp
    total = (_num(r"a total of\s+([\d,]+)\s+tokens")
             or _num(r"requested (?:a total of )?([\d,]+)\s+tokens"))
    if total:
        info["total"] = total
    return info or {}


def _fit_max_tokens_from_actual(model: Model, info: dict,
                                effort: Effort) -> int | None:
    """max_tokens that provably fits, computed from the backend's OWN
    counts. None means the input alone overflows — the caller must shrink
    the conversation, not the completion budget."""
    if info.get("window"):
        learn_context_window(model.id, info["window"])
    window = effective_window(model)
    actual_input = info.get("input_tokens")
    if not actual_input:
        total, comp = info.get("total"), info.get("completion_tokens")
        if total and comp:
            actual_input = total - comp
    if not actual_input:
        return None
    margin = max(4_096, window // 64)
    headroom = window - actual_input - margin
    if headroom < _MIN_COMPLETION_TOKENS:
        return None
    requested = effort.max_tokens or headroom
    return min(requested, headroom)


def _learn_from_usage(usage: dict | None, sent_chars: int,
                     model_id: str = "") -> None:
    """Calibrate the chars/token estimator with the backend's real count."""
    if not usage:
        return
    try:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
    except (TypeError, ValueError):
        return
    if prompt_tokens > 0 and sent_chars > 0:
        learn_token_ratio(sent_chars, prompt_tokens, model_id)


def shrink_tool_outputs(messages: list[dict], keep: int = 1,
                        max_chars: int = 400) -> bool:
    """In-place overflow shrinker for any message list. Used as the
    on_overflow callback by sub-agents (scouts, workers) whose own tool
    loop can bloat their context — the same protection the main agent
    gets. Three escalating passes, all pairing-safe:
      1. truncate the OLDEST tool results to short summaries (keeping the
         newest `keep` verbatim),
      2. if nothing was stale, truncate the newest tool results too,
      3. if still nothing shrank, drop the oldest complete turn unit
         (user message + its assistant reply + tool results) so
         tool_call / tool-response pairing is never broken.
    Returns True if anything actually shrank."""
    tool_idx = [i for i, m in enumerate(messages)
                if m.get("role") == "tool"]
    stale = tool_idx if keep <= 0 else tool_idx[:-keep]

    def _truncate(indices: list[int]) -> bool:
        shrank = False
        for i in indices:
            content = str(messages[i].get("content") or "")
            if len(content) > max_chars:
                messages[i]["content"] = (
                    content[:max_chars]
                    + f"\n[… truncated — {len(content):,} chars originally]")
                shrank = True
        return shrank

    if _truncate(stale):
        return True
    if _truncate(tool_idx):          # even the newest results, if desperate
        return True

    # drop the oldest complete turn unit (never a lone message)
    start = 1 if messages and messages[0].get("role") == "system" else 0
    end = None
    for j in range(start + 1, len(messages)):
        if messages[j].get("role") == "user":
            end = j
            break
    if end is not None and end > start:
        del messages[start:end]
        return True
    return False


def _check_api_key(provider: Provider) -> None:
    """Fail fast with actionable guidance instead of an opaque 401."""
    if not provider.api_key:
        env_name = {"zen": "OPENCODE_API_KEY",
                    "tokenrouter": "TOKENROUTER_API_KEY",
                    "agnes": "AGNES_API_KEY"}.get(
                        provider.key, f"{provider.key.upper()}_API_KEY")
        raise APIError(
            f"no API key configured for {provider.name}. Set the "
            f"{env_name} environment variable (export {env_name}=sk-...) "
            f"and restart.", status=401)


def chat_stream(provider: Provider, model: Model, effort: Effort,
                messages: list[dict], tools: list[dict] | None,
                on_token: Callable[[str], None] | None = None,
                on_reasoning: Callable[[str], None] | None = None,
                on_tool_start: Callable[[str], None] | None = None,
                should_cancel: Callable[[], bool] | None = None,
                on_overflow: Callable[[], bool] | None = None,
                timeout: float = config.DEFAULT_TIMEOUT) -> StreamResult:
    """Send a streaming chat completion request; calls callbacks as tokens
    arrive; returns the fully accumulated result.

    Context-overflow recovery — three escalating layers, so a long session
    on a huge project never dies with a context-length error:
      1. pre-flight clamp with the learned per-model chars/token ratio;
      2. on rejection, parse the backend's REAL token counts, recalibrate,
         re-clamp max_tokens, retry (up to OVERFLOW_RETRIES times);
      3. if the input itself no longer fits, invoke on_overflow (the
         caller shrinks the conversation — e.g. emergency compaction) and
         retry with the shrunken messages.
    The payload invariant is re-asserted before every send: the request
    that goes out always satisfies input + max_tokens <= window."""
    _check_api_key(provider)
    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    current_effort = effort
    shrinks_used = 0

    for overflow_attempt in range(OVERFLOW_RETRIES + OVERFLOW_SHRINKS + 1):
        sent_chars = _prompt_chars(messages, tools)
        try:
            payload = build_payload(model, current_effort, messages, tools,
                                    stream=True)
        except APIError as e:
            # pre-flight refusal (input already over the window). Give the
            # caller a chance to shrink the conversation and retry.
            if (is_context_overflow(str(e)) and on_overflow is not None
                    and shrinks_used < OVERFLOW_SHRINKS and on_overflow()):
                shrinks_used += 1
                continue
            raise
        try:
            result = _chat_stream_with_retries(
                url, headers, payload, on_token, on_reasoning, on_tool_start,
                should_cancel, timeout)
            _learn_from_usage(result.usage, sent_chars, model.id)
            return result
        except APIError as e:
            healed = _heal_overflow(model, current_effort, e,
                                    overflow_attempt, sent_chars)
            if healed is not None:
                current_effort = healed
                continue
            # clamping cannot help — the input itself is too big. Let the
            # caller shrink the conversation, then retry.
            if (is_context_overflow(str(e)) and on_overflow is not None
                    and shrinks_used < OVERFLOW_SHRINKS and on_overflow()):
                shrinks_used += 1
                continue
            raise

    # retries exhausted — surface the last overflow error
    raise APIError(
        f"conversation is too large for {model.label} even after "
        f"re-clamping — start a new session (/new) or rewind (/rewind)")


OVERFLOW_RETRIES = 2   # re-clamp-and-retry attempts (input still fits)
OVERFLOW_SHRINKS = 2   # shrink-the-conversation-and-retry attempts


def _heal_overflow(model: Model, effort: Effort, error: APIError,
                   attempt: int, sent_chars: int = 0) -> Effort | None:
    """Turn a backend overflow error into a tighter Effort, or None when
    the input itself no longer fits (nothing a clamp can fix)."""
    if not is_context_overflow(str(error)):
        return None
    if attempt >= OVERFLOW_RETRIES:
        return None
    info = _parse_overflow(str(error)) or {}
    # the backend just told us the real input size — calibrate the
    # per-model estimator with it so every later clamp is accurate
    if info.get("input_tokens") and sent_chars > 0:
        learn_token_ratio(sent_chars, info["input_tokens"], model.id)
    fitted = _fit_max_tokens_from_actual(model, info, effort)
    if fitted is None:
        return None
    if effort.max_tokens is not None and fitted >= effort.max_tokens:
        return None  # no tightening left to do
    return replace(effort, max_tokens=fitted)


def _chat_stream_with_retries(url: str, headers: dict, payload: dict,
                               on_token, on_reasoning, on_tool_start,
                               should_cancel, timeout: float) -> StreamResult:
    """The plain retry loop (rate limits, timeouts, connection errors, and
    TokenRouter cold-admission rejections)."""
    last_error: Exception | None = None
    cold_retries = 0
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            return _chat_stream_once(url, headers, payload,
                                     on_token, on_reasoning, on_tool_start,
                                     should_cancel, timeout)
        except TurnCancelled:
            raise
        except APIError as e:
            last_error = e
            # Cold-admission rejection: not an HTTP failure and not a hard
            # error — the prompt cache just isn't warm yet. Give it a
            # dedicated backoff budget so a session-opening request on a
            # free-tier endpoint survives until admission opens up.
            if is_cold_admission(str(e)):
                if should_cancel is not None and should_cancel():
                    raise TurnCancelled()
                cold_retries += 1
                if cold_retries <= MAX_COLD_RETRIES:
                    time.sleep(COLD_RETRY_WAIT * cold_retries)
                    continue
                raise
            if e.status in RETRY_STATUSES and attempt < MAX_RETRIES - 1:
                wait = 2.0 * (attempt + 1)
                time.sleep(wait)
                attempt += 1
                continue
            raise
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2.0)
                attempt += 1
                continue
            raise APIError(f"request timed out after {timeout}s") from e
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2.0)
                attempt += 1
                continue
            raise APIError(f"connection failed: {e}") from e
    raise APIError(str(last_error))


def _chat_stream_once(url: str, headers: dict, payload: dict,
                      on_token, on_reasoning, on_tool_start,
                      should_cancel, timeout: float) -> StreamResult:
    result = StreamResult()
    tc_acc: dict[int, ToolCallDelta] = {}
    announced_tools: set[int] = set()

    resp = requests.post(url, headers=headers, json=payload,
                         stream=True, timeout=timeout)
    if resp.status_code != 200:
        body = resp.text
        raise APIError(_extract_error_message(body), status=resp.status_code)
    # requests defaults text/* without charset to ISO-8859-1; the API speaks UTF-8
    resp.encoding = "utf-8"

    try:
        for event in _iter_sse_events(resp):
            if should_cancel is not None and should_cancel():
                raise TurnCancelled()
            if not isinstance(event, dict):
                continue
            if event.get("model"):
                result.model = event["model"]
            if event.get("usage"):
                result.usage = event["usage"]
            if event.get("error"):
                err = event["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise APIError(msg)
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            if choice.get("finish_reason"):
                result.finish_reason = choice["finish_reason"]

            piece = delta.get("content")
            if piece:
                result.content += piece
                if on_token:
                    on_token(piece)

            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                result.reasoning += reasoning
                if on_reasoning:
                    on_reasoning(reasoning)

            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                acc = tc_acc.setdefault(idx, ToolCallDelta())
                if tc.get("id"):
                    acc.id = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc.name += fn["name"]
                    if idx not in announced_tools and on_tool_start:
                        announced_tools.add(idx)
                        on_tool_start(acc.name)
                if fn.get("arguments"):
                    acc.arguments += fn["arguments"]
    finally:
        resp.close()

    for idx in sorted(tc_acc):
        acc = tc_acc[idx]
        result.tool_calls.append({
            "id": acc.id or f"call_{idx}",
            "type": "function",
            "function": {"name": acc.name, "arguments": acc.arguments},
        })

    # Some providers return a non-streamed JSON body even when stream=true.
    if not result.content and not result.tool_calls and not result.reasoning:
        pass
    return result


def _post_with_cold_retries(url: str, headers: dict, payload: dict,
                            timeout: float) -> requests.Response:
    """POST with a dedicated budget for TokenRouter cold-admission
    rejections ("cache-only admission rejected"). Returns a 200 response or
    raises; non-cold errors raise immediately."""
    last: APIError | None = None
    for cold in range(MAX_COLD_RETRIES + 1):
        resp = requests.post(url, headers=headers, json=payload,
                             timeout=timeout)
        if resp.status_code == 200:
            return resp
        err = APIError(_extract_error_message(resp.text),
                       status=resp.status_code)
        if not is_cold_admission(str(err)) or cold >= MAX_COLD_RETRIES:
            raise err
        last = err
        time.sleep(COLD_RETRY_WAIT * (cold + 1))
    raise last


def chat_blocking(provider: Provider, model: Model, effort: Effort,
                  messages: list[dict], tools: list[dict] | None,
                  on_overflow: Callable[[], bool] | None = None,
                  timeout: float = config.DEFAULT_TIMEOUT) -> StreamResult:
    """Non-streaming fallback (used when a provider rejects stream=true).

    Carries the same three-layer context-overflow recovery as chat_stream:
    pre-flight clamp, re-clamp-and-retry on rejection, and (when the input
    itself is too big) caller-driven shrink-and-retry via on_overflow."""
    _check_api_key(provider)
    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    current_effort = effort
    shrinks_used = 0

    for overflow_attempt in range(OVERFLOW_RETRIES + OVERFLOW_SHRINKS + 1):
        sent_chars = _prompt_chars(messages, tools)
        try:
            payload = build_payload(model, current_effort, messages, tools,
                                    stream=False)
        except APIError as e:
            # pre-flight refusal (input already over the window). Give the
            # caller a chance to shrink the conversation and retry.
            if (is_context_overflow(str(e)) and on_overflow is not None
                    and shrinks_used < OVERFLOW_SHRINKS and on_overflow()):
                shrinks_used += 1
                continue
            raise
        try:
            resp = _post_with_cold_retries(url, headers, payload, timeout)
        except APIError as err:
            healed = _heal_overflow(model, current_effort, err,
                                    overflow_attempt, sent_chars)
            if healed is not None:
                current_effort = healed
                continue
            if (is_context_overflow(str(err)) and on_overflow is not None
                    and shrinks_used < OVERFLOW_SHRINKS and on_overflow()):
                shrinks_used += 1
                continue
            raise
        data = resp.json()
        result = StreamResult(model=data.get("model", model.id),
                              usage=data.get("usage"))
        _learn_from_usage(result.usage, sent_chars, model.id)
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            result.content = msg.get("content") or ""
            result.reasoning = msg.get("reasoning_content") or ""
            result.finish_reason = choices[0].get("finish_reason")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                result.tool_calls.append({
                    "id": tc.get("id", "call_0"),
                    "type": "function",
                    "function": {"name": fn.get("name", ""),
                                 "arguments": fn.get("arguments", "")},
                })
        return result

    raise APIError(
        f"conversation is too large for {model.label} even after "
        f"re-clamping — start a new session (/new) or rewind (/rewind)")
