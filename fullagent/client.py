"""OpenAI-compatible streaming chat client (requests + manual SSE parsing)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import requests

from . import config
from .config import Effort, Model, Provider

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 3


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
        payload["max_tokens"] = _clamp_max_tokens(model.provider,
                                                  effort.max_tokens)
    if tools and model.supports_tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if model.supports_reasoning and effort.reasoning_effort:
        payload["reasoning_effort"] = _normalize_reasoning_effort(
            model.provider, effort.reasoning_effort)
    return payload


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


def chat_stream(provider: Provider, model: Model, effort: Effort,
                messages: list[dict], tools: list[dict] | None,
                on_token: Callable[[str], None] | None = None,
                on_reasoning: Callable[[str], None] | None = None,
                on_tool_start: Callable[[str], None] | None = None,
                should_cancel: Callable[[], bool] | None = None,
                timeout: float = config.DEFAULT_TIMEOUT) -> StreamResult:
    """Send a streaming chat completion request; calls callbacks as tokens
    arrive; returns the fully accumulated result."""
    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = build_payload(model, effort, messages, tools, stream=True)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return _chat_stream_once(url, headers, payload,
                                     on_token, on_reasoning, on_tool_start,
                                     should_cancel, timeout)
        except TurnCancelled:
            raise
        except APIError as e:
            last_error = e
            if e.status in RETRY_STATUSES and attempt < MAX_RETRIES - 1:
                wait = 2.0 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2.0)
                continue
            raise APIError(f"request timed out after {timeout}s") from e
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2.0)
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


def chat_blocking(provider: Provider, model: Model, effort: Effort,
                  messages: list[dict], tools: list[dict] | None,
                  timeout: float = config.DEFAULT_TIMEOUT) -> StreamResult:
    """Non-streaming fallback (used when a provider rejects stream=true)."""
    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    payload = build_payload(model, effort, messages, tools, stream=False)
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise APIError(_extract_error_message(resp.text), status=resp.status_code)
    data = resp.json()
    result = StreamResult(model=data.get("model", model.id),
                          usage=data.get("usage"))
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
