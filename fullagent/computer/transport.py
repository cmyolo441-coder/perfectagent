"""Bounded OpenAI-compatible transport for concurrent computer workers.

Uses FullAgent's configured models/providers and payload compatibility,
not eight copies of the heavyweight main Agent. Each HTTP attempt is
visible to the engine's token reservation/retry governor.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import replace
from urllib.parse import urlsplit

import requests

from ..client import StreamResult, build_payload
from .state import ComputerError, safe_text

_PAYLOAD_LOCK = threading.Lock()


class TransportError(ComputerError):
    def __init__(self, message, retryable=False, retry_after=0):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = max(0, min(float(retry_after), 15))


class APIClient:
    def __init__(self, provider, model, effort, timeout=45):
        self.provider, self.model, self.effort = provider, model, effort
        self.timeout = timeout
        self._local = threading.local()
        self._lock = threading.Lock()
        self._sessions = []
        self._include_usage = True
        self._stream = True
        p = urlsplit(provider.base_url)
        if p.scheme not in ("https", "http") or not p.hostname or p.username or p.password:
            raise ComputerError("Provider base_url must be an HTTP(S) endpoint without embedded credentials")
        if not model.supports_tools:
            raise ComputerError(f"Selected model {model.id} does not support tools")
        if not provider.api_key and p.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ComputerError(f"No API key configured for {provider.name}; configure it locally before starting")

    def _session(self):
        if not getattr(self._local, "session", None):
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
            with self._lock:
                self._sessions.append(session)
        return self._local.session

    def close(self):
        with self._lock:
            for session in self._sessions:
                session.close()
            self._sessions.clear()

    def chat(self, messages, schemas, max_tokens, control, on_update=None):
        control()
        # build_payload sanitizes messages and reads shared calibration.
        # Only preparation is locked; all eight HTTP requests overlap.
        with _PAYLOAD_LOCK:
            payload = build_payload(self.model, replace(self.effort, max_tokens=max_tokens),
                                    copy.deepcopy(messages), schemas or None, stream=self._stream)
        payload["max_tokens"] = min(max_tokens, payload.get("max_tokens", max_tokens))
        if self._stream and self._include_usage:
            payload["stream_options"] = {"include_usage": True}
        url = self.provider.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream,application/json"}
        if self.provider.api_key:
            headers["Authorization"] = "Bearer " + self.provider.api_key
        started = time.monotonic()
        try:
            with self._session().post(url, json=payload, headers=headers, stream=True,
                                      timeout=(min(10, self.timeout), min(15, self.timeout))) as response:
                if response.status_code != 200:
                    error_bytes = next(response.iter_content(4096), b"")
                    detail = error_bytes.decode("utf-8", errors="replace")
                    # A compatibility retry is a NEW budgeted attempt.
                    if response.status_code in (400, 422) and "stream_options" in detail and self._include_usage:
                        self._include_usage = False
                        raise TransportError("Provider rejected stream_options; retrying without usage streaming", True)
                    if response.status_code in (400, 422) and "stream" in detail.lower() and self._stream:
                        self._stream = False
                        raise TransportError("Provider rejected streaming; retrying bounded JSON response", True)
                    try:
                        retry_after = float(response.headers.get("Retry-After", "0"))
                    except ValueError:
                        retry_after = 0
                    raise TransportError(f"Provider HTTP {response.status_code}: {safe_text(detail, 350)}",
                                         response.status_code in (408, 429, 500, 502, 503, 504), retry_after)
                ctype = response.headers.get("Content-Type", "").lower()
                limit = min(2_000_000, max(131072, max_tokens * 40))
                raw = bytearray()
                size = 0
                result = StreamResult(model=self.model.id)
                accum = {}
                got_finish = False
                data_lines = []

                def consume(event):
                    nonlocal got_finish
                    if event.strip() == "[DONE]":
                        got_finish = True
                        return
                    if not event.strip():
                        return
                    value = json.loads(event)
                    if value.get("error"):
                        raise TransportError("Provider returned a stream error: " + safe_text(value["error"], 350))
                    if value.get("usage"):
                        result.usage = value["usage"]
                    if value.get("model"):
                        result.model = str(value["model"])
                    choices = value.get("choices") or []
                    if not choices:
                        return
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        result.finish_reason = choice["finish_reason"]
                        got_finish = True
                    delta = choice.get("delta", {})
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        result.content += piece
                        if on_update:
                            on_update("receiving response", len(result.content))
                    # Deliberately do not store or expose private reasoning.
                    for tool in delta.get("tool_calls") or []:
                        index = tool.get("index", 0)
                        if type(index) is not int or not 0 <= index < 8:
                            raise TransportError("Provider emitted too many tool calls")
                        current = accum.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        current["id"] = tool.get("id") or current["id"]
                        fn = tool.get("function") or {}
                        current["function"]["name"] += fn.get("name") or ""
                        current["function"]["arguments"] += fn.get("arguments") or ""
                        if on_update and fn.get("name"):
                            on_update("preparing " + str(fn["name"]), 0)

                for chunk in response.iter_content(4096):
                    control()
                    if time.monotonic() - started > self.timeout:
                        raise TransportError("Model request deadline exceeded")
                    size += len(chunk)
                    if size > limit:
                        raise TransportError("Model response exceeded the bounded buffer")
                    raw.extend(chunk)
                    if "text/event-stream" not in ctype:
                        continue
                    while b"\n" in raw:
                        line, _, rest = raw.partition(b"\n")
                        raw[:] = rest
                        text = line.decode("utf-8", errors="replace").rstrip("\r")
                        if text.startswith("data:"):
                            data_lines.append(text[5:].lstrip())
                        elif not text and data_lines:
                            consume("\n".join(data_lines))
                            data_lines.clear()
                control()
                if "text/event-stream" in ctype:
                    if raw.startswith(b"data:"):
                        data_lines.append(raw[5:].decode("utf-8", errors="replace").strip())
                    if data_lines:
                        consume("\n".join(data_lines))
                    if not got_finish:
                        raise TransportError("Provider stream ended without completion; no partial tool call was executed")
                    result.tool_calls = [accum[k] for k in sorted(accum)]
                else:
                    value = json.loads(bytes(raw).decode("utf-8"))
                    if value.get("error"):
                        raise TransportError("Provider returned an error response")
                    choices = value.get("choices") or []
                    if not choices:
                        raise TransportError("Provider returned no choices")
                    choice = choices[0]
                    message = choice.get("message") or {}
                    result.content = message.get("content") or ""
                    result.tool_calls = message.get("tool_calls") or []
                    result.finish_reason = choice.get("finish_reason")
                    result.usage = value.get("usage")
                if len(result.tool_calls) > 8:
                    raise TransportError("Provider emitted more than eight tool calls in one step")
                if result.finish_reason in ("length", "content_filter"):
                    raise TransportError("Model response was truncated/filtered; no incomplete tool call was executed")
                for i, tc in enumerate(result.tool_calls):
                    if not tc.get("id"):
                        tc["id"] = f"call_{i}"
                return result
        except requests.RequestException as exc:
            # URLs and authorization values are never included in errors.
            raise TransportError(f"Model connection failed ({type(exc).__name__})", retryable=True) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise TransportError(f"Malformed model response ({type(exc).__name__})") from exc
