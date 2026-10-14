"""Reference HTTP driver for hosted ``/v1/systemone``-style endpoints.

For the live TypeSafe (Jev) API use :func:`s1core.backends.typesafe.typesafe_driver`,
which layers the production wire-format mapping onto this transport. Every
hosted decider differs slightly; adapt via ``request_builder`` /
``response_parser`` hooks rather than subclassing. Stdlib ``urllib`` only —
zero dependencies.

The driver also accumulates ``usage_totals`` (requests, input/output tokens)
when a backend reports a ``usage`` object, and retries 429/529/transport
failures with exponential backoff.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable, Mapping

from .base import BackendError, StateHandle

RequestBuilder = Callable[[str, str, Mapping[str, Any], list[dict[str, Any]]], dict[str, Any]]
ResponseParser = Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, dict[str, float]]]

# Transient per TypeSafe API docs (429 rate limit, 529 overloaded).
_RETRYABLE_STATUS_CODES = frozenset({429, 529})


def default_request_builder(
    model: str, endpoint: str, state: Mapping[str, Any], questions: list[dict[str, Any]]
) -> dict[str, Any]:
    return {"model": model, "state": state, "questions": questions}


def default_response_parser(
    body: Mapping[str, Any], batch: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    decisions = body.get("decisions", body)
    results: dict[str, dict[str, float]] = {}
    for question_id in batch:
        if question_id not in decisions:
            raise BackendError(f"backend response missing question '{question_id}'")
        results[question_id] = {
            str(k): float(v) for k, v in decisions[question_id].items()
        }
    return results


class SystemOneHTTPDriver:
    """Hosted System 1 endpoint driver. ``pin_state`` caches the canonical
    envelope (hosted APIs re-receive state per call; the harness interface is
    unchanged, so swapping to a KV-pinning local driver needs no app changes).
    """

    def __init__(
        self,
        base_url: str,
        *,
        model_id: str = "jev-latest",
        endpoint: str = "/v1/systemone",
        api_key: str | None = None,
        timeout_s: float = 15.0,
        headers: Mapping[str, str] | None = None,
        request_builder: RequestBuilder | None = None,
        response_parser: ResponseParser | None = None,
        max_retries: int = 2,
        retry_backoff_s: float = 0.8,
    ):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self._timeout_s = timeout_s
        self._headers = dict(headers or {})
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._request_builder = request_builder or default_request_builder
        self._response_parser = response_parser or default_response_parser
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self.usage_totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    @contextmanager
    def pin_state(self, canonical_state: Mapping[str, Any], state_hash: str) -> Iterator[StateHandle]:
        yield StateHandle(state_hash=state_hash, canonical_state=canonical_state)

    def score_branches(
        self, handle: StateHandle, batch: Mapping[str, Any]
    ) -> dict[str, dict[str, float]]:
        payloads = [compiled.to_payload() for compiled in batch.values()]
        payload = self._request_builder(
            self.model_id, self.endpoint, handle.canonical_state, payloads
        )
        parsed = self._post(payload)
        if isinstance(parsed, Mapping) and isinstance(parsed.get("usage"), Mapping):
            usage = parsed["usage"]
            self.usage_totals["requests"] += 1
            self.usage_totals["input_tokens"] += int(usage.get("input_tokens", 0))
            self.usage_totals["output_tokens"] += int(usage.get("output_tokens", 0))
        return self._response_parser(parsed, batch)

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            request = urllib.request.Request(
                self.base_url + self.endpoint,
                data=body,
                headers={"Content-Type": "application/json", **self._headers},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:500].decode("utf-8", errors="replace")
                if exc.code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                    time.sleep(self._retry_backoff_s * (2**attempt))
                    last_error = BackendError(f"HTTP {exc.code}: {detail}")
                    continue
                raise BackendError(f"backend returned HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff_s * (2**attempt))
                    last_error = exc
                    continue
                raise BackendError(f"backend transport failure: {exc}") from exc
        raise BackendError(f"backend failed after {self._max_retries + 1} attempts: {last_error}")
