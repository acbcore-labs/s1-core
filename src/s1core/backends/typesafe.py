"""TypeSafe (Jev) adapter: wire-format mapping for the live
``POST https://api.typesafe.ai/v1/systemone`` API.

Wire differences from the harness's internal representation, all absorbed
here so specs and gates never see them:

- ``questions`` is a JSON **map** keyed by question id (the id is not repeated
  inside each question object, and per the API docs keys never reach the model).
- ``score`` criteria travel as an **ordered array** of level descriptions and
  the API returns level probabilities 0-indexed (``"0".."k-1"``); the harness
  uses 1..k, so the parser restores harness level keys positionally.
- A ``noul`` answer is a scalar ``{"noul": p}``; the parser expands it to the
  harness's ``{"true": p, "false": 1-p}``.
- Choice answers omit zero-probability options sometimes; the parser fills
  declared criteria keys with 0.0 so gate math sees the full simplex.
- ``usage.input_tokens``/``output_tokens`` accumulate on the driver
  (input-only billing: $0.042 per 1M input tokens, output free).
- 429 (rate limit) and 529 (overloaded) are retried with exponential backoff
  per the API docs.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from .base import BackendError

INPUT_TOKEN_PRICE_PER_MTOK = 0.042  # USD per 1M input tokens; output is free


def typesafe_request_builder(
    model: str, endpoint: str, state: Mapping[str, Any], questions: list[dict[str, Any]]
) -> dict[str, Any]:
    request_questions: dict[str, Any] = {}
    for payload in questions:
        question: dict[str, Any] = {
            "type": payload["type"],
            "instructions": payload["instructions"],
        }
        criteria = payload["criteria"]
        if payload["type"] == "score":
            # Harness levels "1".."k" -> ordered array of descriptions.
            question["criteria"] = [criteria[key] for key in sorted(criteria, key=int)]
        else:
            question["criteria"] = dict(criteria)
        request_questions[payload["question_id"]] = question
    return {"model": model, "state": state, "questions": request_questions}


def typesafe_response_parser(
    body: Mapping[str, Any], batch: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise BackendError(f"TypeSafe response missing 'answers' map: {str(body)[:300]}")
    results: dict[str, dict[str, float]] = {}
    for question_id, compiled in batch.items():
        if question_id not in answers:
            raise BackendError(f"TypeSafe response missing question '{question_id}'")
        answer = answers[question_id]
        primitive = answer.get("type", compiled.primitive)
        if primitive == "noul":
            p_true = float(answer["noul"])
            results[question_id] = {"true": p_true, "false": 1.0 - p_true}
        elif primitive == "choice":
            probs = {str(key): 0.0 for key in compiled.criteria}
            for key, value in answer.get("probabilities", {}).items():
                probs[str(key)] = float(value)
            results[question_id] = probs
        elif primitive == "score":
            # Wire levels are 0-indexed positions; restore harness 1..k keys.
            level_keys = list(compiled.criteria.keys())
            items = sorted(answer["probabilities"].items(), key=lambda kv: int(kv[0]))
            if len(items) != len(level_keys):
                raise BackendError(
                    f"score answer for '{question_id}' has {len(items)} levels, "
                    f"spec declared {len(level_keys)}"
                )
            results[question_id] = {
                level_keys[i]: float(value) for i, (_, value) in enumerate(items)
            }
        else:
            raise BackendError(f"unknown answer type '{primitive}' for '{question_id}'")
    return results


def estimate_cost_usd(input_tokens: int) -> float:
    return input_tokens * INPUT_TOKEN_PRICE_PER_MTOK / 1_000_000.0


def typesafe_driver(
    api_key: str,
    *,
    base_url: str = "https://api.typesafe.ai",
    model: str = "jev-latest",
    timeout_s: float = 15.0,
    max_retries: int = 2,
    retry_backoff_s: float = 0.8,
) -> "SystemOneHTTPDriver":
    from .systemone import SystemOneHTTPDriver

    return SystemOneHTTPDriver(
        base_url,
        model_id=model,
        endpoint="/v1/systemone",
        api_key=api_key,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        request_builder=typesafe_request_builder,
        response_parser=typesafe_response_parser,
    )


__all__ = [
    "typesafe_request_builder",
    "typesafe_response_parser",
    "typesafe_driver",
    "estimate_cost_usd",
    "INPUT_TOKEN_PRICE_PER_MTOK",
]
