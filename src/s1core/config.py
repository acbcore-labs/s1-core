"""Configuration: a zero-dependency `.env` loader and driver factories.

The repo ships an ``example.env.txt`` template; users copy it to ``.env``
(gitignored) and fill in their key. ``load_dotenv`` reads it with the stdlib —
no python-dotenv dependency — and ``driver_from_env`` turns environment
variables into a ready backend driver::

    from s1core.config import load_dotenv, driver_from_env

    load_dotenv()                       # reads ./.env, never overrides real env
    driver = driver_from_env()          # None if TYPESAFE_API_KEY is unset
    backend = driver or LexicalHeuristicDriver()   # toy fallback
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

DEFAULT_ENV_PATH = ".env"

TYPESAFE_API_KEY_VAR = "TYPESAFE_API_KEY"
TYPESAFE_BASE_URL_VAR = "TYPESAFE_BASE_URL"
TYPESAFE_MODEL_VAR = "TYPESAFE_MODEL"
TYPESAFE_DEFAULT_BASE_URL = "https://api.typesafe.ai"
TYPESAFE_DEFAULT_MODEL = "jev-latest"


def load_dotenv(path: str | Path = DEFAULT_ENV_PATH, *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` pairs from a dotenv file into ``os.environ``.

    Existing environment variables win unless ``override=True``. Blank lines
    and ``#`` comments are ignored; surrounding single/double quotes are
    stripped. Returns the pairs that were applied.
    """
    env_path = Path(path)
    if not env_path.exists():
        return {}
    applied: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def typesafe_driver_from_env(env: Mapping[str, str] | None = None) -> Any | None:
    """Build a live TypeSafe (Jev) driver from the environment, or ``None``.

    Reads ``TYPESAFE_API_KEY`` (required), ``TYPESAFE_BASE_URL`` and
    ``TYPESAFE_MODEL`` (optional). Returning ``None`` — rather than raising —
    is what lets application code do ``driver_from_env() or ToyDriver()``
    with one line.
    """
    env = os.environ if env is None else env
    api_key = env.get(TYPESAFE_API_KEY_VAR, "").strip()
    if not api_key:
        return None
    from .backends import typesafe_driver

    return typesafe_driver(
        api_key=api_key,
        base_url=env.get(TYPESAFE_BASE_URL_VAR, TYPESAFE_DEFAULT_BASE_URL).strip(),
        model=env.get(TYPESAFE_MODEL_VAR, TYPESAFE_DEFAULT_MODEL).strip(),
    )


# Short alias for application code.
driver_from_env = typesafe_driver_from_env


__all__ = [
    "DEFAULT_ENV_PATH",
    "TYPESAFE_API_KEY_VAR",
    "TYPESAFE_BASE_URL_VAR",
    "TYPESAFE_MODEL_VAR",
    "TYPESAFE_DEFAULT_BASE_URL",
    "TYPESAFE_DEFAULT_MODEL",
    "load_dotenv",
    "typesafe_driver_from_env",
    "driver_from_env",
]
