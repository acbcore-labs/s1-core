"""Backend drivers: the seam between the harness and any System 1 model."""

from .base import BackendDriver, BackendError, StateHandle
from .lexical import LexicalHeuristicDriver
from .shadow import ShadowTeeDriver
from .systemone import SystemOneHTTPDriver
from .typesafe import (
    estimate_cost_usd,
    typesafe_driver,
    typesafe_request_builder,
    typesafe_response_parser,
)

__all__ = [
    "BackendDriver",
    "BackendError",
    "StateHandle",
    "LexicalHeuristicDriver",
    "SystemOneHTTPDriver",
    "ShadowTeeDriver",
    "typesafe_driver",
    "typesafe_request_builder",
    "typesafe_response_parser",
    "estimate_cost_usd",
]
