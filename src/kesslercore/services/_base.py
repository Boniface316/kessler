"""Manage global context during execution."""

# %% IMPORTS

from __future__ import annotations

import abc

import pydantic as pdt

# %% SERVICES


class Service(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a global service.

    Use services to manage global contexts.
    e.g., logger object, mlflow client, spark context, ...
    """

    @abc.abstractmethod
    def start(self) -> None:
        """Start the service."""

    def stop(self) -> None:
        """Stop the service."""
        # does nothing by default
