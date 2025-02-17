"""Generate signatures for AI/ML models."""

# %% IMPORTS

import abc
import typing as T

import pydantic as pdt
from mlflow.models import signature as ms

from ..io.schemas import Inputs, Outputs

# %% TYPES

Signature: T.TypeAlias = ms.ModelSignature

# %% SIGNERS


class Signer(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for generating model signatures.

    Allow switching between model signing strategies.
    e.g., automatic inference, manual model signature, ...

    https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
    """

    KIND: str

    @abc.abstractmethod
    def sign(self, inputs: Inputs, outputs: Outputs) -> Signature:
        """Generate a model signature from its inputs/outputs.

        Args:
            inputs (schemas.Inputs): inputs data.
            outputs (schemas.Outputs): outputs data.

        Returns:
            Signature: signature of the model.
        """
