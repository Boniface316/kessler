"""Savers, loaders, and registers for model registries."""

# %% IMPORTS

import abc
import typing as T

import mlflow
import pydantic as pdt

from ..signers import Signature
from ..io.schemas import Inputs, Outputs

from ..models import Model


# %% TYPES

# Results of model registry operations
Info: T.TypeAlias = mlflow.models.model.ModelInfo
Alias: T.TypeAlias = mlflow.entities.model_registry.ModelVersion
Version: T.TypeAlias = mlflow.entities.model_registry.ModelVersion

# %% HELPERS


# %% SAVERS
class Saver(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for saving models in registry.

    Separate model definition from serialization.
    e.g., to switch between serialization flavors.

    Parameters:
        path (str): model path inside the Mlflow store.
    """

    KIND: str

    path: str = "model"

    @abc.abstractmethod
    def save(
        self,
        model: Model,
        signature: Signature,
        input_example: Inputs,
    ) -> Info:
        """Save a model in the model registry.

        Args:
            model (models.Model): project model to save.
            signature (signers.Signature): model signature.
            input_example (schemas.Inputs): sample of inputs.

        Returns:
            Info: model saving information.
        """


class Loader(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for loading models from registry.

    Separate model definition from deserialization.
    e.g., to switch between deserialization flavors.
    """

    KIND: str

    class Adapter(abc.ABC):
        """Adapt any model for the project inference."""

        @abc.abstractmethod
        def predict(self, inputs: Inputs) -> Outputs:
            """Generate predictions with the internal model for the given inputs.

            Args:
                inputs (schemas.Inputs): validated inputs for the project model.

            Returns:
                schemas.Outputs: validated outputs of the project model.
            """

    @abc.abstractmethod
    def load(self, uri: str) -> "Loader.Adapter":
        """Load a model from the model registry.

        Args:
            uri (str): URI of a model to load.

        Returns:
            Loader.Adapter: model loaded.
        """


class Register(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for registring models to a location.

    Separate model definition from its registration.
    e.g., to change the model registry backend.

    Parameters:
        tags (dict[str, T.Any]): tags for the model.
    """

    KIND: str

    tags: dict[str, T.Any] = {}

    @abc.abstractmethod
    def register(self, name: str, model_uri: str) -> Version:
        """Register a model given its name and URI.

        Args:
            name (str): name of the model to register.
            model_uri (str): URI of a model to register.

        Returns:
            Version: information about the registered model.
        """
