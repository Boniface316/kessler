import typing as T

import mlflow
from mlflow.pyfunc import PyFuncModel

from ..io.schemas import Inputs, Outputs, OutputsSchema

from ._base import Loader, Register, Saver, Version, Info
from mlflow.pyfunc import PythonModel, PythonModelContext
from ..models import Model
from ..signers import Signature

class CustomLoader(Loader):
    """Loader for custom models using the Mlflow PyFunc module.

    https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
    """

    KIND: T.Literal["CustomLoader"] = "CustomLoader"

    class Adapter(Loader.Adapter):
        """Adapt a custom model for the project inference."""

        def __init__(self, model: PyFuncModel) -> None:
            """Initialize the adapter from an mlflow pyfunc model.

            Args:
                model (PyFuncModel): mlflow pyfunc model.
            """
            self.model = model

        @T.override
        def predict(self, inputs: Inputs) -> Outputs:
            # model validation is already done in predict
            outputs = self.model.predict(data=inputs)
            return T.cast(Outputs, outputs)

    @T.override
    def load(self, uri: str) -> "CustomLoader.Adapter":
        model = mlflow.pyfunc.load_model(model_uri=uri)
        adapter = CustomLoader.Adapter(model=model)
        return adapter


class BuiltinLoader(Loader):
    """Loader for built-in models using the Mlflow PyFunc module.

    Note: use Mlflow PyFunc instead of flavors to use standard API.

    https://mlflow.org/docs/latest/models.html#built-in-model-flavors
    """

    KIND: T.Literal["BuiltinLoader"] = "BuiltinLoader"

    class Adapter(Loader.Adapter):
        """Adapt a builtin model for the project inference."""

        def __init__(self, model: PyFuncModel) -> None:
            """Initialize the adapter from an mlflow pyfunc model.

            Args:
                model (PyFuncModel): mlflow pyfunc model.
            """
            self.model = model

        @T.override
        def predict(self, inputs: Inputs) -> Outputs:
            columns = list(OutputsSchema.to_schema().columns)
            outputs = self.model.predict(data=inputs)  # unchecked data!
            return Outputs(outputs, columns=columns, index=inputs.index)

    @T.override
    def load(self, uri: str) -> "BuiltinLoader.Adapter":
        model = mlflow.pyfunc.load_model(model_uri=uri)
        adapter = BuiltinLoader.Adapter(model=model)
        return adapter

class MlflowRegister(Register):
    """Register for models in the Mlflow Model Registry.

    https://mlflow.org/docs/latest/model-registry.html
    """

    KIND: T.Literal["MlflowRegister"] = "MlflowRegister"

    @T.override
    def register(self, name: str, model_uri: str) -> Version:
        return mlflow.register_model(name=name, model_uri=model_uri, tags=self.tags)

class CustomSaver(Saver):
    """Saver for project models using the Mlflow PyFunc module.

    https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
    """

    KIND: T.Literal["CustomSaver"] = "CustomSaver"

    class Adapter(PythonModel):  # type: ignore[misc]
        """Adapt a custom model to the Mlflow PyFunc flavor for saving operations.

        https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html?#mlflow.pyfunc.PythonModel
        """

        def __init__(self, model: Model):
            """Initialize the custom saver adapter.

            Args:
                model (Model): project model.
            """
            self.model = model

        def predict(
            self,
            context: PythonModelContext,
            model_input: Inputs,
            params: dict[str, T.Any] | None = None,
        ) -> Outputs:
            """Generate predictions with a custom model for the given inputs.

            Args:
                context (mlflow.PythonModelContext): mlflow context.
                model_input (Inputs): inputs for the mlflow model.
                params (dict[str, T.Any] | None): additional parameters.

            Returns:
                Outputs: validated outputs of the project model.
            """
            return self.model.predict(inputs=model_input)

    @T.override
    def save(
        self,
        model: Model,
        signature: Signature,
        input_example: Inputs,
    ) -> Info:
        adapter = CustomSaver.Adapter(model=model)
        return mlflow.pyfunc.log_model(
            python_model=adapter,
            signature=signature,
            artifact_path=self.path,
            input_example=input_example,
        )


class BuiltinSaver(Saver):
    """Saver for built-in models using an Mlflow flavor module.

    https://mlflow.org/docs/latest/models.html#built-in-model-flavors

    Parameters:
        flavor (str): Mlflow flavor module to use for the serialization.
    """

    KIND: T.Literal["BuiltinSaver"] = "BuiltinSaver"

    flavor: str

    @T.override
    def save(
        self,
        model: Model,
        signature: Signature,
        input_example: Inputs,
    ) -> Info:
        builtin_model = model.get_internal_model()
        module = getattr(mlflow, self.flavor)
        return module.log_model(
            builtin_model,
            artifact_path=self.path,
            signature=signature,
            input_example=input_example,
        )