import typing as T
import mlflow

from ._base import Signer, Signature

from ..io.schemas import Inputs, Outputs


class ExampleSigner(Signer):
    """Generate model signatures from inputs/outputs data."""

    KIND: T.Literal["InferSigner"] = "InferSigner"

    @T.override
    def sign(self, inputs: Inputs, outputs: Outputs) -> Signature:
        return mlflow.models.infer_signature(model_input=inputs, model_output=outputs)
