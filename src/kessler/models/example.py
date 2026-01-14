"""Define trainable machine learning models."""

# %% IMPORTS
import numpy as np
import typing as T

from ._base import Model

from ..io.schemas import Inputs, Outputs, OutputsSchema, Targets

# %% TYPES

# Model params
ParamKey = str
ParamValue = T.Any
Params = dict[ParamKey, ParamValue]


class ExampleModel(Model):
    KIND: T.Literal["ExampleModel"] = "ExampleModel"
    params: int | None = None

    @T.override
    def fit(self, inputs: Inputs, targets: Targets) -> "ExampleModel":
        print("Fitting the model with inputs and targets")
        self.params = {"dummy_param": 1}
        return self

    @T.override
    def get_internal_model(self):
        print("Returning the internal model")
        return 0

    @T.override
    def predict(self, inputs: Inputs) -> Targets:
        num_rows = len(inputs)
        prediction = np.random.randint(0, 10, size=num_rows)

        return Outputs({OutputsSchema.prediction: prediction}, index=inputs.index)

    @T.override
    def explain_model(self):
        pass

    @T.override
    def explain_samples(self, inputs: Inputs):
        pass
