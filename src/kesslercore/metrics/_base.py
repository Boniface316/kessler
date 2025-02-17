from __future__ import annotations

import abc
import typing as T

import mlflow
import pandas as pd
import pydantic as pdt
from mlflow.metrics import MetricValue


from ..io.schemas import Inputs, Targets, Outputs, OutputsSchema, TargetsSchema
from ..models import Model

# %% TYPINGS

MlflowMetric: T.TypeAlias = MetricValue
MlflowThreshold: T.TypeAlias = mlflow.models.MetricThreshold
MlflowModelValidationFailedException: T.TypeAlias = (
    mlflow.models.evaluation.validation.ModelValidationFailedException
)

# %% METRICS


class Metric(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a project metric.

    Use metrics to evaluate model performance.
    e.g., accuracy, precision, recall, MAE, F1, ...

    Parameters:
        name (str): name of the metric for the reporting.
        greater_is_better (bool): maximize or minimize result.
    """

    KIND: str

    name: str
    greater_is_better: bool

    @abc.abstractmethod
    def score(self, targets: Targets, outputs: Outputs) -> float:
        """Score the outputs against the targets.

        Args:
            targets (Targets): expected values.
            outputs (Outputs): predicted values.

        Returns:
            float: single result from the metric computation.
        """

    def scorer(self, model: Model, inputs: Inputs, targets: Targets) -> float:
        """Score model outputs against targets.

        Args:
            model (Model): model to evaluate.
            inputs (schemas.Inputs): model inputs values.
            targets (schemas.Targets): model expected values.

        Returns:
            float: single result from the metric computation.
        """
        outputs = model.predict(inputs=inputs)
        score = self.score(targets=targets, outputs=outputs)
        return score

    def to_mlflow(self) -> MlflowMetric:
        """Convert the metric to an Mlflow metric.

        Returns:
            MlflowMetric: the Mlflow metric.
        """

        def eval_fn(
            predictions: pd.Series[int], targets: pd.Series[int]
        ) -> MlflowMetric:
            """Evaluation function associated with the mlflow metric.

            Args:
                predictions (pd.Series): model predictions.
                targets (pd.Series | None): model targets.

            Returns:
                MlflowMetric: the mlflow metric.
            """
            score_targets = Targets({TargetsSchema.cnt: targets}, index=targets.index)
            score_outputs = Outputs(
                {OutputsSchema.prediction: predictions}, index=predictions.index
            )
            sign = 1 if self.greater_is_better else -1  # reverse the effect
            score = self.score(targets=score_targets, outputs=score_outputs)
            return MlflowMetric(aggregate_results={self.name: score * sign})

        return mlflow.metrics.make_metric(
            eval_fn=eval_fn, name=self.name, greater_is_better=self.greater_is_better
        )
