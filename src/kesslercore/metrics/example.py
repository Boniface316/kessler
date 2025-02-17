from __future__ import annotations

import abc
import typing as T

import pydantic as pdt

from ..io.schemas import Targets, Outputs

from ._base import Metric, MlflowThreshold


class ExampleMetric(Metric):
    """Compute metrics with sklearn.

    Parameters:
        name (str): name of the sklearn metric.
        greater_is_better (bool): maximize or minimize.
    """

    KIND: T.Literal["ExampleMetric"] = "ExampleMetric"

    name: str = "Example Metric"
    greater_is_better: bool = False

    @T.override
    def score(self, targets: Targets, outputs: Outputs) -> float:
        return 50.00


# %% THRESHOLDS


class Threshold(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """A project threshold for a metric.

    Use thresholds to monitor model performances.
    e.g., to trigger an alert when a threshold is met.

    Parameters:
        threshold (int | float): absolute threshold value.
        greater_is_better (bool): maximize or minimize result.
    """

    threshold: int | float
    greater_is_better: bool

    def to_mlflow(self) -> MlflowThreshold:
        """Convert the threshold to an mlflow threshold.

        Returns:
            MlflowThreshold: the mlflow threshold.
        """
        return MlflowThreshold(
            threshold=self.threshold, greater_is_better=self.greater_is_better
        )
