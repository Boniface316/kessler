"""Find the best hyperparameters for a model."""

# %% IMPORTS

import abc
import typing as T

import pandas as pd
import pydantic as pdt
from sklearn import model_selection
from ..models import Model, Params, ParamKey, ParamValue
from ..metrics import Metric
from ..io.splitters import Splitter, TrainTestSplits
from ..io.schemas import Inputs, Targets



# %% TYPES

# Grid of model params
Grid = dict[ParamKey, list[ParamValue]]

# Results of a model search
Results = tuple[
    T.Annotated[pd.DataFrame, "details"],
    T.Annotated[float, "best score"],
    T.Annotated[Params, "best params"],
]

# Cross-validation options for searchers
CrossValidation = int | TrainTestSplits | Splitter

# %% SEARCHERS


class Searcher(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a searcher.

    Use searcher to fine-tune models.
    i.e., to find the best model params.

    Parameters:
        param_grid (Grid): mapping of param key -> values.
    """

    KIND: str

    param_grid: Grid

    @abc.abstractmethod
    def search(
        self,
        model: Model,
        metric: Metric,
        inputs: Inputs,
        targets: Targets,
        cv: CrossValidation,
    ) -> Results:
        """Search the best model for the given inputs and targets.

        Args:
            model (models.Model): AI/ML model to fine-tune.
            metric (metrics.Metric): main metric to optimize.
            inputs (schemas.Inputs): model inputs for tuning.
            targets (schemas.Targets): model targets for tuning.
            cv (CrossValidation): choice for cross-fold validation.

        Returns:
            Results: all the results of the searcher execution process.
        """
