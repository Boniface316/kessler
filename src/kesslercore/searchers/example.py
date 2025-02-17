import typing as T
import numpy as np


from ._base import Searcher, CrossValidation, Results, Grid
from ..models import Model
from ..metrics import Metric
from ..io.schemas import Inputs, Targets
import pandas as pd

class ExampleSearcher(Searcher):
    """Grid searcher with cross-fold validation.

    Convention: metric returns higher values for better models.

    Parameters:
        n_jobs (int, optional): number of jobs to run in parallel.
        refit (bool): refit the model after the tuning.
        verbose (int): set the searcher verbosity level.
        error_score (str | float): strategy or value on error.
        return_train_score (bool): include train scores if True.
    """

    KIND: T.Literal["ExampleSearcher"] = "ExampleSearcher"

    param_grid: Grid | None = None


    @T.override
    def search(
        self,
        model: Model,
        metric: Metric,
        inputs: Inputs,
        targets: Targets,
        cv: CrossValidation,
    ) -> Results:
        searcher = MockSearcher(
            estimator=model,
            scoring=metric,
            cv=cv
        )
        searcher.fit(inputs, targets)
        results = pd.DataFrame(searcher.cv_results_)
        return results, searcher.best_score_, searcher.best_params_
    

class MockSearcher:
    """Mock the GridSearchCV class from sklearn."""

    def __init__(self, estimator: Model, scoring: Metric, cv: CrossValidation):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv


    def fit(
        self,
        inputs: Inputs,
        targets: Targets
    ) -> Results:
        
        self.estimator.fit(inputs, targets)
        prediction = self.estimator.predict(inputs)
        self.scoring.score(targets, prediction)

        
        
        self.cv_results_ = {
            'mean_fit_time': np.array([0.00031738, 0.00030251, 0.0002552, 0.00025935]),
            'std_fit_time': np.array([5.70565897e-05, 5.95263921e-06, 1.29806601e-05, 6.48638766e-06]),
            'mean_score_time': np.array([0.00020456, 0.00023165, 0.00017381, 0.00019579]),
            'std_score_time': np.array([2.66939387e-05, 8.87301577e-06, 1.40646755e-06, 5.81936725e-06]),
            'param_C': np.ma.masked_array(data=[1, 1, 10, 10], mask=[False, False, False, False], fill_value=999999),
            'param_kernel': np.ma.masked_array(data=['linear', 'rbf', 'linear', 'rbf'], mask=[False, False, False, False], fill_value=np.str_('?')),
            'params': [{'C': 1, 'kernel': 'linear'}, {'C': 1, 'kernel': 'rbf'}, {'C': 10, 'kernel': 'linear'}, {'C': 10, 'kernel': 'rbf'}],
            'split0_test_score': np.array([0.96666667, 0.96666667, 1., 0.96666667]),
            'split1_test_score': np.array([1., 0.96666667, 1., 1.]),
            'split2_test_score': np.array([0.96666667, 0.96666667, 0.9, 0.96666667]),
            'split3_test_score': np.array([0.96666667, 0.93333333, 0.96666667, 0.96666667]),
            'split4_test_score': np.array([1., 1., 1., 1.]),
            'mean_test_score': np.array([0.98, 0.96666667, 0.97333333, 0.98]),
            'std_test_score': np.array([0.01632993, 0.02108185, 0.03887301, 0.01632993]),
            'rank_test_score': np.array([1, 4, 3, 1], dtype=np.int32)
        }

        self.best_score_ = 0.98
        self.best_params_ = {'C': 1, 'kernel': 'linear'}
