import typing as T
from .._base import Splitter, TrainTestSplits, Index
from ..schemas import Inputs, Targets
from sklearn import model_selection
import numpy as np


class ExampleSplitter(Splitter):
    KIND: T.Literal["ExampleSplitter"] = "ExampleSplitter"

    shuffle: bool = False
    test_size: int | float = 0.2
    random_state: int = 42

    @T.override
    def split(
        self,
        inputs: Inputs,
        targets: Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        index = np.arange(len(inputs))
        train_index, test_index = model_selection.train_test_split(
            index,
            shuffle=self.shuffle,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        yield train_index, test_index

    @T.override
    def get_n_splits(
        self,
        inputs: Inputs,
        targets: Targets,
        groups: Index | None = None,
    ) -> int:
        return 1
    