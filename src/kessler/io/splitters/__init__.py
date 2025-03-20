from .._base import Splitter, TrainTestSplits, Index, TrainTestIndex
from .example import ExampleSplitter

SplitterKind =  ExampleSplitter

__all__ = [
    "ExampleSplitter",
    "SplitterKind",
    "Splitter",
    "TrainTestSplits",
    "Index",
    "TrainTestIndex"
]