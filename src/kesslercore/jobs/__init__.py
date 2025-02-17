from .training import TrainingJob
from .tuning import TuningJob
from .dummy import DummyJob
from ._base import Job

JobKind = TrainingJob | TuningJob | DummyJob

__all__ = ["TrainingJob", "TuningJob", "DummyJob", "JobKind", "Job"]
