from .training import TrainingJob
from .tuning import TuningJob
from ._base import Job

JobKind = TrainingJob | TuningJob 

__all__ = ["TrainingJob", "TuningJob", "JobKind", "Job" ]
