"""Base for high-level project jobs."""

# %% IMPORTS

import abc
import types as TS
import typing as T

import pydantic as pdt
from ..services import MlflowService, LoggerService, AlertsService
import platform
import psutil
import torch
import os
import mlflow
import json
import pyperclip

# %% TYPES

# Local job variables
Locals = T.Dict[str, T.Any]

# %% JOBS


class Job(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a job.

    use a job to execute runs in  context.
    e.g., to define common services like logger

    Parameters:
        logger_service (LoggerService): manage the logger system.
        alerts_service (AlertsService): manage the alerts system.
        mlflow_service (MlflowService): manage the mlflow system.
    """

    KIND: str

    logger_service: LoggerService = LoggerService()
    alerts_service: AlertsService = AlertsService()
    mlflow_service: MlflowService = MlflowService()

    def __enter__(self) -> T.Self:
        """Enter the job context.

        Returns:
            T.Self: return the current object.
        """

        self.logger_service.start()
        logger = self.logger_service.logger()
        logger.debug("\033[92m[START]\033[0m Logger service: {}", self.logger_service)
        self.alerts_service.start()
        logger.debug("\033[33m[START]\033[0m Alerts service: {}", self.alerts_service)
        self.mlflow_service.start()
        logger.debug("\033[36m[START]\033[0m Mlflow service: {}", self.mlflow_service)
        return self

    def __exit__(
        self,
        exc_type: T.Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TS.TracebackType | None,
    ) -> T.Literal[False]:
        """Exit the job context.

        Args:
            exc_type (T.Type[BaseException] | None): ignored.
            exc_value (BaseException | None): ignored.
            exc_traceback (TS.TracebackType | None): ignored.

        Returns:
            T.Literal[False]: always propagate exceptions.
        """
        logger = self.logger_service.logger()
        logger.debug("\033[91m[STOP]\033[0m Mlflow service: {}", self.mlflow_service)
        self.mlflow_service.stop()
        logger.debug("\033[91m[STOP]\033[0m Alerts service: {}", self.alerts_service)
        self.alerts_service.stop()
        logger.debug("\033[91m[STOP]\033[0m Logger service: {}", self.logger_service)
        self.logger_service.stop()
        print("Logger services stopped")
        return False  # re-raise

    @abc.abstractmethod
    def run(self) -> Locals:
        """Run the job in context.

        Returns:
            Locals: local job variables.
        """

    def get_system_info(self) -> dict:
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "cpu": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "N/A",
        }
    
    def log_system_info(self, logger, artifact_uri): 
        system_information = self.get_system_info()
        logger.info("System Info: {}", system_information)

        # Log system information as an artifact
        system_info_path = os.path.join(artifact_uri, "system_info.json")
        with open(system_info_path, "w") as f:
            json.dump(system_information, f)
        mlflow.log_artifact(system_info_path)
        # run_uri = self.run.info.artifact_uri
        artifact_path = artifact_uri.split("/artifacts")[0]
        logger.info("Artifact path: {}", artifact_path)
        log_file = os.path.join(artifact_path, "TrainingJob.log")
        logger.add(log_file)
        pyperclip.copy(log_file)    
