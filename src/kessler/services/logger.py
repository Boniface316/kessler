from ._base import Service

import sys
import typing as T

import loguru


class LoggerService(Service):
    """Service for logging messages.

    https://loguru.readthedocs.io/en/stable/api/logger.html

    Parameters:
        sink (str): logging output.
        level (str): logging level.
        format (str): logging format.
        colorize (bool): colorize output.
        serialize (bool): convert to JSON.
        backtrace (bool): enable exception trace.
        diagnose (bool): enable variable display.
        catch (bool): catch errors during log handling.
    """

    sink: str = "stderr"
    level: str = "DEBUG"
    format: str = (
        "<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green>"
        "<level>[{level}]</level>"
        "<cyan>[{name}:{function}:{line}]</cyan>"
        " <level>{message}</level>"
    )
    colorize: bool = True
    serialize: bool = False
    backtrace: bool = True
    diagnose: bool = False
    catch: bool = True

    @T.override
    def start(self) -> None:
        loguru.logger.remove()
        config = self.model_dump()
        # use standard sinks or keep the original
        sinks = {"stderr": sys.stderr, "stdout": sys.stdout}
        config["sink"] = sinks.get(config["sink"], config["sink"])
        loguru.logger.add(**config)

    def logger(self) -> loguru.logger:
        """Return the main logger.

        Returns:
            loguru.Logger: the main logger.
        """
        return loguru.logger
