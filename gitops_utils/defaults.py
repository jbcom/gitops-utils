import logging
from enum import Enum
from pathlib import Path
from typing import Type

MAX_FILE_LOCK_WAIT = 600
VERBOSE = False
VERBOSITY = 1
DEBUG_MARKERS = []
CONSOLE_LOGGING = True
FILE_LOGGING = False
LOG_DIR = Path.home().joinpath(".config/gitops/logs")
LOG_FILE_NAME = "run.log"
DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_ENCODING = "utf-8"
DEFAULT_LOG_PROPAGATION = False
DEFAULT_LOG_ROTATION_MAX_BYTES = 1024
DEFAULT_LOG_ROTATION_BACKUP_COUNT = 5
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class LoggingHandlerDefaults(Enum):
    """
    Enum class for defining default configurations for logging handlers.
    """

    RichHandler = {"rich_tracebacks": True}

    TimedRotatingFileHandler = {"when": "midnight", "backupCount": 5, "delay": True}

    @classmethod
    def get_handler_defaults(cls, handler_type: Type[logging.Handler]) -> dict:
        """
        Get the default configurations for a logging handler.

        Args:
            handler_type (Type[logging.Handler]): A subclass of `logging.Handler`.

        Returns:
            dict: The default configurations for the handler.
        """
        return cls.__members__.get(handler_type.__name__, {})
