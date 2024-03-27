import hashlib
import logging
import logging.handlers
import secrets
from collections import defaultdict
from copy import copy, deepcopy
from typing import Any, Dict, List, Literal, Optional, Type

from annotated_types import Gt
from pathvalidate import is_valid_filename
from rich.logging import RichHandler

from gitops_utils import defaults
from gitops_utils.cases import all_non_empty, is_nothing
from gitops_utils.exports import format_results
from gitops_utils.types import FilePath


def get_loggers() -> list[str]:
    """
    Returns a list of all loggers in the system.
    """
    return [
        logger for logger in logging.Logger.manager.loggerDict if logger is not None
    ]


class Logs:
    """
    The Logs class represents a logger object that can be used for logging messages to the console and/or a file. It provides methods for initializing the logger with various options, getting the list of logging handlers based on the provided options, and generating a unique signature for the object.

    Attributes:
        logger_name (str): The name of the logger.
        formatter (logging.Formatter): The formatter to use for log messages.
        logger (logging.Logger): The logger object.

    Methods:
        __init__: Initializes the logger with the given parameters.
        get_handlers: Gets the list of logging handlers based on the provided options.
        get_console_handler: Returns a console handler for logging.
        get_file_handler: Returns a file handler for logging.
        get_unique_signature: Returns a unique signature for the object.
    """

    def __init__(
        self,
        logger_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        log_marker: Optional[str] = None,
        logged_statements_allowlist: List[str] = None,
        logged_statements_denylist: List[str] = None,
        formatter: Optional[logging.Formatter] = None,
        handlers: Optional[List[logging.Handler]] = None,
        to_console: Optional[bool] = defaults.CONSOLE_LOGGING,
        to_file: Optional[bool] = defaults.FILE_LOGGING,
        console_handler_type: Type[logging.Handler] = RichHandler,
        console_handler_opts: Optional[Dict[str, Any]] = None,
        console_handler_formatter: Optional[logging.Formatter] = None,
        log_file_name: str = defaults.LOG_FILE_NAME,
        log_dir: FilePath = defaults.LOG_DIR,
        file_handler_type: Type[
            logging.Handler
        ] = logging.handlers.TimedRotatingFileHandler,
        file_handler_opts: Optional[Dict[str, Any]] = None,
        file_handler_formatter: Optional[logging.Formatter] = None,
        verbose: Optional[bool] = None,
        verbosity: Optional[int] = None,
        debug_markers: Optional[List[str]] = None,
    ):
        """
        Initialize the logger with the given parameters.

        Args:
            logger_name (Optional[str]): The name of the logger. If not provided, a unique signature will be generated.
            logger (Optional[logging.Logger]): The logger object to use. If not provided, a new logger will be created.
            formatter (Optional[logging.Formatter]): The formatter to use for log messages. If not provided, the default formatter will be used.
            handlers (Optional[List[logging.Handler]]): The list of handlers to use for the logger. If not provided, the default handlers will be used.
            to_console (Optional[bool]): Whether to log messages to the console. Defaults to True.
            to_file (Optional[bool]): Whether to log messages to a file. Defaults to True.
            console_handler_type (Type[logging.Handler]): The type of handler to use for console logging. Defaults to RichHandler.
            console_handler_opts (Optional[Dict[str, Any]]): Additional options for the console handler. Defaults to None.
            console_handler_formatter (Optional[logging.Formatter]): The formatter to use for console logging. Defaults to None.
            log_file_name (str): The name of the log file. Defaults to "log.txt".
            file_handler_type (Type[logging.Handler]): The type of handler to use for file logging. Defaults to TimedRotatingFileHandler.
            file_handler_opts (Optional[Dict[str, Any]]): Additional options for the file handler. Defaults to None.
            file_handler_formatter (Optional[logging.Formatter]): The formatter to use for file logging. Defaults to None.
        """
        self.logger_name = logger_name or self.get_unique_signature()
        self.formatter = formatter or defaults.DEFAULT_LOG_FORMATTER
        self.logger = logger or logging.getLogger(self.logger_name)
        self.logger.propagate = defaults.DEFAULT_LOG_PROPAGATION
        self.logger.setLevel(defaults.DEFAULT_LOG_LEVEL)
        self.logger.handlers = self.get_handlers(
            handlers=handlers,
            to_console=to_console,
            to_file=to_file,
            console_handler_type=console_handler_type,
            console_handler_opts=console_handler_opts,
            console_handler_formatter=console_handler_formatter,
            log_file_name=log_file_name,
            file_handler_type=file_handler_type,
            file_handler_opts=file_handler_opts,
            file_handler_formatter=file_handler_formatter,
        )

        self.logs = defaultdict(set)
        self.default_log_marker = log_marker
        self.default_logged_statements_allowlist = logged_statements_allowlist
        self.default_logged_statements_denylist = logged_statements_denylist

        self.active_marker = None

        self.VERBOSE = verbose
        self.VERBOSITY = verbosity
        self.DEBUG_MARKERS = debug_markers
        self.LOG_DIR = log_dir
        self.LOG_FILE_NAME = log_file_name

        try:
            self.VERBOSITY = int(self.VERBOSITY)
        except ValueError:
            self.logger.warning(
                f"Cannot use non-numeric verbosity {self.VERBOSITY} as the verbosity for this run, defaulting to {defaults.VERBOSITY}",
                exc_info=True,
            )
            self.VERBOSITY = defaults.VERBOSITY

    def verbosity_exceeded(self, verbose: bool, verbosity: int) -> bool:
        """
        Check if the verbosity level exceeds the specified threshold.

        Parameters:
        - verbose (bool): Whether verbose mode is enabled.
        - verbosity (int): The verbosity level.

        Returns:
        - bool: True if the verbosity level exceeds the threshold, False otherwise.
        """
        if self.active_marker in self.DEBUG_MARKERS:
            return False

        if verbosity > 1:
            verbose = True

        if not self.VERBOSE and verbose:
            return True

        if verbosity > self.VERBOSITY:
            return True

        return False

    def logged_statement(
        self,
        msg: str,
        json_data: Optional[List[Dict[str, Any]] | Dict[str, Any]] = None,
        labeled_json_data: Optional[Dict[str, Dict[str, Any]]] = None,
        identifiers: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        verbosity: Optional[int] = 1,
        active_marker: Optional[str] = None,
        log_level: Literal[
            "debug", "info", "warning", "error", "fatal", "critical"
        ] = "debug",
        log_marker: Optional[str] = None,
        allowlist: Optional[List[str]] = None,
        denylist: Optional[List[str]] = None,
    ):
        if not is_nothing(active_marker):
            self.active_marker = active_marker

        if not is_nothing(self.active_marker):
            msg = f"[{self.active_marker}] {msg}"

        if self.verbosity_exceeded(verbose, verbosity):
            return

        if not is_nothing(identifiers):
            msg += " (" + ", ".join(all_non_empty(*identifiers)) + ")"

        if labeled_json_data is not None:
            for label, jd in deepcopy(labeled_json_data).items():
                if not isinstance(jd, Dict):
                    jd = {label: jd}

                    msg += "\n:" + format_results(jd, format_json=True)

                    continue

                msg += f"\n{label}:\n" + format_results(jd, format_json=True)

        if json_data is not None:
            if isinstance(json_data, list):
                unlabeled_json_data = deepcopy(json_data)
            else:
                unlabeled_json_data = [copy(json_data)]

            for jd in unlabeled_json_data:
                msg += "\n:" + format_results(jd, format_json=True)

        if is_nothing(log_marker):
            log_marker = self.default_log_marker

        if is_nothing(allowlist):
            allowlist = self.default_logged_statements_allowlist

        if allowlist is None:
            allowlist = []

        if is_nothing(denylist):
            denylist = self.default_logged_statements_denylist

        if denylist is None:
            denylist = []

        if (
            not is_nothing(log_marker)
            and (is_nothing(allowlist) or log_level in allowlist)
            and log_level not in denylist
        ):
            if log_level not in ["debug", "info"]:
                self.logs[log_marker].add(f":warning: {msg}")
            else:
                self.logs[log_marker].add(msg)

        logger = getattr(self.logger, log_level)
        logger(msg)
        return msg

    def get_handlers(
        self,
        handlers: Optional[List[logging.Handler]] = None,
        to_console: Optional[bool] = defaults.CONSOLE_LOGGING,
        to_file: Optional[bool] = defaults.FILE_LOGGING,
        console_handler_type: Type[logging.Handler] = RichHandler,
        console_handler_opts: Optional[Dict[str, Any]] = None,
        console_handler_formatter: Optional[logging.Formatter] = None,
        log_file_name: str = defaults.LOG_FILE_NAME,
        file_handler_type: Type[
            logging.Handler
        ] = logging.handlers.TimedRotatingFileHandler,
        file_handler_opts: Optional[Dict[str, Any]] = None,
        file_handler_formatter: Optional[logging.Formatter] = None,
    ):
        """
        Get the list of logging handlers based on the provided options.

        Args:
            handlers: Optional list of existing logging handlers.
            to_console: Optional flag indicating whether to add a console handler.
            to_file: Optional flag indicating whether to add a file handler.
            console_handler_type: Type of console handler to use.
            console_handler_opts: Optional options for the console handler.
            console_handler_formatter: Optional formatter for the console handler.
            log_file_name: Name of the log file.
            file_handler_type: Type of file handler to use.
            file_handler_opts: Optional options for the file handler.
            file_handler_formatter: Optional formatter for the file handler.

        Returns:
            List of logging handlers.
        """
        new_handlers = list(handlers) if handlers else []

        if to_console:
            new_handlers.append(
                self.get_console_handler(
                    console_handler_type=console_handler_type,
                    console_handler_opts=console_handler_opts,
                    console_handler_formatter=console_handler_formatter,
                    formatter=self.formatter,
                )
            )

        if to_file:
            new_handlers.append(
                self.get_file_handler(
                    log_file_name=log_file_name,
                    file_handler_type=file_handler_type,
                    file_handler_opts=file_handler_opts,
                    file_handler_formatter=file_handler_formatter,
                    formatter=self.formatter,
                )
            )

        return new_handlers

    def get_console_handler(
        self,
        console_handler_type: Type[logging.Handler] = RichHandler,
        console_handler_opts: Optional[Dict[str, Any]] = None,
        console_handler_formatter: Optional[logging.Formatter] = None,
        **_,
    ) -> logging.Handler:
        """
        Returns a console handler for logging.

        Args:
            console_handler_type (Type[logging.Handler]): The type of handler to use for console logging. Defaults to RichHandler.
            console_handler_opts (Optional[Dict[str, Any]]): Additional options for the console handler. Defaults to None.
            console_handler_formatter (Optional[logging.Formatter]): The formatter to use for log messages. Defaults to None.

        Returns:
            logging.Handler: The console handler.
        """
        console_handler_opts = (
            console_handler_opts
            or defaults.LoggingHandlerDefaults.get_handler_defaults(
                console_handler_type
            )
        )
        console_handler = console_handler_type(**console_handler_opts)
        console_handler_formatter = console_handler_formatter or self.formatter
        console_handler.setFormatter(console_handler_formatter)
        return console_handler

    def get_file_handler(
        self,
        log_file_name: str = defaults.LOG_FILE_NAME,
        file_handler_type: Type[
            logging.Handler
        ] = logging.handlers.TimedRotatingFileHandler,
        file_handler_opts: Optional[Dict[str, Any]] = None,
        file_handler_formatter: Optional[logging.Formatter] = None,
        **_,
    ) -> logging.Handler:
        file_handler_opts = (
            file_handler_opts
            or defaults.LoggingHandlerDefaults.get_handler_defaults(file_handler_type)
        )
        if not is_valid_filename(log_file_name):
            log_file_name = defaults.LOG_FILE_NAME

        file_handler = file_handler_type(log_file_name, **file_handler_opts)
        file_handler_formatter = file_handler_formatter or self.formatter
        file_handler.setFormatter(file_handler_formatter)
        return file_handler

    def get_unique_signature(
        self, delim: str = ":", maxlen: Optional[Gt(0)] = None
    ) -> str:
        """
        Returns a unique signature for the object.

        Args:
            delim (str): The delimiter to use in the signature. Defaults to ':'.
            maxlen (Optional[Gt(0)]): The maximum length of the signature. Defaults to None.

        Returns:
            str: The unique signature.
        """
        sig = f"{self.__class__.__name__}{delim}{secrets.token_hex(16)}"
        hashed_sig = hashlib.blake2b(sig.encode()).hexdigest()
        return hashed_sig[: maxlen % len(hashed_sig)] if maxlen else hashed_sig
