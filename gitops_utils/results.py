from copy import copy, deepcopy
from typing import Any, Dict, Optional

from werkzeug.utils import secure_filename

from gitops_utils import defaults
from gitops_utils.cases import is_nothing
from gitops_utils.exports import format_results
from gitops_utils.filesystem import Filesystem
from gitops_utils.inputs import Inputs
from gitops_utils.logs import Logs
from gitops_utils.transforms import sanitize_key
from gitops_utils.types import FilePath


class Results(Inputs, Filesystem, Logs):
    def __init__(
        self,
        inputs: Optional[Any] = None,
        from_environment: bool = True,
        from_stdin: bool = False,
        **params,
    ):
        super().__init__(
            inputs=inputs, from_environment=from_environment, from_stdin=from_stdin
        )

        self.log_file_count = 0

        self.errors = []
        self.last_error = None
        self.last_error_message = None

        params = self.merger.merge(
            copy(params),
            self.decode_input("params", default={}, required=False, allow_none=False),
        )
        params["verbose"] = self.get_input(
            "verbose",
            required=False,
            default=params.get("verbose", defaults.VERBOSE),
            input_type=bool,
        )

        params["verbosity"] = self.get_input(
            "verbosity",
            required=False,
            default=params.get("verbosity", defaults.VERBOSITY),
            input_type=int,
        )

        params["debug_markers"] = self.decode_input(
            "debug_markers",
            required=False,
            default=params.get("debug_markers", defaults.DEBUG_MARKERS),
            decode_from_base64=False,
            allow_none=False,
        )

        params["log_dir"] = self.get_input(
            "log_dir",
            required=False,
            default=params.get("log_dir", defaults.LOG_DIR),
            input_type=FilePath,
        )

        params["log_file_name"] = self.get_input(
            "log_file_name",
            required=False,
            default=params.get("log_file_name", defaults.LOG_FILE_NAME),
        )

        super().__init__(**params)

        self.LOG_RESULTS_DIR = self.get_input(
            "log_results_dir",
            required=False,
            default=params.get("log_results_dir"),
            input_type=FilePath,
        )

    def log_results(
        self,
        results: Any,
        log_file_name: str,
        no_formatting: bool = False,
        ext: Optional[str] = None,
        verbose: bool = False,
        verbosity: int = 0,
    ):
        """
        Log the results to a file.

        Args:
            results (Any): The results to be logged.
            log_file_name (str): The name of the log file.
            no_formatting (bool, optional): Whether to skip formatting the results. Defaults to False.
            ext (Optional[str], optional): The extension to be added to the log file name. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            verbosity (int, optional): The level of verbosity. Defaults to 0.

        Returns:
            str: The logged results.

        """
        if self.verbosity_exceeded(verbose, verbosity):
            return

        try:
            if not no_formatting:
                log_file_name += ".json"
                results = format_results(results, format_json=True)
        except TypeError:
            results = str(results)

        if not isinstance(results, str):
            results = str(results)

        if is_nothing(self.LOG_RESULTS_DIR):
            log_dir = self.get_unique_sub_path(self.LOG_DIR)
        else:
            log_dir = self.local_path(self.LOG_RESULTS_DIR)

        log_dir.mkdir(parents=True, exist_ok=True)

        if ext is not None:
            log_file_name += f".{ext}"

        log_file_name = secure_filename(log_file_name)
        log_file_name_with_ext = log_file_name + ".log"
        log_file_path = log_dir.joinpath(log_file_name_with_ext)

        counter = 1
        while log_file_path.exists():
            log_file_name_with_ext = log_file_name + f".{counter}.log"
            log_file_path = log_dir.joinpath(log_file_name_with_ext)
            counter += 1

        with open(log_file_path, "w") as f:
            f.write(results)

        self.logged_statement(f"New results log: {log_file_path}")

        return results

    def sanitize_results(
        self,
        results: Dict[str, Any],
        delim: str = "_",
        max_sanitize_depth: Optional[int] = None,
        depth: int = 0,
    ):
        """
        Sanitizes the results dictionary by replacing non-alphanumeric characters in the keys with a delimiter using regular expressions. This method recursively sanitizes nested dictionaries up to a specified depth.

        Args:
            results (Dict[str, Any]): The dictionary containing the results to be sanitized.
            delim (str, optional): The delimiter to be used for replacing non-alphanumeric characters in the keys. Defaults to "_".
            max_sanitize_depth (Optional[int], optional): The maximum depth to which the sanitization should be applied. If the depth exceeds this value, the method will return the raw dictionary. Defaults to None.
            depth (int, optional): The current depth of the recursion. Defaults to 0.

        Returns:
            Dict[str, Any]: The sanitized dictionary.

        Raises:
            None

        Example:
            results = {
                "key1": "value1",
                "key2": {
                    "nested_key1": "nested_value1",
                    "nested_key2": "nested_value2"
                }
            }
            sanitized_results = sanitize_results(results, delim="_", max_sanitize_depth=2)
            # Output: {
            #     "key1": "value1",
            #     "key2": {
            #         "nested_key1": "nested_value1",
            #         "nested_key2": "nested_value2"
            #     }
            # }

        """
        if depth >= max_sanitize_depth:
            self.logged_statement(
                f"Max sanitize depth of {max_sanitize_depth} exceeded for map, returning raw map"
            )
            return results

        sanitized = {}

        for k, v in results.items():
            new_k = sanitize_key(key=k, delim=delim)
            new_v = deepcopy(v)

            if isinstance(v, Dict):
                new_v = self.sanitize_results(
                    results=v,
                    delim=delim,
                    max_sanitize_depth=max_sanitize_depth,
                    depth=depth + 1,
                )

            if (
                new_k in sanitized
                and isinstance(sanitized[new_k], Dict)
                and isinstance(new_v, Dict)
            ):
                sanitized[new_k] = self.merger.merge(sanitized[new_k], new_v)
                continue

            sanitized[new_k] = new_v

        return sanitized
