import base64
import binascii
import datetime
import json
import logging
import logging.config
import os
import pathlib
import re
import subprocess
import sys
import uuid
from collections import defaultdict
from collections.abc import MutableMapping
from copy import deepcopy, copy
from inspect import getmembers, isfunction
from pathlib import Path
from shlex import split as shlex_split
from typing import Optional, Mapping, List, Any, Union, Dict, Literal

import hcl2
import inflection
import lark.exceptions
import numpy as np
import orjson
import requests
import validators
from case_insensitive_dict import CaseInsensitiveDict
from deepmerge import Merger
from filelock import Timeout, FileLock
from more_itertools import split_before
from ruamel.yaml import YAML, StringIO, YAMLError, scalarstring
from sortedcontainers import SortedDict

from gitops_utils.log_formatter import LogFormatter
from gitops_utils import defaults

FilePath = Union[str, bytes, os.PathLike]


def get_caller():
    return sys._getframe().f_back.f_code.co_name


def is_url(url: FilePath) -> bool:
    if validators.url(str(url).strip()) is True:
        return True

    return False


def titleize_name(name: str):
    proper_name = []

    for n in ["".join(i) for i in split_before(name, pred=lambda s: s.isupper())]:
        proper_name.append(n.title())

    return "".join(proper_name)


def zipmap(a: List[str], b: List[str]):
    zipped = {}

    for idx, val in enumerate(a):
        if idx >= len(b):
            break

        zipped[val] = b[idx]

    return zipped


def filter_methods(methods):
    filtered = []

    for method in methods:
        if method.startswith("_"):
            continue

        filtered.append(method)

    return filtered


def get_available_methods(cls):
    module_name = cls.__module__
    methods = getmembers(cls, isfunction)
    unique_methods = {}

    for method_name, method_signature in methods:
        if (
            "__" in method_name
            or method_signature.__module__ != module_name
            or "NOPARSE" in method_signature.__doc__
        ):
            continue

        unique_methods[method_name] = method_signature.__doc__

    return unique_methods


def get_process_output(cmd):
    try:
        results = subprocess.run(shlex_split(cmd), capture_output=True, text=True)
    except FileNotFoundError:
        return None, None
    return results.stdout, results.stderr


def get_tld():
    stdout, stderr = get_process_output("git rev-parse --show-toplevel")

    if stdout is None:
        return None

    return Path(stdout.strip()).resolve(strict=True)


def lower_first_char(inp: str):
    return inp[:1].lower() + inp[1:] if inp else ""


def upper_first_char(inp: str):
    return inp[:1].upper() + inp[1:] if inp else ""


def get_cloud_call_params(
    max_results: Optional[int] = 10,
    reject_null: bool = True,
    first_letter_to_lower: bool = False,
    first_letter_to_upper: bool = False,
    **kwargs,
):
    params = {k: v for k, v in kwargs.items() if not is_nothing(v) or not reject_null}

    if max_results:
        params["MaxResults"] = max_results

    if not first_letter_to_lower and not first_letter_to_upper:
        return params

    if first_letter_to_lower:
        params = {lower_first_char(k): v for k, v in params.items()}

    if first_letter_to_upper:
        params = {upper_first_char(k): v for k, v in params.items()}

    return params


def strtobool(val, raise_on_error=False):
    if val is True or val is False or val is None:
        return val

    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    elif raise_on_error:
        raise ValueError("invalid truth value %r" % (val,))
    else:
        return val


def truncate(msg, max_length, ender="..."):
    max_length -= len(ender)

    if len(msg) >= max_length:
        return msg[:max_length] + ender

    return msg


def is_nothing(v: Any):
    if v in [None, "", {}, []]:
        return True

    if str(v) == "" or str(v).isspace():
        return True

    if isinstance(v, list | set):
        v = [vv for vv in v if vv not in [None, "", {}, []]]
        if len(v) == 0:
            return True

    return False


def is_partial_match(
    a: Optional[str], b: Optional[str], check_prefix_only: bool = False
):
    if is_nothing(a) or is_nothing(b):
        return False

    a = a.lower()
    b = b.lower()

    if check_prefix_only:
        return a.startswith(b) or b.startswith(a)

    return a in b or b in a


def is_non_empty_match(a: Any, b: Any):
    if is_nothing(a) or is_nothing(b):
        return False

    if type(a) != type(b):
        return False

    if isinstance(a, str):
        a = a.lower()
        b = b.lower()
    elif isinstance(a, Mapping):
        a = orjson.dumps(a, default=str, option=orjson.OPT_SORT_KEYS).decode("utf-8")
        b = orjson.dumps(b, default=str, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    elif isinstance(a, list):
        a.sort()
        b.sort()

    if a != b:
        return False

    return True


def all_non_empty(*vals):
    return [val for val in vals if not is_nothing(val)]


def are_nothing(*vals):
    if len(all_non_empty(*vals)) > 0:
        return False

    return True


def first_non_empty(*vals):
    non_empty_vals = all_non_empty(*vals)
    if len(non_empty_vals) == 0:
        return None

    return non_empty_vals[0]


def any_non_empty(m: Mapping, *keys):
    for k in keys:
        v = m.get(k)
        if not is_nothing(v):
            return {k: v}


def yield_non_empty(m: Mapping, *keys):
    for k in keys:
        v = m.get(k)
        if not is_nothing(v):
            yield {k: v}


def first_non_empty_value_from_map(m: Mapping, *keys):
    try:
        _, val = next(yield_non_empty(m, keys))
        return val
    except StopIteration:
        return None


def make_raw_data_export_safe(raw_data: Any, export_to_yaml: bool = False):
    if isinstance(raw_data, Mapping):
        return {
            k: make_raw_data_export_safe(v, export_to_yaml=export_to_yaml)
            for k, v in raw_data.items()
        }
    elif isinstance(raw_data, (set, list)):
        return [
            make_raw_data_export_safe(v, export_to_yaml=export_to_yaml)
            for v in raw_data
        ]

    exported_data = copy(raw_data)
    if isinstance(exported_data, (datetime.date, datetime.datetime)):
        exported_data = exported_data.isoformat()
    elif isinstance(exported_data, pathlib.Path):
        exported_data = str(exported_data)

    if not export_to_yaml or not isinstance(exported_data, str):
        return exported_data

    exported_data = exported_data.replace("${{ ", "${{").replace(" }}", "}}")
    if (
        len(exported_data.splitlines()) > 1
        or "||" in exported_data
        or "&&" in exported_data
    ):
        return scalarstring.LiteralScalarString(exported_data)

    return exported_data


def deduplicate_map(m: Mapping):
    deduplicated_map = make_raw_data_export_safe(m)

    for k, v in m.items():
        if isinstance(v, list):
            deduplicated_map[k] = []

            for elem in v:
                if elem in deduplicated_map[k]:
                    continue

                deduplicated_map[k].append(elem)

            continue

        if isinstance(v, Mapping):
            deduplicated_map[k] = deduplicate_map(v)
            continue

        if k not in deduplicated_map:
            deduplicated_map[k] = v

    return deduplicated_map


def all_values_from_map(m: Mapping):
    values = []

    for v in m.values():
        if isinstance(v, Mapping):
            values.extend(all_values_from_map(v))
            continue

        values.append(v)

    return values


def flatten_map(dictionary, parent_key=False, separator="."):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_map(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten_map({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def flatten_list(matrix: List[Any]):
    array = np.array(matrix)
    return list(array.flatten())


def match_file_extensions(
    p: FilePath,
    allowed_extensions: Optional[List[str]],
    denied_extensions: Optional[List[str]] = None,
):
    if allowed_extensions is None:
        allowed_extensions = []

    if denied_extensions is None:
        denied_extensions = []

    allowed_extensions = [ext.removeprefix(".") for ext in allowed_extensions]
    denied_extensions = [ext.removeprefix(".") for ext in denied_extensions]

    p = Path(p)
    if p.name.startswith("."):
        suffix = p.name.removeprefix(".")
    else:
        suffix = p.suffix.removeprefix(".")

    if (
        len(allowed_extensions) > 0 and suffix not in allowed_extensions
    ) or suffix in denied_extensions:
        return False

    return True


def format_results(
    results: Any,
    format_json: bool = True,
    **format_opts,
):
    if format_json:
        format_opts["indent"] = format_opts.get("indent", 2)
        format_opts["sort_keys"] = format_opts.get("sort_keys", True)

    indent = format_opts.get("indent")
    sort_keys = format_opts.get("sort_keys")

    orjson_opts = None
    if indent:
        orjson_opts = orjson.OPT_INDENT_2

    if sort_keys:
        if orjson_opts is None:
            orjson_opts = orjson.OPT_SORT_KEYS
        else:
            orjson_opts |= orjson.OPT_SORT_KEYS

    results = make_raw_data_export_safe(results)

    try:
        return orjson.dumps(results, default=str, option=orjson_opts).decode("utf-8")
    except TypeError:
        return json.dumps(results, default=str, **format_opts)


def get_log_level(level):
    if level is None:
        return defaults.DEFAULT_LOG_LEVEL

    if isinstance(level, int):
        return level

    return getattr(logging, level.upper(), defaults.DEFAULT_LOG_LEVEL)


def get_loggers():
    loggers = [logging.getLogger()]
    loggers += [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    return loggers


def find_logger(name: str):
    loggers = get_loggers()
    for logger in loggers:
        if logger.name == name:
            return logger

    return None


def get_default_dict(sorted: bool = False, default_type: Any = dict):
    return defaultdict(lambda: defaultdict(SortedDict() if sorted else default_type))


def file_path_depth(file_path: FilePath):
    depth = []

    for p in Path(file_path).parts:
        if p == ".":
            continue

        depth.append(p)

    return len(depth)


def file_path_rel_to_root(file_path: FilePath):
    depth = file_path_depth(file_path)
    path_rel_to_root = []

    for i in range(depth):
        path_rel_to_root.append("..")

    return "/".join(path_rel_to_root)


def sanitize_key(key: str, delim: str = "_"):
    return "".join(map(lambda x: x if (x.isalnum() or x == delim) else delim, key))


def unhump_map(m: Mapping[str, Any], drop_without_prefix: Optional[str] = None):
    unhumped = {}

    for k, v in m.items():
        if drop_without_prefix is not None and not k.startswith(drop_without_prefix):
            continue

        unhumped_key = inflection.underscore(k)

        if isinstance(v, Mapping):
            unhumped[unhumped_key] = unhump_map(v)
            continue

        unhumped[unhumped_key] = v

    return unhumped


def filter_list(
    l: Optional[List[str]],
    allowlist: List[str] = None,
    denylist: List[str] = None,
):
    if l is None:
        l = []

    if allowlist is None:
        allowlist = []

    if denylist is None:
        denylist = []

    filtered = []

    for elem in l:
        if (len(allowlist) > 0 and elem not in allowlist) or elem in denylist:
            continue

        filtered.append(elem)

    return filtered


class Utils:
    def __init__(
        self,
        inputs: Optional[Any] = None,
        from_environment: bool = True,
        from_stdin: bool = False,
        to_console: bool = False,
        to_file: bool = True,
        logger: Optional[logging.Logger] = None,
        log_marker: Optional[str] = None,
        logged_statements_allowlist: List[str] = None,
        logged_statements_denylist: List[str] = None,
        **kwargs,
    ):
        self.logs = defaultdict(set)
        self.default_log_marker = log_marker
        self.default_logged_statements_allowlist = logged_statements_allowlist
        self.default_logged_statements_denylist = logged_statements_denylist
        self.errors = []
        self.last_error = None
        self.last_error_message = None

        if inputs is None:
            inputs = {}

        if from_environment:
            inputs = os.environ | inputs

        if from_stdin and not strtobool(os.getenv("OVERRIDE_STDIN", False)):
            inputs_from_stdin = sys.stdin.read()

            if not is_nothing(inputs_from_stdin):
                try:
                    inputs = json.loads(inputs_from_stdin) | inputs
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Failed to decode stdin:\n{inputs_from_stdin}"
                    ) from exc

        self.from_stdin = from_stdin
        self.inputs = CaseInsensitiveDict(inputs)
        self.frozen_inputs = CaseInsensitiveDict()

        self.active_marker = None
        self.log_file_count = 0

        self.tld = get_tld()
        self.merger = Merger(
            [(list, ["append"]), (dict, ["merge"]), (set, ["union"])],
            ["override"],
            ["override"],
        )

        self.params = self.get_params(**kwargs)
        for k, v in self.params.items():
            setattr(self, k.upper(), v)

        self.to_console = to_console
        self.to_file = to_file

        self.logger = self.get_logger(logger=logger)

        self.logged_statement(
            "Params",
            json_data=self.params,
            verbose=True,
        )

    def multi_merge(self, *maps):
        merged_maps = {}
        for m in maps:
            merged_maps = self.merger.merge(merged_maps, m)

        return merged_maps

    def local_path(self, file_path: FilePath):
        path = Path(file_path)
        if path.is_absolute():
            return path.resolve()

        if self.tld is None:
            caller = get_caller()
            raise RuntimeError(
                f"[{caller}] CLI is not being run locally and has no top level directory to use with {file_path}"
            )

        return Path(self.tld, file_path).resolve()

    def local_path_exists(self, file_path: FilePath):
        caller = get_caller()

        if is_nothing(file_path):
            raise RuntimeError(f"File path being checked from {caller} is empty")

        local_file_path = self.local_path(file_path)

        if not local_file_path.exists():
            raise NotADirectoryError(
                f"Directory {local_file_path} from {caller} does not exist locally"
            )

        return local_file_path

    def get_repository_dir(self, dir_path: FilePath):
        if self.tld:
            repo_dir_path = self.local_path(dir_path)
            repo_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            repo_dir_path = Path(dir_path)

        return repo_dir_path

    def get_params(self, **kwargs) -> Dict[str, Any]:
        params = self.decode_input(
            "params", default={}, required=False, allow_none=False
        )
        params["verbose"] = kwargs.get(
            "verbose",
            self.get_input(
                "verbose",
                default=params.get("verbose", defaults.VERBOSE),
                is_bool=True,
            ),
        )
        params["verbosity"] = kwargs.get(
            "verbosity",
            self.get_input(
                "verbosity", default=params.get("verbosity", defaults.VERBOSITY)
            ),
        )

        params["debug_markers"] = kwargs.get(
            "debug_markers",
            self.decode_input(
                "debug_markers",
                default=params.get("debug_markers", []),
                decode_from_base64=False,
            ),
        )

        params["log_dir"] = kwargs.get(
            "log_dir",
            self.get_input(
                "log_dir",
                default=params.get("log_dir", defaults.LOG_DIR),
            ),
        )

        params["log_results_dir"] = kwargs.get(
            "log_results_dir",
            self.get_input(
                "log_results_dir",
                default=params.get("log_results_dir"),
            ),
        )

        params["log_file_name"] = kwargs.get(
            "log_file_name",
            self.get_input(
                "log_file_name",
                default=params.get("log_file_name", defaults.LOG_FILE_NAME),
            ),
        )

        return params

    def get_logger(
        self,
        to_console: Optional[bool] = None,
        to_file: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
        logger_name: Optional[str] = None,
        log_file_name: Optional[str] = None,
    ):
        if to_console is None:
            to_console = self.to_console

        if to_file is None:
            to_file = self.to_file

        if log_file_name is None:
            log_file_name = self.LOG_FILE_NAME

        if logger_name is None:
            logger_name = self.get_unique_signature()

        if logger is None:
            logger = logging.getLogger(logger_name)

        logger.propagate = True
        logger.setLevel(logging.DEBUG)

        gunicorn_logger = find_logger("gunicorn.error")
        if gunicorn_logger:
            logger.handlers = gunicorn_logger.handlers
            logger.setLevel(gunicorn_logger.level)
            to_console = False

        if to_console:
            console_handler = logging.StreamHandler(sys.stderr)
            console_formatter = LogFormatter(color=True)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        if to_file:
            log_file_name = re.sub("[^0-9a-zA-Z]+", "_", log_file_name.rstrip(".log"))

            if not log_file_name[:1].isalnum():
                first_alpha = re.search(r"[A-Za-z0-9]", log_file_name)
                if not first_alpha:
                    raise RuntimeError(
                        f"Malformed log file name: {log_file_name} must contain at least one ASCII character"
                    )

                log_file_name = log_file_name[first_alpha.start() :]

            log_file = f"{log_file_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = LogFormatter(color=False)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def get_unique_signature(self, delim="/", maxlen: Optional[int] = None):
        sig = self.__class__.__module__ + delim + self.__class__.__name__
        if maxlen is None:
            return sig

        return sig[:maxlen]

    def verbosity_exceeded(self, verbose: bool, verbosity: int):
        debug_markers = getattr(self, "DEBUG_MARKERS", [])
        if self.active_marker in debug_markers:
            return False

        if verbosity > 1:
            verbose = True

        if not getattr(self, "VERBOSE", None) and verbose:
            return True

        try:
            max_verbosity = int(getattr(self, "VERBOSITY", defaults.VERBOSITY))
        except ValueError:
            max_verbosity = defaults.VERBOSITY

        if verbosity > max_verbosity:
            return True

        return False

    def logged_statement(
        self,
        msg: str,
        json_data: Optional[List[Mapping[str, Any]] | Mapping[str, Any]] = None,
        labeled_json_data: Optional[Mapping[str, Mapping[str, Any]]] = None,
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
                if not isinstance(jd, Mapping):
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

    def get_unique_sub_path(self, dir_path: FilePath):
        local_dir_path = self.local_path(dir_path)

        def get_sub_path():
            return local_dir_path.joinpath(str(uuid.uuid1()))

        local_sub_path = get_sub_path()

        while local_sub_path.exists():
            local_sub_path = get_sub_path()

        return local_sub_path

    def get_rel_to_root(self, dir_path: FilePath):
        try:
            return self.tld.relative_to(dir_path)
        except (ValueError, AttributeError):
            self.logger.warning(
                f"Could not calculate path for directory {dir_path} relative to the repository TLD {self.tld}",
                exc_info=True,
            )

        return None

    def log_results(
        self,
        results,
        log_file_name,
        no_formatting=False,
        ext=None,
        verbose=False,
        verbosity=0,
    ):
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

        log_file_name = log_file_name.replace(" ", "_").lower()
        log_file_name_with_ext = log_file_name + ".log"
        log_file_path = log_dir.joinpath(log_file_name_with_ext)

        while log_file_path.exists():
            self.log_file_count += 1
            log_file_name_with_ext = log_file_name + f".{self.log_file_count}.log"
            log_file_path = log_dir.joinpath(log_file_name_with_ext)

        with open(log_file_path, "w") as fh:
            fh.write(results)

        self.logged_statement(f"New results log: {log_file_path}")

        return results

    def filter_map(
        self,
        m: Optional[Mapping[str, Any]],
        allowlist: List[str] = None,
        denylist: List[str] = None,
    ):
        if m is None:
            m = {}

        if allowlist is None:
            allowlist = []

        if denylist is None:
            denylist = []

        fm = {}
        rm = {}

        for k, v in m.items():
            self.logged_statement(
                f"Checking if {k} is allowed", verbose=True, verbosity=2
            )
            if (len(allowlist) > 0 and k not in allowlist) or k in denylist:
                self.logged_statement(
                    f"Removing {k} from map: {list(m.keys())},"
                    f" either not in allowlist: {allowlist},"
                    f" or in denylist: {denylist}",
                    verbose=True,
                    verbosity=2,
                )
                rm[k] = v
            else:
                self.logged_statement(
                    f"{k} is allowed, allowing its value '{v}' through",
                    verbose=True,
                    verbosity=2,
                )
                fm[k] = v

        return fm, rm

    def get_input(
        self, k, default=None, required=False, is_bool=False, is_integer=False
    ) -> Any:
        inp = self.inputs.get(k, default)

        if is_nothing(inp):
            inp = default

        if is_bool:
            inp = strtobool(inp)

        if is_integer and inp is not None:
            try:
                inp = int(inp)
            except TypeError as exc:
                raise RuntimeError(f"Input {k} not an integer: {inp}") from exc

        if is_nothing(inp) and required:
            raise RuntimeError(
                f"Required input {k} not passed from inputs:\n{self.inputs}"
            )

        return inp

    def decode_input(
        self,
        k: str,
        default: Optional[Any] = None,
        required: bool = False,
        decode_from_json: bool = True,
        decode_from_base64: bool = True,
        allow_none: bool = True,
    ):
        conf = self.get_input(k, default=default, required=required)

        if conf is None:
            return conf

        if conf == default:
            return conf

        if decode_from_base64:
            try:
                conf = base64.b64decode(conf, validate=True).decode("utf-8")
            except binascii.Error as exc:
                raise RuntimeError(f"Failed to decode {conf} from base64") from exc

        if decode_from_json:
            try:
                conf = json.loads(conf)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to decode {conf} from JSON") from exc

        if conf is None and not allow_none:
            return default

        return conf

    def freeze_inputs(self):
        if is_nothing(self.frozen_inputs):
            self.logger.warning("Freezing all inputs")
            self.frozen_inputs = deepcopy(self.inputs)
            self.inputs = {}

        return self.frozen_inputs

    def thaw_inputs(self):
        if is_nothing(self.inputs):
            self.logger.warning("Thawing all inputs")
            self.inputs = deepcopy(self.frozen_inputs)
            self.frozen_inputs = {}
            return self.inputs

        self.logger.warning("Thawing frozen inputs into new inputs")
        self.inputs = self.merger.merge(
            deepcopy(self.inputs), deepcopy(self.frozen_inputs)
        )
        self.frozen_inputs = {}
        return self.inputs

    def shift_inputs(self):
        if is_nothing(self.frozen_inputs):
            return self.freeze_inputs()

        return self.thaw_inputs()

    def get_file(
        self,
        file_path: FilePath,
        decode: Optional[bool] = True,
        return_path: Optional[bool] = False,
        charset: Optional[str] = "utf-8",
        errors: Optional[str] = "strict",
        headers: Optional[Mapping[str, str]] = None,
        raise_on_not_found: bool = False,
    ):
        if headers is None:
            headers = {}

        file_data = ""

        def state_negative_result(result: str):
            self.logger.warning(result)

            if raise_on_not_found:
                raise FileNotFoundError(result)

        try:
            if is_url(str(file_path)):
                self.logged_statement(f"Getting remote URL: {file_path}")
                response = requests.get(file_path, headers=headers)
                if response.ok:
                    file_data = response.content.decode(charset, errors)
                else:
                    state_negative_result(
                        f"URL {file_path} could not be read:"
                        f" [{response.status_code}] {response.reason}"
                    )
            else:
                local_file = self.local_path(file_path)

                self.logged_statement(f"Getting local file: {local_file}")

                if local_file.is_file():
                    file_data = local_file.read_text(charset, errors)
                else:
                    state_negative_result(f"{file_path} does not exist")
        except ValueError as exc:
            self.logger.warning(f"Reading {file_path} not supported: {exc}")
            decode = False

        if decode:
            self.logged_statement(f"Decoding {file_path}")
            file_data = (
                {}
                if is_nothing(file_data)
                else self.decode_file(file_data=file_data, file_path=file_path)
            )

        retval = [file_data]

        if return_path:
            retval.append(file_path)

        if len(retval) == 1:
            return retval[0]

        return tuple(retval)

    def decode_file(
        self,
        file_data: str,
        file_path: Optional[FilePath] = None,
        suffix: Optional[str] = None,
    ):
        yaml = YAML(typ="safe")
        file_data_stream = StringIO(file_data)

        if suffix is None:
            if file_path is not None:
                self.logger.info(f"Decoding file {file_path}")
                suffix = Path(file_path).suffix.lstrip(".").lower()

        if suffix == "yml" or suffix == "yaml":
            self.logger.info(f"Data is being loaded from YAML")

            try:
                return yaml.load(file_data_stream)
            except YAMLError as exc:
                self.logger.warning(
                    f"Decoding data from YAML resulted in error: {exc},"
                    f" trying to read it as multiple documents"
                )

                try:
                    yaml_data = {}

                    for doc in yaml.load_all(file_data_stream):
                        yaml_data = self.merger.merge(yaml_data, doc)

                    return yaml_data
                except YAMLError as exc:
                    raise RuntimeError(f"Failed to parse YAML file") from exc
        elif suffix == "json":
            self.logger.info(f"Data is being loaded from JSON")

            try:
                return json.loads(file_data)
            except (json.JSONDecodeError, TypeError) as exc:
                raise RuntimeError(f"Failed to parse JSON file") from exc
        elif suffix == "tf":
            self.logger.info(f"Data is being loaded from HCL2")

            try:
                return hcl2.load(file_data_stream)
            except lark.exceptions.ParseError as exc:
                raise RuntimeError(f"Failed to parse HCL file") from exc
        else:
            try:
                return json.loads(file_data)
            except json.JSONDecodeError as exc:
                self.logger.info(f"Failed to decode from JSON, trying YAML: {exc}")

                try:
                    return yaml.load(file_data_stream)
                except YAMLError as exc:
                    self.logger.warning(
                        f"Decoding data from YAML resulted in error: {exc},"
                        f" trying to read it as multiple documents"
                    )

                    try:
                        yaml_data = {}

                        for doc in yaml.load_all(file_data_stream):
                            yaml_data = self.merger.merge(yaml_data, doc)

                        return yaml_data
                    except YAMLError as exc:
                        self.logger.info(
                            f"Failed to decode from YAML, trying HCL: {exc}"
                        )
                        try:
                            return hcl2.load(file_data_stream)
                        except lark.exceptions.ParseError as exc:
                            raise RuntimeError(
                                f"Failed to parse raw data, format is unknown and all known parsers failed"
                            ) from exc

    def update_file(
        self,
        file_path: FilePath,
        file_data: Any,
        encode_with_json: Optional[bool] = False,
        allow_empty: Optional[bool] = False,
    ):
        if is_nothing(file_data) and not allow_empty:
            self.logger.warning(f"Empty file data for {file_path} not allowed")
            return None

        if encode_with_json:
            file_data = format_results(file_data, format_json=True)

        if not isinstance(file_data, str):
            file_data = str(file_data)

        self.logged_statement(f"Updating local file {file_path}", verbose=True)
        lock = FileLock(f"{file_path}.lock", timeout=defaults.MAX_FILE_LOCK_WAIT)

        try:
            with lock.acquire():
                local_file = self.local_path(file_path)

                self.logged_statement(f"Updating local file: {local_file}")

                local_file.parent.mkdir(parents=True, exist_ok=True)

                return local_file.write_text(file_data)
        except Timeout:
            raise RuntimeError(
                f"Cannot update file path {file_path},"
                f" another instance of this application currently holds the lock."
            )
        finally:
            lock.release()
            self.delete_file(lock.lock_file)

    def delete_file(self, file_path: FilePath):
        local_file = self.local_path(file_path)
        self.logger.warning(f"Deleting local file {file_path}")
        return local_file.unlink(missing_ok=True)

    def sanitize_map(
        self,
        m: Dict[str, Any],
        delim: str = "_",
        max_sanitize_depth: Optional[int] = None,
        depth: int = 0,
    ):
        if depth >= max_sanitize_depth:
            self.logged_statement(
                f"Max sanitize depth of {max_sanitize_depth} exceeded for map, returning raw map"
            )
            return m

        sanitized = {}

        for k, v in m.items():
            new_k = sanitize_key(key=k, delim=delim)
            new_v = deepcopy(v)

            if isinstance(v, Dict):
                new_v = self.sanitize_map(
                    m=v,
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
