import datetime
import pathlib
import sys
from typing import Any, Dict, Optional, Union

import ruamel.yaml
from orjson import orjson
from ruamel.yaml import scalarstring

from gitops_utils.cases import (
    FirstLetterCase,
    is_nothing,
    recase_first_char_of_map_keys,
)
from gitops_utils.errors import InvalidYAMLSyntaxError
from gitops_utils.patterns import IS_MULTILINE_CONDITIONAL_EXPR


def all_elements_as_strings(raw_data: Any) -> Any:
    """
    Convert all elements in the raw_data to strings.

    Args:
        raw_data: The data to be converted.

    Returns:
        The converted data with all elements as strings.
    """
    if isinstance(raw_data, (datetime.date, datetime.datetime)):
        return raw_data.isoformat()
    elif isinstance(raw_data, (pathlib.Path, complex, float, int)):
        return str(raw_data)
    elif isinstance(raw_data, (bytes, bytearray, memoryview)):
        return raw_data.decode("utf-8", "replace")
    elif isinstance(raw_data, dict):
        return {k: all_elements_as_strings(v) for k, v in raw_data.items()}
    elif isinstance(raw_data, tuple):
        return tuple(all_elements_as_strings(v) for v in raw_data)
    elif isinstance(raw_data, list):
        return [all_elements_as_strings(v) for v in raw_data]
    elif isinstance(raw_data, frozenset):
        return frozenset(all_elements_as_strings(v) for v in raw_data)
    elif isinstance(raw_data, set):
        return set(all_elements_as_strings(v) for v in raw_data)

    return raw_data


def to_json(raw_data: Any, indent: bool = True, sort_keys: bool = True, **_) -> str:
    """
    Convert raw data to a JSON string.

    Args:
        raw_data: The raw data to be converted.
        indent: Whether to intent the keys in the JSON string by 2 spaces (default is True).
        sort_keys: Whether to sort the keys in the JSON string (default is True).

    Returns:
        The JSON string as a string.

    """
    raw_data = all_elements_as_strings(raw_data)
    options = orjson.OPT_INDENT_2 if indent else 0
    options |= orjson.OPT_SORT_KEYS if sort_keys else 0
    json_bytes = orjson.dumps(raw_data, default=str, option=options)
    return json_bytes.decode("utf-8")


def format_results(results: Any, format_json: bool = True) -> str:
    """
    Format the results.

    Args:
        results (Any): The results to be formatted.
        format_json (bool, optional): Whether to format the results as JSON. Defaults to True.

    Returns:
        str: The formatted results.

    """
    indent, sort_keys = format_json
    return to_json(results, indent=indent, sort_keys=sort_keys)


def cleanup_yaml_string(yaml_string: str) -> str:
    """
    Clean up the YAML string.

    Args:
        yaml_string: The YAML string to clean up.

    Returns:
        The cleaned up YAML string.
    """
    cleaned_yaml_string = (
        yaml_string.strip().replace("${{ ", "${{").replace(" }}", "}}")
    )
    return cleaned_yaml_string


def validate_yaml(yaml_string: str) -> None:
    """
    Validate the YAML string.

    Args:
        yaml_string: The YAML string to validate.

    Raises:
        InvalidYAMLSyntaxError: If the YAML syntax is invalid.
    """
    try:
        ruamel.yaml.safe_load(yaml_string)
    except ruamel.yaml.YAMLError as exc:
        raise InvalidYAMLSyntaxError() from exc


def determine_scalarstring_type(
    yaml_string: str,
) -> Union[scalarstring.PlainScalarString, scalarstring.LiteralScalarString]:
    """
    Determine the type of scalarstring to return.

    Args:
        yaml_string: The YAML string.

    Returns:
        The scalarstring type.
    """
    if IS_MULTILINE_CONDITIONAL_EXPR.search(yaml_string):
        return scalarstring.LiteralScalarString(yaml_string)

    return scalarstring.PlainScalarString(yaml_string)


def export_multiline_yaml_string(
    yaml_string: str,
) -> Union[scalarstring.PlainScalarString, scalarstring.LiteralScalarString]:
    """
    Export a multiline YAML string.

    Args:
        yaml_string: The YAML string to export.

    Returns:
        The exported YAML yaml.
    """
    return determine_scalarstring_type(cleanup_yaml_string(yaml_string))


def get_cloud_call_params(
    max_results: Optional[int] = 10,
    reject_null: bool = True,
    first_letter_case: Optional[FirstLetterCase] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Get the parameters for a cloud call.

    Args:
        max_results: The maximum number of results to return.
        reject_null: Whether to reject null values.
        first_letter_case: The case to convert the first letter of the keys to.

    Returns:
        The parameters for the cloud call.
    """
    params = {k: v for k, v in kwargs.items() if not (is_nothing(v) and reject_null)}

    if max_results is not None:
        params["MaxResults"] = max_results

    if not first_letter_case:
        return params

    return recase_first_char_of_map_keys(params, first_letter_case)


def get_caller():
    """
    Returns the name of the calling function.
    If the calling function is not available, returns "unknown".
    """
    caller_frame = sys._getframe().f_back
    if caller_frame is not None:
        return caller_frame.f_code.co_name
    else:
        return "unknown"
