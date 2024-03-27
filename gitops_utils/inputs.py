import base64
import binascii
import json
import os
import sys
from json import JSONDecodeError
from typing import Any, Callable, Optional, Type, Union

import orjson
from case_insensitive_dict import CaseInsensitiveDict
from deepmerge import Merger

from gitops_utils import defaults
from gitops_utils.cases import is_nothing, strtobool
from gitops_utils.errors import (
    Base64DecodeError,
    InputValueDecodingError,
    InvalidBooleanValueError,
    InvalidEncodingError,
    InvalidInputTypeError,
    InvalidInputValuesError,
    RequiredInputError,
)
from gitops_utils.types import FileEncodings, FileTypes


def fetch_environment_variables():
    return os.environ


def decode_json(
    json_data: FileTypes, encoding: FileEncodings = defaults.DEFAULT_ENCODING
) -> Union[dict, list, str, int, float, bool, None]:
    """
    Decode JSON data into a Python object.

    Args:
        json_data (FileTypes): The JSON data to decode.
        encoding (FileEncodings, optional): The encoding of the JSON data. Defaults to defaults.DEFAULT_ENCODING.

    Returns:
        Union[dict, list, str, int, float, bool, None]: The decoded Python object.

    Raises:
        JSONDecodeError: If the JSON data cannot be decoded.
    """
    try:
        return orjson.loads(json_data)
    except JSONDecodeError as e:
        raise JSONDecodeError(
            f"Error decoding JSON: {e}", doc=str(json_data), pos=e.pos
        )


def decode_base64(
    base64_string: str, encoding: FileEncodings = defaults.DEFAULT_ENCODING
) -> str:
    """
    Decodes a base64 string using the specified encoding.

    Parameters:
    - base64_string (str): The base64 string to decode.
    - encoding (FileEncodings, optional): The encoding to use when decoding the base64 string. Defaults to "utf-8".

    Returns:
    - str: The decoded string.

    Raises:
    - Base64DecodeError: If the base64 string is invalid or if there is an encoding error.
    - InvalidInputTypeError: If the base64_string is not a string or the encoding is not one of the allowed values.
    - InvalidEncodingError: If the encoding is not a valid encoding type.

    Example:
        decode_base64("SGVsbG8gd29ybGQ=")  # Returns "Hello world"
        decode_base64("SGVsbG8gd29ybGQ===", encoding="ascii")  # Raises Base64DecodeError
    """
    if not isinstance(base64_string, str):
        raise InvalidInputTypeError("base64_string", str)
    if encoding not in ["utf-8", "ascii", "latin-1"]:
        raise InvalidEncodingError(encoding)
    try:
        return base64.b64decode(base64_string).decode(encoding)
    except binascii.Error as exc:
        error_message = f"Failed to decode base64 string: {str(exc)}"
        raise Base64DecodeError(error_message) from exc
    except UnicodeDecodeError as exc:
        error_message = (
            f"Failed to decode base64 string: Invalid encoding: {encoding}: {exc}"
        )
        raise Base64DecodeError(error_message) from exc


class Inputs:
    """
    A class representing inputs for a task.

    Attributes:
        inputs (CaseInsensitiveDict): A dictionary-like object that stores the inputs.
        frozen_inputs (CaseInsensitiveDict): A dictionary-like object that stores the frozen inputs.
        from_stdin (bool): A flag indicating whether inputs should be read from stdin.
        merger (Merger): An instance of the Merger class used for merging inputs.
        override_stdin (bool): A flag indicating whether stdin should be overridden.

    Methods:
        __init__(inputs: Optional[Any] = None, from_environment: bool = True, from_stdin: bool = False)
            Initializes the Inputs object.
        get_input(k, default=None, required=False, is_bool=False, is_integer=False) -> Any
            Retrieves an input value from the inputs dictionary.
        decode_input(k: str, default: Optional[Any] = None, required: bool = False, decode_from_json: bool = True, decode_from_base64: bool = True, allow_none: bool = True)
            Decodes an input value from the inputs dictionary.
        freeze_inputs()
            Freezes the inputs dictionary and returns a copy of the frozen inputs.
        thaw_inputs()
            Restores the inputs dictionary to its original state.
        toggle_input_freezing()
            Toggles the freezing of inputs.

    """

    def __init__(
        self,
        inputs: Optional[Any] = None,
        from_environment: bool = True,
        from_stdin: bool = False,
    ):
        if inputs is None:
            inputs = {}

        if from_environment:
            inputs.update(fetch_environment_variables())

        self.override_stdin = strtobool(os.getenv("OVERRIDE_STDIN", "false"))

        if from_stdin and not self.override_stdin:
            inputs_from_stdin = sys.stdin.read()

            if not is_nothing(inputs_from_stdin):
                try:
                    inputs.update(json.loads(inputs_from_stdin))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Failed to decode stdin:\n{inputs_from_stdin}"
                    ) from exc

        self.from_stdin = from_stdin
        self.inputs = CaseInsensitiveDict(inputs)
        self.frozen_inputs = CaseInsensitiveDict()

        self.merger = Merger(
            [(list, ["append"]), (dict, ["merge"]), (set, ["union"])],
            ["override"],
            ["override"],
        )

    def get_input(
        self,
        k: str,
        default: Optional[Any] = None,
        required: bool = False,
        input_type: Optional[Union[Type, Callable]] = None,
        allow_none: bool = True,
    ) -> Any:
        """
        Retrieves an input value from the inputs dictionary.

        Args:
            k (str): The key of the input value to retrieve.
            default (Optional[Any]): The default value to return if the input value is not found. Defaults to None.
            required (bool): A flag indicating whether the input value is required. If set to True and the input value is not found, a RequiredInputError will be raised. Defaults to False.
            input_type (Optional[Union[Type, Callable]]): The expected type of the input value. If provided, the input value will be converted to this type before returning. Defaults to None.
            allow_none (bool): A flag indicating whether None is an acceptable value for the input. If set to False and the input value is None, an InvalidInputValuesError will be raised. Defaults to True.

        Returns:
            Any: The retrieved input value.

        Raises:
            RequiredInputError: If the required flag is set to True and the input value is not found.
            InvalidInputTypeError: If the input value cannot be converted to the specified input_type.
            InvalidInputValuesError: If the allow_none flag is set to False and the input value is None.

        """
        inp = self.inputs.get(k)
        inp = default if is_nothing(inp) else inp

        if required and is_nothing(inp):
            raise RequiredInputError(k)

        if input_type is None:
            if allow_none:
                return inp

            raise InvalidInputValuesError(k, None)

        try:
            if input_type is bool:
                return strtobool(inp)
            else:
                return input_type(inp)
        except (InvalidBooleanValueError, ValueError) as exc:
            raise InvalidInputTypeError(k, input_type) from exc

    def decode_input(
        self,
        k: str,
        default: Optional[Any] = None,
        required: bool = False,
        input_type: Optional[Union[Type, Callable]] = None,
        allow_none: bool = True,
        decode_from_json: bool = True,
        decode_from_base64: bool = True,
    ):
        """
        Decodes an input value from the inputs dictionary.

        Args:
            k (str): The key of the input value to retrieve.
            default (Optional[Any]): The default value to return if the input value is not found. Defaults to None.
            required (bool): A flag indicating whether the input value is required. If set to True and the input value is not found, a RequiredInputError will be raised. Defaults to False.
            input_type (Optional[Union[Type, Callable]]): The expected type of the input value. If provided, the input value will be converted to this type before returning. Defaults to None.
            allow_none (bool): A flag indicating whether None is an acceptable value for the input. If set to False and the input value is None, an InvalidInputValuesError will be raised. Defaults to True.
            decode_from_json (bool): A flag indicating whether the input value should be decoded from JSON. If set to True, the input value will be decoded using the decode_json function. Defaults to True.
            decode_from_base64 (bool): A flag indicating whether the input value should be decoded from Base64. If set to True, the input value will be decoded using the decode_base64 function. Defaults to True.

        Returns:
            Any: The decoded input value.

        Raises:
            InputValueDecodingError: If the input value cannot be decoded from either JSON or Base64.
            RequiredInputError: If the required flag is set to True and the input value is not found.
            InvalidInputTypeError: If the input value cannot be converted to the specified input_type.
            InvalidInputValuesError: If the allow_none flag is set to False and the input value is None.
        """
        conf = self.get_input(
            k,
            default=default,
            required=required,
            input_type=input_type,
            allow_none=allow_none,
        )
        if conf == default or not isinstance(conf, str):
            return conf

        methods = []
        try:
            if decode_from_base64:
                methods.append("Base64")
                conf = decode_base64(conf)

            if decode_from_json:
                methods.append("JSON")
                conf = decode_json(conf)
        except (
            Base64DecodeError,
            InvalidInputTypeError,
            InvalidEncodingError,
            JSONDecodeError,
        ):
            raise InputValueDecodingError(k, *methods)

        return conf

    def freeze_inputs(self):
        """
        Freezes the inputs dictionary and returns a copy of the frozen inputs.

        Returns:
            CaseInsensitiveDict: A dictionary-like object that stores the frozen inputs.

        """
        self.frozen_inputs = self.inputs.copy()
        self.inputs.clear()
        return self.frozen_inputs

    def thaw_inputs(self):
        """
        Restores the inputs dictionary to its original state.

        Returns:
            CaseInsensitiveDict: A dictionary-like object that stores the restored inputs.

        """
        self.inputs = self.frozen_inputs.copy()
        self.frozen_inputs.clear()
        return self.inputs

    def toggle_input_freezing(self):
        """
        Toggles the freezing of inputs.

        Returns:
            CaseInsensitiveDict: A dictionary-like object that stores the frozen inputs if inputs are currently not frozen, or a dictionary-like object that stores the restored inputs if inputs are currently frozen.

        """
        if self.inputs_frozen:
            return self.thaw_inputs()

        return self.freeze_inputs()

    @property
    def inputs_frozen(self):
        return bool(self.frozen_inputs)
