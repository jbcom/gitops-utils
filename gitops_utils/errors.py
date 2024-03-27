from json import JSONDecodeError as _JSONDecodeError
from typing import Any, Literal, Type, Union

from git.exc import InvalidGitRepositoryError
from ruamel.yaml import YAMLError


class TopLevelDirectoryNotFoundError(InvalidGitRepositoryError):
    def __init__(self):
        super().__init__(
            "Top-level directory could not be found. Please make sure you are in a Git repository or one of its parent directories."
        )

    def __str__(self):
        return self.args[0]


class InvalidBooleanValueError(TypeError):
    def __init__(self, invalid_value: Any):
        """
        Initialize the InvalidBooleanValueError object.

        Args:
            invalid_value (Any): The invalid boolean value. The value must be a valid boolean string.
        """
        super().__init__(f"Invalid boolean value {invalid_value}")

    def __str__(self):
        return f"Invalid boolean value: {self.args[0]}"


class InvalidYAMLSyntaxError(YAMLError):
    """
    Custom exception class for representing an error in YAML syntax.
    """

    def __init__(
        self, msg: str = "Invalid YAML syntax", line: int = None, column: int = None
    ):
        super().__init__(msg)
        self.line = line
        self.column = column

    def __str__(self):
        return f"Invalid YAML syntax: {self.args[0]}"

    def __repr__(self):
        return f"InvalidYAMLSyntaxError(msg={self.args[0]})"


class Base64DecodeError(ValueError):
    """
    Exception raised for errors during base64 decoding.

    This exception is a subclass of ValueError and is raised when there are errors during base64 decoding.
    """

    def __init__(self, msg: str):
        super().__init__(msg)

    def __str__(self):
        return "Base64 decoding error occurred."


class JSONDecodeError(_JSONDecodeError):
    def __init__(
        self,
        msg: Union[str, bytes, bytearray, memoryview],
        doc: str,
        pos: int,
        encoding: Literal["utf-8", "ascii", "latin-1"] = "utf-8",
    ):
        """
        Initialize the object.

        Args:
            msg (Union[str, bytes, bytearray, memoryview]): The message.
            doc: The document.
            pos: The position.
            encoding (Literal["utf-8", "ascii", "latin-1"], optional): The encoding. Defaults to "utf-8".
        """
        decoded_msg = msg.decode(encoding) if not isinstance(msg, str) else msg

        super().__init__(decoded_msg, doc, pos)

    def __str__(self):
        return "JSON decoding error occurred."


class InvalidInputTypeError(TypeError):
    """
    Custom exception class for invalid input types.
    """

    def __init__(self, attribute: str, expected_type: Type) -> None:
        super().__init__(f"{attribute} must be of type {expected_type}")

    def __str__(self) -> str:
        return f"{self.args[0]}"


class InvalidInputValuesError(TypeError):
    """
    Custom exception class for invalid input values.
    """

    def __init__(self, attribute: str, *invalid_values: Any) -> None:
        super().__init__(
            f"'{attribute}' cannot be any of values: {", ".join(invalid_values)}"
        )

    def __str__(self) -> str:
        return f"{self.args[0]}"


class InputValueDecodingError(ValueError):
    """
    Custom exception class for input values which cannot be decoded.
    """

    def __init__(self, attribute: str, *methods) -> None:
        error_message = f"'{attribute}' cannot be decoded"
        if methods:
            error_message += f" from {', '.join(methods)}"
        super().__init__(error_message)

    def __str__(self) -> str:
        return f"{self.args[0]}"


class RequiredInputError(TypeError):
    """
    Custom exception class for missing required input types.
    """

    def __init__(self, attribute: str) -> None:
        super().__init__(f"'{attribute}' is required")

    def __str__(self) -> str:
        return f"{self.args[0]}"


class ConflictingFunctionInputsError(ValueError):
    """
    Custom exception class for conflicting function inputs.
    """

    def __init__(self, conflicting_value: Any, *inputs: str) -> None:
        if not isinstance(conflicting_value, str):
            conflicting_value = str(conflicting_value)

        super().__init__(f"{', '.join(inputs)} cannot all be '{conflicting_value}'")


class InvalidEncodingError(ValueError):
    """
    Exception raised for invalid encodings.

    Args:
        encoding (str): The invalid encoding.
    """

    def __init__(self, encoding: str) -> None:
        super().__init__(
            f"Invalid encoding '{encoding}'. The encoding must be one of the following: 'utf-8', 'ascii', or 'latin-1'. Please check the encoding and try again."
        )

    def __str__(self) -> str:
        return f"Invalid encoding error occurred: {self.args[0]}"
