import collections.abc
import re
from enum import Enum
from typing import Any, Dict, List, Mapping, Union

from gitops_utils.errors import InvalidBooleanValueError


class FirstLetterCase(Enum):
    LOWER = "lower"
    UPPER = "upper"


def lower_first_char(inp: str) -> str:
    """
    Returns the input string with the first character converted to lowercase.

    Args:
        inp: The input string.

    Returns:
        The input string with the first character converted to lowercase.
    """
    inp = inp.lstrip()
    if not inp or not inp[0].isalpha() or inp[0].islower():
        return inp
    return inp[0].lower() + inp[1:]


def upper_first_char(inp: str) -> str:
    """
    Capitalizes the first character of a string.

    Args:
        inp (str): The input string.

    Returns:
        str: The input string with the first character capitalized.
    """
    inp = inp.lstrip()
    if not inp or not inp[0].isalpha() or inp[0].isupper():
        return inp
    return inp[0].upper() + inp[1:]


def recase_first_char(inp: str, first_letter_case: FirstLetterCase) -> str:
    if not inp or not re.match(r"^[A-Za-z]", inp):
        return inp
    if first_letter_case == FirstLetterCase.LOWER:
        return lower_first_char(inp)
    elif first_letter_case == FirstLetterCase.UPPER:
        return upper_first_char(inp)
    else:
        raise ValueError(
            f"First letter case {first_letter_case} not supported for recasing"
        )


def recase_first_char_of_map_keys(
    m: Mapping[str, Any], first_letter_case: FirstLetterCase
) -> dict:
    """
    Recases the first character of the keys in a mapping.

    Args:
        m (Mapping[str, Any]): The input mapping.
        first_letter_case (FirstLetterCase): A flag indicating whether to convert the first character to lowercase or uppercase.

    Returns:
        dict: The mapping with the recased keys.
    """
    return {recase_first_char(k, first_letter_case): v for k, v in m.items()}


TRUE_VALUES = {"true", "y", "yes", "on", "1"}
FALSE_VALUES = {"false", "n", "no", "off", "0"}


def strtobool(val: Union[str, bool]) -> bool:
    """
    Convert a string or boolean value to a boolean.

    Args:
        val: The value to convert.

    Returns:
        The converted boolean value.

    Raises:
        InvalidBooleanValueError: If the value is not a valid boolean
    """
    if isinstance(val, bool):
        return val

    val = val.casefold()

    if val in TRUE_VALUES:
        return True
    elif val in FALSE_VALUES:
        return False
    else:
        raise InvalidBooleanValueError(val)


def is_nothing(v: Any) -> bool:
    """
    Check if the given value is empty or None.

    Args:
        v: The value to check.

    Returns:
        The result of the condition.
    """
    if v is None:
        return True
    elif isinstance(v, str):
        return len(v.strip()) == 0
    elif isinstance(v, collections.abc.Sized):
        return len(v) == 0
    else:
        return not v


def all_non_empty(*vals: Any) -> List[Any]:
    """
    Returns a list of non-empty values from the given arguments.

    Args:
        *vals: Variable number of arguments

    Returns:
        List of non-empty values
    """
    return [val for val in vals if not is_nothing(val)]


def are_nothing(*vals: Any) -> bool:
    """
    Check if all values are empty.

    Args:
        *vals: Variable number of values to check.

    Returns:
        True if all values are empty, False otherwise.
    """
    return len(all_non_empty(*vals)) == 0


def first_non_empty(*vals: Any) -> Any:
    """
    Return the first non-empty value from the given arguments.

    Args:
        *vals: The values to check.

    Returns:
        The first non-empty value, or None if all values are empty.
    """
    if len(vals) == 0:
        return None

    return next((val for val in vals if not is_nothing(val)), None)


def any_non_empty(m: Dict[str, Any], *keys: str, first_non_empty: bool = False) -> Any:
    """
    Check if any of the given keys in the dictionary 'm' are non-empty.

    Args:
        m (Dict[str, Any]): The dictionary to check.
        keys (str): The keys to check in the dictionary.
        first_non_empty (bool): If True, return the first non-empty value found. Otherwise, return a dictionary of all non-empty values.

    Returns:
        Any: The first non-empty value found, or a dictionary of all non-empty values.
    """
    if is_nothing(m):
        return None if first_non_empty else {}

    result = {}
    for k in keys:
        value = m.get(k)
        if not is_nothing(value):
            if first_non_empty:
                return value

            result[k] = value

    return result or (None if first_non_empty else {})


def capitalize_name(name: str) -> str:
    """
    Capitalizes each word in a given name.

    Args:
        name (str): The name to be capitalized.

    Returns:
        str: The capitalized name.
    """
    if is_nothing(name):
        return ""

    cleaned_name = "".join(c for c in name if c.isalpha() or c.isspace())
    proper_name = [n.title() for n in cleaned_name.split()]
    return " ".join(proper_name)
