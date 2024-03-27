from pathlib import Path
from typing import Any, List, Mapping, Optional

import orjson
import validators

from gitops_utils.cases import is_nothing
from gitops_utils.types import FilePath


def is_prefix(a: Optional[str], b: Optional[str]) -> bool:
    """
    Returns True if either string is a prefix of the other.
    """
    if is_nothing(a) or is_nothing(b):
        return False

    a = a.lower()
    b = b.lower()

    return a.startswith(b) or b.startswith(a)


def is_substring(a: Optional[str], b: Optional[str]) -> bool:
    """
    Check if one string is contained in another.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        True if the condition is met, False otherwise.
    """
    if not a or not b:
        return False

    a = a.strip().lower()
    b = b.strip().lower()

    return b.find(a) != -1


def is_reverse(a: str, b: str) -> bool:
    """
    Check if one string is the reverse of another.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        True if the condition is met, False otherwise.
    """
    if is_nothing(a) or is_nothing(b):
        return False

    if len(a) != len(b):
        return False

    return a.lower() == b[::-1].lower()


def is_non_empty_match(a: Any, b: Any):
    """
    Check if two values are non-empty and match each other.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        True if the values are non-empty and match, False otherwise.
    """
    if is_nothing(a) or is_nothing(b):
        return False

    if not isinstance(a, type(b)) and not isinstance(b, type(a)):
        return False

    type_transformations = {
        str: lambda x: x.lower(),
        Mapping: lambda x: orjson.dumps(
            x, default=str, option=orjson.OPT_SORT_KEYS
        ).decode("utf-8"),
        list: lambda x: sorted(x),
    }

    if type(a) in type_transformations:
        a = type_transformations[type(a)](a)
        b = type_transformations[type(b)](b)

    return a == b


def is_url(url: str) -> bool:
    """
    Check if the given input is a valid URL.

    Args:
        url (str): The input to be checked.

    Returns:
        bool: True if the input is a valid URL, False otherwise.
    """
    try:
        return validators.url(url)
    except validators.ValidationError:
        return False


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
