from typing import Any, Dict, List, Set

import numpy as np
from flatdict import FlatDict

from gitops_utils.patterns import IS_MAGIC_METHOD


def flatten_dict(value: Any, delimiter: str = ".") -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a flat dictionary.

    Parameters:
    - value: The nested dictionary to be flattened.
    - delimiter: The delimiter to use for the keys in the flat dictionary. Defaults to '.'.

    Returns:
    - The flattened dictionary.

    """
    flat_dict = FlatDict(value, delimiter)
    return dict(flat_dict)


def flatten_list(matrix: List[Any]) -> List[Any]:
    """
    Flattens a matrix into a 1D list.

    Args:
        matrix: The matrix to be flattened.

    Returns:
        The flattened matrix as a 1D list.
    """
    array = np.array(matrix, dtype=object)
    return array.ravel().tolist()


def deduplicate_nested_object(obj: Any, seen: set = None) -> Any:
    """
    Remove duplicate elements from a list, tuple, set, or dictionary and all elements contained within.

    Args:
        obj: The object to deduplicate.
        seen: A set to keep track of seen elements.

    Returns:
        The deduplicated object.

    """
    if seen is None:
        seen = set()
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            deduplicate_nested_object(elem, seen)
            for elem in obj
            if elem not in seen and not seen.add(elem)
        )
    elif isinstance(obj, set):
        return {
            deduplicate_nested_object(elem, seen)
            for elem in obj
            if elem not in seen and not seen.add(elem)
        }
    elif isinstance(obj, dict):
        return {
            k: deduplicate_nested_object(v, seen)
            for k, v in obj.items()
            if v not in seen and not seen.add(v)
        }
    else:
        return obj


def truncate(msg: str, max_length: int, ender: str = "...") -> str:
    """
    Truncates a message to a specified maximum length, adding an ender if necessary.

    Args:
        msg (str): The message to be truncated.
        max_length (int): The maximum length of the truncated message.
        ender (str, optional): The string to be added at the end of the truncated message. Defaults to "...".

    Returns:
        str: The truncated message.

    Raises:
        ValueError: If the ender is an empty string.
        ValueError: If the max_length is less than or equal to zero.
    """
    if not ender:
        raise ValueError("ender cannot be an empty string")

    if max_length <= 0:
        raise ValueError("max_length must be greater than zero")

    adjusted_max_length = max_length - len(ender) + 1

    if len(msg.strip()) > adjusted_max_length:
        return msg[:adjusted_max_length].rstrip() + ender

    return msg


def filter_list(
    l: List[str] = None,
    allowlist: List[str] = None,
    denylist: List[str] = None,
):
    """
    Filter a list based on an allowlist and denylist.

    Args:
        l (List[str]): The list to be filtered.
        allowlist (List[str]): The list of elements to allow in the filtered list.
        denylist (List[str]): The list of elements to deny in the filtered list.

    Returns:
        List[str]: The filtered list.
    """
    if l is None:
        l = []
    if allowlist is None:
        allowlist = []
    if denylist is None:
        denylist = []

    allowlist = set(allowlist)
    denylist = set(denylist)

    return list(
        filter(
            lambda elem: (not allowlist or elem in allowlist) and elem not in denylist,
            l,
        )
    )


def filter_dict(
    dct: Dict[str, Any],
    allowlist: Set[str] = None,
    denylist: Set[str] = None,
) -> Dict[str, Any]:
    """
    Filter a dictionary based on an allowlist and denylist.

    Args:
        dct (Dict[str, Any]): The dictionary to filter.
        allowlist (Set[str], optional): The set of keys to allow. Defaults to None.
        denylist (Set[str], optional): The set of keys to deny. Defaults to None.

    Returns:
        Dict[str, Any]: The filtered dictionary.
    """
    if dct is None:
        return {}

    if allowlist is None and denylist is None:
        return dct

    if allowlist is not None:
        dct = {key: dct[key] for key in allowlist.intersection(dct)}

    if denylist is not None:
        dct = {key: dct[key] for key in dct.keys() if key not in denylist}

    return dct


def filter_methods(methods: List[str]) -> List[str]:
    """
    Filter the given methods based on certain conditions.

    Args:
        methods (List[str]): The methods to be filtered.

    Returns:
        List[str]: A list containing the filtered methods.
    """
    return [m for m in methods if isinstance(m, str) and not IS_MAGIC_METHOD.match(m)]
