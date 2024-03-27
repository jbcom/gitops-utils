import re
from typing import Any, Dict, Iterable, Optional

import inflection
from deepmerge import Merger

from gitops_utils.patterns import IS_NON_WORD_CHAR


def sanitize_key(key: str, delim: str = "_") -> str:
    """
    Sanitizes a key by replacing non-alphanumeric characters with a delimiter using regular expressions.

    Args:
        key (str): The key to be sanitized.
        delim (str, optional): The delimiter to be used. Defaults to "_".

    Returns:
        str: The sanitized key.
    """
    if not isinstance(delim, str) or len(delim) != 1:
        delim = "_"
    return re.sub(IS_NON_WORD_CHAR, delim, key)


def unhump_nested_dict(dct: Dict[str, Any], drop_without_prefix: Optional[str] = None):
    """
    Unhumps a nested dictionary by converting the keys to snake_case.

    Args:
        dct (Dict[str, Any]): The nested dictionary to be unhumped.
        drop_without_prefix (Optional[str]): The prefix to filter the keys. If provided, only the keys starting with the prefix will be included.

    Returns:
        Dict[str, Any]: The unhumped dictionary.
    """
    unhumped = {}

    for k, v in dct.items():
        if not drop_without_prefix or k.startswith(drop_without_prefix):
            unhumped_key = inflection.underscore(k)
            if isinstance(v, dict):
                unhumped[unhumped_key] = unhump_nested_dict(v)
            elif isinstance(v, (list, tuple)):
                unhumped[unhumped_key] = [
                    unhump_nested_dict(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                unhumped[unhumped_key] = v

    return unhumped


def zip_dict(a: Iterable[Any], b: Iterable[Any]) -> Dict[Any, Any]:
    """
    Maps the elements of two iterables into a dictionary.

    Args:
        a (Iterable[Any]): The first iterable.
        b (Iterable[Any]): The second iterable.

    Returns:
        Dict[Any, Any]: A dictionary mapping the elements of iterable 'a' to the elements of iterable 'b'.

    Raises:
        ValueError: If the lists 'a' and 'b' have different lengths.
    """
    return dict(zip(a, b, strict=True))


def invert_dict(dct: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Inverts a dictionary by swapping the keys and values.

    Args:
        dct (Dict[Any, Any]): The dictionary to be inverted.

    Returns:
        Dict[Any, Any]: A new dictionary with the keys and values swapped.

    Raises:
        KeyError: If the dictionary values are not unique.
    """
    if len(dct.values()) != len(set(dct.values())):
        raise KeyError("Dictionary values are not unique.")

    return zip_dict(dct.values(), dct.keys())


class Transforms:
    def __init__(self):
        self.merger = Merger(
            [(list, ["append"]), (dict, ["merge"]), (set, ["union"])],
            ["override"],
            ["override"],
        )

    def multi_merge(self, *maps):
        merged_maps = {}
        for m in maps:
            merged_maps = self.merger.merge(merged_maps, m)

        return merged_maps
