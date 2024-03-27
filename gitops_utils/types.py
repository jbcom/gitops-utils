import os
from collections import defaultdict
from functools import partial
from typing import DefaultDict, Literal, Type, Union

FilePath = Union[str, bytes, os.PathLike]
FileEncodings = Literal["utf-8", "ascii", "latin-1"]
FileTypes = Union[str, bytes, bytearray, memoryview]


def create_nested_default_dict(
    default_type: Type = dict,
) -> DefaultDict[Type, DefaultDict]:
    """
    Create a nested dictionary with the specified default value type.

    Args:
        default_type (typing.Type): The default value type for the nested dictionary. Defaults to dict.

    Returns:
        DefaultDict: The nested dictionary with the specified default value type.

    Example:
        >>> create_nested_default_dict()
        defaultdict(<class 'dict'>, {})

        >>> create_nested_default_dict(list)
        defaultdict(<class 'list'>, {})
    """
    return defaultdict(partial(defaultdict, default_type))
