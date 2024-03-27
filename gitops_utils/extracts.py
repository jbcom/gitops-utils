from collections import deque
from collections.abc import Mapping
from typing import Any, Iterable, List


def all_values_from_nested_object(obj: Any) -> List[Any]:
    """
    Return a list of all values from a nested object.

    Parameters:
        obj (Any): The object to extract values from.

    Returns:
        List[Any]: A list containing all values from the nested object.

    Example:
        >>> obj = {'a': 1, 'b': {'c': 2, 'd': [3, 4]}}
        >>> all_values_from_nested_object(obj)
        [1, 2, 3, 4]
    """
    stack = deque((obj,))
    result = []
    while stack:
        current = stack.popleft()
        if isinstance(current, Mapping):
            stack.extend(current.values())
        elif isinstance(current, Iterable) and not isinstance(current, (str, bytes)):
            stack.extend(current)
        else:
            result.append(current)

    return result
