from collections.abc import Iterable

from numpy import array, ndarray
from numpy import float as npfloat


def ensure_np(x, dtype=npfloat):
    if isinstance(x, ndarray):
        return x
    else:
        return array(x, dtype=dtype)


def ensure_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, Iterable):
        return list(x)
    else:
        return [x]
