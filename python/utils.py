"""
"""

import re
from functools import wraps
from typing import List, Callable, NoReturn

import numpy as np


__all__ = [
    "SEED",
    "RNG",
    "set_seed",
    "ReprMixin",
]


SEED = 42
RNG = np.random.default_rng(seed=SEED)


def set_seed(seed: int) -> NoReturn:
    """
    set the seed of the random number generator

    Parameters
    ----------
    seed: int,
        the seed to be set

    """

    global RNG, SEED
    SEED = seed
    RNG = np.random.default_rng(seed=seed)


def default_class_repr(c: object, align: str = "center", depth: int = 1) -> str:
    """

    Parameters
    ----------
    c: object,
        the object to be represented
    align: str, default "center",
        the alignment of the class arguments

    Returns
    -------
    str,
        the representation of the class
    """
    indent = 4 * depth * " "
    closing_indent = 4 * (depth - 1) * " "
    if not hasattr(c, "extra_repr_keys"):
        return repr(c)
    elif len(c.extra_repr_keys()) > 0:
        max_len = max([len(k) for k in c.extra_repr_keys()])
        extra_str = (
            "(\n"
            + ",\n".join(
                [
                    f"""{indent}{k.ljust(max_len, " ") if align.lower() in ["center", "c"] else k} = {default_class_repr(eval(f"c.{k}"),align,depth+1)}"""
                    for k in c.__dir__()
                    if k in c.extra_repr_keys()
                ]
            )
            + f"{closing_indent}\n)"
        )
    else:
        extra_str = ""
    return f"{c.__class__.__name__}{extra_str}"


class ReprMixin(object):
    """
    Mixin for enhanced __repr__ and __str__ methods.
    """

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """ """
        return []


def add_docstring(doc: str, mode: str = "replace") -> Callable:
    """
    decorator to add docstring to a function

    Parameters
    ----------
    doc: str,
        the docstring to be added
    mode: str, default "replace",
        the mode of the docstring,
        can be "replace", "append" or "prepend",
        case insensitive

    """

    def decorator(func: Callable) -> Callable:
        """ """

        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable:
            """ """
            return func(*args, **kwargs)

        pattern = "(\\s^\n){1,}"
        if mode.lower() == "replace":
            wrapper.__doc__ = doc
        elif mode.lower() == "append":
            tmp = re.sub(pattern, "", wrapper.__doc__)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", doc)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            wrapper.__doc__ += new_lines + doc
        elif mode.lower() == "prepend":
            tmp = re.sub(pattern, "", doc)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", wrapper.__doc__)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            wrapper.__doc__ = doc + new_lines + wrapper.__doc__
        else:
            raise ValueError(f"mode {mode} is not supported")
        return wrapper

    return decorator
