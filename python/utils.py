"""
"""

import re
from functools import wraps
from typing import Any, MutableMapping, Optional, List, Callable, NoReturn

import numpy as np


__all__ = [
    "DEFAULTS",
    "set_seed",
    "ReprMixin",
]


class CFG(dict):
    """

    this class is created in order to renew the `update` method,
    to fit the hierarchical structure of configurations

    Examples
    --------
    >>> c = CFG(hehe={"a":1,"b":2})
    >>> c.update(hehe={"a":-1})
    >>> c
    {'hehe': {'a': -1, 'b': 2}}
    >>> c.__update__(hehe={"a":-10})
    >>> c
    {'hehe': {'a': -10}}

    """

    __name__ = "CFG"

    def __init__(self, *args, **kwargs) -> NoReturn:
        """ """
        if len(args) > 1:
            raise TypeError(f"expected at most 1 arguments, got {len(args)}")
        elif len(args) == 1:
            d = args[0]
            assert isinstance(d, MutableMapping)
        else:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            try:
                setattr(self, k, v)
            except Exception:
                dict.__setitem__(self, k, v)
        # Class attributes
        exclude_fields = ["update", "pop"]
        for k in self.__class__.__dict__:
            if (
                not (k.startswith("__") and k.endswith("__"))
                and k not in exclude_fields
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(
        self, new_cfg: Optional[MutableMapping] = None, **kwargs: Any
    ) -> NoReturn:
        """

        the new hierarchical update method

        Parameters
        ----------
        new_cfg : MutableMapping, optional
            the new configuration, by default None
        kwargs : Any, optional
            key value pairs, by default None

        """
        _new_cfg = new_cfg or CFG()
        if len(kwargs) > 0:  # avoid RecursionError
            _new_cfg.update(kwargs)
        for k in _new_cfg:
            # if _new_cfg[k].__class__.__name__ in ["dict", "EasyDict", "CFG"] and k in self:
            if isinstance(_new_cfg[k], MutableMapping) and k in self:
                self[k].update(_new_cfg[k])
            else:
                try:
                    setattr(self, k, _new_cfg[k])
                except Exception:
                    dict.__setitem__(self, k, _new_cfg[k])

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        """

        the updated pop method

        Parameters
        ----------
        key : str
            the key to pop
        default : Any, optional
            the default value, by default None

        """
        if key in self:
            delattr(self, key)
        return super().pop(key, default)


DEFAULTS = CFG()
DEFAULTS.SEED = 42
DEFAULTS.RNG = np.random.default_rng(seed=DEFAULTS.SEED)


def set_seed(seed: int) -> NoReturn:
    """
    set the seed of the random number generator

    Parameters
    ----------
    seed: int,
        the seed to be set

    """

    global DEFAULTS
    DEFAULTS.SEED = seed
    DEFAULTS.RNG = np.random.default_rng(seed=seed)


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
