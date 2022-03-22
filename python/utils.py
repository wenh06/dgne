"""
"""

from typing import List

import numpy as np


__all__ = [
    "SEED", "RNG",
    "ReprMixin",
]


SEED = 42
RNG = np.random.default_rng(seed=SEED)


def default_class_repr(c:object, align:str="center", depth:int=1) -> str:
    """ finished, checked,

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
    indent = 4*depth*" "
    closing_indent = 4*(depth-1)*" "
    if not hasattr(c, "extra_repr_keys"):
        return repr(c)
    elif len(c.extra_repr_keys()) > 0:
        max_len = max([len(k) for k in c.extra_repr_keys()])
        extra_str = "(\n" + \
            ",\n".join([
                f"""{indent}{k.ljust(max_len, " ") if align.lower() in ["center", "c"] else k} = {default_class_repr(eval(f"c.{k}"),align,depth+1)}""" \
                    for k in c.__dir__() if k in c.extra_repr_keys()
                ]) + \
            f"{closing_indent}\n)"
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
        """
        """
        return []
