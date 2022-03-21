"""
Closed Convex Set
"""

from abc import ABC, abstractmethod
from typing import NoReturn, List, Optional
from numbers import Real

import numpy as np
from scipy.spatial import ConvexHull as _ConvexHull

from utils import ReprMixin


__all__ = [
    "CCS",
    "EuclideanSpace",
    "EuclideanPlus",
    "LpBall",
    "L2Ball",
    "L1Ball",
    "LInfBall",
    "Simplex",
    "ConvexHull",
]


class CCS(ReprMixin, ABC):
    """
    Closed Convex Set
    """
    __name__ = "CCS"

    @abstractmethod
    def isin(self, point:np.ndarray) -> bool:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return ["dim",]


class EuclideanSpace(CCS):
    """
    """
    __name__ = "EuclideanSpace"

    def __init__(self, dim:int) -> NoReturn:
        """
        """
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        return point.shape[0] == self.dim

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        return point


class EuclideanPlus(CCS):
    """
    """
    __name__ = "EuclideanPlus"

    def __init__(self, dim:int) -> NoReturn:
        """
        """
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        return point.shape[0] == self.dim \
            and np.all(point >= 0)

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        if self.isin(point):
            return point
        neg_inds = np.where(point < 0)[0]
        proj_point = point.copy()
        proj_point[neg_inds] = 0
        return proj_point


class LpBall(CCS):
    """
    """
    __name__ = "LpBall"

    def __init__(self, p:Real, center:np.ndarray, radius:float) -> NoReturn:
        """
        """
        assert p >= 1, "p must be >= 1 to be convex"
        self.p = p
        self.center = center
        self.radius = radius

    @property
    def dim(self) -> int:
        return self.center.shape[0]

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        return point.shape[0] == self.dim \
            and np.linalg.norm(point-self.center, ord=self.p) <= self.radius

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return super().extra_repr_keys() + ["p", "center", "radius",]


class L2Ball(LpBall):
    """
    """
    __name__ = "L2Ball"

    def __init__(self, center:np.ndarray, radius:float) -> NoReturn:
        """
        """
        super().__init__(p=2, center=center, radius=radius)

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        if self.isin(point):
            return point
        return self.center + (point-self.center) / np.linalg.norm(point-self.center) * self.radius


class L1Ball(LpBall):
    """
    """
    __name__ = "L1Ball"

    def __init__(self, center:np.ndarray, radius:float) -> NoReturn:
        """
        """
        super().__init__(p=1, center=center, radius=radius)

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        raise NotImplementedError


class LInfBall(LpBall):
    """
    """
    __name__ = "LInfBall"

    def __init__(self, center:np.ndarray, radius:float) -> NoReturn:
        """
        """
        super().__init__(p=np.inf, center=center, radius=radius)

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        raise NotImplementedError


class Simplex(CCS):
    """
    """
    __name__ = "Simplex"

    def __init__(self, dim:int) -> NoReturn:
        """
        """
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        return point.shape[0] == self.dim \
            and point.sum() <= 1

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        raise NotImplementedError


class ConvexHull(_ConvexHull, CCS):
    """
    """
    __name__ = "ConvexHull"

    def __init__(self, points:np.ndarray, incremental:bool=False, qhull_options:Optional[str]=None) -> NoReturn:
        """
        """
        super().__init__(points, incremental, qhull_options)

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        raise NotImplementedError

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        raise NotImplementedError
