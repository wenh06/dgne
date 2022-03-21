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
    "HyperPlane",
    "HalfSpace",
    "Polyhedron",
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

    @abstractmethod
    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
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

    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        assert point.shape[0] == self.dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        self._check_validity(point)
        return True

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        self._check_validity(point)
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

    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        assert point.shape[0] == self.dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        self._check_validity(point)
        return np.all(point >= 0)

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        if self.isin(point):
            return point
        neg_inds = np.where(point < 0)[0]
        proj_point = point.copy()
        proj_point[neg_inds] = 0
        return proj_point


class HyperPlane(CCS):
    """
    """
    __name__ = "HyperPlane"

    def __init__(self, normal_vec:np.ndarray, offset:float) -> NoReturn:
        """
        """
        self._normal_vec = np.array(normal_vec)
        self._offset = offset

    @property
    def dim(self) -> int:
        return self._normal_vec.shape[0] - 1

    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        assert point.shape[0] == self.dim + 1

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        self._check_validity(point)
        return np.dot(self._normal_vec, point) == self._offset

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        self._check_validity(point)
        proj_point = point + \
            (self._offset - np.dot(self._normal_vec, point)) * self._normal_vec / np.linalg.norm(self._normal_vec)**2
        return proj_point


class HalfSpace(CCS):
    """
    """
    __name__ = "HalfSpace"

    def __init__(self, normal_vec:np.ndarray, offset:float) -> NoReturn:
        """
        """
        self._normal_vec = np.array(normal_vec)
        self._offset = offset

    @property
    def dim(self) -> int:
        return self._normal_vec.shape[0]

    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        assert point.shape[0] == self.dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        self._check_validity(point)
        return np.dot(self._normal_vec, point) <= self._offset

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        if self.isin(point):
            return point
        proj_point = point + \
            (self._offset - np.dot(self._normal_vec, point)) * self._normal_vec / np.linalg.norm(self._normal_vec)**2
        return proj_point


class Polyhedron(CCS):
    """
    """
    __name__ = "Polyhedron"

    def __init__(self, inequalities:Optional[np.ndarray]=None, equalities:Optional[np.ndarray]=None) -> NoReturn:
        """
        Example:
        >>> inequalities = np.array([[0,-1,0], [0,1,1], [-1,0,0], [1,0,1]])
        >>> ph = Polyhedron(inequalities)
        """
        if inequalities is None:
            self._equalities = np.array(equalities)
            self._inequalities = np.array([]).reshape(0, self._equalities.shape[1])
        elif equalities is None:
            self._inequalities = np.array(inequalities)
            self._equalities = np.array([]).reshape(0, self._inequalities.shape[1])
        else:
            self._inequalities = np.array(inequalities)
            self._equalities = np.array(equalities)
        assert self._equalities.shape[1] == self._inequalities.shape[1]
        if self._equalities.shape[0] > 0:
            assert np.linalg.matrix_rank(self._equalities) == np.linalg.matrix_rank(self._equalities[:, :-1])

    @property
    def dim(self) -> int:
        if self._equalities.shape[0] > 0:
            return self._inequalities.shape[1] - 1 - np.linalg.matrix_rank(self._equalities)
        return self._inequalities.shape[1] - 1

    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        assert point.shape[0] == self._inequalities.shape[1] - 1

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        self._check_validity(point)
        return (np.dot(self._equalities[:, :-1], point) == self._equalities[:, -1]).all() \
            and (np.dot(self._inequalities[:, :-1], point) <= self._inequalities[:, -1]).all()

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        if self.isin(point):
            return point
        raise NotImplementedError


class LpBall(CCS):
    """
    """
    __name__ = "LpBall"

    def __init__(self, p:Real, center:np.ndarray, radius:float) -> NoReturn:
        """
        """
        assert p >= 1, "p must be >= 1 to be convex"
        self.p = p
        self.center = np.array(center)
        self.radius = radius

    @property
    def dim(self) -> int:
        return self.center.shape[0]

    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        assert point.shape[0] == self.dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        self._check_validity(point)
        return np.linalg.norm(point-self.center, ord=self.p) <= self.radius

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

    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        assert point.shape[0] == self.dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        self._check_validity(point)
        return point.sum() <= 1

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

    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        assert point.shape[0] == self.dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        raise NotImplementedError

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        raise NotImplementedError
