"""
Closed Convex Set
"""

from abc import ABC, abstractmethod
from typing import NoReturn, List, Optional, Union, Sequence
from numbers import Real

import numpy as np
from scipy.spatial import ConvexHull as _ConvexHull

from utils import ReprMixin


__all__ = [
    "CCS",
    "EuclideanSpace",
    "NonnegativeOrthant",
    "Polyhedron",
    "HyperPlane",
    "HalfSpace",
    "Rectangle",
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

    @property
    @abstractmethod
    def embedded_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def random_point(self) -> np.ndarray:
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

    @property
    def embedded_dim(self) -> int:
        return self.dim

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

    def random_point(self, bounds:Union[Real,Sequence[Real]]=10) -> np.ndarray:
        """
        """
        if isinstance(bounds, Real):
            bounds = (-abs(bounds), abs(bounds))
        assert len(bounds) == 2 and bounds[0] < bounds[1]
        return np.random.uniform(bounds[0], bounds[1], self.embedded_dim)


class NonnegativeOrthant(CCS):
    """
    """
    __name__ = "NonnegativeOrthant"

    def __init__(self, dim:int) -> NoReturn:
        """
        """
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def embedded_dim(self) -> int:
        return self.dim

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

    def random_point(self, bound:Real=10) -> np.ndarray:
        """
        """
        assert bound > 0
        return np.random.uniform(0, bound, self.embedded_dim)

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

    @property
    def embedded_dim(self) -> int:
        return self._equalities.shape[1] - 1

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


class HyperPlane(Polyhedron):
    """
    """
    __name__ = "HyperPlane"

    def __init__(self, normal_vec:np.ndarray, offset:float) -> NoReturn:
        """
        """
        self._normal_vec = np.array(normal_vec)
        self._offset = offset
        super().__init__(None, np.append(self._normal_vec, self._offset)[np.newaxis, :])

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        self._check_validity(point)
        proj_point = point + \
            (self._offset - np.dot(self._normal_vec, point)) * self._normal_vec / np.linalg.norm(self._normal_vec)**2
        return proj_point

    def random_point(self) -> np.ndarray:
        """
        """
        return self.projection(np.random.uniform(-10, 10, self.embedded_dim))


class HalfSpace(Polyhedron):
    """
    """
    __name__ = "HalfSpace"

    def __init__(self, normal_vec:np.ndarray, offset:float) -> NoReturn:
        """
        """
        self._normal_vec = np.array(normal_vec)
        self._offset = offset
        super().__init__(np.append(self._normal_vec, self._offset)[np.newaxis, :], None)
        self._hyperplane = HyperPlane(self._normal_vec, self._offset)

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        if self.isin(point):
            return point
        return self._hyperplane.projection(point)

    def random_point(self) -> np.ndarray:
        """
        """
        return self._hyperplane.random_point() + \
            np.random.uniform(-10, 0) * self._normal_vec


class Rectangle(Polyhedron):
    """
    """
    __name__ = "Rectangle"

    def __init__(self, lower_bounds:np.ndarray, upper_bounds:np.ndarray) -> NoReturn:
        """
        Example:
        >>> rect = Rectangle(np.zeros(10), np.ones(10))
        """
        self._lower_bounds = np.array(lower_bounds)
        self._upper_bounds = np.array(upper_bounds)
        assert self._lower_bounds.shape[0] == self._upper_bounds.shape[0]
        dim = self._lower_bounds.shape[0]
        inequlities = np.concatenate(
            (-np.eye(dim), -self._lower_bounds[:, np.newaxis]),
            axis=1
        )
        inequlities = np.concatenate(
            (
                inequlities,
                np.concatenate(
                    (np.eye(dim), self._upper_bounds[:, np.newaxis]),
                    axis=1
                )
            ),
            axis=0
        )
        super().__init__(inequlities, None)

    def projection(self, point:np.ndarray) -> np.ndarray:
        """
        """
        return np.array(point).clip(self._lower_bounds, self._upper_bounds)

    def random_point(self) -> np.ndarray:
        """
        """
        return np.random.uniform(self._lower_bounds, self._upper_bounds)

    @property
    def lower_bounds(self) -> np.ndarray:
        return self._lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        return self._upper_bounds

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + ["lower_bounds", "upper_bounds",]


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

    @property
    def embedded_dim(self) -> int:
        return self.dim

    def _check_validity(self, point:np.ndarray) -> NoReturn:
        """
        """
        assert point.shape[0] == self.dim

    def isin(self, point:np.ndarray) -> bool:
        """
        """
        self._check_validity(point)
        return np.linalg.norm(point-self.center, ord=self.p) <= self.radius

    def random_point(self) -> np.ndarray:
        """
        """
        bound = self.radius / np.power(self.embedded_dim, 1/self.p)
        return self.projection(
            self.center + np.random.uniform(-bound, bound, self.embedded_dim)
        )

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

    @property
    def embedded_dim(self) -> int:
        return self.dim

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

    def random_point(self) -> np.ndarray:
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

    @property
    def embedded_dim(self) -> int:
        return self.dim

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

    def random_point(self) -> np.ndarray:
        """
        """
        raise NotImplementedError
