"""
"""

import time
from collections import deque
from typing import NoReturn, List, Dict, Union, Callable, Sequence, Optional, Tuple
from numbers import Real

import numpy as np

from utils import ReprMixin
from graph import Graph
from ccs import (
    CCS,
    EuclideanSpace,
    NonNegativeOrthant,
    NonPositiveOrthant,
)


__all__ = [
    "Agent",
]


class Agent(ReprMixin):
    """ """

    __name__ = "Agent"

    def __init__(
        self,
        agent_id: int,
        feasible_set: CCS,
        ceoff: np.ndarray,
        total_offset: np.ndarray,
        constraint_type: int,
        objective: Callable[[np.ndarray, np.ndarray], float],
        objective_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
        step_sizes: Sequence[float] = [0.1, 0.1, 0.1],
        alpha: Optional[float] = None,
        cache_size: int = -1,
    ) -> NoReturn:
        """

        Parameters
        ----------
        agent_id : int,
            agent id
        feasible_set : CCS,
            closed convex set, of dimension n,
            the feasible set (region) of the agent's decision variable
        ceoff : np.ndarray,
            coefficient of the agent linear constraint,
            of shape (m, n)
        total_offset : np.ndarray,
            total (sum) offset of all agents' linear constraint,
            of shape (m,)
        constraint_type : int,
            type of the constraint,
            1 for offset - ceoff @ x <= 0 (or equivalently coeff @ x >= offset),
            2 for offset - ceoff @ x >= 0 (or equivalently coeff @ x <= offset),
        objective : Callable[[np.ndarray, np.ndarray], float],
            objective function,
            takes self.x and self.decision_profile(others) as input
        objective_grad : Callable[[np.ndarray, np.ndarray], np.ndarray],
            gradient of the objective function
        step_sizes : sequence of float,
            3-tuples of step sizes for x, z, and lam, respectively,
            namely tau, nu, and sigma, respectively
        alpha: float, optional,
            factor for the extrapolation of the variables (x, z, lambda)
            if alpha is None, then the extrapolation is disabled
        cache_size: int, default -1,
            size of the cache for the variables (x, lambda),
            shoule be a positive integer, or -1 for unlimited cache size


        NOTE
        ----
        the linear constraint is assumed to be of the form
            offset - ceoff @ x <= 0

        TODO
        ----
        1. implement the `primal_update` and `dual_update` functions as external functions, so that multiprocessing can be used for parallel computation
        2. add stop criteria for the iteration

        """
        self.agent_id = agent_id
        self._feasible_set = feasible_set
        self._coeff = np.array(ceoff)  # A_i, shape (m, n_i)
        self._total_offset = np.array(total_offset)  # b, shape (m,)
        self._offset = self._total_offset.copy()  # b_i, shape (m,)
        self._objective = objective
        self._objective_grad = objective_grad
        self.tau, self.nu, self.sigma = step_sizes
        self.alpha = alpha
        self.__cache_size = cache_size
        assert isinstance(cache_size, int) and (
            cache_size >= 2 or cache_size == -1
        ), f"cache_size should be at least 2 or be -1, but got {cache_size}"
        if cache_size == -1:
            self.__cache_size = float("inf")
        self.__cache = deque()

        self.constraint_type = constraint_type
        if self.constraint_type == 1:
            self._multiplier_orthant = NonNegativeOrthant(self.b.shape[0])
        elif self.constraint_type == 2:
            self._multiplier_orthant = NonPositiveOrthant(self.b.shape[0])
        else:
            raise ValueError(
                f"constraint_type must be 1 or 2, but got {constraint_type}"
            )
        self._es = EuclideanSpace(self.b.shape[0])

        assert self.dim == self.A.shape[1]  # n_i
        assert self.A.shape[0] == self.b.shape[0]  # m

        self._decision = self.omega.random_point()  # x_i
        # self._multiplier = self._multiplier_orthant.random_point()  # lambda_i
        self._multiplier = np.zeros((self._multiplier_orthant.dim,))
        # self._aux_var = self._es.random_point()  # z_i
        self._aux_var = np.zeros((self._es.dim,))

        self.__step = 0
        self.__dual_step = 0
        prev_var = dict(
            x=self._decision.copy(),
            z=self._aux_var.copy(),
            lam=self._multiplier.copy(),
        )
        self.__cache.append(prev_var)
        self.__metrics = deque()
        self.__timer = time.time()

    def primal_update(
        self,
        others: List["Agent"],
        interference_graph: Graph,
        multiplier_graph: Graph,
    ) -> NoReturn:
        """

        the primal update step of the agent

        Parameters
        ----------
        others : list of Agent,
            list of other agents
        interference_graph : Graph,
            the interference graph
        multiplier_graph : Graph,
            the multiplier graph

        """
        self.add_cache()
        if self.cached_size > self.__cache_size:
            self.__cache.popleft()
        self.start_timing()
        # self.update_offset(others)
        interference_inds = [
            i
            for i, other in enumerate(others)
            if other.agent_id in interference_graph.get_neighbors(self.agent_id)
        ]
        multiplier_inds = [
            i
            for i, other in enumerate(others)
            if other.agent_id in multiplier_graph.get_neighbors(self.agent_id)
        ]
        self._decision = self.omega.projection(
            self.extrapolated_decision
            - self.tau
            * self.objective_grad(self.x, self.decision_profile(others, True))
            - np.matmul(self.A.T, self.lam)
        )

        W = multiplier_graph.adj_mat
        self._aux_var = self.extrapolated_aux_var + self.nu * sum(
            [
                W[self.agent_id, others[j].agent_id] * (self.lam - others[j].lam)
                for j in multiplier_inds
            ]
        )
        self.__metrics.append(
            dict(
                primal_update_time=self.stop_timing(),
            )
        )
        self.__step += 1

    def dual_update(
        self,
        others: List["Agent"],
        multiplier_graph: Graph,
    ) -> NoReturn:
        """

        the dual update step of the agent

        Parameters
        ----------
        others : list of Agent,
            list of other agents
        multiplier_graph : Graph,
            the multiplier graph

        """
        self.start_timing()
        multiplier_inds = [
            i
            for i, other in enumerate(others)
            if other.agent_id in multiplier_graph.get_neighbors(self.agent_id)
        ]
        W = multiplier_graph.adj_mat
        self._multiplier = self._multiplier_orthant.projection(
            self.extrapolated_multiplier
            - self.sigma
            * (
                np.matmul(self.A, 2 * self.x - self.prev_x)
                - self.b
                + sum(
                    [
                        W[self.agent_id, others[j].agent_id]
                        * (
                            2 * (self.z - others[j].z)
                            - (self.prev_z - others[j].prev_z)
                        )
                        for j in multiplier_inds
                    ]
                )
                + sum(
                    [
                        W[self.agent_id, others[j].agent_id]
                        * (self.lam - others[j].lam)
                        for j in multiplier_inds
                    ]
                )
            )
        )
        self.__metrics[-1]["dual_update_time"] = self.stop_timing()
        self.__metrics[-1]["objective"] = self.objective(
            self.x, self.decision_profile(others, True)
        )
        self.__metrics[-1]["objective_grad_norm"] = np.linalg.norm(
            self.objective_grad(self.x, self.decision_profile(others, True))
        )
        self.__dual_step += 1

    # def update_offset(self, others: List["Agent"]) -> NoReturn:
    #     r"""

    #     Update the offset of the linear constraint of the agent via
    #         .. math::
    #             \mathbf{b}_i = \mathbf{b} - \sum_{j\neq i} A_j \cdot \mathbf{x}_j

    #     Parameters
    #     ----------
    #     others : list of Agent,
    #         list of other agents

    #     """
    #     self._offset = self._total_offset - sum([c.Ax for c in others])

    def add_cache(self) -> NoReturn:
        self.__cache.append(
            dict(
                x=self.x.copy(),
                z=self.z.copy(),
                lam=self.lam.copy(),
            )
        )

    def start_timing(self) -> NoReturn:
        self.__timer = time.time()

    def stop_timing(self) -> float:
        time_elapsed = time.time() - self.__timer
        self.__timer = time.time()
        return time_elapsed

    @property
    def dim(self) -> int:
        return self._coeff.shape[1]

    @property
    def decision(self) -> np.ndarray:
        return self._decision

    @property
    def multiplier(self) -> np.ndarray:
        return self._multiplier

    @property
    def aux_var(self) -> np.ndarray:
        return self._aux_var

    @property
    def x(self) -> np.ndarray:
        return self.decision

    @property
    def lam(self) -> np.ndarray:
        return self.multiplier

    @property
    def z(self) -> np.ndarray:
        return self.aux_var

    @property
    def A(self) -> np.ndarray:
        return self._coeff

    @property
    def Ax(self) -> np.ndarray:
        return np.matmul(self.A, self.x)

    @property
    def b(self) -> np.ndarray:
        return self._offset

    @property
    def omega(self) -> np.ndarray:
        return self._feasible_set

    @property
    def prev_decision(self) -> np.ndarray:
        return self.__cache[-1]["x"]

    @property
    def prev_x(self) -> np.ndarray:
        return self.prev_decision

    @property
    def prev_multiplier(self) -> np.ndarray:
        return self.__cache[-1]["lam"]

    @property
    def prev_lam(self) -> np.ndarray:
        return self.prev_multiplier

    @property
    def prev_aux_var(self) -> np.ndarray:
        return self.__cache[-1]["z"]

    @property
    def prev_z(self) -> np.ndarray:
        return self.prev_aux_var

    @property
    def extrapolated_decision(self) -> np.ndarray:
        if self.alpha is None:
            return self.x
        return self.x + self.alpha * (self.x - self.prev_x)

    @property
    def extrapolated_aux_var(self) -> np.ndarray:
        if self.alpha is None:
            return self.z
        return self.z + self.alpha * (self.z - self.prev_z)

    @property
    def extrapolated_multiplier(self) -> np.ndarray:
        if self.alpha is None:
            return self.lam
        return self.lam + self.alpha * (self.lam - self.prev_lam)

    @property
    def step(self) -> int:
        return self.__step

    @property
    def dual_step(self) -> int:
        return self.__dual_step

    def degree(self, adj_mat: np.ndarray) -> Real:
        """ """
        return adj_mat[self.agent_id, :].sum()

    def decision_profile(
        self, others: List["Agent"], except_self: bool = False
    ) -> np.ndarray:
        """ """
        agent_ids = [other.agent_id for other in others]
        inds = np.argsort(agent_ids)
        dp = np.concatenate([others[i].decision for i in inds])
        if not except_self:
            insert_pos = sum(
                [other.dim for other in others if other.agent_id < self.agent_id]
            )
            dp = np.insert(dp, insert_pos, self.decision, axis=0)
        return dp

    @property
    def objective(self) -> Callable:
        """ """
        return self._objective

    @property
    def objective_grad(self) -> Callable:
        """ """
        return self._objective_grad

    @property
    def cached_size(self) -> int:
        """ """
        return len(self.__cache)

    @property
    def cache_size(self) -> int:
        """ """
        if np.isfinite(self.__cache_size):
            return self.__cache_size
        return -1

    def get_prev_cache(self, key: str, step_diff: int) -> np.ndarray:
        """

        Get the previous cache of the agent of specified step difference

        Parameters
        ----------
        key : str,
            the key of the cache, can be one of "x", "z", "lam"
        step_diff : int,
            the step difference, must be a non-negative integer,
            and must be less than the cached size

        """
        assert step_diff >= 0 and step_diff < self.cached_size
        return self.__cache[-1 - step_diff][key]

    def get_cache(
        self,
        key: Optional[str] = None,
        dims: Optional[Union[int, Sequence[int]]] = None,
    ) -> Union[List[Dict[str, np.ndarray]], List[np.ndarray], List[float]]:
        """

        Get cached sequence of variable values of the optimization process

        Parameters
        ----------
        key : str, optional,
            the key of the variable to be returned, by default None
            can be one of "x", "z", "lam"
            if None, return all the cached variable values
        dims : int or sequence of int, optional,
            the dimensions of the variable to be returned, by default None
            if None, return all dimensions of the variable values

        Returns
        -------
        List[Dict[str, np.ndarray]] or List[np.ndarray] or List[float],
            if `key` is None, return a list of dicts of variable values;
            if `key` is not None, return a list of variable values

        """
        if key is None:
            assert dims is None, "dim must be None if key is None"
            return list(self.__cache)
        assert key in [
            "x",
            "z",
            "lam",
        ], f"""key must be one of "x", "z", "lam" or None, but got {key}"""
        if dims is None:
            return [cache[key] for cache in self.__cache]
        return [cache[key][dims] for cache in self.__cache]

    def get_metrics(
        self, key: Optional[str] = None
    ) -> Union[List[Dict[str, float]], List[float]]:
        """

        Get cached metrics collected during the optimization.

        Parameters
        ----------
        key : str, optional,
            the key of the metric to be returned, by default None
            can be one of "primal_update_time", "dual_update_time", "objective", "objective_grad_norm",
            if None, return all the cached metrics

        Returns
        -------
        List[Dict[str, float]] or List[float],
            if `key` is None, return a list of dicts of metrics;
            if `key` is not None, return a list of metrics

        """
        if key is None:
            return list(self.__metrics)
        assert key in [
            "primalupdate_time",
            "dual_update_time",
            "objective",
            "objective_grad_norm",
        ], f"""key must be one of "primal_update_time", "dual_update_time", "objective", "objective_grad_norm" or None, but got {key}"""
        return [metric[key] for metric in self.__metrics]

    def is_convergent(
        self,
        keys: Union[str, Sequence[str]] = "objective_grad_norm",
        func: Callable[[Tuple[np.ndarray, ...]], bool] = lambda a: (
            a[-max(10, len(a) // 10) :] < 1e-6
        ).all(),
    ) -> bool:
        """

        Check if the optimization process is convergent

        Parameters
        ----------
        keys : str or sequence of str, default "objective_grad_norm",
            the keys of the metrics to be checked,
            can be one of "x", "objective", "objective_grad_norm",
        func : function, default lambda a: (a[-max(10, len(a) // 10) :] < 1e-6).all(),
            the function to check the convergence,

        Returns
        -------
        bool,
            True if convergent, False otherwise

        TODO: use KKT conditions to check the convergence

        """
        if isinstance(keys, str):
            keys = [keys]
        values = []
        for key in keys:
            if key == "x":
                values.append(np.array(self.get_cache(key=key)))
            else:
                values.append(np.array(self.get_metrics(key=key)))
        return func(*values)

    def KKT(self, others: List["Agent"], eps: float = 1e-8) -> bool:
        """ """
        # condition on x
        if (
            np.abs(
                self.objective_grad(self.x, self.decision_profile(others, True))
                - np.matmul(self.A.T, self.lam)
            )
            > eps
        ).any():
            return False
        # condition on lam
        if (np.abs(np.dot(self.lam, self.Ax - self.b)) > eps).any():
            return False
        if self.constraint_type == 1:
            if ((self.b - self.Ax) > 0).any():
                return False
        elif self.constraint_type == 2:
            if ((self.b - self.Ax) < 0).any():
                return False
        return True

    def extra_repr_keys(self) -> List[str]:
        return [
            "agent_id",
            "dim",
            "tau",
            "nu",
            "sigma",
        ]
