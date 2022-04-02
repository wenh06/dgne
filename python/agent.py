"""
"""

from collections import deque
from typing import NoReturn, List, Union, Callable, Sequence, Optional
from numbers import Real

import numpy as np
from scipy import sparse

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
        ccs: CCS,
        ceoff: np.ndarray,
        offset: np.ndarray,
        constraint_type: int,
        objective: Callable[[np.ndarray, np.ndarray], float],
        objective_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
        step_sizes: Sequence[float] = [0.1, 0.1, 0.1],
        alpha: Optional[float] = None,
        cache_size: int = 1,
    ) -> NoReturn:
        """

        Parameters
        ----------
        agent_id : int,
            agent id
        ccs : CCS,
            closed convex set, of dimension n
        ceoff : np.ndarray,
            coefficient of the agent linear constraint,
            of shape (m, n)
        offset : np.ndarray,
            offset of the agent linear constraint,
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
        cache_size: int, default 1,
            size of the cache for the variables (x, lambda),
            shoule be a positive integer, or -1 for unlimited cache size


        NOTE: the linear constraint is assumed to be of the form
            offset - ceoff @ x <= 0
        """
        self.agent_id = agent_id
        self.ccs = ccs
        self._coeff = np.array(ceoff)  # A_i, shape (m, n_i)
        self._offset = np.array(offset)  # b_i, shape (m,)
        self._objective = objective
        self._objective_grad = objective_grad
        self.tau, self.nu, self.sigma = step_sizes
        self.alpha = alpha
        self.__cache_size = cache_size
        assert isinstance(cache_size, int) and (
            cache_size > 0 or cache_size == -1
        ), f"cache_size should be a positive integer or -1, but got {cache_size}"
        if cache_size == -1:
            self.__cache_size = float("inf")
        self.__cache = deque()

        if constraint_type == 1:
            self._multiplier_orthant = NonNegativeOrthant(self.b.shape[0])
        elif constraint_type == 2:
            self._multiplier_orthant = NonPositiveOrthant(self.b.shape[0])
        else:
            raise ValueError(
                f"constraint_type must be 1 or 2, but got {constraint_type}"
            )
        self._es = EuclideanSpace(self.b.shape[0])

        assert self.dim == self.A.shape[1]  # n_i
        assert self.A.shape[0] == self.b.shape[0]  # m

        self._decision = self.ccs.random_point()  # x_i
        # self._multiplier = self._multiplier_orthant.random_point()  # lambda_i
        self._multiplier = np.zeros((self._multiplier_orthant.dim,))
        # self._aux_var = self._es.random_point()  # z_i
        self._aux_var = np.zeros((self._es.dim,))
        prev_var = dict(
            x=self._decision.copy(),
            z=self._aux_var.copy(),
            lam=None,
        )
        if self.alpha is not None:
            prev_var["lam"] = self._multiplier.copy()

    def update(
        self,
        others: List["Agent"],
        interference_graph: Graph,
        multiplier_graph: Graph,
    ) -> NoReturn:
        """

        the update step of the agent

        Parameters
        ----------
        others : list of Agent,
            list of other agents
        interference_graph : Graph,
            the interference graph
        multiplier_graph : Graph,
            the multiplier graph

        """
        self.__cache.append(
            dict(
                x=self.x.copy(),
                z=self.z.copy(),
                lam=self.lam.copy(),
            )
        )
        if self.cached_size > self.__cache_size:
            self.__cache.popleft()
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
        self._decision = self.ccs.projection(
            self.extrapolated_decision
            - self.tau
            * self._objective_grad(self.x, self.decision_profile(others, True))
            - np.matmul(self.A.T, self.lam)
        )

        W = multiplier_graph.adj_mat
        self._aux_var = self.extrapolated_aux_var + self.nu * sum(
            [
                W[self.agent_id, others[j].agent_id] * (self.lam - others[j].lam)
                for j in multiplier_inds
            ]
        )

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
    def b(self) -> np.ndarray:
        return self._offset

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

    def extra_repr_keys(self) -> List[str]:
        return [
            "agent_id",
            "dim",
            "tau",
            "nu",
            "sigma",
        ]
