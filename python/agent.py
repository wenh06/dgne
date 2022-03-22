"""
"""

from typing import NoReturn, List, Union, Callable, Sequence, Optional
from numbers import Real

import numpy as np
from scipy import sparse

from utils import ReprMixin
from graph import Graph
from ccs import CCS, NonnegativeOrthant, EuclideanSpace


__all__ = [
    "Agent",
]


class Agent(ReprMixin):
    """
    """
    __name__ = "Agent"

    def __init__(self,
                 agent_id:int,
                 ccs:CCS,
                 ceoff:np.ndarray,
                 offset:np.ndarray,
                 objective:Callable[[np.ndarray, np.ndarray], float],
                 objective_grad:Callable[[np.ndarray, np.ndarray], np.ndarray],
                 step_sizes:Sequence[float]=[0.1,0.1,0.1],
                 alpha:Optional[float]=None) -> NoReturn:
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
        objective : Callable[[np.ndarray, np.ndarray], float],
            objective function,
            takes self.x and self.decision_profile(others) as input
        objective_grad : Callable[[np.ndarray, np.ndarray], np.ndarray],
            gradient of the objective function,
        step_sizes : Sequence[float],
            3-tuples of step sizes for x, z, and lam, respectively,
            namely tau, nu, and sigma, respectively.
        """
        self.agent_id = agent_id
        self.ccs = ccs
        self._coeff = np.array(ceoff)  # A_i, shape (m, n_i)
        self._offset = np.array(offset)  # b_i, shape (m,)
        self._objective = objective
        self._objective_grad = objective_grad
        self.tau, self.nu, self.sigma = step_sizes
        self.alpha = alpha
        self._nno = NonnegativeOrthant(self.b.shape[0])
        self._es = EuclideanSpace(self.b.shape[0])

        assert self.dim == self.A.shape[1]  # n_i
        assert self.A.shape[0] == self.b.shape[0]  # m

        self._decision = self.ccs.random_point()  # x_i
        self._multiplier = self._nno.random_point()  # lambda_i
        self._aux_var = self._es.random_point()  # z_i
        self._prev_decision = self._decision.copy()
        self._prev_aux_var = self._aux_var.copy()
        self._prev_multiplier = None
        if self.alpha is not None:
            self._prev_multiplier = self._multiplier.copy()

    def update(self,
               others:List["Agent"],
               interference_graph:Graph,
               multiplier_graph:Graph,) -> NoReturn:
        """
        """
        self._prev_decision = self.x.copy()
        self._prev_aux_var = self.z.copy()
        interference_inds = [
            i for i, other in enumerate(others) if \
                other.agent_id in interference_graph.get_neighbors(self.agent_id)
        ]
        multiplier_inds = [
            i for i, other in enumerate(others) if \
                other.agent_id in multiplier_graph.get_neighbors(self.agent_id)
        ]
        self._decision = self.ccs.projection(
            self.extrapolated_decision - self.tau * self._objective_grad(
                self.x, self.decision_profile(others, True)
            ) - np.matmul(self.A.T, self.lam)
        )

        W = multiplier_graph.adj_mat
        self._aux_var = self.extrapolated_aux_var + self.nu * sum([
            W[self.agent_id, others[j].agent_id] * \
                (self.lam - others[j].lam) for j in multiplier_inds
        ])

    def dual_update(self,
                    others:List["Agent"],
                    multiplier_graph:Graph,) -> NoReturn:
        """
        """
        multiplier_inds = [
            i for i, other in enumerate(others) if \
                other.agent_id in multiplier_graph.get_neighbors(self.agent_id)
        ]
        W = multiplier_graph.adj_mat
        self._multiplier = self._nno.projection(
            self.extrapolated_multiplier - self.sigma * (
                np.matmul(self.A, 2*self.x - self._prev_decision) - self.b + \
                sum([
                    W[self.agent_id, others[j].agent_id] * (2*(self.z - others[j].z) - (self._prev_aux_var - others[j]._prev_aux_var)) \
                        for j in multiplier_inds
                ]) + \
                sum([W[self.agent_id, others[j].agent_id] * (self.lam - others[j].lam) for j in multiplier_inds])
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
    def extrapolated_decision(self) -> np.ndarray:
        if self.alpha is None:
            return self.x
        return self.x + self.alpha * (self.x - self._prev_decision)

    @property
    def extrapolated_aux_var(self) -> np.ndarray:
        if self.alpha is None:
            return self.z
        return self.z + self.alpha * (self.z - self._prev_aux_var)

    @property
    def extrapolated_multiplier(self) -> np.ndarray:
        if self.alpha is None:
            return self.lam
        return self.lam + self.alpha * (self.lam - self._prev_multiplier)

    def degree(self, adj_mat:np.ndarray) -> Real:
        """
        """
        return adj_mat[self.agent_id, :].sum()

    def decision_profile(self, others:List["Agent"], except_self:bool=False) -> np.ndarray:
        """
        """
        agent_ids = [other.agent_id for other in others]
        inds = np.argsort(agent_ids)
        dp = np.concatenate([others[i].decision for i in inds])
        if not except_self:
            insert_pos = sum([other.dim for other in others if other.agent_id < self.agent_id])
            dp = np.insert(dp, insert_pos, self.decision, axis=0)
        return dp

    def extra_repr_keys(self) -> List[str]:
        return ["agent_id", "dim", "tau", "nu", "sigma",]
