"""
"""

from typing import NoReturn, List, Union, Callable, Sequence, Optional
from numbers import Real

import numpy as np
from scipy import sparse

from utils import ReprMixin
from graph import Graph
from ccs import CCS, EuclideanPlus


class Agent(ReprMixin):
    """
    """
    __name__ = "Agent"

    def __init__(self,
                 agent_id:int,
                 ccs:CCS,
                 ceoff:np.ndarray,
                 intercept:np.ndarray,
                 objective:Callable[[np.ndarray, np.ndarray], float],
                 objective_grad:Callable[[np.ndarray, np.ndarray], np.ndarray],
                 step_sizes:Sequence[float]=[0.1,0.1,0.1],) -> NoReturn:
        """
        """
        self.agent_id = agent_id
        self.ccs = ccs
        self._coeff = np.array(ceoff)  # A_i, shape (m, n_i)
        self._intercept = np.array(intercept)  # b_i, shape (m,)
        self._objective = objective
        self._objective_grad = objective_grad
        self.tau, self.nu, self.sigma = step_sizes
        self._ep = EuclideanPlus(self._intercept.shape[0])

        assert self.dim == self._coeff.shape[1]  # n_i
        assert self._coeff.shape[0] == self._intercept.shape[0]  # m
        assert self._intercept.shape[1] == 1

        self._decision = np.zeros((dim,1))  # x_i
        self._multiplier = np.zeros((self._coeff.shape[0],1))  # lambda_i
        self._aux_var = np.zeros((self._coeff.shape[0],1))  # z_i
        self._prev_decision = self._decision.copy()
        self._prev_aux_var = self._aux_var.copy()

    def update(self,
               others:List["Agent"],
               interference_graph:Graph,
               multiplier_graph:Graph,) -> NoReturn:
        """
        """
        self._prev_decision = self._decision.copy()
        self._prev_aux_var = self._aux_var.copy()
        interference_inds = [
            i for i, other in enumerate(others) if \
                other.agent_id in interference_graph.get_neighbors(self.agent_id)
        ]
        multiplier_inds = [
            i for i, other in enumerate(others) if \
                other.agent_id in multiplier_graph.get_neighbors(self.agent_id)
        ]
        self._decision = self.ccs.projection(
            self.x - self.tau * self._objective_grad(
                self.x, self.decision_profile(others, True)
            ) - np.dot(self._coeff.T, self.lam)
        )

        self._aux_var += self.nu * sum([
            multiplier_graph.adj_mat[self.agent_id, others[j].agent_id] * \
                (self.lam - others[j].lam) for j in multiplier_inds
        ])

    def dual_update(self,
                    multiplier_neighbors:List["Agent"],
                    multiplier_graph:Graph,) -> NoReturn:
        """
        """
        multiplier_inds = [
            i for i, other in enumerate(others) if \
                other.agent_id in multiplier_graph.get_neighbors(self.agent_id)
        ]
        raise NotImplementedError
        self._multiplier = self._ep.projection(
            self.lam - self.sigma * (
                np.dot(self._coeff, 2*self.x - self._prev_aux_var) - self._intercept + \
                sum([])  # TODO
            )
        )

    @property
    def dim(self) -> int:
        return self.ccs.dim

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
