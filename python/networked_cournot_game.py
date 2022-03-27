"""
Networked Cournot Game
"""

from typing import NoReturn, Sequence, Callable, Optional

import numpy as np

from agent import Agent
from ccs import CCS
from utils import ReprMixin


class Company(Agent):
    """ """

    __name__ = "Company"

    def __init__(
        self,
        company_id: int,
        ccs: CCS,
        ceoff: np.ndarray,
        offset: np.ndarray,
        market_price: Callable[[np.ndarray, np.ndarray], np.ndarray],
        market_price_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
        product_cost: Callable[[np.ndarray], float],
        product_cost_grad: Callable[[np.ndarray], np.ndarray],
        step_sizes: Sequence[float] = [0.1, 0.1, 0.1],
        alpha: Optional[float] = None,
    ) -> NoReturn:
        """

        Parameters
        ----------
        company_id: int,
            company id
        ccs: CCS,
            closed convex set, of (embeded) dimension n
        coeff: np.ndarray,
            coefficient of the linear constraint,
            of shape (m, n)
        offset: np.ndarray,
            offset of the linear constraint,
            of shape (m,)
        market_price: Callable[[np.ndarray, np.ndarray], np.ndarray],
            market price function,
            takes self.x and self.decision_profile(others) as input,
            essentially it is a map from R^m to R^m,
            which maps the total supply of each market to its corresponding price
        market_price_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
            gradient of the market price function, as a map from R^m to R^m
        product_cost: Callable[[np.ndarray], float],
            product cost function,
            takes self.x as input,
            essentially it is a map from R^n (more precisely from `ccs`) to R,
        product_cost_grad: Callable[[np.ndarray], np.ndarray],
            gradient of the product cost function, w.r.t. self.x,
        step_sizes : Sequence[float],
            3-tuples of step sizes for x, z, and lam, respectively,
            namely tau, nu, and sigma, respectively
        alpha: float, optional,
            factor for the extrapolation of the variables (x, z, lambda)
        """
        super().__init__(
            company_id,
            ccs,
            ceoff,
            offset,
            2,  # constraint type 2: offset - ceoff @ x >= 0
            None,
            None,
            step_sizes,
            alpha,
        )
        self._market_price = market_price
        self._market_price_grad = market_price_grad
        self._product_cost = product_cost
        self._product_cost_grad = product_cost_grad

        # equation 36:
        # c_i(x_i) - (P(Ax))^T A_ix_i
        self._objective = lambda decision, profile: self._product_cost(
            decision
        ) - np.matmul(
            self._market_price(decision, profile).T, np.matmul(self.A, decision)
        )

        def objective_grad(decision: np.ndarray, profile: np.ndarray) -> np.ndarray:
            """
            gradient of self._objective w.r.t. decision
            """
            num_markets = self.A.shape[0]
            g = np.zeros(self.A.shape[1])
            for k in range(self.dim):
                g[k] = self._product_cost_grad(decision) - sum(
                    [
                        np.dot(
                            self._market_price_grad(decision, profile)[t], self.A[:, k]
                        )
                        * np.dot(self.A[t], decision)
                        + self._market_price(decision, profile)[t] * self.A[t, k]
                        for t in range(num_markets)
                    ]
                )
            return g

        self._objective_grad = objective_grad

    @property
    def num_markets(self) -> int:
        return self._coeff.shape[0]


class NetworkedCournotGame(ReprMixin):
    """ """

    __name__ = "NetworkedCournotGame"

    def __init__(
        self,
        companies: Sequence[Agent],
        max_capacities: np.ndarray,
    ) -> NoReturn:
        """ """
        self._companies = companies
        self._max_capacities = max_capacities

    # @property
    # def market_price(self) -> Callable[[np.ndarray], np.ndarray]:
    #     """
    #     """
    #     return self._companies._market_price


def linear_inverse_demand(
    coeff: np.ndarray, companies: Sequence[Company]
) -> np.ndarray:
    """
    coeff of shape (m, 2), with the first column `offsets`,
    the second column negative `slopes` (being positive)
    """
    x = sum([np.matmul(c.A, c.x) for c in companies])
    return coeff[:, 0] - np.dot(coeff[:, 1], x)


def _linear_inverse_demand(
    coeff: np.ndarray,
    company_coeffs: Sequence[np.ndarray],
    company_decisions: Sequence[np.ndarray],
) -> np.ndarray:
    """
    coeff of shape (m, 2), with the first column `offsets`,
    the second column negative `slopes` (being positive)
    """
    x = sum([np.matmul(A, x) for A, x in zip(company_coeffs, company_decisions)])
    return coeff[:, 0] - np.dot(coeff[:, 1], x)
