"""
Networked Cournot Game
"""

from typing import NoReturn, Sequence, Callable, Optional, List

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from agent import Agent
from ccs import CCS
from graph import Graph
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
        market_price_jac: Callable[[np.ndarray, np.ndarray], np.ndarray],
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
        market_price_jac: Callable[[np.ndarray, np.ndarray], np.ndarray],
            jacobian of the market price function (as function of self.x)
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
        self._market_price_jac = market_price_jac
        self._product_cost = product_cost
        self._product_cost_grad = product_cost_grad

        # equation 36:
        # c_i(x_i) - (P(Ax))^T A_ix_i
        self._objective = lambda decision, profile: self._product_cost(
            decision
        ) - np.matmul(
            self._market_price(decision, profile).T, np.matmul(self.A, decision)
        )

        self._objective_grad = lambda decision, profile: (
            self._product_cost_grad(decision)
            - np.matmul(
                self._market_price_jac(decision, profile).T,
                np.matmul(self.A, decision),
            )
            - np.matmul(self.A.T, self._market_price(decision, profile))
        )

    @property
    def num_markets(self) -> int:
        return self._coeff.shape[0]

    @property
    def market_price(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        market price function,
        takes decision and profile as input,
        essentially it is a map from R^m to R^m,
        which maps the total supply of each market to its corresponding price
        """
        return self._market_price

    @property
    def market_price_jac(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        jacobian of the market price function (as function of the decision),
        taing decision and profile as input
        """
        return self._market_price_jac

    @property
    def product_cost(self) -> Callable[[np.ndarray], float]:
        """
        product cost function,
        takes decision as input,
        essentially it is a map from R^n (more precisely from `ccs`) to R
        """
        return self._product_cost

    @property
    def product_cost_grad(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        gradient of the product cost function, w.r.t. decision
        """
        return self._product_cost_grad


class NetworkedCournotGame(ReprMixin):
    """ """

    __name__ = "NetworkedCournotGame"

    def __init__(
        self,
        companies: Sequence[Company],
        multiplier_graph: Graph,
        interference_graph: Graph,
        market_capacities: np.ndarray,
    ) -> NoReturn:
        """

        Parameters
        ----------
        companies: sequence of Company,
            companies in the game
        multiplier_graph: Graph,
            the multiplier graph
        interference_graph: Graph,
            the interference graph
        market_capacities: np.ndarray,
            market capacities,

        """
        self._companies = companies
        self._market_capacities = market_capacities
        self._multiplier_graph = multiplier_graph
        self._interference_graph = interference_graph

    @property
    def num_companies(self) -> int:
        return len(self._companies)

    @property
    def num_markets(self) -> int:
        return self.market_capacities.shape[0]

    @property
    def companies(self) -> Sequence[Company]:
        return self._companies

    @property
    def market_capacities(self) -> np.ndarray:
        return self._market_capacities

    @property
    def multiplier_graph(self) -> Graph:
        return self._multiplier_graph

    @property
    def interference_graph(self) -> Graph:
        return self._interference_graph

    def run_simulation(self, num_steps: int) -> NoReturn:
        """

        Parameters
        ----------
        num_steps: int,
            number of steps to run the simulation

        """
        with tqdm(
            range(num_steps), total=num_steps, desc="Running simulation", unit="step"
        ) as pbar:
            for i in pbar:
                for company in self.companies:
                    # update
                    company.update(
                        [c for c in self.companies if c.agent_id != company.agent_id],
                        self.interference_graph,
                        self.multiplier_graph,
                    )
                    # dual update
                    company.dual_update(
                        [c for c in self.companies if c.agent_id != company.agent_id],
                        self.multiplier_graph,
                    )

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "num_companies",
            "num_markets",
        ]

    # @property
    # def market_price(self) -> Callable[[np.ndarray], np.ndarray]:
    #     """
    #     """
    #     return self._companies[0]._market_price.func


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
