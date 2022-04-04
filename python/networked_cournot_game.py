"""
Networked Cournot Game
"""

import multiprocessing as mp
from typing import NoReturn, Sequence, Callable, Optional, List, Union, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

import functional as F
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
        feasible_set: CCS,
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
        feasible_set: CCS,
            closed convex set, of (embeded) dimension n,
            the feasible set (region) of the agent's decision variable
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
            essentially it is a map from R^n (more precisely from `feasible_set`) to R,
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
            feasible_set,
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
        essentially it is a map from R^n (more precisely from `feasible_set`) to R
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
        run_parallel: bool = False,
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
        run_parallel: bool, default False,
            whether to run the algorithm in parallel

        """
        self._companies = companies
        self._market_capacities = market_capacities
        self._multiplier_graph = multiplier_graph
        self._interference_graph = interference_graph
        self.run_parallel = run_parallel

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

    def run_simulation(
        self, num_steps: int, run_parallel: Optional[bool] = None
    ) -> NoReturn:
        """

        Parameters
        ----------
        num_steps: int,
            number of steps to run the simulation
        run_parallel: bool, optional,
            whether to run the algorithm in parallel,
            if specified, overrides the value of self.run_parallel

        """
        rp = run_parallel if run_parallel is not None else self.run_parallel
        if rp:
            self._run_simulation_parallel(num_steps)
            return
        with tqdm(
            range(num_steps), total=num_steps, desc="Running simulation", unit="step"
        ) as pbar:
            for i in pbar:
                for company in self.companies:
                    # primal update
                    company.primal_update(
                        [c for c in self.companies if c.agent_id != company.agent_id],
                        self.interference_graph,
                        self.multiplier_graph,
                    )
                for company in self.companies:
                    # dual update
                    company.dual_update(
                        [c for c in self.companies if c.agent_id != company.agent_id],
                        self.multiplier_graph,
                    )

    def _run_simulation_parallel(self, num_steps: int) -> NoReturn:
        """NOT implemented yet

        Parameters
        ----------
        num_steps: int,
            number of steps to run the simulation

        """
        raise NotImplementedError
        with tqdm(
            range(num_steps), total=num_steps, desc="Running simulation", unit="step"
        ) as pbar:
            for i in pbar:
                pass

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

        """
        return all([c.is_convergent(keys, func) for c in self.companies])

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "num_companies",
            "num_markets",
            "run_parallel",
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
