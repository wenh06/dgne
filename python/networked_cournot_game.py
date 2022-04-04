"""
Networked Cournot Game
"""

import multiprocessing as mp
from typing import NoReturn, Sequence, Callable, Optional, List, Union, Tuple, Dict

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


__all__ = [
    "NetworkedCournotGame",
    "Company",
]


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

    @property
    def company_id(self) -> int:
        return self.agent_id


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
        self._companies = sorted((c for c in companies), key=lambda c: c.company_id)
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
    def r(self) -> np.ndarray:
        """alias of market_capacities"""
        return self.market_capacities

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

    @property
    def x(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray,
            the concatenation of decision vectors of all companies,
            of shape (n,), where n is the total number of company-market connections

        """
        return np.concatenate([c.x for c in self.companies])

    @property
    def z(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray,
            the concatenation of aux variables of all companies,
            of shape (num_markets * num_companies,)

        """
        return np.concatenate([c.z for c in self.companies])

    @property
    def lam(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray,
            the concatenation of lagrange multipliers of all companies,
            of shape (num_markets * num_companies,)

        """
        return np.concatenate([c.lam for c in self.companies])

    @property
    def A(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray,
            the concatenation of the constraint coeff. matrix of all companies,
            of shape (num_markets, n),
            where n is the total number of company-market connections

        """
        return np.concatenate([c.A for c in self.companies], axis=1)

    @property
    def Ax(self) -> np.ndarray:
        return self.A @ self.x

    @property
    def omega(self) -> np.ndarray:
        return np.concatenate([self.x, self.z, self.lam])

    def get_cache(
        self,
        key: Optional[str] = None,
    ) -> Union[List[Dict[str, np.ndarray]], List[np.ndarray], List[float]]:
        """

        Get cached sequence of variable values of the optimization process

        Parameters
        ----------
        key : str, optional,
            the key of the variable to be returned, by default None
            can be one of "x", "z", "lam", "omega",
            if None, return all the cached variable values

        Returns
        -------
        List[Dict[str, np.ndarray]] or List[np.ndarray] or List[float],
            if `key` is None, return a list of dicts of variable values;
            if `key` is not None, return a list of variable values

        """
        if key is not None:
            assert key in [
                "x",
                "z",
                "lam",
                "omega",
            ], f"""key must be one of "x", "z", "lam", "omega" or None, but got {key}"""
        if key is None:
            return dict(
                x=self.get_cache("x"),
                z=self.get_cache("z"),
                lam=self.get_cache("lam"),
                omega=self.get_cache("omega"),
            )
        elif key == "omega":
            cache = [
                np.concatenate((x, z, lam))
                for x, z, lam in zip(
                    self.get_cache("x"), self.get_cache("z"), self.get_cache("lam")
                )
            ]
            return cache
        else:
            _cache = [c.get_cache(key) for c in self.companies]
            cache = [
                np.concatenate(
                    [
                        _cache[company_idx][step_idx]
                        for company_idx in range(len(self.companies))
                    ]
                )
                for step_idx in range(len(_cache[0]))
            ]
            del _cache
            return cache

    def get_metrics(
        self, key: Optional[str] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """NOT finished yet,

        Computes a set of predefined metrics.

        Parameters
        ----------
        key : str, optional,
            the key of the metric to be returned, by default None
            can be one of "xxx" (to add),
            if None, return all the cached metrics

        Returns
        -------
        Dict[str, np.ndarray] or np.ndarray,
            if `key` is None, return a list of dicts of metrics;
            if `key` is not None, return a list of metrics

        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Company:
        return self.companies[index]

    def __len__(self) -> int:
        return len(self.companies)

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
