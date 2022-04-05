"""
"""

import argparse
from functools import partial
from typing import NoReturn, Sequence, Optional

import numpy as np

from graph import Graph
from ccs import Rectangle
from networked_cournot_game import Company, NetworkedCournotGame
from utils import RNG


__all__ = [
    "setup_simulation",
    "run_simulation",
]


# section 7


###############################################################################
# setups
_num_companies, _num_markets = 20, 7
###############################################################################


###############################################################################
# Fig. 1
# fmt: off
_market_company_connections = (
    np.array(
        [
            [1, 1], [1, 2], [1, 5], [1, 6], [1, 10],
            [2, 2], [2, 3], [2, 6],
            [3, 4], [3, 6], [3, 8], [3, 9],
            [4, 6], [4, 7], [4, 8], [4, 10], [4, 11],
            [5, 11], [5, 12], [5, 13], [5, 15], [5, 16],
            [6, 10], [6, 14], [6, 15], [6, 17],
            [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 20],
        ],
        dtype=int,
    )
    - 1
)
# fmt: on
###############################################################################

###############################################################################
# Market M_j has a maximal capacity of r_j randomly drawn from [0.5, 1].
_market_capacities = RNG.uniform(0.5, 1, _num_markets)
###############################################################################


def setup_simulation(
    num_companies: int = _num_companies,
    num_markets: int = _num_markets,
    market_company_connections: np.ndarray = _market_company_connections,
    market_capacities: np.ndarray = _market_capacities,
    step_sizes: Sequence[float] = (0.03, 0.2, 0.03),
    alpha: Optional[float] = None,
    **kwargs: dict,
) -> NetworkedCournotGame:
    """

    Setup the simulation according to section 7 of the paper.

    Parameters
    ----------
    num_companies: int,
        number of companies
    num_markets: int,
        number of markets
    market_company_connections: np.ndarray,
        market-company connection matrix,
        of shape (num_connections, 2),
        each element is a pair of market and company indices
    market_capacities: np.ndarray,
        market capacities,
        of shape (num_markets,),
    step_sizes: sequence of float,
        3-tuples of step sizes for x, z, and lam, respectively,
        namely tau, nu, and sigma, respectively
    alpha: float, optional,
        factor for the extrapolation of the variables (x, z, lambda)
        if alpha is None, then the extrapolation is disabled
    kwargs: dict,
        additional keyword arguments,
        for changing the parameters of the game, including:
            - multiplier_edge_set:
                sequence of 2-tuples of int,
                of variable length
            - market_P: sequence of float,
                of length num_markets
            - market_D: sequence of float,
                of length num_markets
            - product_cost_parameters: dict,
                parameters for the product cost function,
                with items "pi", "b",
                "pi" is a sequence of int, of length num_companies
                "b" is a sequence of np.ndarray, of length num_companies
            - verbose: int, default 0,
                verbosity level for printing the simulation parameters

    Returns
    -------
    networked_cournot_game: NetworkedCournotGame,
        an instance of NetworkedCournotGame

    NOTE
    ----
    Most parameters of the game have default values in this function

    """

    verbose = kwargs.get("verbose", 0)

    # assertions
    assert market_capacities.shape == (num_markets,)
    assert (
        market_company_connections.ndim == 2
        and market_company_connections.shape[1] == 2
    )

    ###############################################################################
    # interference edge set from Fig. 1
    interference_edge_set = []
    for i in range(num_companies - 1):
        for m in market_company_connections[
            np.where(market_company_connections[:, 1] == i)[0]
        ][:, 0]:
            for c in market_company_connections[
                np.where(market_company_connections[:, 0] == m)[0]
            ][:, 1]:
                if i < c and [i, c] not in interference_edge_set:
                    interference_edge_set.append([i, c])
    interference_edge_set = np.array(interference_edge_set, dtype=int)

    """
    interference_edge_set = np.array([
        [1,2], [1,5], [1,6], [1,10],
        [2,3], [2,5], [2,6], [2,10],
        [3,6],
        [4,6], [4,8], [4,9],
        [5,6], [5,10],
        [6,7], [6,8], [6,9], [6,10], [6,11],
        [7,8], [7,10], [7,11],
        [8,9], [8,10], [8,11],
        [10,11], [10,14], [10,15], [10,17],
        [11,12], [11,13], [11,15], [11,16],
        [12,13], [12,15], [12,16],
        [13,15], [13,16],
        [14,15], [14,17],
        [15,16], [15,17], [15,18], [15,19], [15,20],
        [16,17], [16,18], [16,19], [16,20],
        [17,18], [17,19], [17,20],
        [18,19], [18,20],
        [19,20],
    ], dtype=int) - 1
    """

    # construct interference graph
    interference_graph = Graph(
        num_vertices=num_companies, edge_set=interference_edge_set
    )
    # interference_graph.random_weights()
    if verbose >= 2:
        print(f"interference_graph: {interference_graph}")

    # Player i decides its strategy in the competition
    # in n_i markets by delivering x_i ∈ R^{n_i} amount of products to the
    # markets it connects with
    num_company_market_connection = np.array(
        [(market_company_connections[:, 1] == i).sum() for i in range(num_companies)],
        dtype=int,
    )  # the n_i's
    ###############################################################################

    ###############################################################################
    # The j-th column of A_i, denoted by [A_i]_{:j}, has only one element as
    # 1, all the other ones being 0; [A_i]_{:j} has its k-th element equal to
    # 1 if and only if player i delivers [x_i]_j production to M_k

    company_parameters = {
        "ceoff": [],
        "offset": [],
    }
    for i in range(num_companies):
        coeff = np.zeros((num_markets, num_company_market_connection[i]))
        current_markets = market_company_connections[
            np.where(market_company_connections[:, 1] == i)[0]
        ][:, 0]
        for j, m in enumerate(current_markets):
            coeff[m, j] = 1
        offset = np.zeros((num_markets,))
        company_parameters["ceoff"].append(coeff)
        company_parameters["offset"].append(offset)
    total_coeff = np.concatenate(company_parameters["ceoff"], axis=1)
    total_offset = sum(company_parameters["offset"])
    ###############################################################################

    ###############################################################################
    # multiplier edge set, decribed in section 7.2 as
    # "We adopt a ring graph arranged in alphabetical order
    # with additional edges (2, 15), (6, 13) as the multiplier graph"
    multiplier_edge_set = kwargs.get("multiplier_edge_set", None)
    if multiplier_edge_set is None:
        multiplier_edge_set = np.array(
            [[i, i + 1] for i in range(num_companies - 1)]
            + [[num_companies - 1, 0], [2 - 1, 15 - 1], [6 - 1, 13 - 1]],
            dtype=int,
        )

    # construct multiplier graph
    multiplier_graph = Graph(num_vertices=num_companies, edge_set=multiplier_edge_set)
    # multiplier_graph.random_weights()
    if verbose >= 2:
        print(f"multiplier_graph: {multiplier_graph}")
    ###############################################################################

    ###############################################################################
    # The market price is taken as a linear function P − DAx
    # known as a linear inverse demand function in economics
    # P, D randomly drawn from [2, 4] and [0.5, 1]
    market_P = kwargs.get("market_P", RNG.uniform(2, 4, num_markets))
    market_D = kwargs.get("market_D", RNG.uniform(0.5, 1, num_markets))
    if verbose >= 1:
        print(f"market_P (shape {market_P.shape}): {market_P}")
        print(f"market_D (shape {market_D.shape}): {market_D}")

    def _martket_price(supply: np.ndarray) -> np.ndarray:
        """

        Linear inverse demand function: P(s) = p − Ds,
        a function from R^m to R^m

        Parameters
        ----------
        supply : np.ndarray
            the supply vector

        Returns
        -------
        np.ndarray,
            the market price vector

        """
        return market_P - market_D * supply

    def market_price(
        company_id: int,
        decision: np.ndarray,
        profile: np.ndarray,
        num_cmc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""

        market price function from the perspective of company i,
        i.e. the map
            .. math::
                P(x_i) := P(x_i, x_{-i}) = P(x) = p − DAx = p − D \left( \sum\limits_{i=1}^{N} A_ix_i \right)

        Parameters
        ----------
        company_id: int,
            the id of the company
        decision: np.ndarray,
            the decision of the company
        profile: np.ndarray,
            the concatenated decision profile of the other companies in the market
        num_cmc: np.ndarray, optional
            the array of company market connection numbers,
            of shape (num_companies,),
            if None, default to `num_company_market_connection`

        Returns
        -------
        mp: np.ndarray,
            the market price

        """
        if num_cmc is None:
            num_cmc = num_company_market_connection
        split_inds = np.append(0, np.cumsum(num_cmc))
        mp = _martket_price(
            supply=np.matmul(
                total_coeff, np.insert(profile, split_inds[company_id], decision)
            )
        )
        return mp

    def _market_price_jac(supply: np.ndarray) -> np.ndarray:
        r"""

        jacobbian of the function `market_price` as a function from R^m to R^m
        i.e. the jacobian of the map
            .. math::
                P(s) = P − Ds

        Parameters
        ----------
        supply: np.ndarray,
            the supply vector

        Returns
        -------
        np.ndarray,
            the jacobbian of the market price function `_market_price`
        """
        return -np.diag(market_D)

    def market_price_jac(
        company_id: int,
        decision: np.ndarray,
        profile: np.ndarray,
        num_cmc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""

        jacobian of the market price function from the perspective of company i,
        i.e. jacobian of the map
            .. math::
                P(x_i) := P(x_i, x_{-i}) = P(x) = p − DAx = p − D \left( \sum\limits_{i=1}^{N} A_ix_i \right)

        Parameters
        ----------
        company_id: int,
            the id of the company
        decision: np.ndarray,
            the decision of the company
        profile: np.ndarray,
            the concatenated decision profile of the other companies in the market
        num_cmc: np.ndarray, optional
            the array of company market connection numbers,
            of shape (num_companies,),
            if None, default to `num_company_market_connection`

        Returns
        -------
        np.ndarray,
            the Jacobian of the market price as a function of `decision`

        """
        split_inds = np.append(0, np.cumsum(num_cmc or num_company_market_connection))
        supply = np.matmul(
            total_coeff, np.insert(profile, split_inds[company_id], decision)
        )
        A_i = total_coeff[:, split_inds[company_id] : split_inds[company_id + 1]]
        return np.matmul(_market_price_jac(supply), A_i)

    ###############################################################################

    ###############################################################################
    # local cost function of palyer i, denoted by c_i(x_i)
    # in section 7.2 as "c_i(x_i) = \pi \left(∑_{j = 1}^{n_i} [x_i]_j \right)^2 + b_i^T x_i"
    # π_i is randomly drawn from [1, 8],
    # and each component of b_i is randomly drawn from [0.1, 0.6].
    product_cost_parameters = kwargs.get("product_cost_parameters", None)
    if product_cost_parameters is None:
        product_cost_parameters = dict(
            pi=[RNG.integers(1, 8, endpoint=True) for _ in range(num_companies)],
            b=[RNG.uniform(0.1, 0.6, n) for n in num_company_market_connection],
        )
    if verbose >= 1:
        print(f"product_cost_parameters: {product_cost_parameters}")

    def product_cost(
        pi: int,
        b: np.ndarray,
        decision: np.ndarray,
    ) -> float:
        r"""

        local cost function of palyer i, defined as

            .. math::
                c_i(x_i) = \pi \left(∑_{j = 1}^{n_i} [x_i]_j \right)^2 + b_i^T x_i

        """
        decision = np.array(decision).flatten()
        # return pi * np.linalg.norm(decision) ** 2 + np.dot(decision, b)
        return pi * decision.sum() ** 2 + np.dot(decision, b)

    def product_cost_grad(
        pi: int,
        b: np.ndarray,
        decision: np.ndarray,
    ) -> np.ndarray:
        """
        gradient of `product_cost` w.r.t. decision
        """
        decision = np.array(decision).flatten()
        # return 2 * pi * decision + b
        return 2 * pi * decision.sum() + b

    ###############################################################################

    companies = [
        Company(
            company_id=company_id,
            feasible_set=Rectangle(
                # Player i has a local constraint 0 < x_i < Θ_i
                # and each component of Θ_i is randomly drawn from [1, 1.5].
                np.zeros(
                    (num_company_market_connection[company_id]),
                ),
                RNG.uniform(1, 1.5, num_company_market_connection[company_id]),
            ),
            ceoff=company_parameters["ceoff"][company_id],
            offset=market_capacities,
            market_price=partial(market_price, company_id),
            market_price_jac=partial(market_price_jac, company_id),
            product_cost=partial(
                product_cost,
                product_cost_parameters["pi"][company_id],
                product_cost_parameters["b"][company_id],
            ),
            product_cost_grad=partial(
                product_cost_grad,
                product_cost_parameters["pi"][company_id],
                product_cost_parameters["b"][company_id],
            ),
            step_sizes=step_sizes,
        )
        for company_id in range(num_companies)
    ]

    networked_cournot_game = NetworkedCournotGame(
        companies=companies,
        multiplier_graph=multiplier_graph,
        interference_graph=interference_graph,
        market_capacities=market_capacities,
    )

    return networked_cournot_game


def run_simulation(num_steps: int) -> NoReturn:
    """

    Parameters
        ----------
        num_steps: int,
            number of steps to run the simulation

    """
    networked_cournot_game = setup_simulation(
        _num_companies, _num_markets, _market_company_connections
    )
    networked_cournot_game.run_simulation(num_steps)


def get_parser() -> dict:
    """ """
    description = "Run a simulation of the Networked Cournot Game"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-steps",
        type=int,
        default=10000,
        help="number of simulation steps",
        dest="num_steps",
    )

    args = vars(parser.parse_args())

    return args


if __name__ == "__main__":
    args = get_parser()
    run_simulation(args["num_steps"])
