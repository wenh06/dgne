"""
"""

from functools import partial
from typing import NoReturn

import numpy as np

from graph import Graph
from ccs import Rectangle
from networked_cournot_game import (
    Company, NetworkedCournotGame
)
from utils import RNG


__all__ = [
    "run_simulation",
]


# section 7


###############################################################################
# setups
num_companies, num_markets = 20, 7
###############################################################################


###############################################################################
# Fig. 1
market_company_connections = np.array([
    [1,1], [1,2], [1,5], [1,6], [1,10],
    [2,2], [2,3], [2,6],
    [3,4], [3,6], [3,8], [3,9],
    [4,6], [4,7], [4,8], [4,10], [4,11],
    [5,11], [5,12], [5,13], [5,15], [5,16],
    [6,10], [6,14], [6,15], [6,17],
    [7,15], [7,16], [7,17], [7,18], [7,19], [7,20],
], dtype=int) - 1
###############################################################################


###############################################################################
# interference edge set from Fig. 1
interference_edge_set = []
for i in range(num_companies-1):
    for m in market_company_connections[np.where(market_company_connections[:,1] == i)[0]][:,0]:
        for c in market_company_connections[np.where(market_company_connections[:,0] == m)[0]][:,1]:
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
interference_graph = Graph(num_vertices=num_companies, edge_set=interference_edge_set)
# interference_graph.random_weights()


# Player i decides its strategy in the competition
# in n_i markets by delivering x_i ∈ R^n_i amount of products to the
# markets it connects with
num_company_market_connection = np.array([
    (market_company_connections[:,1]==i).sum() for i in range(num_companies)
], dtype=int)  # the n_i's
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
    current_markets = \
        market_company_connections[np.where(market_company_connections[:,1] == i)[0]][:,0]
    for j, m in enumerate(current_markets):
        coeff[m,j] = 1
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
multiplier_edge_set = np.array([
    [i, i+1] for i in range(num_companies-1)
] + [[num_companies-1, 0], [2-1,15-1], [6-1,13-1]], dtype=int)

# construct multiplier graph
multiplier_graph = Graph(num_vertices=num_companies, edge_set=multiplier_edge_set)
# multiplier_graph.random_weights()
###############################################################################


###############################################################################
# Market M_j has a maximal capacity of r_j randomly drawn from [0.5, 1].
market_capacities = RNG.uniform(0.5, 1, num_markets)
###############################################################################


###############################################################################
# randomly drawn from [2, 4] and [0.5, 1]
market_P = RNG.uniform(2, 4, num_markets)
market_D = RNG.uniform(0.5, 1, num_markets)

def market_price(company_id:int, decision:np.ndarray, profile:np.ndarray) -> float:
    """
    """
    split_inds = np.append(0, np.cumsum(num_company_market_connection))
    mp = market_P - market_D * np.matmul(
        total_coeff,
        np.insert(profile, split_inds[company_id], decision)
    )
    return mp

def market_price_grad(company_id:int, decision:np.ndarray, profile:np.ndarray) -> np.ndarray:
    """
    gradient as a function from R^m to R^m
    """
    mpg = -np.diag(market_D)
    return mpg
###############################################################################


###############################################################################
# π_i is randomly drawn from [1, 8],
# and each component of b_i is randomly drawn from [0.1, 0.6].
product_cost_parameters = dict(
    pi = [RNG.integers(1, 8, endpoint=True) for _ in range(num_companies)],
    b = [RNG.uniform(0.1, 0.6, n) for n in num_company_market_connection],
)

def product_cost(pi:int, b:np.ndarray, decision:np.ndarray,) -> float:
    """
    """
    decision = np.array(decision).flatten()
    return pi * np.linalg.norm(decision)**2 + np.dot(decision, b)

def product_cost_grad(pi:int, b:np.ndarray, decision:np.ndarray,) -> np.ndarray:
    """
    gradient of `product_cost` w.r.t. decision
    """
    decision = np.array(decision).flatten()
    return 2 * pi * decision + b
###############################################################################


companies = [
    Company(
        company_id=i,
        ccs=Rectangle(
            # Player i has a local constraint 0 < x_i < Θ_i
            # and each component of Θ_i is randomly drawn from [1, 1.5].
            np.zeros((num_company_market_connection[i]),),
            RNG.uniform(1, 1.5, num_company_market_connection[i]),
        ),
        ceoff=company_parameters["ceoff"][i],
        offset=market_capacities,
        market_price=partial(market_price, i),
        market_price_grad=partial(market_price_grad, i),
        product_cost=partial(product_cost, product_cost_parameters["pi"][i], product_cost_parameters["b"][i]),
        product_cost_grad=partial(product_cost_grad, product_cost_parameters["pi"][i], product_cost_parameters["b"][i]),
        step_sizes=(0.03,0.2,0.03),
    ) for i in range(num_companies)
]


def run_simulation() -> NoReturn:
    """
    """
    pass


if __name__ == '__main__':
    run_simulation()
