"""
"""

from functools import partial

import pytest
import numpy as np

from graph import Graph, is_connected
from agent import Agent  # noqa: F401
from ccs import (  # noqa: F401
    Rectangle,
    L2Ball,
    NonNegativeOrthant,
    NonPositiveOrthant,
)
from utils import DEFAULTS
from simulation import setup_simulation, run_simulation  # noqa: F401
from minimal_example import (  # noqa: F401
    setup_minimal_example,
    market_capacities_homo,
    market_D_homo,
    market_P_homo,
    product_cost_parameters_homo,
    num_markets_mono,
    market_capacities_homo_mono,
    market_D_homo_mono,
    market_P_homo_mono,
    product_cost_parameters_homo_mono,
    market_company_connections_homo_mono,
)


_ABS_EPS = 1e-5  # noqa: F401


def test_ccs():
    """ """
    # test rectangle
    space = Rectangle(np.zeros(10), np.ones(10))
    assert space.isin(np.zeros(10))
    assert not space.isin(2 * np.ones(10))
    point = np.random.uniform(0, 1, 10)
    assert space.isin(point)
    assert (space.projection(point) == point).all()
    point = np.ones(10)
    point[3] = -1
    _point = point.copy()
    _point[3] = 0
    assert (space.projection(point) == _point).all()
    point[3] = 2
    _point[3] = 1
    assert (space.projection(point) == _point).all()

    # test non-negative orthant
    space = NonNegativeOrthant(20)
    point = np.random.uniform(0, 1, 20)
    assert space.isin(point)
    assert (space.projection(point) == point).all()
    point[3] *= -1
    _point = point.copy()
    _point[3] = 0
    assert (space.projection(point) == _point).all()

    # test non-positive orthant
    space = NonPositiveOrthant(20)
    point = -np.random.uniform(0, 1, 20)
    assert space.isin(point)
    assert (space.projection(point) == point).all()
    point[3] *= -1
    _point = point.copy()
    _point[3] = 0
    assert (space.projection(point) == _point).all()

    print("test_ccs passed!")


def test_graph():
    g = Graph.random(num_vertices=1000, num_neighbors=(4, 16))
    assert is_connected(g)
    assert g.is_weighted is False
    assert g.num_vertices == 1000
    assert all(
        [4 <= len(g.get_neighbors(vertex_id)) for vertex_id in range(g.num_vertices)]
    )
    g.random_weights(generator=partial(DEFAULTS.RNG.uniform, 1, 4))
    assert g.is_weighted is True

    print("test_graph passed!")


def test_networked_cournot_game():
    ncg = setup_simulation()
    ncg.run_simulation(500)
    # assert ncg.is_convergent()

    print("test_networked_cournot_game passed!")


def test_parallel_running():
    ncg = setup_simulation(run_parallel=True)
    ncg.run_simulation(500)
    # assert ncg.is_convergent()

    print("test_parallel_running passed!")


def test_minimal_example():
    me = setup_minimal_example(verbose=2)
    me.run_simulation(1000)
    assert me.is_convergent()

    print("test_minimal_example passed!")


def test_minimal_example_homo():
    me_homo = setup_minimal_example(
        market_capacities=market_capacities_homo,
        market_P=market_P_homo,
        market_D=market_D_homo,
        product_cost_parameters=product_cost_parameters_homo,
        verbose=2,
    )
    me_homo.run_simulation(1000)
    c1, c2 = me_homo.companies
    assert me_homo.is_convergent()

    print("test_minimal_example_homo passed!")


def test_minimal_example_homo_mono():
    me_homo_mono = setup_minimal_example(
        num_markets=num_markets_mono,
        market_company_connections=market_company_connections_homo_mono,
        market_capacities=market_capacities_homo_mono,
        market_P=market_P_homo_mono,
        market_D=market_D_homo_mono,
        product_cost_parameters=product_cost_parameters_homo_mono,
        verbose=2,
    )
    me_homo_mono.run_simulation(1000)
    c1, c2 = me_homo_mono.companies
    assert 0.6 == pytest.approx(c1.x[0]) == pytest.approx(c2.x[0])
    assert (
        -0.72
        == pytest.approx(c1.objective(c1.x, c2.x))
        == pytest.approx(c2.objective(c2.x, c1.x))
    )
    assert me_homo_mono.is_convergent()

    print("test_minimal_example_homo_mono passed!")


if __name__ == "__main__":
    test_ccs()
    test_graph()
    test_networked_cournot_game()
    test_parallel_running()
    test_minimal_example()
    test_minimal_example_homo()
    test_minimal_example_homo_mono()
