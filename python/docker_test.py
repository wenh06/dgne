"""
"""

import numpy as np

from graph import Graph
from agent import Agent
from ccs import (
    Rectangle, L2Ball,
    NonNegativeOrthant, NonPositiveOrthant,
)


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
    pass


def test_agent():
    pass


def test_networked_cournot_game():
    pass


if __name__ == "__main__":
    test_ccs()
    test_graph()
    test_agent()
    test_networked_cournot_game()
