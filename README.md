# An operator splitting approach for distributed generalized Nash equilibria computation

![pytest](https://github.com/wenh06/dgne/actions/workflows/run-pytest.yml/badge.svg)
![format_check](https://github.com/wenh06/dgne/actions/workflows/check-formatting.yml/badge.svg)

A re-implementation of the paper \[[1](#ref1)\] (currently python, TODO matlab)


## [Python re-implementation](/python/)

Agents are implemented as a class in [python/agent.py](/python/agent.py), conducting update and dual update. Variables and statistics (time consumption, value of objective function, norm of the gradient of the objective function, etc.) during the optimization process are maintained in this class as well.

The class [`CCS`](/python/ccs.py) (abbreviation for closed convex set) is mainly used to do projection to this closed convex set in the algorithm. `Rectangle`, `NonNegativeOrthant`, `NonPositiveOrthant` are the most frequently used closed convex sets.

The class [`Graph`](/python/graph.py) implements the connected symmetric interference graph and multiplier graph, as well as their operations.

The class [`NetworkedCournotGame`](/python/networked_cournot_game.py), along with [simulation.py](/python/simulation.py), implements the networked cournot game described in section 7 of \[[1](#ref1)\].

The file [`minimal_example.py`](/python/minimal_example.py) provides some minimal examples containing networked cournot games consisting of 2 companies and 1 or 2 market(s).

The file [`functional.py`](/python/functional.py) re-implements the `primal update` and `dual update` of the agents (players), so as to facilitate the usage of `multiprocessing` for parallel (or distributed) computation.

The [test file](/python/docker_test.py), used along with [GitHub Pytest Action](https://github.com/wenh06/dgne/actions/workflows/run-pytest.yml) provides automatic test for the classes and algorithms.


### TODO
1. (Done) ~~implement the `primal_update` and `dual_update` functions as external functions in [functional.py](/python/functional.py), instead of member methods of the `Agent` class, so that multiprocessing can be used for parallel computation.~~
2. add stop criteria for the iteration


## [Matlab re-implementation](/matlab/)


## [Complementary materials](/tex/)


## References
1. <a name="ref1"></a> Yi P, Pavel L. An operator splitting approach for distributed generalized Nash equilibria computation[J]. Automatica, 2019, 102: 111-121.
