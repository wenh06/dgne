"""
"""

from typing import Sequence, Tuple, Callable

import numpy as np

from ccs import CCS


__all__ = [
    "primal_update",
    "dual_update",
    "agent_update",
]


def primal_update(
    agent_id: int,
    A: np.ndarray,
    W: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    lam: np.ndarray,
    prev_x: np.ndarray,
    prev_z: np.ndarray,
    profile: np.ndarray,
    feasible_set: CCS,
    alpha: float,
    tau: float,
    nu: float,
    others_agent_id: Sequence[int],
    others_lam: Sequence[np.ndarray],
    objective_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    x = feasible_set.projection(
        x
        + alpha * (x - prev_x)
        - tau * objective_grad(x, profile)
        - np.matmul(A.T, lam)
    )
    z = (
        z
        + alpha * (z - prev_z)
        + nu
        * sum(
            [
                W[agent_id, oai] * (lam - ol)
                for oai, ol in zip(others_agent_id, others_lam)
            ]
        )
    )
    return x, z


def dual_update(
    agent_id: int,
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    lam: np.ndarray,
    prev_x: np.ndarray,
    prev_z: np.ndarray,
    prev_lam: np.ndarray,
    multiplier_orthant: CCS,
    alpha: float,
    sigma: float,
    others_agent_id: Sequence[int],
    others_z: Sequence[np.ndarray],
    others_prev_z: Sequence[np.ndarray],
    others_lam: Sequence[np.ndarray],
) -> np.ndarray:
    """ """
    lam = multiplier_orthant.projection(
        lam
        + alpha * (lam - prev_lam)
        - sigma
        * (
            np.matmul(A, 2 * x - prev_x)
            - b
            + sum(
                [
                    W[agent_id, oai] * (2 * (z - oz) - (prev_z - opz))
                    for oai, oz, opz in zip(others_agent_id, others_z, others_prev_z)
                ]
            )
            + sum(
                [
                    W[agent_id, oai] * (lam - ol)
                    for oai, ol in zip(others_agent_id, others_lam)
                ]
            )
        )
    )
    return lam


def agent_update(
    dual: bool,
    agent_id: int,
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    lam: np.ndarray,
    prev_x: np.ndarray,
    prev_z: np.ndarray,
    prev_lam: np.ndarray,
    profile: np.ndarray,
    feasible_set: CCS,
    multiplier_orthant: CCS,
    alpha: float,
    tau: float,
    nu: float,
    sigma: float,
    others_agent_id: Sequence[int],
    others_z: Sequence[np.ndarray],
    others_prev_z: Sequence[np.ndarray],
    others_lam: Sequence[np.ndarray],
    objective_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ """
    if dual:  # dual update
        lam = multiplier_orthant.projection(
            lam
            + alpha * (lam - prev_lam)
            - sigma
            * (
                np.matmul(A, 2 * x - prev_x)
                - b
                + sum(
                    [
                        W[agent_id, oai] * (2 * (z - oz) - (prev_z - opz))
                        for oai, oz, opz in zip(
                            others_agent_id, others_z, others_prev_z
                        )
                    ]
                )
                + sum(
                    [
                        W[agent_id, oai] * (lam - ol)
                        for oai, ol in zip(others_agent_id, others_lam)
                    ]
                )
            )
        )
    else:  # primal update
        x = feasible_set.projection(
            x
            + alpha * (x - prev_x)
            - tau * objective_grad(x, profile)
            - np.matmul(A.T, lam)
        )
        z = (
            z
            + alpha * (z - prev_z)
            + nu
            * sum(
                [
                    W[agent_id, oai] * (lam - ol)
                    for oai, ol in zip(others_agent_id, others_lam)
                ]
            )
        )
    return x, z, lam
