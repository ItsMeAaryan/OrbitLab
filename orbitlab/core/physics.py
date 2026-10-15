"""Newtonian gravitational calculations for OrbitLab."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from orbitlab.core.body import Body


DEFAULT_GRAVITATIONAL_CONSTANT = 6.67430e-11
FloatArray = NDArray[np.float64]


def compute_forces(
    bodies: Iterable[Body],
    gravitational_constant: float = DEFAULT_GRAVITATIONAL_CONSTANT,
    softening: float = 0.0,
) -> FloatArray:
    """Calculate the net Newtonian force on every body.

    Each unique pair is evaluated once. For a pair ``i, j``, the force on
    ``i`` points from ``i`` toward ``j`` and the equal opposite force is
    applied to ``j``. Softening replaces ``r^3`` with
    ``(r^2 + softening^2) ** 1.5``.
    """
    body_list = _validate_inputs(bodies, gravitational_constant, softening)
    dimensions = body_list[0].dimensions
    forces = np.zeros((len(body_list), dimensions), dtype=np.float64)
    positions = np.array([body.position for body in body_list], dtype=np.float64)
    masses = np.array([body.mass for body in body_list], dtype=np.float64)
    softening_squared = softening * softening

    for first_index in range(len(body_list) - 1):
        for second_index in range(first_index + 1, len(body_list)):
            displacement = positions[second_index] - positions[first_index]
            distance_squared = float(np.dot(displacement, displacement))
            if distance_squared == 0.0 and softening == 0.0:
                raise ValueError("coincident bodies require positive softening")

            denominator = (distance_squared + softening_squared) ** 1.5
            pair_force = (
                gravitational_constant
                * masses[first_index]
                * masses[second_index]
                * displacement
                / denominator
            )
            forces[first_index] += pair_force
            forces[second_index] -= pair_force

    return forces


def compute_accelerations(
    bodies: Iterable[Body],
    gravitational_constant: float = DEFAULT_GRAVITATIONAL_CONSTANT,
    softening: float = 0.0,
) -> FloatArray:
    """Calculate each body's acceleration from its net gravitational force."""
    body_list = _validate_inputs(bodies, gravitational_constant, softening)
    forces = compute_forces(body_list, gravitational_constant, softening)
    masses = np.array([body.mass for body in body_list], dtype=np.float64)
    return forces / masses[:, np.newaxis]


def _validate_inputs(
    bodies: Iterable[Body],
    gravitational_constant: float,
    softening: float,
) -> list[Body]:
    """Validate shared inputs for force and acceleration calculations."""
    try:
        body_list = list(bodies)
    except TypeError as error:
        raise ValueError("bodies must be a non-empty iterable of Body objects") from error
    if not body_list:
        raise ValueError("bodies must not be empty")
    if not all(isinstance(body, Body) for body in body_list):
        raise ValueError("bodies must contain only Body objects")

    dimensions = body_list[0].dimensions
    if any(body.dimensions != dimensions for body in body_list[1:]):
        raise ValueError("all bodies must have matching dimensions")

    try:
        gravitational_constant = float(gravitational_constant)
        softening = float(softening)
    except (TypeError, ValueError) as error:
        raise ValueError("gravitational_constant and softening must be real numbers") from error
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0.0:
        raise ValueError("gravitational_constant must be finite and strictly positive")
    if not np.isfinite(softening) or softening < 0.0:
        raise ValueError("softening must be finite and non-negative")
    return body_list