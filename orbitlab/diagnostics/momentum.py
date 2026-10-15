"""Linear and angular momentum diagnostics."""

import numpy as np
from numpy.typing import NDArray

from orbitlab.integrators.base import SystemState


FloatArray = NDArray[np.float64]


def linear_momentum(state: SystemState) -> FloatArray:
    """Return total linear momentum as a vector."""
    return np.sum(state.masses[:, np.newaxis] * state.velocities, axis=0)


def angular_momentum(state: SystemState) -> float | FloatArray:
    """Return angular momentum as a 2D scalar or 3D vector."""
    weighted_velocities = state.masses[:, np.newaxis] * state.velocities
    if state.dimensions == 2:
        return float(
            np.sum(
                state.positions[:, 0] * weighted_velocities[:, 1]
                - state.positions[:, 1] * weighted_velocities[:, 0]
            )
        )
    return np.sum(np.cross(state.positions, weighted_velocities), axis=0)
