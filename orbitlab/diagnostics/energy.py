"""Energy diagnostics for Newtonian N-body systems."""

import numpy as np

from orbitlab.core.physics import DEFAULT_GRAVITATIONAL_CONSTANT
from orbitlab.integrators.base import SystemState


def kinetic_energy(state: SystemState) -> float:
    """Return ``1/2 * sum(m_i * |v_i|^2)``."""
    return float(0.5 * np.sum(state.masses * np.sum(state.velocities**2, axis=1)))


def gravitational_potential_energy(
    state: SystemState,
    gravitational_constant: float = DEFAULT_GRAVITATIONAL_CONSTANT,
    softening: float = 0.0,
) -> float:
    """Return pairwise gravitational potential energy.

    The softened separation is ``sqrt(distance**2 + softening**2)``, matching
    the force law used by the Phase 1 physics module.
    """
    constant = _validate_gravitational_constant(gravitational_constant)
    softening_value = _validate_softening(softening)
    energy = 0.0
    for first_index in range(state.number_of_bodies - 1):
        for second_index in range(first_index + 1, state.number_of_bodies):
            displacement = state.positions[second_index] - state.positions[first_index]
            distance_squared = float(np.dot(displacement, displacement))
            if distance_squared == 0.0 and softening_value == 0.0:
                raise ValueError("coincident bodies require positive softening")
            separation = np.sqrt(distance_squared + softening_value**2)
            energy -= (
                constant
                * state.masses[first_index]
                * state.masses[second_index]
                / separation
            )
    return float(energy)


def total_energy(
    state: SystemState,
    gravitational_constant: float = DEFAULT_GRAVITATIONAL_CONSTANT,
    softening: float = 0.0,
) -> float:
    """Return kinetic plus gravitational potential energy."""
    return kinetic_energy(state) + gravitational_potential_energy(
        state, gravitational_constant, softening
    )


def _validate_gravitational_constant(value: float) -> float:
    try:
        constant = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "gravitational_constant must be finite and strictly positive"
        ) from error
    if not np.isfinite(constant) or constant <= 0.0:
        raise ValueError("gravitational_constant must be finite and strictly positive")
    return constant


def _validate_softening(value: float) -> float:
    try:
        softening = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("softening must be finite and non-negative") from error
    if not np.isfinite(softening) or softening < 0.0:
        raise ValueError("softening must be finite and non-negative")
    return softening
