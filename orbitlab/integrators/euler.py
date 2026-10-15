"""Explicit Euler numerical integration."""

from orbitlab.core.physics import (
    DEFAULT_GRAVITATIONAL_CONSTANT,
    compute_accelerations,
)
from orbitlab.integrators.base import (
    SystemState,
    bodies_from_state,
    validate_timestep,
)


class ExplicitEuler:
    """Advance an N-body state with the explicit Euler method."""

    def step(
        self,
        state: SystemState,
        dt: float,
        gravitational_constant: float = DEFAULT_GRAVITATIONAL_CONSTANT,
        softening: float = 0.0,
    ) -> SystemState:
        """Return the state one timestep after ``state``."""
        timestep = validate_timestep(dt)
        accelerations = compute_accelerations(
            bodies_from_state(state), gravitational_constant, softening
        )
        positions = state.positions + state.velocities * timestep
        velocities = state.velocities + accelerations * timestep
        return SystemState(
            state.masses,
            positions,
            velocities,
            state.time + timestep,
        )
