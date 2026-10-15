"""Velocity Verlet numerical integration."""

from orbitlab.core.physics import (
    DEFAULT_GRAVITATIONAL_CONSTANT,
    compute_accelerations,
)
from orbitlab.integrators.base import (
    SystemState,
    bodies_from_state,
    validate_timestep,
)


class VelocityVerlet:
    """Advance an N-body state with velocity Verlet integration."""

    def step(
        self,
        state: SystemState,
        dt: float,
        gravitational_constant: float = DEFAULT_GRAVITATIONAL_CONSTANT,
        softening: float = 0.0,
    ) -> SystemState:
        """Return the state one timestep after ``state``.

        Acceleration is evaluated at both the old and newly updated
        positions, as required by velocity Verlet.
        """
        timestep = validate_timestep(dt)
        initial_accelerations = compute_accelerations(
            bodies_from_state(state), gravitational_constant, softening
        )
        positions = (
            state.positions
            + state.velocities * timestep
            + 0.5 * initial_accelerations * timestep**2
        )
        updated_position_state = type(state)(
            state.masses, positions, state.velocities, state.time
        )
        updated_accelerations = compute_accelerations(
            bodies_from_state(updated_position_state),
            gravitational_constant,
            softening,
        )
        velocities = state.velocities + 0.5 * (
            initial_accelerations + updated_accelerations
        ) * timestep
        return type(state)(state.masses, positions, velocities, state.time + timestep)
