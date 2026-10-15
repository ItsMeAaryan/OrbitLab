"""The OrbitLab simulation engine."""

from collections.abc import Callable

import numpy as np

from orbitlab.core.physics import DEFAULT_GRAVITATIONAL_CONSTANT
from orbitlab.integrators.base import SystemState, validate_timestep
from orbitlab.simulation.result import SimulationResult


Integrator = Callable[..., SystemState]


class Simulation:
    """Repeatedly advance an initial state with one numerical integrator."""

    def __init__(
        self,
        initial_state: SystemState,
        integrator: object,
        dt: float,
        gravitational_constant: float = DEFAULT_GRAVITATIONAL_CONSTANT,
        softening: float = 0.0,
    ) -> None:
        """Validate and store a simulation configuration."""
        if not isinstance(initial_state, SystemState):
            raise ValueError("initial_state must be a SystemState")
        if not callable(getattr(integrator, "step", None)):
            raise ValueError("integrator must provide a callable step method")

        self._initial_state = SystemState(
            initial_state.masses,
            initial_state.positions,
            initial_state.velocities,
            initial_state.time,
        )
        self.integrator = integrator
        self.dt = validate_timestep(dt)
        self.gravitational_constant = _validate_gravitational_constant(
            gravitational_constant
        )
        self.softening = _validate_softening(softening)

    @property
    def initial_state(self) -> SystemState:
        """Return an independent copy of the configured initial state."""
        return SystemState(
            self._initial_state.masses,
            self._initial_state.positions,
            self._initial_state.velocities,
            self._initial_state.time,
        )

    def run(
        self,
        *,
        steps: int | None = None,
        duration: float | None = None,
    ) -> SimulationResult:
        """Run the configured simulation and record every state.

        Exactly one of ``steps`` or ``duration`` must be provided. A duration
        must represent an integral number of configured timesteps.
        """
        number_of_steps = _resolve_step_count(self.dt, steps, duration)
        state = self.initial_state
        times = np.empty(number_of_steps + 1, dtype=np.float64)
        positions = np.empty(
            (number_of_steps + 1, state.number_of_bodies, state.dimensions),
            dtype=np.float64,
        )
        velocities = np.empty_like(positions)
        times[0] = state.time
        positions[0] = state.positions
        velocities[0] = state.velocities

        for step_index in range(1, number_of_steps + 1):
            state = self.integrator.step(
                state,
                self.dt,
                self.gravitational_constant,
                self.softening,
            )
            times[step_index] = state.time
            positions[step_index] = state.positions
            velocities[step_index] = state.velocities

        return SimulationResult(
            self._initial_state.masses,
            times,
            positions,
            velocities,
            gravitational_constant=self.gravitational_constant,
            softening=self.softening,
            integrator_name=type(self.integrator).__name__,
        )


def _resolve_step_count(
    dt: float,
    steps: int | None,
    duration: float | None,
) -> int:
    """Validate run length and convert duration to an exact step count."""
    if (steps is None) == (duration is None):
        raise ValueError("provide exactly one of steps or duration")
    if steps is not None:
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
            raise ValueError("steps must be a non-negative integer")
        if steps < 0:
            raise ValueError("steps must be a non-negative integer")
        return int(steps)

    try:
        duration_value = float(duration)
    except (TypeError, ValueError) as error:
        raise ValueError("duration must be finite and non-negative") from error
    if not np.isfinite(duration_value) or duration_value < 0.0:
        raise ValueError("duration must be finite and non-negative")
    step_count = duration_value / dt
    rounded_count = round(step_count)
    if not np.isclose(step_count, rounded_count, rtol=0.0, atol=1e-12):
        raise ValueError("duration must be an integral number of timesteps")
    return int(rounded_count)


def _validate_gravitational_constant(value: float) -> float:
    """Validate and normalize the gravitational constant."""
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
    """Validate and normalize the gravitational softening length."""
    try:
        softening = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("softening must be finite and non-negative") from error
    if not np.isfinite(softening) or softening < 0.0:
        raise ValueError("softening must be finite and non-negative")
    return softening
