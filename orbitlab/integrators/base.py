"""Shared state representation and validation for numerical integrators."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orbitlab.core.body import Body


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SystemState:
    """The complete state of a non-empty N-body system at one time.

    Arrays are copied during construction. The dataclass itself is frozen,
    while its arrays are kept private from callers through those copies and
    through each integrator returning a new state.
    """

    masses: FloatArray
    positions: FloatArray
    velocities: FloatArray
    time: float = 0.0

    def __post_init__(self) -> None:
        """Validate and copy all state values."""
        try:
            masses = np.array(self.masses, dtype=np.float64, copy=True)
            positions = np.array(self.positions, dtype=np.float64, copy=True)
            velocities = np.array(self.velocities, dtype=np.float64, copy=True)
            time = float(self.time)
        except (TypeError, ValueError) as error:
            raise ValueError("state values must be finite numeric values") from error

        if masses.ndim != 1 or masses.size == 0:
            raise ValueError("masses must be a non-empty one-dimensional array")
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError("positions must have shape (number_of_bodies, 2 or 3)")
        if velocities.shape != positions.shape:
            raise ValueError("positions and velocities must have matching shapes")
        if positions.shape[0] != masses.size:
            raise ValueError("masses and positions must contain the same number of bodies")
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("masses must be finite and strictly positive")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain only finite values")
        if not np.all(np.isfinite(velocities)):
            raise ValueError("velocities must contain only finite values")
        if not np.isfinite(time):
            raise ValueError("time must be finite")

        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "time", time)

    @property
    def number_of_bodies(self) -> int:
        """Return the number of bodies represented by the state."""
        return self.masses.size

    @property
    def dimensions(self) -> int:
        """Return the spatial dimensionality of the state."""
        return self.positions.shape[1]


def validate_timestep(dt: float) -> float:
    """Return a validated positive timestep."""
    try:
        timestep = float(dt)
    except (TypeError, ValueError) as error:
        raise ValueError("dt must be a finite, strictly positive number") from error
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("dt must be a finite, strictly positive number")
    return timestep


def bodies_from_state(state: SystemState) -> list[Body]:
    """Convert a validated state into the Phase 1 body representation."""
    return [
        Body(mass, position, velocity)
        for mass, position, velocity in zip(
            state.masses, state.positions, state.velocities
        )
    ]

