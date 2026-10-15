"""Structured results produced by the OrbitLab simulation engine."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orbitlab.integrators.base import SystemState


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationResult:
    """Recorded states and physical metadata for a completed simulation."""

    masses: FloatArray
    times: FloatArray
    positions: FloatArray
    velocities: FloatArray
    gravitational_constant: float | None = None
    softening: float | None = None
    integrator_name: str | None = None

    def __post_init__(self) -> None:
        """Validate and copy the recorded arrays."""
        try:
            masses = np.array(self.masses, dtype=np.float64, copy=True)
            times = np.array(self.times, dtype=np.float64, copy=True)
            positions = np.array(self.positions, dtype=np.float64, copy=True)
            velocities = np.array(self.velocities, dtype=np.float64, copy=True)
            gravitational_constant = (
                None
                if self.gravitational_constant is None
                else float(self.gravitational_constant)
            )
            softening = None if self.softening is None else float(self.softening)
        except (TypeError, ValueError) as error:
            raise ValueError("simulation result values must be numeric") from error

        if masses.ndim != 1 or masses.size == 0:
            raise ValueError("masses must be a non-empty one-dimensional array")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a non-empty one-dimensional array")
        if positions.ndim != 3 or positions.shape[2] not in (2, 3):
            raise ValueError("positions must have shape (records, bodies, 2 or 3)")
        if velocities.shape != positions.shape:
            raise ValueError("positions and velocities must have matching shapes")
        if positions.shape[0] != times.size:
            raise ValueError("times and state histories must have matching lengths")
        if positions.shape[1] != masses.size:
            raise ValueError("masses and state histories must have matching body counts")
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("masses must be finite and strictly positive")
        if not np.all(np.isfinite(times)):
            raise ValueError("times must contain only finite values")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain only finite values")
        if not np.all(np.isfinite(velocities)):
            raise ValueError("velocities must contain only finite values")
        if gravitational_constant is not None and (
            not np.isfinite(gravitational_constant) or gravitational_constant <= 0.0
        ):
            raise ValueError(
                "gravitational_constant must be finite and strictly positive"
            )
        if softening is not None and (
            not np.isfinite(softening) or softening < 0.0
        ):
            raise ValueError("softening must be finite and non-negative")
        if self.integrator_name is not None and not self.integrator_name:
            raise ValueError("integrator_name must be non-empty when provided")

        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "gravitational_constant", gravitational_constant)
        object.__setattr__(self, "softening", softening)
        object.__setattr__(self, "integrator_name", self.integrator_name)

    @property
    def number_of_steps(self) -> int:
        """Return the number of integration steps represented."""
        return self.times.size - 1

    @property
    def dimensions(self) -> int:
        """Return the spatial dimensionality of the recorded states."""
        return self.positions.shape[2]

    def state_at(self, index: int) -> SystemState:
        """Return an independent ``SystemState`` for one history record."""
        return SystemState(
            self.masses,
            self.positions[index],
            self.velocities[index],
            self.times[index],
        )
