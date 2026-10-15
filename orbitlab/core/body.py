"""Data model for celestial bodies in an OrbitLab simulation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Body:
    """A celestial body with mass, position, and velocity.

    Positions and velocities must be two- or three-dimensional vectors.
    Input arrays are converted to independent floating-point copies so later
    mutation of the caller's arrays cannot change the body state.
    """

    mass: float
    position: FloatArray
    velocity: FloatArray

    def __post_init__(self) -> None:
        """Validate physical values and normalize vector storage."""
        try:
            mass = float(self.mass)
        except (TypeError, ValueError) as error:
            raise ValueError("mass must be a finite, strictly positive number") from error

        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("mass must be a finite, strictly positive number")

        position = self._copy_vector(self.position, "position")
        velocity = self._copy_vector(self.velocity, "velocity")

        if position.shape != velocity.shape:
            raise ValueError("position and velocity must have matching dimensions")

        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)

    @staticmethod
    def _copy_vector(vector: NDArray[np.generic], name: str) -> FloatArray:
        """Validate and copy one position or velocity vector."""
        try:
            copied = np.array(vector, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite NumPy vector") from error

        if copied.ndim != 1 or copied.size not in (2, 3):
            raise ValueError(f"{name} must be a one-dimensional 2D or 3D vector")
        if not np.all(np.isfinite(copied)):
            raise ValueError(f"{name} must contain only finite values")
        return copied

    @property
    def dimensions(self) -> int:
        """Return the spatial dimensionality of the body."""
        return self.position.size