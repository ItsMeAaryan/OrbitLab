"""Shared analytical references and measurement helpers for Phase 5."""

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from orbitlab import SimulationResult, SystemState, analyze_history


@dataclass(frozen=True)
class CircularOrbitConfig:
    """Normalized two-body circular-orbit configuration."""

    gravitational_constant: float = 1.0
    mass_one: float = 1.0
    mass_two: float = 1.0
    separation: float = 1.0
    softening: float = 0.0

    @property
    def total_mass(self) -> float:
        """Return the total system mass."""
        return self.mass_one + self.mass_two

    @property
    def angular_frequency(self) -> float:
        """Return the analytical relative-orbit angular frequency."""
        return np.sqrt(
            self.gravitational_constant
            * self.total_mass
            / self.separation**3
        )

    @property
    def period(self) -> float:
        """Return the analytical relative-orbit period."""
        return 2.0 * np.pi / self.angular_frequency

    @property
    def relative_speed(self) -> float:
        """Return the relative tangential speed for a circular orbit."""
        return self.angular_frequency * self.separation


def circular_two_body_state(config: CircularOrbitConfig) -> SystemState:
    """Create center-of-mass circular-orbit initial conditions.

    The bodies begin on the x-axis with velocities perpendicular to it.
    Softening is intentionally required to be zero for this analytical
    Newtonian circular reference.
    """
    if config.softening != 0.0:
        raise ValueError("analytical circular reference requires zero softening")
    radius_one = config.mass_two / config.total_mass * config.separation
    radius_two = config.mass_one / config.total_mass * config.separation
    speed_one = config.mass_two / config.total_mass * config.relative_speed
    speed_two = config.mass_one / config.total_mass * config.relative_speed
    return SystemState(
        masses=[config.mass_one, config.mass_two],
        positions=[[-radius_one, 0.0], [radius_two, 0.0]],
        velocities=[[0.0, -speed_one], [0.0, speed_two]],
    )


def circular_two_body_reference(
    times: np.ndarray,
    config: CircularOrbitConfig,
) -> np.ndarray:
    """Return exact continuous circular positions at the requested times."""
    times = np.asarray(times, dtype=np.float64)
    phase = config.angular_frequency * times
    relative = np.column_stack((np.cos(phase), np.sin(phase)))
    scale_one = -config.mass_two / config.total_mass * config.separation
    scale_two = config.mass_one / config.total_mass * config.separation
    return np.stack((scale_one * relative, scale_two * relative), axis=1)


def maximum_position_error(
    result: SimulationResult,
    reference_positions: np.ndarray,
) -> float:
    """Return the maximum Euclidean position error over all records/bodies."""
    reference = np.asarray(reference_positions, dtype=np.float64)
    if reference.shape != result.positions.shape:
        raise ValueError("reference positions must match the result shape")
    return float(np.max(np.linalg.norm(result.positions - reference, axis=2)))


def final_position_error(
    result: SimulationResult,
    reference_positions: np.ndarray,
) -> float:
    """Return the final-record maximum body position error."""
    reference = np.asarray(reference_positions, dtype=np.float64)
    if reference.shape != result.positions.shape:
        raise ValueError("reference positions must match the result shape")
    return float(np.max(np.linalg.norm(result.positions[-1] - reference[-1], axis=1)))


def estimate_orbital_period(result: SimulationResult) -> float:
    """Estimate the relative orbital period from unwrapped position angle."""
    if result.positions.shape[1:] != (2, 2):
        raise ValueError("orbital period estimation requires a 2D two-body result")
    relative = result.positions[:, 1] - result.positions[:, 0]
    phase_change = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))[-1]
    phase_change -= np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))[0]
    if phase_change <= 0.0:
        raise ValueError("result does not contain a forward orbital revolution")
    return float(2.0 * np.pi * (result.times[-1] - result.times[0]) / phase_change)


def measure_conservation(
    result: SimulationResult,
) -> dict[str, float]:
    """Measure maximum relative energy, linear momentum, and angular drift."""
    if result.gravitational_constant is None or result.softening is None:
        raise ValueError("result physical metadata is required for diagnostics")
    history = analyze_history(
        result,
        result.gravitational_constant,
        result.softening,
    )
    linear_delta = history.linear_momentum - history.linear_momentum[0]
    angular_delta = history.angular_momentum - history.angular_momentum[0]
    return {
        "max_relative_energy_drift": float(np.max(history.relative_energy_drift)),
        "max_linear_momentum_drift": float(np.max(np.linalg.norm(linear_delta, axis=1))),
        "max_angular_momentum_drift": float(np.max(np.abs(angular_delta))),
    }


def minimum_pairwise_separation(positions: np.ndarray) -> float:
    """Return the minimum separation over all recorded body pairs."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1] < 2:
        raise ValueError("positions must have shape (records, bodies, dimensions)")
    separations = [
        np.linalg.norm(positions[:, j] - positions[:, i], axis=1)
        for i in range(positions.shape[1])
        for j in range(i + 1, positions.shape[1])
    ]
    return float(np.min(np.concatenate(separations)))


def measure_three_body(result: SimulationResult) -> dict[str, float]:
    """Measure conservation and closest approach for a three-body result."""
    if result.positions.shape[1] != 3:
        raise ValueError("three-body measurements require exactly three bodies")
    return {
        **measure_conservation(result),
        "final_relative_energy_drift": float(
            analyze_history(
                result,
                result.gravitational_constant,
                result.softening,
            ).relative_energy_drift[-1]
        ),
        "minimum_pairwise_separation": minimum_pairwise_separation(result.positions),
    }


def estimate_loglog_order(step_sizes: np.ndarray, errors: np.ndarray) -> float:
    """Estimate a convergence order from a log-log least-squares fit."""
    step_sizes = np.asarray(step_sizes, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if step_sizes.shape != errors.shape or step_sizes.size < 2:
        raise ValueError("step sizes and errors need at least two matching values")
    if np.any(step_sizes <= 0.0) or np.any(errors <= 0.0):
        raise ValueError("step sizes and errors must be strictly positive")
    slope, _ = np.polyfit(np.log(step_sizes), np.log(errors), 1)
    return float(slope)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a deterministic list of measurement dictionaries to CSV."""
    if not rows:
        raise ValueError("cannot write an empty measurement table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    """Write JSON experiment data with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
