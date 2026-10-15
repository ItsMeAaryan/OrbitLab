"""Shared validation and Matplotlib helpers for visualization modules."""

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from orbitlab.simulation.result import SimulationResult


def validate_result(result: SimulationResult) -> None:
    """Reject values that are not simulation results."""
    if not isinstance(result, SimulationResult):
        raise ValueError("result must be a SimulationResult")


def validate_2d_result(result: SimulationResult) -> None:
    """Require a two-dimensional simulation history."""
    validate_result(result)
    if result.dimensions != 2:
        raise ValueError("this visualization requires a 2D SimulationResult")


def save_figure(figure: Figure, save_path: str | Path | None) -> None:
    """Save a figure when a path was explicitly supplied."""
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")


def stable_limits(
    positions: np.ndarray,
    *,
    padding_fraction: float = 0.05,
) -> tuple[float, float, float, float]:
    """Return fixed, padded x/y limits for all recorded 2D positions."""
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions must have shape (records, bodies, 2)")
    minimum = np.min(positions, axis=(0, 1))
    maximum = np.max(positions, axis=(0, 1))
    span = maximum - minimum
    scale = max(float(np.max(span)), 1.0)
    padding = max(scale * padding_fraction, np.finfo(float).eps)
    return (
        float(minimum[0] - padding),
        float(maximum[0] + padding),
        float(minimum[1] - padding),
        float(maximum[1] + padding),
    )


def validate_comparison_results(
    results: Mapping[str, SimulationResult],
) -> None:
    """Validate labels, result dimensions, and identical time grids."""
    if not results:
        raise ValueError("results must contain at least one labeled simulation")
    if any(not label for label in results):
        raise ValueError("comparison labels must be non-empty")
    values = list(results.values())
    for result in values:
        validate_result(result)
    reference = values[0]
    if (
        reference.gravitational_constant is None
        or reference.softening is None
    ):
        raise ValueError(
            "comparison requires results with physical metadata: "
            "gravitational_constant and softening"
        )
    for result in values[1:]:
        if result.dimensions != reference.dimensions:
            raise ValueError("compared simulations must have matching dimensions")
        if result.masses.shape != reference.masses.shape:
            raise ValueError("compared simulations must have matching body counts")
        if not np.allclose(result.masses, reference.masses, rtol=1e-12, atol=1e-15):
            raise ValueError("compared simulations must have matching masses")
        if not np.allclose(
            result.positions[0],
            reference.positions[0],
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("compared simulations must have matching initial positions")
        if not np.allclose(
            result.velocities[0],
            reference.velocities[0],
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("compared simulations must have matching initial velocities")
        if not np.allclose(result.times, reference.times, rtol=1e-12, atol=1e-15):
            raise ValueError("compared simulations must use identical time grids")
        if result.gravitational_constant is None or result.softening is None:
            raise ValueError(
                "comparison requires physical metadata on every result"
            )
        if not np.isclose(
            result.gravitational_constant,
            reference.gravitational_constant,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("compared simulations must have matching gravitational constants")
        if not np.isclose(
            result.softening,
            reference.softening,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("compared simulations must have matching softening")


def new_figure_axes(
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    """Create a consistently sized Matplotlib figure and axes."""
    return plt.subplots(figsize=figsize)
