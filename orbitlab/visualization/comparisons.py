"""Plots comparing simulation histories from different integrators."""

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from orbitlab.core.physics import DEFAULT_GRAVITATIONAL_CONSTANT
from orbitlab.diagnostics.analysis import analyze_history
from orbitlab.simulation.result import SimulationResult
from orbitlab.visualization._common import (
    save_figure,
    stable_limits,
    validate_2d_result,
    validate_comparison_results,
)


def plot_trajectory_comparison(
    results: Mapping[str, SimulationResult],
    *,
    figsize: tuple[float, float] = (9.0, 8.0),
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare 2D trajectories, using one line style per integrator."""
    validate_comparison_results(results)
    for result in results.values():
        validate_2d_result(result)
    all_positions = np.concatenate(
        [result.positions for result in results.values()], axis=1
    )
    figure, axes = plt.subplots(figsize=figsize)
    colors = plt.get_cmap("tab10")
    for index, (label, result) in enumerate(results.items()):
        for body_index in range(result.masses.size):
            trajectory = result.positions[:, body_index]
            axes.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors(body_index % 10),
                linestyle=("-", "--", ":")[index % 3],
                label=f"{label} — Body {body_index + 1}",
            )
    limits = stable_limits(all_positions)
    axes.set_xlim(limits[0:2])
    axes.set_ylim(limits[2:4])
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x position")
    axes.set_ylabel("y position")
    axes.set_title("Integrator trajectory comparison")
    axes.legend()
    axes.grid(True, alpha=0.3)
    save_figure(figure, save_path)
    return figure, axes


def plot_energy_comparison(
    results: Mapping[str, SimulationResult],
    *,
    gravitational_constant: float | None = None,
    softening: float | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare total energy histories for compatible simulations."""
    validate_comparison_results(results)
    reference = next(iter(results.values()))
    effective_gravitational_constant = (
        reference.gravitational_constant
        if gravitational_constant is None
        else gravitational_constant
    )
    effective_softening = reference.softening if softening is None else softening
    figure, axes = plt.subplots(figsize=figsize)
    for label, result in results.items():
        history = analyze_history(
            result, effective_gravitational_constant, effective_softening
        )
        axes.plot(result.times, history.total_energy, label=label)
    axes.set_title("Integrator total-energy comparison")
    axes.set_xlabel("Simulation time")
    axes.set_ylabel("Total energy")
    axes.legend()
    axes.grid(True, alpha=0.3)
    save_figure(figure, save_path)
    return figure, axes


def plot_relative_energy_drift_comparison(
    results: Mapping[str, SimulationResult],
    *,
    gravitational_constant: float | None = None,
    softening: float | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare relative total-energy drift for compatible simulations."""
    validate_comparison_results(results)
    reference = next(iter(results.values()))
    effective_gravitational_constant = (
        reference.gravitational_constant
        if gravitational_constant is None
        else gravitational_constant
    )
    effective_softening = reference.softening if softening is None else softening
    figure, axes = plt.subplots(figsize=figsize)
    for label, result in results.items():
        history = analyze_history(
            result, effective_gravitational_constant, effective_softening
        )
        axes.plot(result.times, history.relative_energy_drift, label=label)
    axes.set_title("Integrator relative-energy drift comparison")
    axes.set_xlabel("Simulation time")
    axes.set_ylabel("Relative energy drift")
    axes.legend()
    axes.grid(True, alpha=0.3)
    save_figure(figure, save_path)
    return figure, axes
