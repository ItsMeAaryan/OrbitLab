"""Energy diagnostic plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from orbitlab.diagnostics.analysis import (
    DiagnosticHistory,
    analyze_history,
    validate_diagnostic_history,
)
from orbitlab.core.physics import DEFAULT_GRAVITATIONAL_CONSTANT
from orbitlab.simulation.result import SimulationResult
from orbitlab.visualization._common import save_figure, validate_result


def plot_energy_diagnostics(
    result: SimulationResult,
    diagnostics: DiagnosticHistory | None = None,
    *,
    gravitational_constant: float | None = None,
    softening: float | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    save_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot kinetic, potential, total energy, and relative energy drift."""
    validate_result(result)
    effective_gravitational_constant = (
        result.gravitational_constant
        if gravitational_constant is None and result.gravitational_constant is not None
        else (
            DEFAULT_GRAVITATIONAL_CONSTANT
            if gravitational_constant is None
            else gravitational_constant
        )
    )
    effective_softening = (
        result.softening
        if softening is None and result.softening is not None
        else (0.0 if softening is None else softening)
    )
    history = diagnostics or analyze_history(
        result, effective_gravitational_constant, effective_softening
    )
    if diagnostics is not None:
        validate_diagnostic_history(
            result,
            diagnostics,
            effective_gravitational_constant,
            effective_softening,
        )
    figure, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
    plots = (
        (history.kinetic_energy, "Kinetic energy", "Energy"),
        (history.potential_energy, "Potential energy", "Energy"),
        (history.total_energy, "Total energy", "Energy"),
        (history.relative_energy_drift, "Relative total-energy drift", "Relative drift"),
    )
    for axis, (values, title, ylabel) in zip(axes.flat, plots):
        axis.plot(result.times, values)
        axis.set_title(title)
        axis.set_xlabel("Simulation time")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    save_figure(figure, save_path)
    return figure, axes
