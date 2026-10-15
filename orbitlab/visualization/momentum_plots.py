"""Linear and angular momentum plots."""

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


def plot_momentum_diagnostics(
    result: SimulationResult,
    diagnostics: DiagnosticHistory | None = None,
    *,
    gravitational_constant: float | None = None,
    softening: float | None = None,
    figsize: tuple[float, float] = (10.0, 7.0),
    save_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot linear momentum components and angular momentum."""
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
    figure, axes = plt.subplots(2, 1, figsize=figsize, squeeze=False)
    linear_axis = axes[0, 0]
    for component_index in range(result.dimensions):
        linear_axis.plot(
            result.times,
            history.linear_momentum[:, component_index],
            label=f"P{('x', 'y', 'z')[component_index]}",
        )
    linear_axis.set_title("Linear momentum")
    linear_axis.set_xlabel("Simulation time")
    linear_axis.set_ylabel("Momentum")
    linear_axis.legend()
    linear_axis.grid(True, alpha=0.3)

    angular_axis = axes[1, 0]
    if result.dimensions == 2:
        angular_axis.plot(result.times, history.angular_momentum, label="Lz")
    else:
        for component_index, label in enumerate(("Lx", "Ly", "Lz")):
            angular_axis.plot(
                result.times,
                history.angular_momentum[:, component_index],
                label=label,
            )
    angular_axis.set_title("Angular momentum")
    angular_axis.set_xlabel("Simulation time")
    angular_axis.set_ylabel("Angular momentum")
    angular_axis.legend()
    angular_axis.grid(True, alpha=0.3)
    figure.tight_layout()
    save_figure(figure, save_path)
    return figure, axes
