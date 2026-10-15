"""Matplotlib visualizations for OrbitLab simulation results."""

from orbitlab.visualization.animation import create_animation, save_animation
from orbitlab.visualization.comparisons import (
    plot_energy_comparison,
    plot_relative_energy_drift_comparison,
    plot_trajectory_comparison,
)
from orbitlab.visualization.energy_plots import plot_energy_diagnostics
from orbitlab.visualization.momentum_plots import plot_momentum_diagnostics
from orbitlab.visualization.trajectories import plot_trajectories

__all__ = [
    "create_animation",
    "plot_energy_comparison",
    "plot_energy_diagnostics",
    "plot_momentum_diagnostics",
    "plot_relative_energy_drift_comparison",
    "plot_trajectories",
    "plot_trajectory_comparison",
    "save_animation",
]
