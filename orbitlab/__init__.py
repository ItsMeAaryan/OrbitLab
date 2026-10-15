"""OrbitLab: a Newtonian N-body gravitational simulation foundation."""

from orbitlab.core import (
    Body,
    DEFAULT_GRAVITATIONAL_CONSTANT,
    compute_accelerations,
    compute_forces,
)
from orbitlab.integrators import (
    ExplicitEuler,
    SemiImplicitEuler,
    SystemState,
    VelocityVerlet,
)
from orbitlab.simulation import Simulation, SimulationResult
from orbitlab.diagnostics import (
    DiagnosticHistory,
    analyze_history,
    angular_momentum,
    gravitational_potential_energy,
    kinetic_energy,
    linear_momentum,
    total_energy,
)
from orbitlab.visualization import (
    create_animation,
    plot_energy_comparison,
    plot_energy_diagnostics,
    plot_momentum_diagnostics,
    plot_relative_energy_drift_comparison,
    plot_trajectories,
    plot_trajectory_comparison,
    save_animation,
)

__all__ = [
    "Body",
    "DEFAULT_GRAVITATIONAL_CONSTANT",
    "DiagnosticHistory",
    "ExplicitEuler",
    "Simulation",
    "SimulationResult",
    "SemiImplicitEuler",
    "SystemState",
    "VelocityVerlet",
    "analyze_history",
    "angular_momentum",
    "compute_accelerations",
    "compute_forces",
    "create_animation",
    "gravitational_potential_energy",
    "kinetic_energy",
    "linear_momentum",
    "plot_energy_comparison",
    "plot_energy_diagnostics",
    "plot_momentum_diagnostics",
    "plot_relative_energy_drift_comparison",
    "plot_trajectories",
    "plot_trajectory_comparison",
    "save_animation",
    "total_energy",
]