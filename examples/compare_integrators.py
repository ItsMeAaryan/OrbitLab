"""Compare the three Phase 2 integrators on identical initial conditions."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab import (
    ExplicitEuler,
    SemiImplicitEuler,
    Simulation,
    SystemState,
    VelocityVerlet,
)
from orbitlab.visualization import (
    plot_energy_comparison,
    plot_relative_energy_drift_comparison,
    plot_trajectory_comparison,
)


OUTPUT_DIR = Path("outputs")


def main() -> None:
    """Run and plot all three integrators with the same configuration."""
    initial = SystemState([1.0, 1.0], [[-0.5, 0.0], [0.5, 0.0]], [[0.0, 0.7], [0.0, -0.7]])
    integrators = {
        "Explicit Euler": ExplicitEuler(),
        "Semi-Implicit Euler": SemiImplicitEuler(),
        "Velocity Verlet": VelocityVerlet(),
    }
    results = {
        label: Simulation(initial, integrator, 0.01, gravitational_constant=1.0).run(
            steps=200
        )
        for label, integrator in integrators.items()
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    trajectory_figure, _ = plot_trajectory_comparison(
        results, save_path=OUTPUT_DIR / "integrator_trajectories.png"
    )
    energy_figure, _ = plot_energy_comparison(
        results, gravitational_constant=1.0, save_path=OUTPUT_DIR / "integrator_energy.png"
    )
    drift_figure, _ = plot_relative_energy_drift_comparison(
        results,
        gravitational_constant=1.0,
        save_path=OUTPUT_DIR / "integrator_energy_drift.png",
    )
    trajectory_figure.clf()
    energy_figure.clf()
    drift_figure.clf()


if __name__ == "__main__":
    main()
