"""Generate energy and momentum diagnostic plots."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab import Simulation, SystemState, VelocityVerlet, analyze_history
from orbitlab.visualization import plot_energy_diagnostics, plot_momentum_diagnostics


OUTPUT_DIR = Path("outputs")


def main() -> None:
    """Run a two-body simulation and save diagnostic figures."""
    state = SystemState([1.0, 1.0], [[-0.5, 0.0], [0.5, 0.0]], [[0.0, 0.7], [0.0, -0.7]])
    result = Simulation(state, VelocityVerlet(), 0.01, gravitational_constant=1.0).run(
        steps=200
    )
    diagnostics = analyze_history(result, gravitational_constant=1.0)
    OUTPUT_DIR.mkdir(exist_ok=True)
    energy_figure, _ = plot_energy_diagnostics(
        result, diagnostics, save_path=OUTPUT_DIR / "two_body_energy.png"
    )
    momentum_figure, _ = plot_momentum_diagnostics(
        result, diagnostics, save_path=OUTPUT_DIR / "two_body_momentum.png"
    )
    energy_figure.clf()
    momentum_figure.clf()


if __name__ == "__main__":
    main()
