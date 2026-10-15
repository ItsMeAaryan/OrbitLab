"""Generate a static two-body trajectory plot."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab import Simulation, SystemState, VelocityVerlet
from orbitlab.visualization import plot_trajectories


OUTPUT_DIR = Path("outputs")


def main() -> None:
    """Run a short two-body simulation and save its trajectory."""
    state = SystemState(
        [1.0, 1.0],
        [[-0.5, 0.0], [0.5, 0.0]],
        [[0.0, 0.7], [0.0, -0.7]],
    )
    result = Simulation(state, VelocityVerlet(), 0.01, gravitational_constant=1.0).run(
        steps=200
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    figure, _ = plot_trajectories(result, save_path=OUTPUT_DIR / "two_body_trajectories.png")
    figure.clf()


if __name__ == "__main__":
    main()
