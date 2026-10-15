"""Run a deterministic three-body gravitational experiment."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab import Simulation, SystemState, VelocityVerlet, analyze_history  # noqa: E402
from orbitlab.visualization import (  # noqa: E402
    plot_energy_diagnostics,
    plot_momentum_diagnostics,
    plot_trajectories,
)
from experiments.common import measure_three_body, write_json  # noqa: E402


OUTPUT_DIR = Path("outputs") / "phase5"


def three_body_initial_state() -> SystemState:
    """Return the deterministic equal-mass figure-eight initial state."""
    return SystemState(
        masses=[1.0, 1.0, 1.0],
        positions=[
            [-0.97000436, 0.24308753],
            [0.0, -0.48617507],
            [0.97000436, 0.24308753],
        ],
        velocities=[
            [0.466203685, 0.43236573],
            [-0.93240737, -0.86473146],
            [0.466203685, 0.43236573],
        ],
    )


def main() -> None:
    """Run and save a normalized, collision-free three-body experiment."""
    gravitational_constant = 1.0
    initial = three_body_initial_state()
    result = Simulation(
        initial,
        VelocityVerlet(),
        0.0025,
        gravitational_constant=gravitational_constant,
    ).run(steps=2528)
    diagnostics = analyze_history(result, gravitational_constant)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_trajectories(
        result,
        save_path=OUTPUT_DIR / "three_body_trajectories.png",
        title="Three-body trajectories",
    )[0].clf()
    plot_energy_diagnostics(
        result,
        diagnostics,
        save_path=OUTPUT_DIR / "three_body_energy.png",
    )[0].clf()
    plot_momentum_diagnostics(
        result,
        diagnostics,
        save_path=OUTPUT_DIR / "three_body_momentum.png",
    )[0].clf()
    write_json(
        OUTPUT_DIR / "three_body_summary.json",
        {
            "masses": initial.masses.tolist(),
            "initial_positions": initial.positions.tolist(),
            "initial_velocities": initial.velocities.tolist(),
            "gravitational_constant": gravitational_constant,
            "dt": 0.0025,
            "steps": 2528,
            "final_time": float(result.times[-1]),
            "conservation": measure_three_body(result),
        },
    )
    print(measure_three_body(result))


if __name__ == "__main__":
    main()
