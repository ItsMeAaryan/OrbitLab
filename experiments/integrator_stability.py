"""Compare long-run behavior of the three existing integrators."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab import (  # noqa: E402
    ExplicitEuler,
    SemiImplicitEuler,
    Simulation,
    VelocityVerlet,
)
from orbitlab.visualization import (  # noqa: E402
    plot_energy_comparison,
    plot_relative_energy_drift_comparison,
    plot_trajectory_comparison,
)
from experiments.common import (  # noqa: E402
    CircularOrbitConfig,
    circular_two_body_state,
    measure_conservation,
    write_csv,
)


OUTPUT_DIR = Path("outputs") / "phase5"


def main() -> None:
    """Run all integrators for twenty analytical periods."""
    config = CircularOrbitConfig()
    initial = circular_two_body_state(config)
    dt = config.period / 200.0
    duration = 20.0 * config.period
    integrators = {
        "Explicit Euler": ExplicitEuler(),
        "Semi-Implicit Euler": SemiImplicitEuler(),
        "Velocity Verlet": VelocityVerlet(),
    }
    results = {
        name: Simulation(
            initial,
            integrator,
            dt,
            gravitational_constant=config.gravitational_constant,
        ).run(duration=duration)
        for name, integrator in integrators.items()
    }
    rows = [
        {
            "integrator": name,
            "dt": dt,
            "duration": duration,
            **measure_conservation(result),
        }
        for name, result in results.items()
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "integrator_stability.csv", rows)
    plot_trajectory_comparison(
        results, save_path=OUTPUT_DIR / "integrator_stability_orbits.png"
    )[0].clf()
    plot_energy_comparison(
        results,
        save_path=OUTPUT_DIR / "integrator_stability_energy.png",
    )[0].clf()
    plot_relative_energy_drift_comparison(
        results,
        save_path=OUTPUT_DIR / "integrator_stability_energy_drift.png",
    )[0].clf()
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
