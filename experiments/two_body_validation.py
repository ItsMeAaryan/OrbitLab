"""Validate a circular two-body orbit with all three integrators."""

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
    analyze_history,
)
from orbitlab.visualization import (  # noqa: E402
    plot_energy_diagnostics,
    plot_momentum_diagnostics,
    plot_trajectory_comparison,
)
from experiments.common import (  # noqa: E402
    CircularOrbitConfig,
    circular_two_body_reference,
    circular_two_body_state,
    final_position_error,
    estimate_orbital_period,
    measure_conservation,
    write_csv,
)


OUTPUT_DIR = Path("outputs") / "phase5"


def main() -> None:
    """Run and save the circular two-body validation measurements."""
    config = CircularOrbitConfig()
    initial = circular_two_body_state(config)
    duration = 4.0 * config.period
    dt = config.period / 400.0
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
            softening=config.softening,
        ).run(duration=duration)
        for name, integrator in integrators.items()
    }
    reference = {
        name: circular_two_body_reference(result.times, config)
        for name, result in results.items()
    }
    rows = []
    for name, result in results.items():
        measurements = measure_conservation(result)
        rows.append(
            {
                "integrator": name,
                "analytical_period": config.period,
                "measured_orbital_period": estimate_orbital_period(result),
                "simulated_duration": result.times[-1],
                "final_position_error": final_position_error(result, reference[name]),
                **measurements,
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "two_body_validation.csv", rows)
    plot_trajectory_comparison(
        results,
        save_path=OUTPUT_DIR / "two_body_orbits.png",
    )[0].clf()
    verlet_result = results["Velocity Verlet"]
    diagnostics = analyze_history(
        verlet_result,
        config.gravitational_constant,
        config.softening,
    )
    plot_energy_diagnostics(
        verlet_result,
        diagnostics,
        save_path=OUTPUT_DIR / "two_body_energy.png",
    )[0].clf()
    plot_momentum_diagnostics(
        verlet_result,
        diagnostics,
        save_path=OUTPUT_DIR / "two_body_momentum.png",
    )[0].clf()
    print(f"analytical_period={config.period:.12g}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
