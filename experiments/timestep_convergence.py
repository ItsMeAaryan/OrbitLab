"""Measure Velocity Verlet timestep convergence against a circular reference."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab import Simulation, VelocityVerlet  # noqa: E402
from experiments.common import (  # noqa: E402
    CircularOrbitConfig,
    circular_two_body_reference,
    circular_two_body_state,
    estimate_loglog_order,
    final_position_error,
    measure_conservation,
    write_csv,
    write_json,
)


OUTPUT_DIR = Path("outputs") / "phase5"


def main() -> None:
    """Run decreasing-timestep simulations at one common final time."""
    config = CircularOrbitConfig()
    duration = 2.0 * config.period
    step_sizes = np.array(
        [config.period / 100.0, config.period / 200.0, config.period / 400.0]
    )
    rows = []
    errors = []
    for dt in step_sizes:
        result = Simulation(
            circular_two_body_state(config),
            VelocityVerlet(),
            float(dt),
            gravitational_constant=config.gravitational_constant,
        ).run(duration=duration)
        reference = circular_two_body_reference(result.times, config)
        error = final_position_error(result, reference)
        errors.append(error)
        rows.append(
            {
                "dt": float(dt),
                "final_time": float(result.times[-1]),
                "final_position_error": error,
                **measure_conservation(result),
            }
        )
    observed_order = estimate_loglog_order(step_sizes, np.array(errors))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "timestep_convergence.csv", rows)
    write_json(
        OUTPUT_DIR / "timestep_convergence_summary.json",
        {"observed_order": observed_order, "rows": rows},
    )
    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    axes.loglog(step_sizes, errors, "o-", label=f"observed order = {observed_order:.4f}")
    axes.set_xlabel("Timestep")
    axes.set_ylabel("Final position error")
    axes.set_title("Velocity Verlet timestep convergence")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "timestep_convergence.png", bbox_inches="tight")
    figure.clf()
    print(f"observed_order={observed_order:.12g}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
