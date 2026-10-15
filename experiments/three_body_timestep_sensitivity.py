"""Measure timestep sensitivity for the deterministic three-body experiment."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab import Simulation, VelocityVerlet  # noqa: E402
from experiments.common import measure_three_body, write_csv  # noqa: E402
from experiments.three_body import three_body_initial_state  # noqa: E402


OUTPUT_DIR = Path("outputs") / "phase5"
TIMESTEPS = (0.01, 0.005, 0.0025, 0.00125)
FINAL_TIME = 6.32


def main() -> None:
    """Run every timestep at the same physical final time."""
    rows = []
    for timestep in TIMESTEPS:
        result = Simulation(
            three_body_initial_state(),
            VelocityVerlet(),
            timestep,
            gravitational_constant=1.0,
        ).run(steps=round(FINAL_TIME / timestep))
        rows.append({"dt": timestep, "final_time": result.times[-1], **measure_three_body(result)})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "three_body_timestep_sensitivity.csv", rows)
    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    axes.loglog(
        [row["dt"] for row in rows],
        [row["max_relative_energy_drift"] for row in rows],
        "o-",
    )
    axes.set_xlabel("Timestep")
    axes.set_ylabel("Maximum relative energy drift")
    axes.set_title("Three-body timestep sensitivity")
    axes.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "three_body_timestep_sensitivity.png", bbox_inches="tight")
    figure.clf()
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
