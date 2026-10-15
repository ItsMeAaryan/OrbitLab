"""Quantify conservation behavior for a controlled isolated two-body system."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab import ExplicitEuler, SemiImplicitEuler, Simulation, VelocityVerlet  # noqa: E402
from experiments.common import (  # noqa: E402
    CircularOrbitConfig,
    circular_two_body_state,
    measure_conservation,
    write_json,
)


OUTPUT_DIR = Path("outputs") / "phase5"


def main() -> None:
    """Save conservation measurements and their scientific interpretation."""
    config = CircularOrbitConfig()
    initial = circular_two_body_state(config)
    dt = config.period / 300.0
    duration = 5.0 * config.period
    integrators = {
        "Explicit Euler": ExplicitEuler(),
        "Semi-Implicit Euler": SemiImplicitEuler(),
        "Velocity Verlet": VelocityVerlet(),
    }
    measurements = {}
    for name, integrator in integrators.items():
        result = Simulation(
            initial,
            integrator,
            dt,
            gravitational_constant=config.gravitational_constant,
        ).run(duration=duration)
        measurements[name] = measure_conservation(result)
    write_json(
        OUTPUT_DIR / "conservation_analysis.json",
        {
            "system": "isolated Newtonian two-body system",
            "expected_ideal_conservation": [
                "total energy",
                "linear momentum",
                "angular momentum",
            ],
            "note": "Finite timestep integration introduces numerical error.",
            "measurements": measurements,
        },
    )
    print(measurements)


if __name__ == "__main__":
    main()
