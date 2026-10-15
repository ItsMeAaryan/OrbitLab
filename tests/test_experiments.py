import csv
import json

import numpy as np
import pytest

from experiments.common import (
    CircularOrbitConfig,
    circular_two_body_reference,
    circular_two_body_state,
    estimate_loglog_order,
    estimate_orbital_period,
    final_position_error,
    maximum_position_error,
    measure_three_body,
    minimum_pairwise_separation,
    write_csv,
    write_json,
)
from orbitlab import ExplicitEuler, Simulation, SystemState, VelocityVerlet


def test_circular_orbit_period_and_initial_conditions() -> None:
    config = CircularOrbitConfig()

    assert config.period == pytest.approx(2.0 * np.pi / np.sqrt(2.0))
    state = circular_two_body_state(config)

    np.testing.assert_allclose(state.positions.sum(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(state.velocities.sum(axis=0), [0.0, 0.0])
    assert np.linalg.norm(state.positions[1] - state.positions[0]) == pytest.approx(1.0)
    assert np.linalg.norm(state.velocities[1] - state.velocities[0]) == pytest.approx(
        np.sqrt(2.0)
    )


def test_circular_reference_and_error_helpers() -> None:
    config = CircularOrbitConfig()
    times = np.array([0.0, config.period / 4.0])
    reference = circular_two_body_reference(times, config)
    result = Simulation(
        circular_two_body_state(config),
        VelocityVerlet(),
        config.period / 4.0,
        gravitational_constant=config.gravitational_constant,
    ).run(steps=1)

    assert reference.shape == result.positions.shape
    assert final_position_error(result, reference) >= 0.0
    assert maximum_position_error(result, reference) >= final_position_error(result, reference)


def test_estimate_loglog_order() -> None:
    step_sizes = np.array([0.1, 0.05, 0.025])
    errors = step_sizes**2

    assert estimate_loglog_order(step_sizes, errors) == pytest.approx(2.0)


def test_estimate_orbital_period_from_reference() -> None:
    config = CircularOrbitConfig()
    result = Simulation(
        circular_two_body_state(config),
        VelocityVerlet(),
        config.period / 100.0,
        gravitational_constant=config.gravitational_constant,
    ).run(duration=config.period)

    assert estimate_orbital_period(result) == pytest.approx(config.period, rel=2e-3)


def test_experiment_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        circular_two_body_state(CircularOrbitConfig(softening=0.1))
    with pytest.raises(ValueError):
        estimate_loglog_order(np.array([0.1]), np.array([0.01]))
    with pytest.raises(ValueError):
        estimate_loglog_order(np.array([0.1, 0.05]), np.array([0.0, 0.01]))


def test_measurement_serialization_round_trip(tmp_path) -> None:
    csv_path = tmp_path / "measurements.csv"
    json_path = tmp_path / "measurements.json"
    write_csv(csv_path, [{"dt": 0.1, "error": 0.01}])
    write_json(json_path, {"order": 2.0})

    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [{"dt": "0.1", "error": "0.01"}]
    with json_path.open(encoding="utf-8") as handle:
        assert json.load(handle) == {"order": 2.0}


def test_three_body_measurements_and_reproducibility() -> None:
    state = SystemState(
        masses=[1.0, 1.0, 1.0],
        positions=[[-1.0, 0.0], [0.0, 0.8], [1.0, 0.0]],
        velocities=[[0.0, 0.3], [0.0, -0.6], [0.0, 0.3]],
    )
    first = Simulation(state, VelocityVerlet(), 0.01, gravitational_constant=1.0).run(
        steps=10
    )
    second = Simulation(state, VelocityVerlet(), 0.01, gravitational_constant=1.0).run(
        steps=10
    )

    np.testing.assert_array_equal(first.positions, second.positions)
    measurements = measure_three_body(first)
    assert measurements["minimum_pairwise_separation"] == pytest.approx(
        minimum_pairwise_separation(first.positions)
    )
    assert measurements["final_relative_energy_drift"] >= 0.0
