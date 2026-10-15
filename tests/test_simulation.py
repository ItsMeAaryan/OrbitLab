import numpy as np
import pytest

from orbitlab import (
    Body,
    ExplicitEuler,
    Simulation,
    SimulationResult,
    SystemState,
    VelocityVerlet,
    analyze_history,
    angular_momentum,
    gravitational_potential_energy,
    kinetic_energy,
    linear_momentum,
    total_energy,
)
from orbitlab.integrators.semi_implicit_euler import SemiImplicitEuler


def two_body_state() -> SystemState:
    return SystemState(
        masses=[1.0, 2.0],
        positions=[[0.0, 0.0], [1.0, 0.0]],
        velocities=[[0.0, 0.5], [0.0, -0.25]],
    )


@pytest.mark.parametrize("integrator", [ExplicitEuler(), SemiImplicitEuler(), VelocityVerlet()])
def test_simulation_records_initial_state_and_all_steps(integrator: object) -> None:
    initial = two_body_state()
    result = Simulation(initial, integrator, 0.1, gravitational_constant=1.0).run(
        steps=3
    )

    assert isinstance(result, SimulationResult)
    assert result.number_of_steps == 3
    np.testing.assert_allclose(result.times, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(result.positions[0], initial.positions)
    np.testing.assert_allclose(result.velocities[0], initial.velocities)


def test_duration_and_final_time() -> None:
    result = Simulation(
        two_body_state(), ExplicitEuler(), 0.25, gravitational_constant=1.0
    ).run(duration=0.75)

    assert result.number_of_steps == 3
    assert result.times[-1] == pytest.approx(0.75)


def test_initial_state_is_preserved() -> None:
    initial = two_body_state()
    positions = initial.positions.copy()
    velocities = initial.velocities.copy()

    Simulation(initial, SemiImplicitEuler(), 0.1, gravitational_constant=1.0).run(
        steps=4
    )

    np.testing.assert_array_equal(initial.positions, positions)
    np.testing.assert_array_equal(initial.velocities, velocities)
    assert initial.time == 0.0


def test_simulation_is_deterministic() -> None:
    configuration = dict(steps=5)
    first = Simulation(
        two_body_state(), VelocityVerlet(), 0.05, gravitational_constant=1.0
    ).run(**configuration)
    second = Simulation(
        two_body_state(), VelocityVerlet(), 0.05, gravitational_constant=1.0
    ).run(**configuration)

    np.testing.assert_array_equal(first.times, second.times)
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.velocities, second.velocities)


def test_kinetic_potential_and_total_energy() -> None:
    state = SystemState([2.0, 3.0], [[0.0, 0.0], [4.0, 0.0]], [[3.0, 0.0], [0.0, 2.0]])

    assert kinetic_energy(state) == pytest.approx(15.0)
    assert gravitational_potential_energy(state, gravitational_constant=1.0) == pytest.approx(
        -1.5
    )
    assert total_energy(state, gravitational_constant=1.0) == pytest.approx(13.5)


def test_momentum_and_angular_momentum_in_2d() -> None:
    state = SystemState([2.0, 3.0], [[1.0, 0.0], [0.0, 2.0]], [[0.0, 3.0], [-2.0, 0.0]])

    np.testing.assert_allclose(linear_momentum(state), [ -6.0, 6.0])
    assert angular_momentum(state) == pytest.approx(18.0)


def test_angular_momentum_in_3d() -> None:
    state = SystemState([2.0], [[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]])

    np.testing.assert_allclose(angular_momentum(state), [-6.0, 12.0, -6.0])


def test_history_diagnostics_and_relative_energy_drift() -> None:
    result = Simulation(
        two_body_state(), VelocityVerlet(), 0.01, gravitational_constant=1.0
    ).run(steps=4)

    diagnostics = analyze_history(result, gravitational_constant=1.0)

    assert diagnostics.total_energy.shape == (5,)
    assert diagnostics.linear_momentum.shape == (5, 2)
    assert diagnostics.angular_momentum.shape == (5,)
    assert diagnostics.relative_energy_drift[0] == pytest.approx(0.0)
    assert np.all(diagnostics.relative_energy_drift >= 0.0)


def test_zero_initial_energy_drift_is_rejected() -> None:
    result = SimulationResult(
        masses=[1.0],
        times=[0.0],
        positions=[[[0.0, 0.0]]],
        velocities=[[[0.0, 0.0]]],
    )

    with pytest.raises(ValueError, match="relative energy drift"):
        analyze_history(result, gravitational_constant=1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": -1}, "steps"),
        ({"steps": 1.5}, "steps"),
        ({"steps": 1, "duration": 0.1}, "exactly one"),
        ({}, "exactly one"),
        ({"duration": 0.15}, "integral"),
    ],
)
def test_invalid_run_configuration(kwargs: dict[str, object], match: str) -> None:
    simulation = Simulation(two_body_state(), ExplicitEuler(), 0.1, gravitational_constant=1.0)

    with pytest.raises(ValueError, match=match):
        simulation.run(**kwargs)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_invalid_simulation_timestep(dt: float) -> None:
    with pytest.raises(ValueError):
        Simulation(two_body_state(), ExplicitEuler(), dt)


def test_invalid_simulation_configuration() -> None:
    with pytest.raises(ValueError):
        Simulation(two_body_state(), object(), 0.1)
    with pytest.raises(ValueError):
        Simulation(two_body_state(), ExplicitEuler(), 0.1, gravitational_constant=0.0)
    with pytest.raises(ValueError):
        Simulation(two_body_state(), ExplicitEuler(), 0.1, softening=-1.0)


def test_momentum_conservation_behavior() -> None:
    initial = SystemState(
        [1.0, 1.0],
        [[-0.5, 0.0], [0.5, 0.0]],
        [[0.0, 1.0], [0.0, -1.0]],
    )
    result = Simulation(
        initial, VelocityVerlet(), 0.005, gravitational_constant=1.0
    ).run(steps=20)
    momentum = np.array([linear_momentum(result.state_at(i)) for i in range(len(result.times))])

    np.testing.assert_allclose(momentum, np.zeros_like(momentum), atol=1e-12)


def test_velocity_verlet_energy_behavior_improves_with_smaller_timestep() -> None:
    initial = SystemState(
        [1.0, 1.0],
        [[-0.5, 0.0], [0.5, 0.0]],
        [[0.0, 0.7], [0.0, -0.7]],
    )
    coarse = Simulation(
        initial, VelocityVerlet(), 0.02, gravitational_constant=1.0
    ).run(steps=50)
    fine = Simulation(
        initial, VelocityVerlet(), 0.01, gravitational_constant=1.0
    ).run(steps=100)
    coarse_drift = analyze_history(coarse, gravitational_constant=1.0).relative_energy_drift[-1]
    fine_drift = analyze_history(fine, gravitational_constant=1.0).relative_energy_drift[-1]

    assert fine_drift < coarse_drift
