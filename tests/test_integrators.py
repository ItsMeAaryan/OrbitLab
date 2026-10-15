import numpy as np
import pytest

from orbitlab import (
    Body,
    ExplicitEuler,
    SemiImplicitEuler,
    SystemState,
    VelocityVerlet,
)
from orbitlab.core.physics import compute_accelerations


def stationary_acceleration_state() -> SystemState:
    return SystemState(
        masses=np.array([1.0]),
        positions=np.array([[0.0, 0.0]]),
        velocities=np.array([[2.0, 3.0]]),
    )


def gravitational_state() -> SystemState:
    return SystemState(
        masses=np.array([1.0, 1.0]),
        positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
        velocities=np.array([[0.0, 0.0], [0.0, 0.0]]),
    )


def test_explicit_euler_position_update() -> None:
    state = stationary_acceleration_state()

    updated = ExplicitEuler().step(state, 0.5, gravitational_constant=1.0)

    np.testing.assert_allclose(updated.positions, [[1.0, 1.5]])


def test_explicit_euler_velocity_update() -> None:
    state = gravitational_state()

    updated = ExplicitEuler().step(state, 0.5, gravitational_constant=1.0)

    np.testing.assert_allclose(updated.velocities, [[0.5, 0.0], [-0.5, 0.0]])


def test_semi_implicit_euler_position_update() -> None:
    state = gravitational_state()

    updated = SemiImplicitEuler().step(state, 0.5, gravitational_constant=1.0)

    np.testing.assert_allclose(updated.positions, [[0.25, 0.0], [0.75, 0.0]])


def test_semi_implicit_euler_velocity_update() -> None:
    state = gravitational_state()

    updated = SemiImplicitEuler().step(state, 0.5, gravitational_constant=1.0)

    np.testing.assert_allclose(updated.velocities, [[0.5, 0.0], [-0.5, 0.0]])


def test_velocity_verlet_position_update() -> None:
    state = gravitational_state()

    updated = VelocityVerlet().step(state, 0.5, gravitational_constant=1.0)

    np.testing.assert_allclose(updated.positions, [[0.125, 0.0], [0.875, 0.0]])


def test_velocity_verlet_velocity_update() -> None:
    state = gravitational_state()
    initial_acceleration = compute_accelerations(
        [
            Body(mass, position, velocity)
            for mass, position, velocity in zip(
                state.masses, state.positions, state.velocities
            )
        ],
        gravitational_constant=1.0,
    )
    moved_positions = state.positions + 0.5 * initial_acceleration * 0.5**2
    moved_state = SystemState(state.masses, moved_positions, state.velocities)
    updated_acceleration = compute_accelerations(
        [
            Body(mass, position, velocity)
            for mass, position, velocity in zip(
                moved_state.masses, moved_state.positions, moved_state.velocities
            )
        ],
        gravitational_constant=1.0,
    )
    expected = 0.5 * (initial_acceleration + updated_acceleration) * 0.5

    updated = VelocityVerlet().step(state, 0.5, gravitational_constant=1.0)

    np.testing.assert_allclose(updated.velocities, expected)


def test_velocity_verlet_uses_updated_acceleration() -> None:
    state = gravitational_state()
    updated = VelocityVerlet().step(state, 0.5, gravitational_constant=1.0)

    old_acceleration_velocity = 0.5 * compute_accelerations(
        [
            Body(mass, position, velocity)
            for mass, position, velocity in zip(
                state.masses, state.positions, state.velocities
            )
        ],
        gravitational_constant=1.0,
    ) * 0.5

    assert not np.allclose(updated.velocities, old_acceleration_velocity)


@pytest.mark.parametrize(
    ("integrator", "dt"),
    [
        (ExplicitEuler(), 0.0),
        (SemiImplicitEuler(), -1.0),
        (VelocityVerlet(), np.nan),
        (ExplicitEuler(), np.inf),
    ],
)
def test_invalid_timestep_is_rejected(integrator: object, dt: float) -> None:
    with pytest.raises(ValueError):
        integrator.step(gravitational_state(), dt, gravitational_constant=1.0)


def test_invalid_state_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        SystemState(
            masses=[1.0, np.nan],
            positions=[[0.0, 0.0], [1.0, 0.0]],
            velocities=[[0.0, 0.0], [0.0, 0.0]],
        )


def test_simulation_time_advances_and_original_state_is_unchanged() -> None:
    state = gravitational_state()
    original_positions = state.positions.copy()
    original_velocities = state.velocities.copy()

    updated = SemiImplicitEuler().step(state, 0.25, gravitational_constant=1.0)

    assert updated.time == 0.25
    assert state.time == 0.0
    np.testing.assert_array_equal(state.positions, original_positions)
    np.testing.assert_array_equal(state.velocities, original_velocities)


def test_state_copies_input_arrays() -> None:
    masses = np.array([1.0])
    positions = np.array([[1.0, 2.0]])
    velocities = np.array([[3.0, 4.0]])
    state = SystemState(masses, positions, velocities)

    masses[0] = 9.0
    positions[0, 0] = 9.0
    velocities[0, 0] = 9.0

    np.testing.assert_array_equal(state.masses, [1.0])
    np.testing.assert_array_equal(state.positions, [[1.0, 2.0]])
    np.testing.assert_array_equal(state.velocities, [[3.0, 4.0]])


@pytest.mark.parametrize(
    "state",
    [
        (
            [1.0],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ),
        (
            [1.0],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ),
    ],
)
def test_2d_and_3d_compatibility(
    state: tuple[list[float], list[list[float]], list[list[float]]],
) -> None:
    masses, positions, velocities = state
    updated = VelocityVerlet().step(
        SystemState(masses, positions, velocities),
        0.1,
        gravitational_constant=1.0,
    )

    assert updated.positions.shape == np.asarray(positions).shape


def test_single_body_behavior() -> None:
    state = SystemState([2.0], [[1.0, 2.0]], [[3.0, 4.0]])

    updated = VelocityVerlet().step(state, 0.5, gravitational_constant=1.0)

    np.testing.assert_allclose(updated.positions, [[2.5, 4.0]])
    np.testing.assert_allclose(updated.velocities, state.velocities)


def test_zero_velocity_initial_conditions() -> None:
    state = gravitational_state()

    updated = ExplicitEuler().step(state, 0.1, gravitational_constant=1.0)

    np.testing.assert_array_equal(updated.positions, state.positions)


def test_methods_produce_distinct_expected_gravitational_updates() -> None:
    state = gravitational_state()

    explicit = ExplicitEuler().step(state, 0.5, gravitational_constant=1.0)
    semi_implicit = SemiImplicitEuler().step(state, 0.5, gravitational_constant=1.0)
    verlet = VelocityVerlet().step(state, 0.5, gravitational_constant=1.0)

    assert explicit.positions[0, 0] == 0.0
    assert semi_implicit.positions[0, 0] == 0.25
    np.testing.assert_allclose(verlet.positions[0, 0], 0.125)
    assert not np.array_equal(explicit.positions, semi_implicit.positions)
