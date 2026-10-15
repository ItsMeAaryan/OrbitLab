import numpy as np
import pytest

from orbitlab.core.body import Body
from orbitlab.core.physics import compute_accelerations, compute_forces


def test_analytical_two_body_force_and_direction() -> None:
    first = Body(2.0, [0.0, 0.0], [0.0, 0.0])
    second = Body(3.0, [4.0, 0.0], [0.0, 0.0])

    forces = compute_forces([first, second], gravitational_constant=1.0)

    np.testing.assert_allclose(forces, [[3.0 / 8.0, 0.0], [-3.0 / 8.0, 0.0]])


def test_newtons_third_law_and_zero_total_internal_force() -> None:
    bodies = [
        Body(2.0, [0.0, 0.0], [0.0, 0.0]),
        Body(3.0, [4.0, 0.0], [0.0, 0.0]),
        Body(5.0, [0.0, 3.0], [0.0, 0.0]),
    ]

    forces = compute_forces(bodies, gravitational_constant=1.0)

    np.testing.assert_allclose(np.sum(forces, axis=0), [0.0, 0.0], atol=1e-15)


def test_acceleration_uses_force_divided_by_mass() -> None:
    bodies = [
        Body(2.0, [0.0, 0.0], [0.0, 0.0]),
        Body(3.0, [4.0, 0.0], [0.0, 0.0]),
    ]

    accelerations = compute_accelerations(bodies, gravitational_constant=1.0)

    np.testing.assert_allclose(accelerations, [[3.0 / 16.0, 0.0], [-1.0 / 8.0, 0.0]])


def test_single_body_has_zero_force_and_acceleration() -> None:
    body = Body(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    np.testing.assert_array_equal(
        compute_forces([body], gravitational_constant=1.0),
        np.zeros((1, 3)),
    )
    np.testing.assert_array_equal(
        compute_accelerations([body], gravitational_constant=1.0),
        np.zeros((1, 3)),
    )


def test_force_supports_3d_bodies() -> None:
    bodies = [
        Body(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        Body(1.0, [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]),
    ]

    forces = compute_forces(bodies, gravitational_constant=1.0)

    np.testing.assert_allclose(forces, [[0.0, 0.0, 0.25], [0.0, 0.0, -0.25]])


def test_empty_collection_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_forces([], gravitational_constant=1.0)


def test_mixed_dimensions_are_rejected() -> None:
    bodies = [
        Body(1.0, [0.0, 0.0], [0.0, 0.0]),
        Body(1.0, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
    ]

    with pytest.raises(ValueError):
        compute_forces(bodies, gravitational_constant=1.0)


def test_coincident_bodies_are_rejected_without_softening() -> None:
    bodies = [
        Body(1.0, [0.0, 0.0], [0.0, 0.0]),
        Body(1.0, [0.0, 0.0], [0.0, 0.0]),
    ]

    with pytest.raises(ValueError):
        compute_forces(bodies, gravitational_constant=1.0)


def test_softening_handles_coincident_bodies() -> None:
    bodies = [
        Body(2.0, [0.0, 0.0], [0.0, 0.0]),
        Body(3.0, [0.0, 0.0], [0.0, 0.0]),
    ]

    np.testing.assert_array_equal(
        compute_forces(bodies, gravitational_constant=1.0, softening=1.0),
        np.zeros((2, 2)),
    )


def test_softening_reduces_force_magnitude() -> None:
    bodies = [
        Body(1.0, [0.0, 0.0], [0.0, 0.0]),
        Body(1.0, [1.0, 0.0], [0.0, 0.0]),
    ]

    force_without_softening = compute_forces(bodies, gravitational_constant=1.0)
    force_with_softening = compute_forces(
        bodies, gravitational_constant=1.0, softening=1.0
    )

    np.testing.assert_allclose(force_without_softening, [[1.0, 0.0], [-1.0, 0.0]])
    np.testing.assert_allclose(
        force_with_softening, [[1.0 / (2.0**1.5), 0.0], [-1.0 / (2.0**1.5), 0.0]]
    )


@pytest.mark.parametrize("constant", [0.0, -1.0, np.nan, np.inf])
def test_invalid_gravitational_constant(constant: float) -> None:
    body = Body(1.0, [0.0, 0.0], [0.0, 0.0])

    with pytest.raises(ValueError):
        compute_forces([body], gravitational_constant=constant)


@pytest.mark.parametrize("softening", [-1.0, np.nan, np.inf])
def test_invalid_softening(softening: float) -> None:
    body = Body(1.0, [0.0, 0.0], [0.0, 0.0])

    with pytest.raises(ValueError):
        compute_forces([body], gravitational_constant=1.0, softening=softening)