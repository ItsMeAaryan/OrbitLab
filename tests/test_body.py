import numpy as np
import pytest

from orbitlab.core.body import Body


def test_valid_2d_body() -> None:
    body = Body(2.0, np.array([1.0, 2.0]), np.array([3.0, 4.0]))

    assert body.mass == 2.0
    assert body.dimensions == 2
    np.testing.assert_array_equal(body.position, [1.0, 2.0])
    np.testing.assert_array_equal(body.velocity, [3.0, 4.0])


def test_valid_3d_body() -> None:
    body = Body(2.0, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    assert body.dimensions == 3


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_mass(mass: float) -> None:
    with pytest.raises(ValueError):
        Body(mass, [0.0, 0.0], [0.0, 0.0])


@pytest.mark.parametrize("vector", [[0.0], [0.0, 0.0, 0.0, 0.0], [[0.0, 0.0]]])
def test_invalid_dimensions(vector: list[float]) -> None:
    with pytest.raises(ValueError):
        Body(1.0, vector, vector)


def test_position_velocity_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        Body(1.0, [0.0, 0.0], [0.0, 0.0, 0.0])


@pytest.mark.parametrize("position", [[0.0, np.nan], [0.0, np.inf]])
def test_non_finite_position(position: list[float]) -> None:
    with pytest.raises(ValueError):
        Body(1.0, position, [0.0, 0.0])


@pytest.mark.parametrize("velocity", [[0.0, np.nan], [0.0, -np.inf]])
def test_non_finite_velocity(velocity: list[float]) -> None:
    with pytest.raises(ValueError):
        Body(1.0, [0.0, 0.0], velocity)


def test_input_arrays_are_copied() -> None:
    position = np.array([1.0, 2.0])
    velocity = np.array([3.0, 4.0])
    body = Body(1.0, position, velocity)

    position[0] = 99.0
    velocity[0] = 99.0

    assert body.position[0] == 1.0
    assert body.velocity[0] == 3.0