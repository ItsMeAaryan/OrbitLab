import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from orbitlab import (
    ExplicitEuler,
    Simulation,
    SimulationResult,
    SystemState,
    VelocityVerlet,
    analyze_history,
)
from orbitlab.diagnostics.analysis import DiagnosticHistory
from orbitlab.visualization import (
    create_animation,
    plot_energy_comparison,
    plot_energy_diagnostics,
    plot_momentum_diagnostics,
    plot_relative_energy_drift_comparison,
    plot_trajectories,
    plot_trajectory_comparison,
)


def result_2d() -> SimulationResult:
    state = SystemState([1.0, 2.0], [[-1.0, 0.0], [1.0, 0.0]], [[0.0, 0.5], [0.0, -0.5]])
    return Simulation(state, VelocityVerlet(), 0.05, gravitational_constant=1.0).run(
        steps=4
    )


def result_3d() -> SimulationResult:
    return SimulationResult(
        [1.0],
        [0.0, 1.0],
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        [[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]],
    )


def test_trajectory_plot_single_and_multiple_bodies(tmp_path) -> None:
    figure, axes = plot_trajectories(result_2d(), save_path=tmp_path / "trajectory.png")

    assert len(axes.lines) == 2
    assert len(axes.collections) == 4
    assert axes.get_aspect() == 1.0
    assert (tmp_path / "trajectory.png").exists()
    figure.clf()


def test_trajectory_plot_rejects_3d() -> None:
    with pytest.raises(ValueError, match="2D"):
        plot_trajectories(result_3d())


def test_energy_and_momentum_plots(tmp_path) -> None:
    result = result_2d()
    diagnostics = analyze_history(result, gravitational_constant=1.0)
    energy_figure, energy_axes = plot_energy_diagnostics(
        result, diagnostics, save_path=tmp_path / "energy.png"
    )
    momentum_figure, momentum_axes = plot_momentum_diagnostics(
        result, diagnostics, save_path=tmp_path / "momentum.png"
    )

    assert len(energy_axes.flat) == 4
    assert len(momentum_axes[0, 0].lines) == 2
    assert len(momentum_axes[1, 0].lines) == 1
    assert (tmp_path / "energy.png").exists()
    assert (tmp_path / "momentum.png").exists()
    energy_figure.clf()
    momentum_figure.clf()


def test_3d_momentum_plot() -> None:
    figure, axes = plot_momentum_diagnostics(
        result_3d(),
        analyze_history(result_3d(), gravitational_constant=1.0),
        gravitational_constant=1.0,
    )

    assert len(axes[0, 0].lines) == 3
    assert len(axes[1, 0].lines) == 3
    figure.clf()


def test_animation_has_stable_limits_and_recorded_frames(tmp_path) -> None:
    result = result_2d()
    animation, figure, axes = create_animation(result, interval=25, trail_length=2)

    assert isinstance(animation, FuncAnimation)
    initial_limits = (axes.get_xlim(), axes.get_ylim())
    animation._func(result.times.size - 1)
    assert (axes.get_xlim(), axes.get_ylim()) == initial_limits
    np.testing.assert_allclose(
        axes.collections[0].get_offsets(), result.positions[-1]
    )
    animation.save(tmp_path / "audit.gif", writer="pillow", fps=10)
    assert (tmp_path / "audit.gif").exists()
    figure.clf()


def test_animation_invalid_values() -> None:
    with pytest.raises(ValueError):
        create_animation(result_2d(), interval=0)
    with pytest.raises(ValueError):
        create_animation(result_2d(), trail_length=0)


def test_comparison_plots_and_incompatible_time_grids() -> None:
    initial = SystemState([1.0, 1.0], [[-0.5, 0.0], [0.5, 0.0]], [[0.0, 0.7], [0.0, -0.7]])
    results = {
        "Explicit Euler": Simulation(initial, ExplicitEuler(), 0.05, gravitational_constant=1.0).run(steps=4),
        "Velocity Verlet": Simulation(initial, VelocityVerlet(), 0.05, gravitational_constant=1.0).run(steps=4),
    }
    figures = [
        plot_trajectory_comparison(results)[0],
        plot_energy_comparison(results, gravitational_constant=1.0)[0],
        plot_relative_energy_drift_comparison(results, gravitational_constant=1.0)[0],
    ]
    assert all(figure.axes for figure in figures)
    for figure in figures:
        figure.clf()

    incompatible = dict(results)
    incompatible["different dt"] = Simulation(
        initial, ExplicitEuler(), 0.1, gravitational_constant=1.0
    ).run(steps=4)
    with pytest.raises(ValueError, match="time grids"):
        plot_energy_comparison(incompatible, gravitational_constant=1.0)


def test_comparisons_reject_different_physical_initial_conditions_and_parameters() -> None:
    initial = SystemState(
        [1.0, 1.0],
        [[-0.5, 0.0], [0.5, 0.0]],
        [[0.0, 0.7], [0.0, -0.7]],
    )
    base = Simulation(
        initial,
        ExplicitEuler(),
        0.05,
        gravitational_constant=1.0,
        softening=0.1,
    ).run(steps=4)

    cases = [
        Simulation(
            SystemState([2.0, 1.0], initial.positions, initial.velocities),
            ExplicitEuler(),
            0.05,
            gravitational_constant=1.0,
            softening=0.1,
        ).run(steps=4),
        Simulation(
            SystemState(initial.masses, [[-0.4, 0.0], [0.5, 0.0]], initial.velocities),
            ExplicitEuler(),
            0.05,
            gravitational_constant=1.0,
            softening=0.1,
        ).run(steps=4),
        Simulation(
            SystemState(initial.masses, initial.positions, [[0.0, 0.8], [0.0, -0.7]]),
            ExplicitEuler(),
            0.05,
            gravitational_constant=1.0,
            softening=0.1,
        ).run(steps=4),
        Simulation(
            initial,
            ExplicitEuler(),
            0.05,
            gravitational_constant=2.0,
            softening=0.1,
        ).run(steps=4),
        Simulation(
            initial,
            ExplicitEuler(),
            0.05,
            gravitational_constant=1.0,
            softening=0.2,
        ).run(steps=4),
    ]

    for different in cases:
        with pytest.raises(ValueError):
            plot_energy_comparison(
                {"base": base, "different": different},
                gravitational_constant=1.0,
                softening=0.1,
            )


def test_comparisons_reject_missing_metadata_and_allow_integrator_identity_changes() -> None:
    initial = SystemState(
        [1.0, 1.0],
        [[-0.5, 0.0], [0.5, 0.0]],
        [[0.0, 0.7], [0.0, -0.7]],
    )
    explicit = Simulation(
        initial, ExplicitEuler(), 0.05, gravitational_constant=1.0
    ).run(steps=4)
    verlet = Simulation(
        initial, VelocityVerlet(), 0.05, gravitational_constant=1.0
    ).run(steps=4)

    figure, _ = plot_energy_comparison(
        {"Explicit": explicit, "Verlet": verlet},
        gravitational_constant=1.0,
    )
    figure.clf()

    missing = SimulationResult(
        explicit.masses,
        explicit.times,
        explicit.positions,
        explicit.velocities,
    )
    with pytest.raises(ValueError, match="physical metadata"):
        plot_energy_comparison(
            {"complete": explicit, "missing": missing},
            gravitational_constant=1.0,
        )


def test_comparison_limits_include_all_trajectories() -> None:
    initial = SystemState([1.0], [[0.0, 0.0]], [[0.0, 0.0]])
    near = Simulation(initial, ExplicitEuler(), 1.0, gravitational_constant=1.0).run(
        steps=1
    )
    far = Simulation(
        SystemState([1.0], [[0.0, 0.0]], [[10.0, 0.0]]),
        ExplicitEuler(),
        1.0,
        gravitational_constant=1.0,
    ).run(steps=1)
    # Make the histories physically comparable while retaining a far trajectory.
    far = SimulationResult(
        near.masses,
        near.times,
        np.array([[[0.0, 0.0]], [[10.0, 0.0]]]),
        near.velocities,
        gravitational_constant=1.0,
        softening=0.0,
        integrator_name="Other",
    )
    figure, axes = plot_trajectory_comparison({"near": near, "far": far})
    assert axes.get_xlim()[1] > 9.0
    figure.clf()


def test_supplied_diagnostic_history_must_match_result() -> None:
    result = result_2d()
    valid = analyze_history(result, gravitational_constant=1.0)
    mismatched = DiagnosticHistory(
        valid.kinetic_energy,
        valid.potential_energy,
        valid.total_energy,
        valid.linear_momentum,
        valid.angular_momentum,
        valid.relative_energy_drift,
        times=valid.times + 1.0,
        gravitational_constant=1.0,
        softening=0.0,
    )

    plot_energy_diagnostics(result, valid, gravitational_constant=1.0)
    with pytest.raises(ValueError, match="timestamps"):
        plot_energy_diagnostics(result, mismatched, gravitational_constant=1.0)
