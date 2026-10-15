"""Diagnostics calculated across a recorded simulation history."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orbitlab.core.physics import DEFAULT_GRAVITATIONAL_CONSTANT
from orbitlab.diagnostics.energy import (
    gravitational_potential_energy,
    kinetic_energy,
)
from orbitlab.diagnostics.momentum import angular_momentum, linear_momentum
from orbitlab.simulation.result import SimulationResult


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DiagnosticHistory:
    """Energy and momentum values corresponding to each result timestamp."""

    kinetic_energy: FloatArray
    potential_energy: FloatArray
    total_energy: FloatArray
    linear_momentum: FloatArray
    angular_momentum: FloatArray
    relative_energy_drift: FloatArray
    times: FloatArray | None = None
    gravitational_constant: float | None = None
    softening: float | None = None


def analyze_history(
    result: SimulationResult,
    gravitational_constant: float = DEFAULT_GRAVITATIONAL_CONSTANT,
    softening: float = 0.0,
    *,
    zero_energy_tolerance: float = 1e-14,
) -> DiagnosticHistory:
    """Calculate diagnostics for every state recorded in a result.

    Relative drift raises ``ValueError`` when the initial energy is zero or
    within ``zero_energy_tolerance`` of zero, because a relative ratio would
    otherwise be physically misleading.
    """
    states = [result.state_at(index) for index in range(result.times.size)]
    kinetic = np.array([kinetic_energy(state) for state in states])
    potential = np.array(
        [
            gravitational_potential_energy(
                state, gravitational_constant, softening
            )
            for state in states
        ]
    )
    total = kinetic + potential
    initial_magnitude = abs(total[0])
    try:
        tolerance = float(zero_energy_tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("zero_energy_tolerance must be finite and non-negative") from error
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_energy_tolerance must be finite and non-negative")
    if initial_magnitude <= tolerance:
        raise ValueError(
            "relative energy drift is undefined for zero or near-zero initial energy"
        )

    linear = np.array([linear_momentum(state) for state in states])
    angular_values = [angular_momentum(state) for state in states]
    angular = np.asarray(angular_values, dtype=np.float64)
    drift = np.abs(total - total[0]) / initial_magnitude
    return DiagnosticHistory(
        kinetic,
        potential,
        total,
        linear,
        angular,
        drift,
        times=result.times.copy(),
        gravitational_constant=gravitational_constant,
        softening=softening,
    )


def validate_diagnostic_history(
    result: SimulationResult,
    diagnostics: DiagnosticHistory,
    gravitational_constant: float,
    softening: float,
) -> None:
    """Validate a supplied diagnostic history against a simulation result."""
    if not isinstance(diagnostics, DiagnosticHistory):
        raise ValueError("diagnostics must be a DiagnosticHistory")
    record_count = result.times.size
    if diagnostics.times is None:
        raise ValueError(
            "diagnostics must include timestamps to verify result compatibility"
        )
    if diagnostics.times.shape != result.times.shape or not np.allclose(
        diagnostics.times, result.times, rtol=1e-12, atol=1e-15
    ):
        raise ValueError("diagnostic timestamps do not match the SimulationResult")
    if diagnostics.kinetic_energy.shape != (record_count,):
        raise ValueError("kinetic energy history has an invalid shape")
    if diagnostics.potential_energy.shape != (record_count,):
        raise ValueError("potential energy history has an invalid shape")
    if diagnostics.total_energy.shape != (record_count,):
        raise ValueError("total energy history has an invalid shape")
    if diagnostics.relative_energy_drift.shape != (record_count,):
        raise ValueError("relative energy drift history has an invalid shape")
    if diagnostics.linear_momentum.shape != (record_count, result.dimensions):
        raise ValueError("linear momentum history has an invalid shape")
    angular_shape = (record_count,) if result.dimensions == 2 else (
        record_count,
        3,
    )
    if diagnostics.angular_momentum.shape != angular_shape:
        raise ValueError("angular momentum history has an invalid shape")
    if diagnostics.gravitational_constant is None or diagnostics.softening is None:
        raise ValueError(
            "diagnostics must include physical metadata to verify compatibility"
        )
    if not np.isclose(
        diagnostics.gravitational_constant,
        gravitational_constant,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("diagnostic gravitational constant does not match")
    if not np.isclose(diagnostics.softening, softening, rtol=1e-12, atol=1e-15):
        raise ValueError("diagnostic softening does not match")
    if result.gravitational_constant is not None and not np.isclose(
        result.gravitational_constant,
        diagnostics.gravitational_constant,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("diagnostic gravitational constant does not match result")
    if result.softening is not None and not np.isclose(
        result.softening,
        diagnostics.softening,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("diagnostic softening does not match result")
