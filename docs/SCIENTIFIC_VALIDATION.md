# OrbitLab Phase 5 Scientific Validation

This report is generated from the deterministic experiments in `experiments/`.
The numerical values in the results table are produced by executing the
experiment scripts, not by analytical substitution.

## Methodology

The experiments use normalized units with:

```text
G = 1
mass_1 = mass_2 = 1
initial separation = 1
softening = 0
```

For the circular two-body experiment, the center of mass is at the origin.
The analytical relative angular frequency and period are:

```text
omega = sqrt(G (m1 + m2) / r^3)
T = 2 pi / omega
```

The analytical continuous reference is used only as a benchmark. Numerical
trajectories are not expected to be exact.

## Reproducibility

Run from the project root:

```powershell
.\.venv\Scripts\python.exe experiments\two_body_validation.py
.\.venv\Scripts\python.exe experiments\three_body.py
.\.venv\Scripts\python.exe experiments\three_body_timestep_sensitivity.py
.\.venv\Scripts\python.exe experiments\timestep_convergence.py
.\.venv\Scripts\python.exe experiments\integrator_stability.py
.\.venv\Scripts\python.exe experiments\conservation_analysis.py
```

Outputs are written under `outputs/phase5/`.

## Executed results

The scripts were executed with Python 3.13.15, NumPy 2.5.4, and the project
virtual environment. The circular two-body period was
`T = 4.442882938158366`. Over four periods with
`dt = T / 400`, the measured final position errors and maximum relative
energy drifts were:

| Integrator | Final position error | Maximum relative energy drift |
|---|---:|---:|
| Explicit Euler | 0.7363112228 | 0.3330462229 |
| Semi-Implicit Euler | 0.0021963137 | 0.0002467553 |
| Velocity Verlet | 0.0010334452 | 0.0000000152 |

The measured orbital periods from unwrapped relative-position angles were
`6.3149073651` for Explicit Euler, `4.4436595700` for Semi-Implicit Euler,
and `4.4432483463` for Velocity Verlet. This angle-based estimate is a
cycle-average over the four-period run; it does not imply that the
eccentricity or phase error is zero. The Explicit Euler value is strongly
affected by its energy growth and radial drift.

All three methods preserved linear momentum to the measured floating-point
precision in this pairwise implementation. Explicit Euler showed a much
larger angular-momentum drift (`0.1583685774`) than the other two methods
(below `9e-15` in this run).

For timestep convergence, Velocity Verlet was run for two periods at
`T/100`, `T/200`, and `T/400`. The final position errors were
`0.0082559056`, `0.0020663075`, and `0.0005167227`, respectively. A
least-squares log-log fit gave an observed order of `1.9989821940`, close to
the expected second order for this smooth circular problem. This is a
measurement for this configuration, not a guarantee for every problem.

The long-run integrator comparison used `dt = T/200` for 20 periods. Maximum
relative energy drifts were `0.6599050615` (Explicit Euler),
`0.0009872024` (Semi-Implicit Euler), and `0.0000002432` (Velocity Verlet).
These results illustrate different numerical behavior at this timestep; they
do not establish a universal ranking independent of problem or metric.

## Three-body audit and revised experiment

The original experiment used masses `(1.0, 0.8, 0.5)`, `dt = 0.002`, and
2500 steps to final time 5.0. Its reported maximum relative energy drift was
`73.8609104693`. A direct sensitivity study at the same final time measured:

| Timestep | Maximum drift | Final drift | Minimum separation |
|---:|---:|---:|---:|
| 0.002 | 73.8609104693 | 71.4958165511 | 0.0108658840 |
| 0.001 | 35.3297054828 | 26.8724192484 | 0.0092296409 |
| 0.0005 | 742.0110122123 | 736.4453929748 | 0.0043712639 |

The non-monotonic response and very small closest approaches show that the
old configuration enters close encounters that are not adequately resolved by
this timestep. The result is retained here as an audit finding, not hidden or
normalized away. The approximately zero momentum drift is expected from the
pairwise equal-and-opposite force update and does not validate the energy
accuracy.

The demonstration now uses the deterministic equal-mass figure-eight
choreography: three unit masses, `G = 1`, and the positions and velocities
recorded in `three_body_summary.json`. This is a better-conditioned example
because it is symmetric, has zero center-of-mass position and momentum to the
precision of the published constants, and maintains a nonzero closest
approach. It is still a numerical approximation, not an exact engine
solution.

The revised run uses `dt = 0.0025`, 2528 steps, and final time 6.32. The
timestep sensitivity study uses the same initial conditions and final time:

| Timestep | Maximum drift | Final drift | Linear drift | Angular drift | Minimum separation |
|---:|---:|---:|---:|---:|---:|
| 0.0100 | 0.0401557759 | 0.0000264342 | 2.0175841e-15 | 9.9920073e-16 | 0.1561002515 |
| 0.0050 | 0.0102063964 | 0.0000066579 | 1.4130832e-15 | 3.4416914e-15 | 0.1552701204 |
| 0.0025 | 0.0025904273 | 0.0000016674 | 2.2452871e-15 | 7.9936058e-15 | 0.1548492712 |
| 0.00125 | 0.0006482227 | 0.0000004170 | 3.0546269e-15 | 4.3298698e-15 | 0.1547989396 |

The revised measurements decrease consistently with timestep over the tested
range. The experiment does not claim chaotic behavior: a complicated-looking
three-body trajectory is not, by itself, a chaos test.

## Scientific interpretation

An isolated Newtonian system conserves total energy, linear momentum, and
angular momentum in the continuous equations. Discrete integrators introduce
finite-timestep error. The magnitude and character of that error depend on the
method and timestep; no integrator is universally best for every problem.

General three-body motion does not have a simple closed-form solution. Complex
trajectories alone are not evidence of chaos; a controlled sensitivity
experiment would be needed to establish that behavior.

The convergence experiment compares all methods at a common final time and
estimates an observed log-log slope from measured errors. The expected
Velocity Verlet order is approximately second order under smooth conditions,
but the measured order is reported rather than forced to equal two.
