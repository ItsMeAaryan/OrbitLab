# OrbitLab

OrbitLab is a modular Python engine for simulating Newtonian gravitational
interactions between multiple celestial bodies. It includes validated state
models, pairwise force and acceleration calculations, three numerical
integrators, simulation-history diagnostics, Matplotlib visualizations, and
reproducible scientific-validation experiments.

This project is intended for scientific computing education and portfolio
demonstration. It does not claim production-grade accuracy for arbitrary
close-encounter or long-duration astrophysical simulations.

## Features

- Validated 2D and 3D `Body` and `SystemState` models.
- Newtonian pairwise gravity with optional softening.
- Explicit Euler, semi-implicit Euler, and velocity Verlet integrators.
- Recorded simulation results with physical-parameter metadata.
- Kinetic, potential, total-energy, linear-momentum, and angular-momentum
  diagnostics.
- Agg-compatible trajectory, energy, momentum, comparison, and animation
  plotting.
- Scientific experiments for orbital validation, convergence, integrator
  stability, conservation, and three-body timestep sensitivity.

## Requirements

- Windows 11 or another Python-compatible operating system.
- Python 3.11 or newer.
- NumPy, Matplotlib, Pillow, and pytest as declared in
  [requirements.txt](requirements.txt).

## Windows setup

PowerShell activation is optional. The commands below use the virtual
environment interpreter directly:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an editable package installation instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
```

## Run the tests

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The suite covers core physics, integrators, simulation and diagnostics,
visualization, comparison validation, and experiment helpers.

## Public API example

```python
from orbitlab import Simulation, SystemState, VelocityVerlet

state = SystemState(
    masses=[1.0, 1.0],
    positions=[[-0.5, 0.0], [0.5, 0.0]],
    velocities=[[0.0, -0.7], [0.0, 0.7]],
)
simulation = Simulation(
    state,
    VelocityVerlet(),
    dt=0.01,
    gravitational_constant=1.0,
)
result = simulation.run(steps=1000)
```

## Scientific validation experiments

Run the reproducible experiments from the project root:

```powershell
.\.venv\Scripts\python.exe experiments\two_body_validation.py
.\.venv\Scripts\python.exe experiments\three_body.py
.\.venv\Scripts\python.exe experiments\three_body_timestep_sensitivity.py
.\.venv\Scripts\python.exe experiments\timestep_convergence.py
.\.venv\Scripts\python.exe experiments\integrator_stability.py
.\.venv\Scripts\python.exe experiments\conservation_analysis.py
```

Generated plots and measurement tables are written to `outputs/phase5/`.
The scientific discussion and executed measurements are documented in
[docs/SCIENTIFIC_VALIDATION.md](docs/SCIENTIFIC_VALIDATION.md).

The three-body validation explicitly includes the original close-encounter
configuration audit and a better-conditioned deterministic figure-eight
demonstration. Complex three-body trajectories are not labelled chaotic
without a controlled sensitivity experiment.

## Project structure

```text
orbitlab/
  core/          Body model and Newtonian physics
  integrators/   Numerical integration methods and system state
  simulation/    Simulation engine and recorded results
  diagnostics/   Energy and momentum analysis
  visualization/ Matplotlib plotting and animation
experiments/     Reproducible numerical experiments
examples/        Small runnable demonstrations
tests/           Automated pytest suite
docs/            Scientific validation report
```

## Scope and limitations

- The engine uses direct pairwise force calculation, so computational cost
  grows quadratically with body count.
- Numerical results depend on timestep, integrator, initial conditions, and
  close encounters.
- Relative energy drift is undefined for zero or near-zero initial energy and
  is rejected explicitly.
- Trajectory and animation helpers currently target 2D positions; momentum
  plots support both 2D and 3D.
- GIF export uses Pillow when installed. Other animation encoders, such as
  FFmpeg, are not assumed to be available.

## License

OrbitLab is distributed under the [MIT License](LICENSE).
