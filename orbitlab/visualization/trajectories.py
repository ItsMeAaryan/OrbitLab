"""Static trajectory plots."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from orbitlab.simulation.result import SimulationResult
from orbitlab.visualization._common import save_figure, validate_2d_result


def plot_trajectories(
    result: SimulationResult,
    *,
    figsize: tuple[float, float] = (8.0, 8.0),
    save_path: str | Path | None = None,
    title: str = "Body trajectories",
) -> tuple[Figure, Axes]:
    """Plot every 2D body trajectory with initial and final markers."""
    validate_2d_result(result)
    figure, axes = plt.subplots(figsize=figsize)
    colors = plt.get_cmap("tab10")
    for body_index in range(result.masses.size):
        color = colors(body_index % 10)
        trajectory = result.positions[:, body_index]
        label = f"Body {body_index + 1}"
        axes.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            label=label,
        )
        axes.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            color=color,
            marker="o",
            s=45,
            zorder=3,
        )
        axes.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            color=color,
            marker="X",
            s=65,
            zorder=3,
        )
    axes.set_xlabel("x position")
    axes.set_ylabel("y position")
    axes.set_title(title)
    axes.set_aspect("equal", adjustable="box")
    axes.legend()
    axes.grid(True, alpha=0.3)
    save_figure(figure, save_path)
    return figure, axes
