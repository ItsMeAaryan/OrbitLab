"""Recorded-history animations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from orbitlab.simulation.result import SimulationResult
from orbitlab.visualization._common import stable_limits, validate_2d_result


def create_animation(
    result: SimulationResult,
    *,
    interval: int = 50,
    trail_length: int | None = None,
    figsize: tuple[float, float] = (8.0, 8.0),
    title: str = "OrbitLab simulation",
) -> tuple[FuncAnimation, Figure, Axes]:
    """Create an animation from recorded 2D states without running physics."""
    validate_2d_result(result)
    if isinstance(interval, bool) or interval <= 0:
        raise ValueError("interval must be a positive integer in milliseconds")
    if trail_length is not None and (
        isinstance(trail_length, bool) or trail_length <= 0
    ):
        raise ValueError("trail_length must be a positive integer or None")

    figure, axes = plt.subplots(figsize=figsize)
    axes.set_xlim(stable_limits(result.positions)[0:2])
    axes.set_ylim(stable_limits(result.positions)[2:4])
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x position")
    axes.set_ylabel("y position")
    axes.set_title(title)

    colors = plt.get_cmap("tab10")
    mass_scale = 30.0 + 170.0 * np.sqrt(result.masses / np.max(result.masses))
    scatter = axes.scatter(
        result.positions[0, :, 0],
        result.positions[0, :, 1],
        s=mass_scale,
        c=[colors(index % 10) for index in range(result.masses.size)],
    )
    trails = [
        axes.plot([], [], color=colors(index % 10), alpha=0.65)[0]
        for index in range(result.masses.size)
    ]
    time_text = axes.text(0.02, 0.95, "", transform=axes.transAxes)

    def update(frame_index: int) -> tuple[object, ...]:
        """Update markers, trails, and displayed simulation time."""
        positions = result.positions
        start = 0 if trail_length is None else max(0, frame_index + 1 - trail_length)
        scatter.set_offsets(positions[frame_index])
        artists: list[object] = [scatter, time_text]
        for body_index, line in enumerate(trails):
            line.set_data(
                positions[start : frame_index + 1, body_index, 0],
                positions[start : frame_index + 1, body_index, 1],
            )
            artists.append(line)
        time_text.set_text(f"t = {result.times[frame_index]:.6g}")
        return tuple(artists)

    animation = FuncAnimation(
        figure,
        update,
        frames=result.times.size,
        interval=interval,
        blit=False,
        repeat=False,
    )
    return animation, figure, axes


def save_animation(
    animation: FuncAnimation,
    save_path: str | Path,
    *,
    writer: str | object | None = None,
    fps: int = 20,
) -> None:
    """Save an animation and report unavailable writers explicitly."""
    try:
        animation.save(save_path, writer=writer, fps=fps)
    except (ImportError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"could not save animation to {save_path!s}; "
            "ensure a compatible Matplotlib writer is installed"
        ) from error
