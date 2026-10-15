"""Create a two-body GIF when Matplotlib's Pillow writer is available."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitlab import Simulation, SystemState, VelocityVerlet
from orbitlab.visualization import create_animation, save_animation


OUTPUT_DIR = Path("outputs")


def main() -> None:
    """Run a short simulation and attempt a GIF export."""
    state = SystemState([1.0, 1.0], [[-0.5, 0.0], [0.5, 0.0]], [[0.0, 0.7], [0.0, -0.7]])
    result = Simulation(state, VelocityVerlet(), 0.01, gravitational_constant=1.0).run(
        steps=100
    )
    animation, figure, _ = create_animation(result, interval=40, trail_length=50)
    OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        save_animation(animation, OUTPUT_DIR / "two_body.gif", writer="pillow", fps=25)
    except RuntimeError as error:
        print(f"Animation export unavailable: {error}")
    finally:
        figure.clf()


if __name__ == "__main__":
    main()
