import argparse
import os
import sys
from typing import Any

import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')

import torch

from config import Config
from model import CASunGroup
from viz import capture_snapshot, generate_nca_colors
from world import World


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize trained adversarial NCAs")
    parser.add_argument(
        "--model-path", required=True,
        help="Path to trained model directory (containing config.json, model.pt, etc.)"
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of simulation steps to run"
    )
    parser.add_argument(
        "--update-every", type=int, default=5,
        help="Update visualization every N steps"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default="cpu",
        help="Device to use for inference"
    )

    return parser.parse_args()


def load_trained_model(model_path: str, device: str) -> tuple[Config, CASunGroup, World]:
    """Load trained model from directory."""
    # Load config
    config_path = os.path.join(model_path, "config.json")
    config = Config.from_file(config_path)
    config.device = device

    # Create models
    group = CASunGroup(config)
    world = World(config)

    # Load trained weights
    group.load(model_path)

    return config, group, world


def run_visualization(config: Config, group: CASunGroup, world: World, steps: int, update_every: int) -> None:
    """Run live visualization of trained NCA behavior."""
    print(f"Starting visualization: {config.n_ncas} NCAs, {config.grid_size} grid")
    print(f"Running for {steps} steps, updating every {update_every} steps")

    nca_colors = generate_nca_colors(config.n_ncas)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    im = None

    grid = world.get_seed()

    try:
        for step in range(steps):
            stats, grid, grids = world.step(group, grid)

            if step % update_every == 0:
                snapshot = capture_snapshot(grid, nca_colors)

                if im is None:
                    im = ax.imshow(snapshot.permute(1, 2, 0))
                    ax.set_title(f"Trained NCA Behavior - Step: {step}")
                    ax.axis("off")
                    plt.show()
                else:
                    im.set_data(snapshot.permute(1, 2, 0))
                    ax.set_title(f"Trained NCA Behavior - Step: {step}")

                plt.draw()
                plt.pause(0.01)

                growth_stats = [f"{g:.2f}" for g in stats["growth"]]
                growth_str = ", ".join(growth_stats)
                print(f"Step {step:6d} | Growth: [{growth_str}] | Loss: {stats['loss']:.2f}")

    except KeyboardInterrupt:
        print(f"\nVisualization interrupted at step {step}")

    # Cleanup
    plt.ioff()
    plt.close()
    print("Visualization completed!")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        sys.exit(1)

    try:
        config, group, world = load_trained_model(args.model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    run_visualization(config, group, world, args.steps, args.update_every)


if __name__ == "__main__":
    main()
