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
from viz import (
    build_plotly_slice_stack_animation,
    build_plotly_slice_stack,
    compute_population_percent,
    capture_snapshot,
    create_territory_volume,
    generate_nca_colors,
    plot_slice_stack_matplotlib,
)
from world import World


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize trained adversarial NCAs (2D/3D, optional Plotly)"
    )
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
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs="+",
        help="Override grid size as D H W (or H W for 2D)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size for visualization",
    )
    parser.add_argument(
        "--slice-axis", choices=["depth", "height", "width"],
        help="Axis to slice for 3D visualization (viz-only)"
    )
    parser.add_argument(
        "--slice-stride", type=int,
        help="Stride between slices for 3D visualization (viz-only)"
    )
    parser.add_argument(
        "--slice-spacing",
        "--viz-slice-spacing",
        dest="slice_spacing",
        type=float,
        help="Spacing between slices (viz-only)",
    )
    parser.add_argument(
        "--slice-alpha", type=float,
        help="Slice opacity (viz-only)"
    )
    parser.add_argument(
        "--max-slices", type=int,
        help="Maximum number of slices to draw (viz-only)"
    )
    parser.add_argument(
        "--plotly", action="store_true",
        help="Show a final interactive Plotly view with playback (mouse controls)"
    )
    parser.add_argument(
        "--plotly-speed",
        type=float,
        default=0.5,
        help="Plotly playback speed multiplier (1.0 = default speed)",
    )
    parser.add_argument(
        "--no-plotly", action="store_true",
        help="Disable final Plotly view even for 3D grids"
    )

    return parser.parse_args()


def load_trained_model(
    model_path: str,
    device: str,
    grid_size: tuple[int, int] | tuple[int, int, int] | None = None,
    batch_size: int | None = None,
) -> tuple[Config, CASunGroup, World]:
    """Load trained model from directory."""
    # Load config
    config_path = os.path.join(model_path, "config.json")
    config = Config.from_file(config_path)
    config.device = device
    if grid_size is not None:
        config.grid_size = grid_size
    if batch_size is not None:
        config.batch_size = batch_size
    if grid_size is not None or batch_size is not None:
        config.__post_init__()

    # Create models
    group = CASunGroup(config)
    world = World(config)

    # Load trained weights
    group.load(model_path)

    return config, group, world


def run_visualization(
    config: Config,
    group: CASunGroup,
    world: World,
    steps: int,
    update_every: int,
    slice_axis: str,
    slice_stride: int,
    slice_spacing: float,
    slice_alpha: float,
    max_slices: int | None,
    show_plotly: bool,
    plotly_speed: float,
) -> None:
    """Run live visualization of trained NCA behavior."""
    print(f"Starting visualization: {config.n_ncas} NCAs, {config.grid_size} grid")
    print(f"Running for {steps} steps, updating every {update_every} steps")

    nca_colors = generate_nca_colors(config.n_ncas)

    plt.ion()
    use_3d_viz = config.grid_size[0] > 1
    if use_3d_viz:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
    im = None

    grid = world.get_seed()
    plotly_grids: list[torch.Tensor] = []
    plotly_populations: list[list[float]] = []
    if show_plotly and use_3d_viz:
        plotly_grids.append(grid[0].detach().cpu())
        plotly_populations.append(
            compute_population_percent(grid[0], config.n_ncas)
        )

    try:
        for step in range(steps):
            stats, grid, grids = world.step(group, grid)

            if step % update_every == 0:
                if use_3d_viz:
                    volume = create_territory_volume(grid[0], nca_colors)
                    plot_slice_stack_matplotlib(
                        ax,
                        volume,
                        axis=slice_axis,
                        stride=slice_stride,
                        spacing=slice_spacing,
                        max_slices=max_slices,
                        alpha=slice_alpha,
                    )
                    ax.set_title(f"Trained NCA Behavior - Step: {step}")
                else:
                    snapshot = capture_snapshot(
                        grid,
                        nca_colors,
                        slice_stride=slice_stride,
                        max_slices=max_slices,
                    )
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

                if show_plotly and use_3d_viz:
                    plotly_grids.append(grid[0].detach().cpu())
                    plotly_populations.append(
                        compute_population_percent(grid[0], config.n_ncas)
                    )

                growth_stats = [f"{g:.2f}" for g in stats["growth"]]
                growth_str = ", ".join(growth_stats)
                print(f"Step {step:6d} | Growth: [{growth_str}] | Loss: {stats['loss']:.2f}")

    except KeyboardInterrupt:
        print(f"\nVisualization interrupted at step {step}")

    # Cleanup
    plt.ioff()
    plt.close()
    print("Visualization completed!")

    if show_plotly and use_3d_viz:
        frame_duration_ms = max(10, int(160 / max(0.05, plotly_speed)))
        fig = build_plotly_slice_stack_animation(
            plotly_grids,
            nca_colors,
            axis=slice_axis,
            stride=slice_stride,
            spacing=slice_spacing,
            max_slices=max_slices,
            alpha=slice_alpha,
            frame_duration_ms=frame_duration_ms,
            population_history=plotly_populations,
        )
        fig.show()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        sys.exit(1)

    try:
        grid_size = None
        if args.grid_size:
            if len(args.grid_size) == 2:
                grid_size = (args.grid_size[0], args.grid_size[1])
            elif len(args.grid_size) == 3:
                grid_size = (
                    args.grid_size[0],
                    args.grid_size[1],
                    args.grid_size[2],
                )
            else:
                raise ValueError("[config] --grid-size expects 2 or 3 integers")

        config, group, world = load_trained_model(
            args.model_path, args.device, grid_size, args.batch_size
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    slice_axis = args.slice_axis or config.viz_slice_axis
    slice_stride = args.slice_stride or config.viz_slice_stride
    slice_spacing = args.slice_spacing or config.viz_slice_spacing
    slice_alpha = args.slice_alpha or config.viz_slice_alpha
    max_slices = args.max_slices or config.viz_max_slices
    use_plotly = args.plotly or (config.grid_size[0] > 1 and not args.no_plotly)

    run_visualization(
        config,
        group,
        world,
        args.steps,
        args.update_every,
        slice_axis,
        slice_stride,
        slice_spacing,
        slice_alpha,
        max_slices,
        use_plotly,
        args.plotly_speed,
    )


if __name__ == "__main__":
    main()
