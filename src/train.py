import argparse
import datetime
from typing import Any

import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from model import CASunGroup
from viz import (
    capture_snapshot,
    colors,
    generate_nca_colors,
    get_shannon_entropy,
    get_compression_ratios
)


from world import World


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace containing config path and overrides.
    """
    parser = argparse.ArgumentParser(description="Train adversarial NCAs")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--n-ncas", type=int, help="Number of NCAs")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], help="Device to use"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--live-viz", action="store_true", help="Enable live matplotlib visualization during training")

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    """Load and configure based on arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Validated configuration object with CLI overrides applied.
    """
    # Load base config
    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config(
            n_ncas=args.n_ncas or 3, device=args.device or "mps", wandb=args.wandb
        )

    # Apply CLI overrides
    if args.n_ncas:
        config.n_ncas = args.n_ncas
        print(f"[config] updated n_ncas to {config.n_ncas}")
    if args.epochs:
        config.epochs = args.epochs
        print(f"[config] updated epochs to {config.epochs}")
    if args.device:
        config.device = args.device
        print(f"[config] updated device to {config.device}")
    if args.wandb:
        config.wandb = args.wandb
        print(f"[config] updated wandb to {config.wandb}")
    if args.live_viz:
        config.live_viz = args.live_viz
        print(f"[config] updated live_viz to {config.live_viz}")

    # Validate after modifications
    config.__post_init__()
    return config


def setup_experiment(
    config: Config,
) -> tuple[Any | None, World, CASunGroup, colors]:
    """Initialize wandb and create world/group.

    Args:
        config: Configuration object for the experiment.

    Returns:
        Tuple containing (wandb run, world, group, nca_colors).
    """
    # Setup wandb
    if config.wandb:
        run = wandb.init(project="adversarial-nca", config=config.__dict__)
    else:
        run = None

    # Create world and group
    world = World(config)
    group = CASunGroup(config)

    # Generate visualization colors
    nca_colors = generate_nca_colors(config.n_ncas)

    return run, world, group, nca_colors


def log_metrics(
    run: Any | None,
    epoch: int,
    stats: dict[str, Any],
    frames: list[torch.Tensor],
    nca_colors: colors,
    grid: torch.Tensor,
) -> None:
    """Log metrics and visualizations to wandb if needed, otherwise just log in terminal.

    Args:
        run: Wandb run object (None if wandb disabled).
        epoch: Current training epoch.
        stats: Training statistics dictionary.
        frames: List of visualization frames.
        nca_colors: Color mapping for each NCA.
        grid: Current grid state.
    """
    avg_grad_norm = stats["grad_norm"].cpu().numpy().mean()

    if run:
        metrics = {"epoch": epoch}

        # Growth metrics
        metrics["growth/sun"] = stats["growth"][0]
        for i, growth in enumerate(stats["growth"][1:]):
            metrics[f"growth/nca_{i:02d}"] = growth

        # Training metrics
        metrics["training/avg_grad_norm"] = avg_grad_norm
        metrics["training/loss"] = stats["loss"]

        # Individual grad norms
        # for i, grad_norm in enumerate(stats["grad_norms"]):
        #     metrics[f"training/grad_norm_nca_{i:02d}"] = grad_norm

        # Visualizations
        frame_images = [
            wandb.Image(frame, caption=f"Step {i}") for i, frame in enumerate(frames)
        ]
        metrics["viz/frame_sequence"] = frame_images
        metrics["viz/final_territory"] = wandb.Image(capture_snapshot(grid, nca_colors))

        # Create video if we have multiple frames
        if len(frames) > 1:
            video_frames = (torch.stack(frames) * 255).to(torch.uint8)
            video_array = video_frames.detach().cpu().numpy()
            metrics["viz/growth"] = wandb.Video(video_array, format="gif")

        # Log to wandb
        run.log(metrics)

    # Terminal logging
    growth_stats = [f"{g:.2f}" for g in stats["growth"]]
    growth_str = ", ".join(growth_stats)
    print(
        f"Epoch {epoch:6d} | Growth: [{growth_str}] | Grad: {avg_grad_norm:.3f} | Loss: {stats['loss']:.2f}"
    )


def should_log(epoch: int, config: Config) -> bool:
    """Determine if we should log this epoch.

    Args:
        epoch: Current epoch number.
        config: Configuration with log_every parameter.

    Returns:
        True if this epoch should be logged.
    """
    return epoch % config.log_every == 0


def train_loop(config: Config) -> None:
    """Main training loop.

    Args:
        config: Configuration object containing all training parameters.
    """
    print(
        f"Starting training: {config.n_ncas} NCAs, {config.grid_size} grid, {config.epochs} epochs"
    )

    # Setup experiment
    run, world, group, nca_colors = setup_experiment(config)

    if config.live_viz:
        plt.rcParams['figure.raise_window'] = False
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except Exception as e:
            import traceback
            print(f"Error:{traceback.format_exc()}")
            raise ValueError(f"Error: {e}")

        plt.ion()
        fig = plt.figure(figsize=(11, 7))

        gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.6)

        ax_grid = fig.add_subplot(gs[:2, :2])
        im_grid = None

        ax_entropy = fig.add_subplot(gs[0, 2])
        ax_compression = fig.add_subplot(gs[0, 3])
        ax_population = fig.add_subplot(gs[1, 2])
        ax_loss = fig.add_subplot(gs[1, 3])
        ax_grad = fig.add_subplot(gs[2, :])

        epochs_history = []
        entropy_history = []
        compression_history = []
        population_history = []
        loss_history = []
        grad_history = []

        line_entropy = None
        line_compression = None
        line_population = None
        line_loss = None
        line_grad = None

    run_name = (
        run.name if run else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    try:
        for epoch in range(config.epochs + 1):
            # Initialize
            grid = world.get_seed()

            # Capture initial frame if logging
            frames = []
            if should_log(epoch, config):
                frames.append(capture_snapshot(grid, nca_colors))

            # Training step
            stats, grid, grids = world.step(group, grid)

            metrics_calculated = False
            snapshot = None
            entropy = None
            compression = None
            populations = []

            if config.live_viz and epoch % 10 == 0:
                snapshot = capture_snapshot(grid, nca_colors).permute(1, 2, 0)
                entropy = get_shannon_entropy(grid[0].detach())  
                compression = get_compression_ratios(grid[0].detach())

                for nca_idx in range(config.n_ncas):
                    aliveness = grid[0, nca_idx + 1] 
                    alive_cells = (aliveness > 0.1).sum().item()
                    populations.append(alive_cells)
                metrics_calculated = True

            if config.live_viz and epoch % 10 == 0 and metrics_calculated:
                epochs_history.append(epoch)
                entropy_history.append(entropy.mean().item())
                compression_history.append(compression.mean())
                population_history.append(populations)
                loss_history.append(stats["loss"])
                grad_history.append(stats["grad_norm"].cpu().numpy().mean())

                if im_grid is None:
                    im_grid = ax_grid.imshow(snapshot)
                    ax_grid.set_title(f"NCA Territories - Epoch: {epoch}")
                    ax_grid.axis("off")
                else:
                    im_grid.set_data(snapshot)
                    ax_grid.set_title(f"NCA Territories - Epoch: {epoch}")

                if line_entropy is None:
                    line_entropy, = ax_entropy.plot(epochs_history, entropy_history, 'b-', label='Shannon Entropy')
                    ax_entropy.set_title("Entropy")
                    ax_entropy.set_xlabel("Epoch")
                    ax_entropy.set_ylabel("Entropy (bits)")
                    ax_entropy.grid(True, alpha=0.3)
                else:
                    line_entropy.set_data(epochs_history, entropy_history)
                    ax_entropy.relim()
                    ax_entropy.autoscale_view()

                if line_compression is None:
                    line_compression, = ax_compression.plot(epochs_history, compression_history, 'r-', label='Compression')
                    ax_compression.set_title("Compression")
                    ax_compression.set_xlabel("Epoch")
                    ax_compression.set_ylabel("Ratio")
                    ax_compression.grid(True, alpha=0.3)
                else:
                    line_compression.set_data(epochs_history, compression_history)
                    ax_compression.relim()
                    ax_compression.autoscale_view()

                populations_array = np.array(population_history)
                if line_population is None:
                    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                    line_population = []
                    for i in range(config.n_ncas):
                        line, = ax_population.plot(epochs_history, populations_array[:, i],
                                                 color=colors[i % len(colors)],
                                                 label=f'NCA {i}')
                        line_population.append(line)
                    ax_population.set_title("Population")
                    ax_population.set_xlabel("Epoch")
                    ax_population.set_ylabel("Cell Count")
                    ax_population.legend()
                    ax_population.grid(True, alpha=0.3)
                else:
                    for i, line in enumerate(line_population):
                        line.set_data(epochs_history, populations_array[:, i])
                    ax_population.relim()
                    ax_population.autoscale_view()

                if line_loss is None:
                    line_loss, = ax_loss.plot(epochs_history, loss_history, 'g-', label='Loss')
                    ax_loss.set_title("Loss")
                    ax_loss.set_xlabel("Epoch")
                    ax_loss.set_ylabel("Loss")
                    ax_loss.grid(True, alpha=0.3)
                else:
                    line_loss.set_data(epochs_history, loss_history)
                    ax_loss.relim()
                    ax_loss.autoscale_view()

                if line_grad is None:
                    line_grad, = ax_grad.plot(epochs_history, grad_history, 'm-', label='Grad Norm')
                    ax_grad.set_title("Grad Norm")
                    ax_grad.set_xlabel("Epoch")
                    ax_grad.set_ylabel("Gradient Norm")
                    ax_grad.grid(True, alpha=0.3)
                else:
                    line_grad.set_data(epochs_history, grad_history)
                    ax_grad.relim()
                    ax_grad.autoscale_view()

                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)

            if should_log(epoch, config):
                for st in range(grids.shape[0]):
                    frames.append(capture_snapshot(grids[st], nca_colors))
                log_metrics(run, epoch, stats, frames, nca_colors, grid)

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}")
        group.save(config, run_name)
        world.save(config, run_name)
        if run:
            wandb.finish()
        print("Saved model!")

    # Save model and world
    # TODO: Improve separate saves
    group.save(config, run_name)
    world.save(config, run_name)

    if config.live_viz:
        plt.ioff()
        plt.close('all')

    if run:
        wandb.finish()

    print("Training completed!")


def main() -> None:
    """Main entry point for training script."""
    args = parse_args()
    config = load_config(args)
    train_loop(config)


if __name__ == "__main__":
    main()
