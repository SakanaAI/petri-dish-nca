import argparse
import datetime
from typing import Any

import torch
import wandb

from config import Config
from model import CASunGroup
from viz import capture_snapshot, colors, generate_nca_colors
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

            # Capture final frame and log if needed
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
