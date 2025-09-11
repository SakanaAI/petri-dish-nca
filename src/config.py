import json
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class Config:
    """Configuration class for adversarial NCA training experiments.

    This dataclass manages all hyperparameters and system settings for training
    competing Neural Cellular Automata. It includes automatic device detection,
    validation, and seed management for reproducible experiments.
    """

    # Grid
    grid_size: tuple[int, int] = (10, 10)
    n_seeds: int = 1

    # World
    cell_state_dim: int = 4
    cell_hidden_dim: int = 4

    # New world paradigm
    world_blend_alpha: float = 0.9
    seed_dist: Literal["scatter"] = "scatter"
    # Whether it's set to 1 in everything or random noise
    seed_mode: Literal["solid", "random"] = "random"
    alive_visible: bool = True  # Whether NCAs can see where things are alive

    # Burn-in config
    # NOTE: For burn-in, we're currently updating both the steps_per_update and steps_before_update
    burn_in: bool = False
    burn_in_increment_epochs: int = (
        0  # How many steps before you try to increase the steps per
    )
    burn_in_increment: int = 0  # How many steps to increase by each time, it seems that

    # NCAs
    n_ncas: int = 3
    n_hidden_layers: int = 0
    hidden_dim: int = 32
    model_kernel_size: int = 3
    model_dropout_per: float = 0.0
    per_hid_upd: float = 1.0  # Percentage of hidden channels each model can update

    # Training
    optimizer: Literal["AdamW", "Adam", "RMSProp", "SGD"] = "RMSProp"
    learning_rate: float = 1e-4
    batch_size: int = 64
    pool_size: int = 2048
    epochs: int = 10_000
    log_every: int = 100
    wandb: bool = False

    # Sun
    sun_update_epoch_wait: int = 2

    # Multi-world
    steps_before_update: int = 0
    steps_per_update: int = 2

    # General system
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    seed: int = 42
    mode: Literal["train", "eval", "frozen_eval"] = "train"

    def __post_init__(self) -> None:
        """Validate configuration and initialize system settings.

        Performs validation checks on configuration parameters, handles device
        availability fallbacks, and sets random seeds for reproducibility.

        Raises:
            AssertionError: If cell_state_dim is not even or batch_size > pool_size.
        """
        assert self.cell_state_dim % 2 == 0, "[config] cell_state_dim must be even"
        assert self.batch_size <= self.pool_size, "[config] batch_size > pool_size"
        assert self.n_seeds * self.n_ncas <= self.total_grid_size, (
            "[config] n_seeds * n_ncas > self.total_grid_size"
        )

        # Device availability check
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[warning] CUDA not available, falling back to CPU")
            object.__setattr__(self, "device", "cpu")
        elif self.device == "mps" and not torch.backends.mps.is_available():
            print("[warning] MPS not available, falling back to CPU")
            object.__setattr__(self, "device", "cpu")

        self._set_random_seed()

    def _set_random_seed(self) -> None:
        """Set all random seeds for reproducibility"""
        import random

        import numpy as np
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @property
    def cell_dim(self) -> int:
        """Total cell dimension including state, hidden, aliveness, and NCA channels.

        Returns:
            Combined dimension of cell_state_dim + cell_hidden_dim + n_ncas + 1.
        """
        return self.cell_state_dim + self.cell_hidden_dim + self.n_ncas + 1

    @property
    def cell_wo_alive_dim(self) -> int:
        """Cell dimension w/o NCA channels (just state and hidden)

        Returns:
            Combined dimension of cell_state_dim + cell_hidden_dim
        """
        return self.cell_state_dim + self.cell_hidden_dim

    @property
    def alive_dim(self) -> int:
        """Dimension for aliveness channels.

        Returns:
            Number of aliveness channels (n_ncas + 1 for sun).
        """
        return self.n_ncas + 1

    @property
    def total_grid_size(self) -> int:
        """Total number of cells in the grid.

        Returns:
            Product of grid dimensions (width * height).
        """
        return self.grid_size[0] * self.grid_size[1]

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file.

        Returns:
            Config instance with parameters loaded from file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: str) -> None:
        """Save configuration to JSON file.

        Args:
            path: Output path for JSON configuration file.

        Raises:
            IOError: If the file cannot be written.
        """
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
