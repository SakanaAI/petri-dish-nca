import colorsys
import gzip
import io
from typing import Any

import einops
import imageio
import numpy as np
import torch
from PIL import Image

color = tuple[float, float, float]
colors = list[tuple[float, float, float]]


def generate_nca_colors(n_ncas: int) -> colors:
    """Generate evenly spaced colors around HSV wheel.

    Args:
        n_ncas: Number of NCAs to generate colors for.

    Returns:
        List of RGB color tuples, one for each NCA.
    """
    colors = []
    for i in range(n_ncas):
        hue = i / n_ncas  # Evenly space around color wheel
        # High saturation, high value for vibrant colors
        color = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(color)
    return colors


def create_territory_visualization(
    grid: torch.Tensor, nca_colors: colors
) -> torch.Tensor:
    """Convert grid to RGB visualization.

    Creates a territory map showing which NCA controls each cell,
    with alpha blending based on control strength.

    Args:
        grid: Grid tensor of shape [channels, height, width].
        nca_colors: List of (r, g, b) tuples for each NCA.

    Returns:
        RGBA visualization tensor of shape [4, height, width].
    """
    C, H, W = grid.shape
    n_ncas = len(nca_colors)

    # TODO: I also need to deal with the fact it might be sun might be winning
    aliveness = grid[: n_ncas + 1]  # [n_ncas, height, width]
    winners = torch.argmax(aliveness, dim=0)  # [batch, height, width]
    control_strength = torch.max(aliveness, dim=0)[0]  # [batch, height, width]

    # Create RGB image
    rgb_image = torch.zeros(H, W, 4)

    for nca_idx, color in enumerate(nca_colors):
        mask = winners == nca_idx + 1
        rgb_image[mask] = torch.tensor([color[0], color[1], color[2], 1.0])

    # Apply alpha to all pixels
    rgb_image[:, :, 3] = control_strength

    return einops.rearrange(rgb_image, "h w c -> c h w")


def capture_snapshot(grid: torch.Tensor, nca_colors: colors) -> torch.Tensor:
    """Return a snapshot of the first grid in a batch.

    Args:
        grid: Batch of grids, uses first element.
        nca_colors: Color mapping for visualization.

    Returns:
        Territory visualization of the first grid.
    """
    return create_territory_visualization(grid[0].detach().clone().cpu(), nca_colors)


def get_int_grid(grid: torch.Tensor) -> np.ndarray:
    normalized = (grid.detach().float().cpu().numpy() + 1.0) / 2.0
    uint8_grid = (normalized * 255).astype(np.uint8)

    return uint8_grid


def get_compression_ratios(grid: torch.Tensor, img_mode: bool = True) -> np.ndarray:
    """Returns how well each channel of the grid compresses

    Args:
        grid: Self explanatory (values must be between -1, 1 to work for this function!)
    
    Returns:
        List of compression ratios (smaller is more compressible)
    """
    C, _, _ = grid.shape

    uint8_grid = get_int_grid(grid)

    original_sizes = [uint8_grid[i].nbytes for i in range(C)]

    def compress_png(channel):
        buffer = io.BytesIO()
        Image.fromarray(channel, mode='L').save(buffer, format='PNG', optimize=True, compress_level=9)
        return len(buffer.getvalue())

    if img_mode:
        # PNG compression - vectorized using map
        compressed_sizes = list(map(compress_png, uint8_grid))
    else:
        # GZIP compression - vectorized using map
        compressed_sizes = list(map(lambda ch: len(gzip.compress(ch.tobytes())), uint8_grid))

    return np.array([compressed_sizes[i] / original_sizes[i] for i in range(C)])


def get_shannon_entropy(grid: torch.Tensor) -> np.ndarray:
    """Returns shannon entropy of grids

    """
    uint8_grid = get_int_grid(grid)
    uint8_flat = uint8_grid.reshape(uint8_grid.shape[0], -1)

    grid_size = uint8_flat.shape[-1]
    freq = np.apply_along_axis(np.bincount, 1, uint8_flat, minlength=256) / grid_size  # [C, 256]
    ent = np.where(freq, -freq * np.log2(freq, where=(freq > 0)), 0)
    ent_sum = ent.sum(axis=1)

    return ent_sum

def higher_order_entropy(grid: torch.Tensor, img_mode: bool = True) -> np.ndarray:
    """Returns the higher order entropy for each channel of the grid
    
    Args:
        grid: Self explanatory (values must be between -1, 1 to work for this function!)
    
    Returns:
        List of HOE
    """
    kolmogorov_estimates = get_compression_ratios(grid, img_mode) * 8.0
    entropy_calcs = get_shannon_entropy(grid)

    return entropy_calcs - kolmogorov_estimates


def create_video(
    frames: list[Any], output_path: str = "output.mp4", fps: int = 10
) -> None:
    """Create video from a sequence of frames.

    Args:
        frames: List of frame arrays/tensors.
        output_path: Output file path for the video.
        fps: Frames per second for the output video.
    """
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f"Video saved as {output_path}")
