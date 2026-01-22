import colorsys
import gzip
import io
import math
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


def _ensure_3d_grid(grid: torch.Tensor) -> torch.Tensor:
    if grid.dim() == 3:
        return grid.unsqueeze(1)
    if grid.dim() == 4:
        return grid
    raise ValueError(f"Expected grid with 3 or 4 dims, got {grid.dim()}")


def _get_slice_indices(
    count: int, stride: int = 1, max_slices: int | None = None
) -> list[int]:
    indices = list(range(0, count, stride))
    if max_slices is not None and len(indices) > max_slices:
        pick = np.linspace(0, len(indices) - 1, num=max_slices, dtype=int)
        indices = [indices[i] for i in pick]
    return indices


def _compute_slice_axis_ticks(
    slice_indices: list[int],
    spacing: float,
    slice_count: int,
) -> tuple[list[float], list[str]]:
    if not slice_indices or slice_count <= 0:
        return [], []

    spacing_int = int(round(spacing))
    indices_set = set(slice_indices)
    if spacing_int > 0 and math.isclose(spacing, spacing_int, rel_tol=1e-6, abs_tol=1e-6):
        axis_max = (slice_count - 1) * spacing_int
        tick_vals = list(range(0, int(axis_max) + 1))
        tick_labels: list[str] = []
        for value in tick_vals:
            if value % spacing_int == 0:
                idx = value // spacing_int
                tick_labels.append(str(idx) if idx in indices_set else "")
            else:
                tick_labels.append("")
        return [float(value) for value in tick_vals], tick_labels

    tick_vals = [idx * spacing for idx in slice_indices]
    tick_labels = [str(idx) for idx in slice_indices]
    return tick_vals, tick_labels


def _compute_territory_winners(
    grid: torch.Tensor, n_ncas: int
) -> tuple[torch.Tensor, torch.Tensor]:
    aliveness = grid[: n_ncas + 1]  # [n_ncas + 1, depth, height, width]
    winners = torch.argmax(aliveness, dim=0)
    control_strength = torch.max(aliveness, dim=0)[0]
    return winners, control_strength


def compute_population_percent(grid: torch.Tensor, n_ncas: int) -> list[float]:
    """Compute occupied percent for each NCA based on territory winners."""
    if grid.dim() == 5:
        grid = grid[0]
    grid = _ensure_3d_grid(grid).detach().cpu()
    winners, _ = _compute_territory_winners(grid, n_ncas)
    counts = torch.bincount(winners.flatten(), minlength=n_ncas + 1).float()
    total = counts.sum().clamp(min=1.0)
    percents = (counts / total * 100.0).tolist()
    return percents[1:]


def _plotly_population_traces(
    population_history: list[list[float]],
    nca_colors: colors,
    showlegend: bool,
):
    import plotly.graph_objects as go

    if not population_history:
        return []

    history = np.array(population_history, dtype=float)
    if history.ndim == 1:
        history = history.reshape(1, -1)
    steps = list(range(history.shape[0]))
    traces = []
    for idx, color in enumerate(nca_colors):
        if idx >= history.shape[1]:
            break
        rgb = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
        traces.append(
            go.Scatter(
                x=steps,
                y=history[:, idx],
                mode="lines",
                name=f"NCA {idx}",
                line=dict(color=rgb, width=2),
                showlegend=showlegend,
            )
        )
    return traces


def create_territory_volume(
    grid: torch.Tensor, nca_colors: colors, sun_color: color = (0.0, 0.0, 0.0)
) -> torch.Tensor:
    """Convert grid to an RGBA volume.

    Args:
        grid: Grid tensor of shape [channels, depth, height, width] or [channels, height, width].
        nca_colors: List of (r, g, b) tuples for each NCA.
        sun_color: Color for the sun/background channel.

    Returns:
        RGBA volume tensor of shape [depth, height, width, 4].
    """
    grid = _ensure_3d_grid(grid).detach().clone().cpu()
    n_ncas = len(nca_colors)

    winners, control_strength = _compute_territory_winners(grid, n_ncas)

    color_lut = torch.tensor(
        [sun_color, *nca_colors], dtype=torch.float32
    )  # [n_ncas + 1, 3]
    rgba_volume = torch.zeros(
        winners.shape[0], winners.shape[1], winners.shape[2], 4
    )
    rgba_volume[..., :3] = color_lut[winners]
    rgba_volume[..., 3] = control_strength

    return rgba_volume


def create_territory_visualization(
    grid: torch.Tensor, nca_colors: colors
) -> torch.Tensor:
    """Convert grid to RGBA visualization (single slice).

    Args:
        grid: Grid tensor of shape [channels, height, width] or [channels, depth, height, width].
        nca_colors: List of (r, g, b) tuples for each NCA.

    Returns:
        RGBA visualization tensor of shape [4, height, width].
    """
    volume = create_territory_volume(grid, nca_colors)
    slice_2d = volume[0]
    return einops.rearrange(slice_2d, "h w c -> c h w")


def create_slice_montage(
    volume: torch.Tensor, stride: int = 1, max_slices: int | None = None
) -> torch.Tensor:
    """Create a 2D montage from a depth stack.

    Args:
        volume: RGBA volume [depth, height, width, 4].
        stride: Step between slices.
        max_slices: Optional maximum slices to include.

    Returns:
        Montage image [height * rows, width * cols, 4].
    """
    depth, height, width, _ = volume.shape
    indices = _get_slice_indices(depth, stride=stride, max_slices=max_slices)
    slices = volume[indices]
    n_slices = slices.shape[0]

    cols = math.ceil(math.sqrt(n_slices))
    rows = math.ceil(n_slices / cols)
    montage = torch.zeros(rows * height, cols * width, 4)

    for idx, slice_2d in enumerate(slices):
        row = idx // cols
        col = idx % cols
        montage[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
        ] = slice_2d

    return montage


def capture_snapshot(
    grid: torch.Tensor,
    nca_colors: colors,
    slice_stride: int = 1,
    max_slices: int | None = None,
) -> torch.Tensor:
    """Return a snapshot of the first grid in a batch.

    Args:
        grid: Batch of grids, uses first element if batched.
        nca_colors: Color mapping for visualization.
        slice_stride: Step between depth slices for montage.
        max_slices: Optional maximum slices to include in montage.

    Returns:
        Territory visualization as a 2D RGBA image [4, height, width].
    """
    if grid.dim() == 5:
        grid = grid[0]
    volume = create_territory_volume(grid, nca_colors)
    if volume.shape[0] == 1:
        image = volume[0]
    else:
        image = create_slice_montage(
            volume, stride=slice_stride, max_slices=max_slices
        )
    return einops.rearrange(image, "h w c -> c h w")


def _extract_slice(volume: np.ndarray | torch.Tensor, axis: str, idx: int):
    if axis == "depth":
        return volume[idx]
    if axis == "height":
        return volume[:, idx]
    if axis == "width":
        return volume[:, :, idx]
    raise ValueError(f"Unknown slice axis: {axis}")


def plot_slice_stack_matplotlib(
    ax,
    volume: torch.Tensor,
    axis: str = "depth",
    stride: int = 1,
    spacing: float = 1.2,
    max_slices: int | None = None,
    alpha: float = 0.9,
) -> None:
    """Render a slice stack in a Matplotlib 3D axis."""
    volume_np = volume.detach().cpu().numpy()
    depth, height, width, _ = volume_np.shape
    indices = _get_slice_indices(
        depth if axis == "depth" else height if axis == "height" else width,
        stride=stride,
        max_slices=max_slices,
    )

    ax.cla()
    for idx in indices:
        slice_rgba = _extract_slice(volume_np, axis, idx).copy()
        slice_rgba[..., 3] *= alpha

        if axis == "depth":
            h, w = slice_rgba.shape[:2]
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(X, idx * spacing, dtype=float)
        elif axis == "height":
            d, w = slice_rgba.shape[:2]
            x = np.arange(w)
            z = np.arange(d)
            X, Z = np.meshgrid(x, z)
            Y = np.full_like(X, idx * spacing, dtype=float)
        else:
            d, h = slice_rgba.shape[:2]
            y = np.arange(h)
            z = np.arange(d)
            Y, Z = np.meshgrid(y, z)
            X = np.full_like(Y, idx * spacing, dtype=float)

        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=slice_rgba,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

    x_max = (width - 1) * spacing if axis == "width" else width - 1
    y_max = (height - 1) * spacing if axis == "height" else height - 1
    z_max = (depth - 1) * spacing if axis == "depth" else depth - 1
    ax.set_xlim(0, max(0, x_max))
    ax.set_ylim(0, max(0, y_max))
    ax.set_zlim(0, max(0, z_max))
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.set_zlabel("depth")

    ax.set_box_aspect(
        [max(1.0, x_max + 1), max(1.0, y_max + 1), max(1.0, z_max + 1)]
    )

    slice_count = depth if axis == "depth" else height if axis == "height" else width
    tick_vals, tick_labels = _compute_slice_axis_ticks(indices, spacing, slice_count)
    if tick_vals:
        if axis == "depth":
            ax.set_zticks(tick_vals)
            ax.set_zticklabels(tick_labels)
        elif axis == "height":
            ax.set_yticks(tick_vals)
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_xticks(tick_vals)
            ax.set_xticklabels(tick_labels)


def build_plotly_slice_stack(
    grid: torch.Tensor,
    nca_colors: colors,
    axis: str = "depth",
    stride: int = 1,
    spacing: float = 1.2,
    max_slices: int | None = None,
    alpha: float = 0.9,
    population_history: list[list[float]] | None = None,
):
    """Build a Plotly figure with slice surfaces for interactive viewing."""
    import plotly.graph_objects as go

    grid = _ensure_3d_grid(grid).detach().cpu()
    winners, _ = _compute_territory_winners(grid, len(nca_colors))
    winners_np = winners.numpy()

    surfaces, extents, slice_ticks, slice_indices = _plotly_slice_surfaces(
        winners_np,
        nca_colors,
        axis=axis,
        stride=stride,
        spacing=spacing,
        max_slices=max_slices,
        alpha=alpha,
    )

    if population_history:
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            specs=[[{"type": "scene"}], [{"type": "xy"}]],
            row_heights=[0.8, 0.2],
            vertical_spacing=0.02,
        )
        for surface in surfaces:
            fig.add_trace(surface, row=1, col=1)
        pop_traces = _plotly_population_traces(
            population_history, nca_colors, showlegend=True
        )
        for trace in pop_traces:
            fig.add_trace(trace, row=2, col=1)
        fig.update_xaxes(
            title_text=None,
            title_standoff=0,
            row=2,
            col=1,
            range=[0, max(0, len(population_history) - 1)],
        )
        y_max = float(np.max(population_history)) if population_history else 100.0
        if y_max <= 0:
            y_max = 1.0
        y_max *= 1.02
        fig.update_yaxes(title_text="Occupied %", row=2, col=1, range=[0, y_max])
    else:
        fig = go.Figure(data=surfaces)
    _plotly_apply_layout(fig, extents, slice_axis=axis, slice_ticks=slice_ticks)
    _plotly_apply_slice_slider(fig, slice_indices)
    return fig


def build_plotly_slice_stack_animation(
    grids: list[torch.Tensor] | torch.Tensor,
    nca_colors: colors,
    axis: str = "depth",
    stride: int = 1,
    spacing: float = 1.2,
    max_slices: int | None = None,
    alpha: float = 0.9,
    frame_duration_ms: int = 160,
    population_history: list[list[float]] | None = None,
):
    """Build a Plotly figure with animated slice stacks."""
    import plotly.graph_objects as go

    grid_sequence = _normalize_grid_sequence(grids)
    frames = []
    extents = None
    slice_ticks = None
    slice_indices: list[int] | None = None
    has_populations = bool(population_history)

    for idx, grid in enumerate(grid_sequence):
        grid = _ensure_3d_grid(grid).detach().cpu()
        winners, _ = _compute_territory_winners(grid, len(nca_colors))
        winners_np = winners.numpy()
        surfaces, extents, slice_ticks, frame_indices = _plotly_slice_surfaces(
            winners_np,
            nca_colors,
            axis=axis,
            stride=stride,
            spacing=spacing,
            max_slices=max_slices,
            alpha=None,
        )
        if slice_indices is None:
            slice_indices = frame_indices
        if has_populations and population_history:
            frame_traces = surfaces + _plotly_population_traces(
                population_history[: idx + 1],
                nca_colors,
                showlegend=False,
            )
        else:
            frame_traces = surfaces
        frames.append(go.Frame(data=frame_traces, name=str(idx)))

    if not frames:
        return go.Figure()

    base_grid = _ensure_3d_grid(grid_sequence[0]).detach().cpu()
    winners, _ = _compute_territory_winners(base_grid, len(nca_colors))
    winners_np = winners.numpy()
    base_surfaces, extents, slice_ticks, slice_indices = _plotly_slice_surfaces(
        winners_np,
        nca_colors,
        axis=axis,
        stride=stride,
        spacing=spacing,
        max_slices=max_slices,
        alpha=alpha,
    )

    if has_populations and population_history:
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            specs=[[{"type": "scene"}], [{"type": "xy"}]],
            row_heights=[0.8, 0.2],
            vertical_spacing=0.02,
        )
        for surface in base_surfaces:
            fig.add_trace(surface, row=1, col=1)
        base_pop_traces = _plotly_population_traces(
            population_history[:1], nca_colors, showlegend=True
        )
        for trace in base_pop_traces:
            fig.add_trace(trace, row=2, col=1)
        fig.update_xaxes(
            title_text=None,
            title_standoff=0,
            row=2,
            col=1,
            range=[0, max(0, len(population_history) - 1)],
        )
        y_max = float(np.max(population_history)) if population_history else 100.0
        if y_max <= 0:
            y_max = 1.0
        y_max *= 1.02
        fig.update_yaxes(title_text="Occupied %", row=2, col=1, range=[0, y_max])
        fig.frames = frames
    else:
        fig = go.Figure(data=base_surfaces, frames=frames)

    _plotly_apply_layout(fig, extents, slice_axis=axis, slice_ticks=slice_ticks)
    _plotly_apply_animation_controls(fig, frame_duration_ms)
    if slice_indices is not None:
        _plotly_apply_slice_slider(fig, slice_indices)
    return fig


def _normalize_grid_sequence(
    grids: list[torch.Tensor] | torch.Tensor,
) -> list[torch.Tensor]:
    if isinstance(grids, list):
        return grids
    if not isinstance(grids, torch.Tensor):
        raise ValueError("Expected grids as list[Tensor] or Tensor")

    if grids.dim() == 6:
        return [grids[t, 0] for t in range(grids.shape[0])]
    if grids.dim() == 5:
        return [grids[t] for t in range(grids.shape[0])]
    if grids.dim() == 4:
        return [grids]
    raise ValueError(f"Unsupported grids tensor dim: {grids.dim()}")


def _plotly_colorscale(nca_colors: colors) -> list[list[float | str]]:
    colorscale = []
    for i, color in enumerate([(0.0, 0.0, 0.0), *nca_colors]):
        pos = i / max(1, len(nca_colors))
        colorscale.append(
            [pos, f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"]
        )
    return colorscale


def _plotly_slice_surfaces(
    winners_np: np.ndarray,
    nca_colors: colors,
    axis: str,
    stride: int,
    spacing: float,
    max_slices: int | None,
    alpha: float | None,
):
    import plotly.graph_objects as go

    depth, height, width = winners_np.shape
    slice_count = depth if axis == "depth" else height if axis == "height" else width
    indices = _get_slice_indices(slice_count, stride=stride, max_slices=max_slices)
    slice_ticks = _compute_slice_axis_ticks(indices, spacing, slice_count)

    colorscale = _plotly_colorscale(nca_colors)
    surfaces = []

    for idx in indices:
        slice_vals = _extract_slice(winners_np, axis, idx)

        if axis == "depth":
            h, w = slice_vals.shape
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(X, idx * spacing, dtype=float)
        elif axis == "height":
            d, w = slice_vals.shape
            x = np.arange(w)
            z = np.arange(d)
            X, Z = np.meshgrid(x, z)
            Y = np.full_like(X, idx * spacing, dtype=float)
        else:
            d, h = slice_vals.shape
            y = np.arange(h)
            z = np.arange(d)
            Y, Z = np.meshgrid(y, z)
            X = np.full_like(Y, idx * spacing, dtype=float)

        surface_kwargs = dict(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=slice_vals,
            cmin=0,
            cmax=len(nca_colors),
            colorscale=colorscale,
            showscale=False,
        )
        if alpha is not None:
            surface_kwargs["opacity"] = alpha
        surfaces.append(go.Surface(**surface_kwargs))

    x_max = (width - 1) * spacing if axis == "width" else width - 1
    y_max = (height - 1) * spacing if axis == "height" else height - 1
    z_max = (depth - 1) * spacing if axis == "depth" else depth - 1

    return surfaces, (x_max, y_max, z_max), slice_ticks, indices


def _plotly_apply_layout(
    fig,
    extents: tuple[float, float, float] | None,
    slice_axis: str | None = None,
    slice_ticks: tuple[list[float], list[str]] | None = None,
) -> None:
    if extents is None:
        extents = (0, 0, 0)
    x_max, y_max, z_max = extents
    x_extent = max(0.0, x_max) + 1.0
    y_extent = max(0.0, y_max) + 1.0
    z_extent = max(0.0, z_max) + 1.0
    max_extent = max(x_extent, y_extent, z_extent, 1.0)
    scene = dict(
        xaxis=dict(title="width", range=[0, max(0, x_max)], autorange=False),
        yaxis=dict(title="height", range=[0, max(0, y_max)], autorange=False),
        zaxis=dict(title="depth", range=[0, max(0, z_max)], autorange=False),
        aspectmode="manual",
        aspectratio=dict(
            x=x_extent / max_extent,
            y=y_extent / max_extent,
            z=z_extent / max_extent,
        ),
        uirevision="slice-view",
        camera=dict(eye=dict(x=1.9, y=1.9, z=1.4)),
    )
    if slice_axis and slice_ticks and slice_ticks[0]:
        axis_key = {"width": "xaxis", "height": "yaxis", "depth": "zaxis"}[slice_axis]
        scene[axis_key].update(
            dict(tickmode="array", tickvals=slice_ticks[0], ticktext=slice_ticks[1])
        )
    fig.update_layout(
        scene=scene,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="slice-view",
    )


def _plotly_apply_animation_controls(fig, frame_duration_ms: int) -> None:
    loop_mode = "immediate"
    speed_options = [0.25, 0.5, 1.0, 2.0]
    speed_buttons = [
        dict(
            label=f"{speed}x",
            method="animate",
            args=[
                None,
                {
                    "frame": {"duration": max(10, int(frame_duration_ms / speed)), "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0},
                    "mode": loop_mode,
                },
            ],
        )
        for speed in speed_options
    ]
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "mode": loop_mode,
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            ),
            dict(
                type="buttons",
                showactive=True,
                x=0.0,
                y=1.12,
                direction="right",
                buttons=speed_buttons,
            ),
        ],
        sliders=[
            dict(
                active=0,
                y=-0.015,
                yanchor="top",
                x=0.1,
                len=0.9,
                currentvalue=dict(prefix="Step: "),
                steps=[
                    dict(
                        method="animate",
                        label=frame.name,
                        args=[
                            [frame.name],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    )
                    for frame in fig.frames
                ],
            )
        ],
        margin=dict(l=0, r=0, t=0, b=90),
    )


def _plotly_apply_slice_slider(fig, slice_indices: list[int]) -> None:
    if len(slice_indices) <= 1:
        return

    steps = []
    n_traces = len(slice_indices)
    total_traces = len(fig.data)
    base_opacities = []
    for trace in fig.data:
        opacity = getattr(trace, "opacity", None)
        base_opacities.append(1.0 if opacity is None else opacity)
    for idx in range(min(n_traces, total_traces)):
        fig.data[idx].visible = True
    initial_slice = slice_indices[-1]
    for step_idx, slice_idx in reversed(list(enumerate(slice_indices))):
        opacities = list(base_opacities)
        for idx in range(step_idx + 1, n_traces):
            if idx < total_traces:
                opacities[idx] = 0.0
        steps.append(
            dict(
                method="restyle",
                label=str(slice_idx),
                args=["opacity", opacities],
            )
        )

    slider = dict(
        active=0,
        y=-0.13,
        yanchor="top",
        x=0.1,
        len=0.9,
        currentvalue=dict(prefix="Slice end: "),
        steps=steps,
    )

    existing = list(getattr(fig.layout, "sliders", []) or [])
    existing.append(slider)
    fig.update_layout(
        sliders=existing,
        margin=dict(l=0, r=0, t=0, b=90),
    )


def get_int_grid(grid: torch.Tensor) -> np.ndarray:
    grid = _ensure_3d_grid(grid)
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
    grid = _ensure_3d_grid(grid)
    C, _, _, _ = grid.shape

    uint8_grid = get_int_grid(grid)

    original_sizes = [uint8_grid[i].nbytes for i in range(C)]

    def compress_png(channel):
        buffer = io.BytesIO()
        Image.fromarray(channel, mode='L').save(buffer, format='PNG', optimize=True, compress_level=9)
        return len(buffer.getvalue())

    if img_mode:
        if uint8_grid.ndim == 4 and uint8_grid.shape[1] == 1:
            uint8_grid = uint8_grid[:, 0]
        elif uint8_grid.ndim == 4:
            img_mode = False

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
