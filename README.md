# Petri Dish NCA

What happens if we are able to put multiple different NCAs in a single substrate that compete for space?

## Setup 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh         # Or install uv another way

git clone https://github.com/SakanaAI/petri-dish-nca
cd petri-dish-nca
uv sync
```

## Commands

- basic: `uv run python src/train.py --n-ncas 3 --epochs 1000 --device cpu`
- wandb logging: `uv run python src/train.py --n-ncas 3 --epochs 10000 --device cuda --wandb`
- run with config: `uv run python src/train.py --config configs/example.json`
- live viz training: `uv run python src/train.py --n-ncas 3 --epochs 1000 --device cpu --live-viz`
- 3D example: `uv run python src/train.py --n-ncas 3 --epochs 200 --device cpu --live-viz --viz-slice-axis depth`
- 3D grid size via CLI: `uv run python src/train.py --n-ncas 3 --epochs 200 --grid-size 10 10 10`
- 3D view with grid override: `uv run python src/visualize_trained.py --model-path <run_dir> --grid-size 10 10 10`
- 3D plotly view with mouse controls + playback: `uv run python src/visualize_trained.py --model-path <run_dir> --plotly`
- Plotly playback speed + loop: `uv run python src/visualize_trained.py --model-path <run_dir> --plotly --plotly-speed 0.5 --plotly-loop`

3D visualization options (viz-only) include:
`--viz-slice-axis`, `--viz-slice-stride`, `--viz-slice-spacing`, `--viz-slice-alpha`, `--viz-max-slices`
for training, and `--slice-axis`, `--slice-stride`, `--slice-spacing`, `--slice-alpha`, `--max-slices`
for `visualize_trained.py`.

## Configs

For additional configurations, you can load a JSON config file. Any parameters not specified in the config file will be set their default value in `src/config.py`.
`grid_size` is `(D, H, W)`; 2D configs `(H, W)` are still accepted and promoted to `(1, H, W)`.
