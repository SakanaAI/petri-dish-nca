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

basic: `uv run python src/train.py --n-ncas 3 --epochs 1000 --device cpu`
wandb logging: `uv run python src/train.py --n-ncas 3 --epochs 10000 --device cuda --wandb`

## Configs

For additional configurations, you can load a JSON config file.