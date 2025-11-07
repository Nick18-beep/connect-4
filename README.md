# Connect-4 Gumbel AlphaZero

Implementation of a Connect‑4 agent trained with a Gumbel Top‑k variant of AlphaZero. It includes:

- Bitboard Connect‑4 environment with fast numba win detection.
- PyTorch policy‑value network with residual + SE blocks.
- Gumbel Top‑k MCTS (batchable, Dirichlet noise, virtual loss, threat detection caches).
- Self‑play trainer with prioritized replay, EMA weights, optional reanalysis workers.
- CLI utilities for training, inference, and evaluation vs. random/minimax.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (typical dependencies: `numpy`, `torch`, `numba`, `pytest`, etc.)

## Quick Start

```bash
# 1. Create environment
python -m venv .venv
. .venv/Scripts/activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
python -m pytest tests

# 4. Train (example)
python Gumbel\ Aplha\ Zero/train.py --sims 800 --episodes 100000 --use_gumbel_actions --dynamic_gumbel_k

# 5. Evaluate/infer
python Gumbel\ Aplha\ Zero/eval.py --load c4_advanced.pt --eval_games 20
python Gumbel\ Aplha\ Zero/infer.py --load c4_advanced.pt --sims 2000 --human_first
```

Adjust `--num_workers`, `--gumbel_scale`, checkpoints, and device flags as needed. For inference or evaluation, point `--load` to the trained `.pt` file.
