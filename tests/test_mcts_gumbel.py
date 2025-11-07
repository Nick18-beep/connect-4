from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from connect4 import C4State
from mcts_gumbel import GumbelMCTS


class DummyModel(nn.Module):
    """Minimal policy-value network with deterministic outputs."""

    def __init__(self, num_actions: int = 7):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, x):  # type: ignore[override]
        batch = x.shape[0]
        logits = torch.zeros((batch, self.num_actions), dtype=torch.float32, device=x.device)
        values = torch.zeros((batch, 1), dtype=torch.float32, device=x.device)
        return logits, values


def test_mcts_policy_supports_only_legal_moves():
    model = DummyModel()
    mcts = GumbelMCTS(
        model=model,
        num_simulations=8,
        batch_size=4,
        gumbel_scale=0.0,
        root_dirichlet_alpha=None,
        device="cpu",
    )

    state = C4State.initial()
    # Fill column 0 so it becomes illegal
    for _ in range(6):
        state = state.apply(0)

    pi, value = mcts.run(state, temperature=1.0)
    assert np.isclose(pi.sum(), 1.0, atol=1e-6)
    assert pi[0] == 0.0  # column 0 is full
    legal = state.legal_actions()
    assert all(pi[a] >= 0.0 for a in legal)
    assert any(pi[a] > 0 for a in legal)
    assert value == pytest.approx(0.0, abs=1e-6)


def _play_sequence(seq):
    state = C4State.initial()
    for move in seq:
        state = state.apply(move)
    return state


def test_mcts_fast_path_wins_immediately():
    state = _play_sequence([0, 1, 0, 1, 0, 2])  # +1 to move with 3 in column 0
    model = DummyModel()
    mcts = GumbelMCTS(model=model, num_simulations=4, batch_size=2, gumbel_scale=0.0, device="cpu")

    pi, value = mcts.run(state, temperature=1.0)
    assert np.argmax(pi) == 0
    assert pi[0] == pytest.approx(1.0)
    assert value == pytest.approx(1.0)


def test_mcts_blocks_unique_opponent_threat():
    state = _play_sequence([0, 0, 1, 0, 1, 0])  # Opponent (-1) threatens column 0 only
    model = DummyModel()
    mcts = GumbelMCTS(model=model, num_simulations=4, batch_size=2, gumbel_scale=0.0, device="cpu")

    pi, value = mcts.run(state, temperature=1.0)
    assert np.argmax(pi) == 0
    assert pi[0] == pytest.approx(1.0)
    assert value == pytest.approx(0.0)


def test_mcts_handles_no_legal_moves():
    state = _play_sequence([0, 1, 0, 1, 0, 1, 2, 1, 2, 1, 3])
    state = state.apply(1)  # opponent just won
    model = DummyModel()
    mcts = GumbelMCTS(model=model, num_simulations=2, batch_size=1, gumbel_scale=0.0, device="cpu")
    pi, value = mcts.run(state, temperature=1.0)
    assert np.isclose(pi.sum(), 1.0)
    assert abs(value) == pytest.approx(1.0)
