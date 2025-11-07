from __future__ import annotations

import torch

from model import PolicyValueNet


@torch.no_grad()
def test_policy_value_net_shapes_and_ranges():
    model = PolicyValueNet(channels=32, n_blocks=2)
    x = torch.randn(4, 4, 6, 7, dtype=torch.float32)
    logits, values = model(x)
    assert logits.shape == (4, 7)
    assert values.shape == (4,)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.abs(values) <= 1.0 + 1e-6)


@torch.no_grad()
def test_policy_head_respects_column_bias():
    model = PolicyValueNet(channels=32, n_blocks=2)
    empty = torch.zeros(1, 4, 6, 7)
    full_center = empty.clone()
    full_center[:, 0, :, 3] = 1  # occupy center column
    logits_empty, _ = model(empty)
    logits_center, _ = model(full_center)
    # Column occupancy should affect logits (bias not zero)
    diff = logits_center - logits_empty
    assert torch.any(torch.abs(diff) > 1e-6)
