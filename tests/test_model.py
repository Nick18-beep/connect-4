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
