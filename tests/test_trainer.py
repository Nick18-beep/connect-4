from __future__ import annotations

import numpy as np
import pytest

from trainer import ReplayBuffer, SumTree, Sample, Trainer
from model import PolicyValueNet
import torch


def make_sample(idx: int) -> Sample:
    obs = np.zeros((4, 6, 7), dtype=np.float32)
    pi = np.zeros(7, dtype=np.float32)
    pi[idx % 7] = 1.0
    z = float((idx % 2) * 2 - 1)
    return Sample(obs=obs, pi=pi, z=z)


def test_sumtree_updates_total_priority():
    tree = SumTree(capacity=4)
    for i in range(4):
        tree.push(priority=i + 1, data=make_sample(i))
    assert tree.total_priority == float(sum(range(1, 5)))
    leaf_idx, priority, _ = tree.get_leaf(tree.total_priority * 0.5)
    assert priority > 0.0
    tree.update(leaf_idx, 10.0)
    assert tree.total_priority == float(sum(range(1, 5)) - priority + 10.0)


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=32)
    for i in range(20):
        buffer.push(make_sample(i))

    idxs, obs, pi, z, w = buffer.sample(batch_size=8)
    assert len(idxs) == 8
    assert obs.shape == (8, 4, 6, 7)
    assert pi.shape == (8, 7)
    assert z.shape == (8,)
    assert w.shape == (8,)
    assert np.all(w > 0)


def test_replay_buffer_sample_empty_returns_zero_batch():
    buffer = ReplayBuffer(capacity=8)
    idxs, obs, pi, z, w = buffer.sample(batch_size=4)
    assert len(idxs) == 0
    assert obs.shape == (0, 4, 6, 7)
    assert pi.shape == (0, 7)
    assert z.shape == (0,)
    assert w.shape == (0,)


def test_replay_buffer_overwrites_old_entries():
    buffer = ReplayBuffer(capacity=3)
    for i in range(5):
        buffer.push(make_sample(i))
    stored = {slot.pi.argmax() for slot in buffer.tree.data if slot is not None}
    assert stored == {2, 3, 4}
    assert len(buffer) == 3


def test_batch_augment_horizontal_flip(monkeypatch):
    obs = np.arange(4 * 6 * 7, dtype=np.float32).reshape(1, 4, 6, 7)
    pi = np.arange(7, dtype=np.float32).reshape(1, 7)
    z = np.array([1.0], dtype=np.float32)

    calls = iter(
        [
            np.array([0.0], dtype=np.float64),  # trigger flip
            np.array([1.0], dtype=np.float64),  # skip color invert
        ]
    )

    def fake_rand(size):
        return next(calls)

    monkeypatch.setattr(np.random, "rand", fake_rand)

    obs_aug, pi_aug, z_aug = Trainer._batch_augment_in_learner(None, obs.copy(), pi.copy(), z.copy())
    expected_obs = obs[:, :, :, ::-1]
    expected_pi = pi[:, ::-1]

    assert np.allclose(obs_aug, expected_obs)
    assert np.allclose(pi_aug, expected_pi)
    assert np.allclose(z_aug, z)


def test_trainer_forward_pass_consistency():
    trainer = Trainer(device="cpu", sims=1, gumbel_scale=0.0, updates_per_iter=1, compile_model=False)
    batch = 8
    obs = np.random.randn(batch, 4, 6, 7).astype(np.float32)
    pi = np.full((batch, 7), 1.0 / 7, dtype=np.float32)
    z = np.linspace(-1, 1, num=batch, dtype=np.float32)

    trainer.buffer.push(Sample(obs[0], pi[0], z[0]))
    idxs, obs_b, pi_b, z_b, w = trainer.buffer.sample(batch_size=1)
    assert obs_b.shape == (1, 4, 6, 7)

    model = PolicyValueNet()
    logits, values = model(torch.from_numpy(obs))
    assert logits.shape == (batch, 7)
    assert values.shape == (batch,)
    assert torch.all(torch.abs(values) <= 1.0 + 1e-6)
