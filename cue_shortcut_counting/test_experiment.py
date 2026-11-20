"""Correctness and methodological checks for the GRU cue experiment."""

import math

import pytest
import torch
from torch import nn

from experiment import (
    CountingGRU,
    binary_entropy_nats,
    evaluate,
    fit_pre_cue_probe,
    flip_cue,
    run_single,
    sample_batch,
)


def test_exact_content_counts():
    x, y = sample_batch(512, generator=torch.Generator().manual_seed(1))
    content = x[:, :64]
    counts = content.sum(dim=1)
    assert x.shape == (512, 65)
    assert torch.all(counts[y == 0] == 31)
    assert torch.all(counts[y == 1] == 33)
    assert set(content.unique().tolist()).issubset({0, 1})
    assert set(x[:, -1].unique().tolist()).issubset({2, 3})


def test_flip_cue_preserves_content_and_inverts_cue():
    x, _ = sample_batch(128)
    changed = flip_cue(x)
    assert torch.equal(x[:, :64], changed[:, :64])
    assert torch.all(x[:, -1] != changed[:, -1])
    assert torch.equal(flip_cue(changed), x)


def test_cue_probability_distribution():
    generator = torch.Generator().manual_seed(0)
    x, y = sample_batch(20_000, cue_prob=0.95, generator=generator)
    agree_rate = ((x[:, -1] - 2) == y).float().mean().item()
    assert abs(agree_rate - 0.95) < 0.01


def test_hidden_state_alignment_and_cue_invariance():
    model = CountingGRU().eval()
    x, _ = sample_batch(16)
    logits, states = model(x, return_states=True)
    _, flipped_states = model(flip_cue(x), return_states=True)
    assert states.shape == (16, 65, 24)
    assert torch.allclose(logits, model.head(states[:, -1, :]), atol=1e-6)
    assert torch.allclose(states[:, -2, :], flipped_states[:, -2, :], atol=1e-6)


def test_binary_entropy_floor():
    assert math.isclose(binary_entropy_nats(0.95), 0.19851524, rel_tol=1e-5)
    assert math.isclose(binary_entropy_nats(0.50), math.log(2.0), rel_tol=1e-6)
    assert binary_entropy_nats(0) == binary_entropy_nats(1) == 0


@pytest.mark.parametrize("p", [-0.01, 1.01, float("nan")])
def test_invalid_cue_probabilities(p):
    with pytest.raises(ValueError):
        sample_batch(2, cue_prob=p)
    with pytest.raises(ValueError):
        binary_entropy_nats(p)


@pytest.mark.parametrize(
    "arguments",
    [
        {"n": 0},
        {"n": 2, "seq_len": 32},
        {"n": 2, "count_0": 33, "count_1": 31},
        {"n": 2, "count_0": -1},
    ],
)
def test_invalid_data_configurations(arguments):
    with pytest.raises(ValueError):
        sample_batch(**arguments)


def test_independent_generators_preserve_training_batches():
    train_rng = torch.Generator().manual_seed(7)
    expected_first = sample_batch(8, generator=train_rng)
    expected_second = sample_batch(8, generator=train_rng)

    train_rng = torch.Generator().manual_seed(7)
    actual_first = sample_batch(8, generator=train_rng)
    sample_batch(32, generator=torch.Generator().manual_seed(10_000))
    actual_second = sample_batch(8, generator=train_rng)

    for expected, actual in zip(expected_first + expected_second, actual_first + actual_second):
        assert torch.equal(expected, actual)


class RuleModel(nn.Module):
    """Diagnostic classifier using either the cue or the exact content count."""

    def __init__(self, cue_only: bool):
        super().__init__()
        self.cue_only = cue_only

    def forward(self, x: torch.Tensor, return_states: bool = False):
        if self.cue_only:
            labels = x[:, -1] - 2
            confidence = math.log(0.95 / 0.05)
        else:
            labels = (x[:, :-1].sum(dim=1) == 33).long()
            confidence = 20.0
        logits = nn.functional.one_hot(labels, 2).float() * confidence
        if return_states:
            states = torch.zeros(x.shape[0], x.shape[1], 24, device=x.device)
            if not self.cue_only:
                states[:, -2, 0] = x[:, :-1].sum(dim=1).float() - 32.0
            return logits, states
        return logits


def test_evaluation_distinguishes_cue_and_count_rules():
    generator = torch.Generator().manual_seed(100)
    eval_data = sample_batch(2048, generator=generator)
    probe_train = sample_batch(256, generator=generator)
    probe_test = sample_batch(256, generator=generator)
    cue_agrees = ((eval_data[0][:, -1] - 2) == eval_data[1]).float().mean().item()

    cue_metrics = evaluate(RuleModel(True), 0, 1, 0.0, eval_data, probe_train, probe_test)
    assert math.isclose(cue_metrics.original_acc, cue_agrees, abs_tol=1e-6)
    assert math.isclose(cue_metrics.flipped_acc, 1.0 - cue_agrees, abs_tol=1e-6)
    assert cue_metrics.flip_rate == 1.0
    assert cue_metrics.cue_agree_acc == 1.0
    assert cue_metrics.cue_disagree_acc == 0.0
    assert abs(cue_metrics.eval_loss - binary_entropy_nats(0.95)) < 0.05
    assert 0.35 <= cue_metrics.pre_cue_probe_acc <= 0.65

    count_metrics = evaluate(RuleModel(False), 0, 1, 0.0, eval_data, probe_train, probe_test)
    assert count_metrics.original_acc == count_metrics.flipped_acc == 1.0
    assert count_metrics.flip_rate == 0.0
    assert count_metrics.cue_agree_acc == count_metrics.cue_disagree_acc == 1.0
    assert count_metrics.pre_cue_probe_acc == 1.0


@pytest.mark.parametrize("ridge", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_probe_ridge(ridge):
    model = CountingGRU()
    data = sample_batch(4)
    with pytest.raises(ValueError, match="ridge"):
        fit_pre_cue_probe(model, data, data, ridge=ridge)


@pytest.mark.parametrize("steps,eval_steps", [(2, {3}), (2, set()), (0, {1})])
def test_rejects_unreachable_checkpoints(steps, eval_steps):
    with pytest.raises(ValueError, match="eval_steps"):
        run_single(seed=0, steps=steps, eval_steps=eval_steps, cue_prob=0.95)


def test_extra_evaluation_does_not_change_later_checkpoint():
    settings = dict(seed=3, steps=3, cue_prob=0.95, batch_size=8, n_eval=256, n_probe=32)
    with_early_eval = run_single(eval_steps={1, 3}, **settings)
    without_early_eval = run_single(eval_steps={3}, **settings)
    assert with_early_eval[-1] == without_early_eval[0]
    assert with_early_eval[-1].seed == 3
    assert 0.0 <= with_early_eval[-1].pre_cue_probe_acc <= 1.0
