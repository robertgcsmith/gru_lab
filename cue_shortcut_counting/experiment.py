"""Test whether a GRU uses a predictive cue instead of the true token counts.

The task distinguishes sequences containing 31 versus 33 As among 64 content
symbols. An additional final token indicates the correct class with probability
0.95. Cue interventions and a pre-cue linear probe track what the model uses.
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass

import torch
from torch import nn


Batch = tuple[torch.Tensor, torch.Tensor]


def sample_batch(
    n: int,
    cue_prob: float = 0.95,
    seq_len: int = 64,
    count_0: int = 31,
    count_1: int = 33,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Batch:
    """Sample labelled content strings and a potentially misleading final cue.

    Content tokens are 0 (B) and 1 (A); cue tokens are 2 (class 0) and
    3 (class 1). Only the content determines the ground-truth label.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if not 0.0 <= cue_prob <= 1.0:
        raise ValueError("cue_prob must be between 0 and 1")
    if not 0 <= count_0 < count_1 <= seq_len:
        raise ValueError("counts must satisfy 0 <= count_0 < count_1 <= seq_len")

    y = torch.randint(0, 2, (n,), device=device, generator=generator)
    count_a = count_0 + (count_1 - count_0) * y

    # Sorting random ranks selects exactly count_a uniformly placed A tokens.
    ranks = torch.rand(n, seq_len, device=device, generator=generator).argsort(dim=1)
    content = (ranks < count_a[:, None]).long()

    agrees = torch.rand(n, device=device, generator=generator) < cue_prob
    cue_label = torch.where(agrees, y, 1 - y)
    x = torch.cat([content, (2 + cue_label)[:, None]], dim=1)
    return x, y


def flip_cue(x: torch.Tensor) -> torch.Tensor:
    """Flip only the final cue (2 <-> 3), preserving the content and label."""
    changed = x.clone()
    changed[:, -1] = 5 - changed[:, -1]
    return changed


def binary_entropy_nats(p: float) -> float:
    """Population cross-entropy of an optimal cue-only predictor."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


class CountingGRU(nn.Module):
    def __init__(self, vocab_size: int = 4, embed_dim: int = 8, hidden_dim: int = 24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(
        self, x: torch.Tensor, return_states: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        outputs, hidden = self.encoder(self.embedding(x))
        logits = self.head(hidden[-1])
        if return_states:
            return logits, outputs
        return logits


@dataclass
class EvalMetrics:
    seed: int
    step: int
    train_loss: float  # Mean minibatch loss over the last <= 50 training steps.
    eval_loss: float
    original_acc: float
    flipped_acc: float
    flip_rate: float
    cue_agree_acc: float
    cue_disagree_acc: float
    pre_cue_probe_acc: float


@torch.no_grad()
def fit_pre_cue_probe(
    model: CountingGRU,
    train_data: Batch,
    test_data: Batch,
    ridge: float = 1e-2,
) -> float:
    """Measure linear decodability of the class from h_64, not exact counting.

    The probe fits on frozen states from one dataset and scores on another.
    Neither dataset is used to update the GRU.
    """
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be positive")
    model.eval()
    x_tr, y_tr = train_data
    x_te, y_te = test_data
    _, states_tr = model(x_tr, return_states=True)
    _, states_te = model(x_te, return_states=True)

    # The cue is token 65: -2 denotes the state after content token 64.
    h_tr = states_tr[:, -2, :]
    h_te = states_te[:, -2, :]

    X_tr = torch.cat([h_tr, torch.ones_like(h_tr[:, :1])], dim=1)
    X_te = torch.cat([h_te, torch.ones_like(h_te[:, :1])], dim=1)
    targets = (2 * y_tr - 1).to(dtype=X_tr.dtype)

    reg = ridge * torch.eye(X_tr.shape[1], device=X_tr.device, dtype=X_tr.dtype)
    reg[-1, -1] = 0.0  # Leave the intercept unregularised.
    weights = torch.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ targets)
    predictions = (X_te @ weights >= 0).long()
    return (predictions == y_te).float().mean().item()


@torch.no_grad()
def evaluate(
    model: CountingGRU,
    seed: int,
    step: int,
    train_loss: float,
    eval_data: Batch,
    probe_train_data: Batch,
    probe_test_data: Batch,
) -> EvalMetrics:
    """Evaluate the same held-out examples at every training checkpoint."""
    model.eval()
    x, y = eval_data
    original_logits = model(x)
    original_predictions = original_logits.argmax(dim=1)
    flipped_predictions = model(flip_cue(x)).argmax(dim=1)

    cue_agrees = (x[:, -1] - 2) == y
    cue_disagrees = ~cue_agrees

    def conditional_accuracy(mask: torch.Tensor) -> float:
        if not mask.any():
            return float("nan")
        return (original_predictions[mask] == y[mask]).float().mean().item()

    return EvalMetrics(
        seed=seed,
        step=step,
        train_loss=train_loss,
        eval_loss=nn.functional.cross_entropy(original_logits, y).item(),
        original_acc=(original_predictions == y).float().mean().item(),
        flipped_acc=(flipped_predictions == y).float().mean().item(),
        flip_rate=(original_predictions != flipped_predictions).float().mean().item(),
        cue_agree_acc=conditional_accuracy(cue_agrees),
        cue_disagree_acc=conditional_accuracy(cue_disagrees),
        pre_cue_probe_acc=fit_pre_cue_probe(model, probe_train_data, probe_test_data),
    )


def run_single(
    seed: int,
    steps: int,
    eval_steps: set[int],
    cue_prob: float,
    batch_size: int = 128,
    lr: float = 2e-3,
    hidden_dim: int = 24,
    device: torch.device | str = "cpu",
    n_eval: int = 4096,
    n_probe: int = 2048,
) -> list[EvalMetrics]:
    """Train one model; hold evaluation/probe sets constant across seeds and steps."""
    if steps < 1 or not eval_steps or any(not 1 <= s <= steps for s in eval_steps):
        raise ValueError("eval_steps must be nonempty and within 1..steps")
    if batch_size < 1 or n_eval < 1 or n_probe < 1 or hidden_dim < 1:
        raise ValueError("batch sizes and hidden_dim must be positive")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not 0.0 <= cue_prob <= 1.0:
        raise ValueError("cue_prob must be between 0 and 1")

    torch.manual_seed(seed)
    model = CountingGRU(hidden_dim=hidden_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Separate streams prevent evaluation from changing future training batches.
    train_rng = torch.Generator(device=device).manual_seed(seed)
    eval_rng = torch.Generator(device=device).manual_seed(10_000)
    eval_data = sample_batch(n_eval, cue_prob=cue_prob, device=device, generator=eval_rng)
    probe_train = sample_batch(n_probe, cue_prob=cue_prob, device=device, generator=eval_rng)
    probe_test = sample_batch(n_probe, cue_prob=cue_prob, device=device, generator=eval_rng)

    history: list[EvalMetrics] = []
    recent_losses: deque[float] = deque(maxlen=50)
    for step in range(1, steps + 1):
        model.train()
        x, y = sample_batch(batch_size, cue_prob=cue_prob, device=device, generator=train_rng)
        loss = nn.functional.cross_entropy(model(x), y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        recent_losses.append(loss.item())

        if step in eval_steps:
            history.append(
                evaluate(
                    model=model,
                    seed=seed,
                    step=step,
                    train_loss=sum(recent_losses) / len(recent_losses),
                    eval_data=eval_data,
                    probe_train_data=probe_train,
                    probe_test_data=probe_test,
                )
            )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU cue-shortcut counting experiment")
    parser.add_argument("--cue-prob", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[240, 500, 1000, 2000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.steps < 1 or any(not 1 <= s <= args.steps for s in args.eval_steps):
        parser.error("--eval-steps must fall within 1..--steps")
    if len(set(args.eval_steps)) != len(args.eval_steps):
        parser.error("--eval-steps must not contain duplicates")

    print(
        f"Cue reliability: {args.cue_prob:.2f}; "
        f"cue-only population loss: {binary_entropy_nats(args.cue_prob):.4f} nats; "
        f"seeds: {args.seeds}"
    )
    all_runs = [
        run_single(
            seed=seed,
            steps=args.steps,
            eval_steps=set(args.eval_steps),
            cue_prob=args.cue_prob,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )
        for seed in args.seeds
    ]

    print("\nPer-seed checkpoints (train loss = preceding 50 minibatches or fewer):")
    print(" seed  step  train_loss  eval_loss  orig_acc  flip_acc  flip_rate  disagree  h64_probe")
    for run in all_runs:
        for metric in run:
            print(
                f"{metric.seed:>5} {metric.step:>5} {metric.train_loss:>11.4f} "
                f"{metric.eval_loss:>10.4f} {metric.original_acc:>9.3f} "
                f"{metric.flipped_acc:>9.3f} {metric.flip_rate:>10.3f} "
                f"{metric.cue_disagree_acc:>9.3f} {metric.pre_cue_probe_acc:>10.3f}"
            )

    print("\nAcross seeds (mean±sample SD):")
    print(" step   train_loss   eval_loss   orig_acc   flip_acc   flip_rate   disagree   h64_probe")
    for i, step in enumerate(sorted(set(args.eval_steps))):
        metrics = [run[i] for run in all_runs]

        def summary(field: str) -> str:
            values = torch.tensor([getattr(m, field) for m in metrics])
            sd = values.std(unbiased=True).item() if len(values) > 1 else 0.0
            return f"{values.mean().item():.3f}±{sd:.3f}"

        print(
            f"{step:>5} {summary('train_loss'):>12} {summary('eval_loss'):>11} "
            f"{summary('original_acc'):>10} {summary('flipped_acc'):>10} "
            f"{summary('flip_rate'):>11} {summary('cue_disagree_acc'):>10} "
            f"{summary('pre_cue_probe_acc'):>11}"
        )


if __name__ == "__main__":
    main()
