"""
Implements a synthetic token-counting task with several controlled evaluation distributions and mechanistic diagnostics.


Exploratory counting, representation, gating, and dynamics diagnostics.
Requires PyTorch with torch.func.jacrev.
"""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import torch
from torch import nn

from gru_vanilla import GRUScratch

# Counting model and data, with the GRU receiving one feature per timestep
class CountingGRUSystem(nn.Module):
    """ Recurrent weights and readout weights are initialised according to different schemes.
        A linear readout on the final hidden state. By design, the readout has no activation function;
        it is an affine map, so its output is not restricted to [0,1] or integer values. 
        It also uses PyTorch's nn.Linear default initialisation.
    """
    def __init__(self, num_hiddens=16, sigma=0.01):
        super().__init__()
        self.gru = GRUScratch(1, num_hiddens, sigma)
        self.readout = nn.Linear(num_hiddens, 1)

    def forward(
        self, X, H=None, return_states=False,
        record_gates=False, z_override=None,):

        """Interpretability hooks: if return_states=True -> prediction, final state, and state trajectory.
            if record_gates=True -> prediction, final state, and gate trajectories.
            They are then bundled into a auxiliary dictionary, returned alongside a tuple: prediction, info.
                        
            In ordinary training mode, with both flags disabled, it returns only the predicted count tensor.
        """
        result = self.gru(X, H, record_gates, z_override)
        outputs, H_final = result[:2] # Extract the final step hidden state and the list of outputs
        prediction = self.readout(H_final).squeeze(-1)  # Pass into linear readout regression layer (B,)

        if not return_states and not record_gates:
            return prediction

        info = {"H_final": H_final.detach()}
        if return_states:
            info["states"] = (
                torch.stack([h.detach() for h in outputs]) if outputs
                else X.new_empty((0, X.shape[1], self.gru.num_hiddens)))  # (T, B, K), after each token
        if record_gates:
            info["gates"] = result[2]
        return prediction, info


def _model_input(model, X):
    """.to(parameter) aligns device and dtype, without copying matching inputs."""
    return X.to(next(model.parameters()))

# Statistical definition of the counting task
def generate_counting_batch(
    T, batch_size=64, p=0.5, device="cpu", generator=None,):
    """This function performs three mathematical operations:
    sampling binary tokens, calculating the cumulative counts, and extracting the final counts.

    Return: X with shape(T,B,1); y_steps[-1] is the final number of ones (B,); 
    and y_steps is the cumulative number of ones after every timestep (T,B).

    The counting target is computed directly from the sampled tokens.
    """
    if T < 1 or batch_size < 1 or not 0 <= p <= 1:
        raise ValueError("Require T >= 1, batch_size >= 1, and 0 <= p <= 1.")

    # Sample Bernoulli tokens and map against p
    X = (torch.rand(T, batch_size, 1, generator=generator) < p).to(
        device=device, dtype=torch.float32)
    y_steps = X[..., 0].cumsum(0)
    return X, y_steps[-1], y_steps


def length_batches(lengths, batch_size=256, p=0.5, device="cpu", seed=100):
    """Takes a collection of sequence lengths,
    generates a separate batch of binary sequences for each length,
    and returns a dictionary mapping each length to its corresponding batch of sequences.
    """
    generator = torch.Generator().manual_seed(seed)
    return {
        T: generate_counting_batch(T, batch_size, p, device, generator)[0]
        for T in lengths
    }


def exact_count_batch(
    T, count, batch_size=64, order="random", device="cpu", generator=None,
):
    """Return binary X (T,B,1), with exactly count ones per sequence.

    Random order samples subsets of positions; other orders are deterministic.
    Prefix counts, when needed, are X[..., 0].cumsum(0).
    """
    if T < 1 or batch_size < 1 or not 0 <= count <= T:
        raise ValueError("Require T, batch_size >= 1 and 0 <= count <= T.")
    if order not in ("random", "front", "back", "spread"):
        raise ValueError("order must be random/front/back/spread.")
    X = torch.zeros(T, batch_size)
    if count:
        if order == "random":
            indices = torch.rand(T, batch_size, generator=generator).argsort(0)[:count]
        else:
            if order == "front":
                positions = torch.arange(count)
            elif order == "back":
                positions = torch.arange(T - count, T)
            else:
                positions = torch.linspace(0, T - 1, count).round().long()
            indices = positions[:, None].expand(-1, batch_size)
        X.scatter_(0, indices, 1.0)
    return X.unsqueeze(-1).to(device)

# Synthetic tasks grouped into four stress test vectors
def controlled_batches(device="cpu", batch_size=64, seed=200):
    generator = torch.Generator().manual_seed(seed)
    specifications = [
        (f"matched_count/T={T}/k=6", T, 6, "random")
        for T in (10, 20, 40, 80)
    ] + [
        (f"matched_length/T=20/k={k}", 20, k, "random")
        for k in (0, 2, 6, 10, 20)
    ] + [
        (f"matched_frequency/f=0.25/T={T}", T, T // 4, "random")
        for T in (8, 20, 40, 80)
    ] + [
        (f"matched_multiset/order={order}", 40, 6, order)
        for order in ("front", "back", "spread", "random")
    ]
    return {
        label: exact_count_batch(T, k, batch_size, order, device, generator)
        for label, T, k, order in specifications
    }


# Training
def train_counting(
    updates=1000, seed=42, data_seed=43, val_seed=44, #seeds are isolated to prevent spurious correlations
    device=None, p=0.5, batch_size=64, lr=1e-3, log_every=100,
    return_initial=False,
):
    """Final-count MSE training; optionally return a frozen initial model third.

    The initial copy is made before optimisation, pairing the representation
    comparison by initialisation while preserving the default two-value return.
    """
    if updates < 1 or log_every < 1:
        raise ValueError("updates and log_every must be positive.")
    
    device = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(seed)
    model = CountingGRUSystem(num_hiddens=16, sigma=0.01).to(device)
    initial_model = deepcopy(model).eval().requires_grad_(False) if return_initial else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # core in-distribution range for the counting task is 5 <= T <= 20
    train_rng = torch.Generator().manual_seed(data_seed)
    val_rng = torch.Generator().manual_seed(val_seed)

    # Reusing independent validation draws
    validation = []
    for T in range(5, 21):
        Xv, yv, _ = generate_counting_batch(T, 256, p, device, val_rng)
        Xv = _model_input(model, Xv)
        validation.append((Xv, yv.to(Xv)))
    history, loss_sum, interval_updates = [], 0.0, 0

    # Optimisation loop, evaluate mean squared error on the terminal scalar count, execute backpropagation
    for step in range(1, updates + 1):
        model.train()
        T = int(torch.randint(5, 21, (), generator=train_rng).item())
        X, target, _ = generate_counting_batch(
            T, batch_size, p, device, train_rng
        )
        X = _model_input(model, X)
        target = target.to(X)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(X) - target).square().mean()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        interval_updates += 1

        # Evaluation
        if step % log_every == 0 or step == updates:
            model.eval()
            with torch.no_grad():
                errors = torch.cat([model(Xv) - yv for Xv, yv in validation])
                val_mse = errors.square().mean().item()
            row = {
                "update": step,
                "train_mse": loss_sum / interval_updates,
                "val_mse": val_mse,
            }
            history.append(row)
            print(row)
            loss_sum, interval_updates = 0.0, 0

    if return_initial:
        return model.eval(), history, initial_model
    return model.eval(), history


# Level 1 prediction
def regression_metrics(prediction, target, rounded_accuracy=False):
    prediction = prediction.detach().double().reshape(-1)
    target = target.detach().to(prediction).reshape(-1)

    # validation check: ensure that prediction and target have equal, non-zero sizes
    if prediction.numel() != target.numel() or target.numel() == 0:
        raise ValueError("Prediction and target must have equal, nonzero sizes.")

    error = prediction - target
    sst = (target - target.mean()).square().sum()

    # Return a dictionary of regression metrics, including MSE, MAE, bias, R^2, and optionally rounded accuracy
    result = {
        "n": target.numel(),
        "mse": error.square().mean().item(),
        "mae": error.abs().mean().item(),
        "bias": error.mean().item(),
        "r2": None if sst.item() == 0
        else (1 - error.square().sum() / sst).item(),
    }
    if rounded_accuracy:
        result["rounded_accuracy"] = (
            prediction.round() == target
        ).double().mean().item()
    return result

# Multi-length stress tests
@torch.no_grad()
def evaluate_predictive(model, batches, p=0.5, train_range=(5, 20)):
    """pT is a Bernoulli reference, not an oracle for exact-count controls."""
    reports = {}
    for label, inputs in batches.items():
        X = _model_input(model, inputs)
        T = X.shape[0]
        target = X[..., 0].sum(0)
        prediction = model(X)

        # create reference prediction baseline of p*T for comparison
        baseline = torch.full_like(target, p * T)

        # Compute regression metrics for each unique count in the target, reveal boundary vulnerabilities
        by_count = {}
        for count in target.unique(sorted=True):
            mask = target == count
            by_count[int(count.item())] = regression_metrics(
                prediction[mask], target[mask], True
            )

        # Label the report with sequence length, whether in-distribition or out-of-distribution
        reports[label] = {
            "T": T,
            "length_region": (
                "ID_length" if train_range[0] <= T <= train_range[1]
                else "OOD_length"
            ),
            "model": regression_metrics(prediction, target, True),
            "length_only_pT": regression_metrics(baseline, target, True),
            "model_by_true_count": by_count,
        }
    return reports


def check_id_learning(id_report, minimum_accuracy=0.95):
    """Sample-weighted pooled accuracy, plus diagnostics for each ID length.

    Supply one held-out Bernoulli report per length. Only the pooled threshold
    defines PASS; the per-length minimum is reported separately, not a new rule.
    """
    n = sum(row["model"]["n"] for row in id_report.values())
    accuracy = sum(
        row["model"]["rounded_accuracy"] * row["model"]["n"]
        for row in id_report.values()
    ) / n
    by_length = {
        row["T"]: row["model"]["rounded_accuracy"] for row in id_report.values()} # weakest link diagnostic
    
    minimum_per_length = min(by_length.values())
    passed = accuracy >= minimum_accuracy
    message = (
        f"Pooled ID rounded accuracy {accuracy:.3f}; "
        f"minimum per-length accuracy {minimum_per_length:.3f}; "
        f"pooled threshold {minimum_accuracy:.3f}: "
        + ("PASS." if passed else
           "FAIL. WARNING: ID learning is not established."))
    return {"passed": passed, "accuracy": accuracy,
            "accuracy_by_length": by_length,
            "minimum_per_length_accuracy": minimum_per_length,
            "threshold": minimum_accuracy, "message": message}


# Level 2: ID-fitted probes

def fit_ridge(features, targets, alpha=1e-3):
    """
    Ridge regression.
    Minimise sum squared errors + alpha*||w||^2; intercept unpenalised.

    Centre A and y to separate the intercept. Appending sqrt(alpha)*I and
    zero targets adds alpha*||w||^2 to the least-squares objective. CPU float64
    SVD avoids an explicit inverse and the conditioning cost of normal equations.
    """
    if alpha < 0:
        raise ValueError("alpha must be nonnegative.")
    A = features.detach().to(device="cpu", dtype=torch.float64)
    y = targets.detach().to(device="cpu", dtype=torch.float64)
    A_mean, y_mean = A.mean(0), y.mean(0)
    Ac, yc = A - A_mean, y - y_mean

    # trick: if alpha > 0, append sqrt(alpha)*I to A and zeros to y, then solve least squares
    if alpha:
        K = A.shape[1]
        Ac = torch.cat([Ac, alpha**0.5 * torch.eye(K, dtype=A.dtype)])
        yc = torch.cat([yc, yc.new_zeros((K,) + yc.shape[1:])])
    weight = torch.linalg.lstsq(Ac, yc, driver="gelsd").solution
    return weight, y_mean - A_mean @ weight


def ridge_predict(fit, features):
    weight, bias = fit
    return features.to(weight) @ weight + bias


@torch.no_grad()
def collect_probe_pairs(model, batches):
    """Flatten time-major states, counts, and times in matching (t,b) order."""
    hidden, counts, times = [], [], []
    for inputs in batches.values():
        X = _model_input(model, inputs)
        T, B, _ = X.shape
        _, info = model(X, return_states=True)

        # State/count reshape and repeat_interleave(B) align every (t,b) sample.
        hidden.append(info["states"].reshape(T * B, -1).to(
            device="cpu", dtype=torch.float64))
        counts.append(X[..., 0].cumsum(0).reshape(-1).to(
            device="cpu", dtype=torch.float64))
        times.append(torch.arange(1, T + 1, dtype=torch.float64).repeat_interleave(B))
    return torch.cat(hidden), torch.cat(counts), torch.cat(times)


def fit_accumulator_probes(
    model, id_batches, p=0.5, alpha=(1e-6, 1e-3, 1.0, 1000.0),
    train_range=(5, 20), *, validation_batches,
):
    """ID-fit standardisation and select each probe's alpha by ID validation MSE.

    Supply independently sampled fitting and validation batches. alpha accepts
    a fixed candidate grid (or one scalar); ties retain the first candidate.
    Direct probes decode count. Residual probes decode y-pt; reconstructing
    count supplies elapsed time externally. Neither establishes causal use.
    """
    for batches in (id_batches, validation_batches):
        if not all(
            train_range[0] <= X.shape[0] <= train_range[1]
            for X in batches.values()
        ):
            raise ValueError("Probe fitting and calibration require ID lengths.")
    candidates = (alpha,) if isinstance(alpha, (int, float)) else tuple(alpha)
    if not candidates or min(candidates) < 0:
        raise ValueError("Provide a nonempty grid of nonnegative penalties.")

    H, y, t = collect_probe_pairs(model, id_batches)
    H_val, y_val, t_val = collect_probe_pairs(model, validation_batches)
    mean, scale = H.mean(0), H.std(0, unbiased=False)

    # Hidden coordinates are bounded; avoid amplifying nearly constant noise.
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    H = (H - mean) / scale
    H_val = (H_val - mean) / scale  # Freeze ID fitting statistics everywhere else.

    # set up direct comparison between direct probe and residual probe
    targets = torch.stack((y, y - p * t), dim=1)
    validation_targets = torch.stack((y_val, y_val - p * t_val), dim=1)

    probes = {
        "feature_mean": mean, "feature_scale": scale,
        "alpha": {}, "validation_mse": {"direct": float("inf"), "residual": float("inf")},
        "p": p, "train_max": train_range[1],}

    # Evaluate all candidates for both probes, retaining the best by validation MSE
    for candidate in candidates:
        # Both targets share one feature matrix and one least-squares solve.
        weight, bias = fit_ridge(H, targets, candidate)
        losses = (ridge_predict((weight, bias), H_val) - validation_targets).square().mean(0)
        for j, name in enumerate(("direct", "residual")):
            score = losses[j].item()
            if score < probes["validation_mse"][name]:
                probes[name] = (weight[:, j], bias[j])
                probes["alpha"][name] = float(candidate)
                probes["validation_mse"][name] = score
    return probes

# Subject frozen probes to OOD timelines, track how linear map degrades
def evaluate_probes(
    model, probes, batches, conditional_times=(5, 10, 20, 40, 80, 160),
):
    """Use frozen fitting statistics/coefficients"""
    reports = {}
    for label, X in batches.items():
        H, y, t = collect_probe_pairs(model, {label: X})
        H = (H - probes["feature_mean"]) / probes["feature_scale"]
        direct = ridge_predict(probes["direct"], H)
        time_only = probes["p"] * t
        residual_prediction = ridge_predict(probes["residual"], H)
        residual_target = y - time_only

        def report(mask):
            return {
                "hidden_to_count": regression_metrics(direct[mask], y[mask]), # measure how well the hidden state predicts the count
                "time_only": regression_metrics(time_only[mask], y[mask]), # measure how well the time-only baseline predicts the count
                "hidden_to_count_residual": regression_metrics(
                    residual_prediction[mask], residual_target[mask]),
                "zero_residual_baseline": regression_metrics(
                    torch.zeros_like(residual_target[mask]), residual_target[mask]),
                "time_plus_hidden_residual": regression_metrics(
                    (time_only + residual_prediction)[mask], y[mask]),
            }

        row = {"all_steps": report(torch.ones_like(t, dtype=torch.bool))}
        late = t > probes["train_max"]
        if late.any():
            row["steps_beyond_training_horizon"] = report(late)
        row["by_elapsed_time"] = {
            elapsed: report(t == elapsed)
            for elapsed in conditional_times if (t == elapsed).any()
        }
        reports[label] = row
    return reports


# Level 3: gate interventions

def clamp_z_on_zero(X, H, Z, t):
    """force update gate to 1 (fully open) when the input token is zero, otherwise keep Z unchanged."""
    zero_token = (X == 0).all(dim=-1, keepdim=True)
    return torch.where(zero_token, torch.ones_like(Z), Z)


def identity_z(X, H, Z, t):
    return Z

@torch.no_grad()
def gate_statistics(X, gates):
    tokens = X[..., 0]
    groups = {
        "all": torch.ones_like(tokens, dtype=torch.bool),
        "zero": tokens == 0,
        "one": tokens == 1,
    }
    result = {}
    for group, mask in groups.items():
        result[group] = {"positions": int(mask.sum().item())}
        if not mask.any():
            continue
        for name, trajectory in gates.items():
            values = trajectory[mask].float()  # (positions, K)
            quantiles = values.new_tensor([0.05, 0.5, 0.95])
            result[group][name] = {
                "mean": values.mean().item(),
                "std": values.std(unbiased=False).item(),
                "q05_q50_q95": torch.quantile(values.flatten(), quantiles).tolist(),
                "per_unit_mean": values.mean(0).tolist(),
            }
    return result


@torch.no_grad()
def compare_gate_intervention(model, X, control_seed=300):
    """Paired zero clamp and a token-blind intervention-budget control.

    For each sequence, uniformly choose as many distinct positions as there
    are zero tokens. Conditional on that number, selection ignores token
    identity. This matches the number of suppressed transitions, not their
    semantic role: clamping ones can prevent legitimate count increments.
    Overlap with zero positions is allowed; this does not isolate gate leakage.
    """
    X = _model_input(model, X)
    T, B, _ = X.shape
    zero_mask = X[..., 0] == 0
    zero_counts = zero_mask.sum(0).cpu()
    generator = torch.Generator().manual_seed(control_seed)
    matched_mask = torch.zeros(T, B, dtype=torch.bool)
    for b, count in enumerate(zero_counts.tolist()):
        matched_mask[torch.randperm(T, generator=generator)[:count], b] = True
    matched_mask = matched_mask.to(X.device)

    def matched_clamp(X, H, Z, t):
        return torch.where(matched_mask[t, :, None], torch.ones_like(Z), Z)

    target = X[..., 0].sum(0)
    runs = {}
    for name, intervention in (
        ("ordinary", None), ("clamped", clamp_z_on_zero),
        ("matched_random", matched_clamp),
    ):
        prediction, info = model(
            X, return_states=True, record_gates=True, z_override=intervention
        )
        info["prediction"] = prediction
        info["step_predictions"] = model.readout(info["states"]).squeeze(-1)
        info["state_norms"] = info["states"].norm(dim=-1)
        info["gate_summary"] = gate_statistics(X, info["gates"])
        info["metrics"] = regression_metrics(prediction, target, True)
        runs[name] = info
    runs["identity_control_max_abs_difference"] = (
        model(X, z_override=identity_z) - runs["ordinary"]["prediction"]
    ).abs().max().item()
    runs["zero_mask"] = zero_mask
    runs["matched_random_mask"] = matched_mask
    return runs


@torch.no_grad()
def zero_tail_drift(model, prefix, tail_lengths=(0, 1, 5, 20, 80)):
    """Start both branches at the same ordinarily processed prefix state.

    Drift is relative to its prediction, which need not equal the true count;
    count-error metrics are reported separately.
    """
    prefix = _model_input(model, prefix)
    _, H_start = model.gru(prefix)
    start_prediction = model.readout(H_start).squeeze(-1)
    target = prefix[..., 0].sum(0)
    rows = []
    for length in tail_lengths:
        if length < 0:
            raise ValueError("Tail lengths must be nonnegative.")
        tail = prefix.new_zeros((length, prefix.shape[1], 1))
        row = {"zero_tail_length": length}
        for name, intervention in (
            ("ordinary", None), ("clamped", clamp_z_on_zero)
        ):
            _, H_end = model.gru(tail, H_start, z_override=intervention)
            prediction = model.readout(H_end).squeeze(-1)
            drift = prediction - start_prediction
            row[name] = {
                **regression_metrics(prediction, target, True),
                "signed_prediction_drift": drift.mean().item(),
                "absolute_prediction_drift": drift.abs().mean().item(),
                "hidden_displacement": (H_end - H_start).norm(dim=-1).mean().item(),
            }
        rows.append(row)
    return rows


# Level 3: local dynamics
@torch.no_grad()
def origin_jacobian(gru):
    """Special case reference at x=h=0 with zero gate/candidate biases."""
    if any(torch.count_nonzero(b).item() for b in (gru.b_z, gru.b_r, gru.b_h)):
        raise ValueError("The origin formula requires zero biases.")
    I = torch.eye(gru.num_hiddens, device=gru.W_hh.device, dtype=gru.W_hh.dtype)
    row_map = 0.5 * I + 0.25 * gru.W_hh
    return {"row_perturbation_map": row_map, "column_jacobian": row_map.T}


def recurrent_step(gru, x, h, z_override=None, t=0):
    """x (D,), h (K,) -> next state (K,); retain the global time index."""
    override = None
    if z_override is not None:
        override = lambda X, H, Z, local_t: z_override(X, H, Z, t)
    _, H_next = gru(x.reshape(1, 1, -1), h.reshape(1, -1), z_override=override)
    return H_next[0]


def dynamical_diagnostics(
    gru, X, batch_index=0, steps=None, windows=(), z_override=None,
):
    """One trajectory from h_0=0. J[i,j] = d h_next[i] / d h_previous[j].

    steps are zero-based; windows are half-open (start, stop) intervals.
    Use small windows. enable_grad supports outer no_grad, not inference_mode.
    Local norms describe one step; product norms describe multistep sensitivity.
    """
    X = _model_input(gru, X)
    sequence = X[:, batch_index:batch_index + 1]
    T = sequence.shape[0]
    if T < 1 or sequence.shape[1] != 1:
        raise ValueError("Select a valid, nonempty sequence.")
    steps = sorted(set((0, T // 2, T - 1) if steps is None else steps))
    needed = set(steps)  # Union of reported steps and every product-window step.
    for start, stop in windows:
        if not 0 <= start < stop <= T:
            raise ValueError("Invalid Jacobian-product window.")
        needed.update(range(start, stop))
    if any(t < 0 or t >= T for t in needed):
        raise ValueError("Invalid transition index.")

    with torch.no_grad():
        outputs, _ = gru(sequence, z_override=z_override)
        states = torch.stack(outputs)[:, 0]
        # previous[t] is pre-token state h_t; t=0 starts from the zero state.
        previous = torch.cat([states.new_zeros((1, gru.num_hiddens)), states[:-1]])
    jacobians = {}
    with torch.enable_grad():
        for t in sorted(needed):
            J = torch.func.jacrev(
                lambda h: recurrent_step(gru, sequence[t, 0], h, z_override, t)
            )(previous[t].detach())
            jacobians[t] = J.detach().to(device="cpu", dtype=torch.float64)

    local = [{
        "transition": t + 1,
        "spectral_radius": torch.linalg.eigvals(jacobians[t]).abs().max().item(),
        "spectral_norm": torch.linalg.matrix_norm(jacobians[t], ord=2).item(),
    } for t in steps]
    products = []
    for start, stop in windows:
        P = torch.eye(gru.num_hiddens, dtype=torch.float64)
        norms = []
        for t in range(start, stop):
            P = jacobians[t] @ P  # Column convention: J_last ... J_first.
            norms.append(torch.linalg.matrix_norm(P, ord=2).item())
        products.append({
            "first_transition": start + 1, "last_transition": stop,
            "prefix_product_norms": norms,
        })
    return {
        "states": states.cpu(), "state_norms": states.norm(dim=-1).cpu(),
        "jacobians": jacobians, "local": local, "products": products,
    }


# Lightweight checks
def check_shapes_and_recording():
    """Freeze parameter dictionary"""
    gru = GRUScratch(1, 16)
    assert set(dict(gru.named_parameters())) == {
        "W_xz", "W_hz", "b_z", "W_xr", "W_hr", "b_r", "W_xh", "W_hh", "b_h"
    }
    assert len(list(gru.parameters())) == 9
    assert sum(p.numel() for p in gru.parameters()) == 864
    # Smoke test: validate sequence generator
    X, final_target, step_targets = generate_counting_batch(
        7, 3, generator=torch.Generator().manual_seed(10))
    
    assert X.shape == (7, 3, 1) and final_target.shape == (3,)
    assert step_targets.shape == (7, 3)
    assert torch.equal(step_targets, X[..., 0].cumsum(0))
    assert torch.equal(final_target, step_targets[-1])

    H0 = torch.linspace(-0.3, 0.3, 48).reshape(3, 16)
    result = gru(X, H0)
    assert len(result) == 2 and isinstance(result[0], list)
    outputs, final = result

    recorded, recorded_final, gates = gru(X, H0, record_gates=True)
    assert len(outputs) == 7 and final.shape == (3, 16)
    assert torch.equal(final, recorded_final)
    for actual, diagnostic in zip(outputs, recorded):
        assert actual.shape == (3, 16) and torch.equal(actual, diagnostic)
    previous = H0
    for t, token in enumerate(X):
        expected_z = torch.sigmoid(token @ gru.W_xz + previous @ gru.W_hz + gru.b_z)
        expected_r = torch.sigmoid(token @ gru.W_xr + previous @ gru.W_hr + gru.b_r)
        assert torch.equal(gates["z_computed"][t], expected_z)
        assert torch.equal(gates["z_effective"][t], expected_z)
        assert torch.equal(gates["r"][t], expected_r)
        previous = outputs[t]
    assert all(G.shape == (7, 3, 16) and not G.requires_grad for G in gates.values())

    model = CountingGRUSystem()
    prediction = model(X)
    diagnostic, info = model(X, return_states=True, record_gates=True)
    assert prediction.shape == (3,) and torch.equal(prediction, diagnostic)
    assert info["states"].shape == (7, 3, 16)
    assert sum(p.numel() for p in model.parameters()) == 881

    # Known one-step result distinguishes reset-before-matmul from reset-after.
    analytical = GRUScratch(1, 2).double()
    with torch.no_grad():
        for parameter in analytical.parameters():
            parameter.zero_()
        analytical.b_r.copy_(torch.logit(torch.tensor([0.25, 0.75], dtype=torch.float64)))
        analytical.W_hh.copy_(torch.tensor([[0., 1.], [2., 0.]], dtype=torch.float64))
    H0 = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    _, actual = analytical(torch.zeros(1, 1, 1, dtype=torch.float64), H0)
    expected = 0.5 * H0 + 0.5 * torch.tanh(
        torch.tensor([[-0.6, 0.05]], dtype=torch.float64)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def check_interventions():
    model = CountingGRUSystem()
    gru = model.gru
    X = torch.tensor([[0., 1.], [1., 0.], [0., 1.], [0., 0.], [1., 1.]])[..., None]
    H0 = torch.linspace(-0.4, 0.4, 32).reshape(2, 16)
    states, final = gru(
        X, H0, z_override=lambda X, H, Z, t: torch.ones_like(Z)
    )
    assert torch.equal(final, H0) and all(torch.equal(h, H0) for h in states)
    states, _, gates = gru(X, H0, record_gates=True, z_override=clamp_z_on_zero)
    previous = H0
    for t, state in enumerate(states):
        mask = X[t, :, 0] == 0
        assert torch.equal(state[mask], previous[mask])
        assert torch.equal(gates["z_effective"][t, mask], torch.ones_like(state[mask]))
        assert torch.equal(gates["z_effective"][t, ~mask], gates["z_computed"][t, ~mask])
        previous = state
    ones = torch.ones_like(X)
    assert torch.equal(gru(ones, H0)[1], gru(ones, H0, z_override=clamp_z_on_zero)[1])
    assert torch.equal(gru(X, H0)[1], gru(X, H0, z_override=identity_z)[1])

    parameters_before = [p.detach().clone() for p in model.parameters()]
    paired = compare_gate_intervention(model, X, control_seed=12)
    assert paired["identity_control_max_abs_difference"] == 0
    assert torch.equal(paired["zero_mask"].sum(0), paired["matched_random_mask"].sum(0))
    repeated = compare_gate_intervention(model, X, control_seed=12)
    assert torch.equal(paired["matched_random_mask"], repeated["matched_random_mask"])
    assert all(torch.equal(old, new) for old, new in zip(parameters_before, model.parameters()))
    tail = zero_tail_drift(model, X, tail_lengths=(0, 3))[-1]["clamped"]
    assert tail["hidden_displacement"] == 0 and tail["absolute_prediction_drift"] == 0


def check_gradient_propagation():
    model = CountingGRUSystem()
    X, target, _ = generate_counting_batch(
        10, 8, generator=torch.Generator().manual_seed(11)
    )
    (model(X) - target).square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.gru.W_xh.grad.norm().item() > 0
    assert model.readout.weight.grad.norm().item() > 0


def check_jacobian(gru=None, x=None, h=None, epsilon=1e-6):
    """Central differences on a double-precision copy"""
    gru = (deepcopy(gru) if gru is not None else GRUScratch(1, 16)).cpu().double()
    K = gru.num_hiddens
    x = torch.ones(gru.num_inputs, dtype=torch.float64) if x is None else x.detach().cpu().double()
    h = torch.linspace(-0.2, 0.2, K, dtype=torch.float64) if h is None else h.detach().cpu().double()
    with torch.enable_grad():
        J_ad = torch.func.jacrev(lambda state: recurrent_step(gru, x, state))(h).detach()
    with torch.no_grad():
        J_fd = torch.stack([
            (recurrent_step(gru, x, h + epsilon * direction)
             - recurrent_step(gru, x, h - epsilon * direction)) / (2 * epsilon)
            for direction in torch.eye(K, dtype=torch.float64)
        ], dim=1)
    torch.testing.assert_close(J_ad, J_fd, rtol=1e-5, atol=1e-7)

    if all(torch.count_nonzero(b).item() == 0 for b in (gru.b_z, gru.b_r, gru.b_h)):
        with torch.enable_grad():
            J_zero = torch.func.jacrev(
                lambda state: recurrent_step(gru, torch.zeros_like(x), state)
            )(torch.zeros_like(h))
        torch.testing.assert_close(
            J_zero, origin_jacobian(gru)["column_jacobian"], rtol=1e-10, atol=1e-12
        )

    # Compare the ordered product against differentiation through two steps.
    sequence = torch.stack((x, torch.zeros_like(x)), dim=0).unsqueeze(1)  # (2, 1, D)
    with torch.no_grad():
        diagnostics = dynamical_diagnostics(gru, sequence, steps=(0, 1), windows=((0, 2),))
    with torch.enable_grad():
        J_two = torch.func.jacrev(
            lambda state: gru(sequence, state[None])[1][0]
        )(torch.zeros(K, dtype=torch.float64))
    product = diagnostics["jacobians"][1] @ diagnostics["jacobians"][0]
    torch.testing.assert_close(product, J_two, rtol=1e-10, atol=1e-12)


def check_methodological_revisions():
    """Small calibration checks and a one-minibatch paired-snapshot smoke test."""
    torch.manual_seed(71)
    expected = {name: value.clone() for name, value in CountingGRUSystem().state_dict().items()}
    with redirect_stdout(StringIO()):
        trained, _, initial = train_counting(
            updates=1, seed=71, data_seed=72, val_seed=73,
            device="cpu", return_initial=True,
        )
    assert all(not parameter.requires_grad for parameter in initial.parameters())
    assert all(torch.equal(initial.state_dict()[name], value) for name, value in expected.items())
    assert any(not torch.equal(trained.state_dict()[name], value) for name, value in expected.items())

    fitting = length_batches((5, 6), batch_size=4, seed=74)
    validation = length_batches((5, 6), batch_size=4, seed=75)
    probes = fit_accumulator_probes(
        initial, fitting, alpha=(1e-3, 1.0), validation_batches=validation
    )
    H, _, _ = collect_probe_pairs(initial, fitting)
    torch.testing.assert_close(probes["feature_mean"], H.mean(0))
    scales = H.std(0, unbiased=False)
    torch.testing.assert_close(
        probes["feature_scale"], torch.where(scales > 1e-12, scales, torch.ones_like(scales))
    )
    frozen = {name: probes[name].clone() for name in ("feature_mean", "feature_scale")}
    evaluate_probes(initial, probes, {21: torch.ones(21, 1, 1)})
    assert all(torch.equal(probes[name], value) for name, value in frozen.items())
    assert all(torch.equal(initial.state_dict()[name], value) for name, value in expected.items())
    assert all(value in (1e-3, 1.0) for value in probes["alpha"].values())
    try:
        fit_accumulator_probes(initial, fitting, validation_batches={21: torch.zeros(21, 1, 1)})
    except ValueError:
        pass
    else:
        raise AssertionError("OOD lengths were accepted for probe calibration.")

    constant = CountingGRUSystem()
    with torch.no_grad():
        for parameter in constant.parameters():
            parameter.zero_()
    constant_probes = fit_accumulator_probes(
        constant, fitting, alpha=1e-3, validation_batches=validation
    )
    assert torch.equal(constant_probes["feature_scale"], torch.ones(16, dtype=torch.float64))
    assert all(torch.isfinite(value).all() for value in constant_probes["direct"])

    # A poor rare length must be visible without changing the pooled criterion.
    summary = check_id_learning({
        5: {"T": 5, "model": {"n": 100, "rounded_accuracy": 1.0}},
        20: {"T": 20, "model": {"n": 1, "rounded_accuracy": 0.0}},
    })
    assert summary["accuracy"] == 100 / 101 and summary["passed"]
    assert summary["accuracy_by_length"] == {5: 1.0, 20: 0.0}
    assert summary["minimum_per_length_accuracy"] == 0.0


def run_sanity_checks():
    """CPU checks, including one minibatch for snapshot verification, not a study."""
    results = {}
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(123)
        for check in (
            check_shapes_and_recording, check_interventions,
            check_gradient_propagation, check_jacobian, check_methodological_revisions,
        ):
            check()
            results[check.__name__] = "passed"
    return results