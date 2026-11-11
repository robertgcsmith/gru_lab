"""Single layer GRU
following classic formulation via Cho et al., 2014,
with optional detached gate diagnostics.

The architecture is designed to track and isolate gate mechanics
for OOD sequential tasks. These diagnostics are research instrumentation, 
not a change to the basic GRU recurrence.

Note: the standard deiviation of the initial weight distribution is set to 0.01, following the original paper;
but this is not to suggest it is optimal for counting. That is a separate optimisation question we defer until later."""

import torch
from torch import nn


class GRUScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs # input feature dimension
        self.num_hiddens = num_hiddens # hidden-state dimension K
        self.sigma = sigma # standard deviation of initial weight distribution	0.01

        def init_weight(*shape):
            return nn.Parameter(torch.randn(*shape) * sigma) # multiplication by sigma rescales samples; sample from N(0, sigma^2)

        # Initialise the weight matrices and bias vector
        def triple():
            return (
                init_weight(num_inputs, num_hiddens), # input-to-hidden weight matrix
                init_weight(num_hiddens, num_hiddens), # hidden-to-hidden weight matrix
                nn.Parameter(torch.zeros(num_hiddens)), # bias vector
            )

        self.W_xz, self.W_hz, self.b_z = triple() # Update gate
        self.W_xr, self.W_hr, self.b_r = triple() # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple() # Candidate hidden state

    def forward(self, inputs, H=None, record_gates=False, z_override=None):
        """
        Accepts a complete sequence of inputs, and an optional initial hidden state.
        Inputs: (T, B, D); H: (B, K).
        If no hidden state is provided, it is intialised to zeros.

        Default return: (outputs, H_final), with outputs a list of (B, K).
        record_gates is a diagnostic flag that adds a dictionary of detached (T, B, K) gate tensors.
        z_override(X, H_previous, Z_computed, t) must return (B, K),
        without modifying its arguments; t is zero-based.

        Note: initialisation is H_0= mathbf 0_{B times K}}, which establishes
        the initial condition for the recurrence H_t = F_theta(X_t,H_{t-1}).
        When H_0 = 0, the first recurrent weight matrices have no direct contribution 
        to the first timestep's affine preactivations.
        """
        if H is None:
            H = inputs.new_zeros((inputs.shape[1], self.num_hiddens))

        # Containers to collect state representations and gate values
        outputs = []
        gates = (
            {"z_computed": [], "z_effective": [], "r": []}
            if record_gates else None
        )
        for t, X in enumerate(inputs):
            Z = torch.sigmoid(
                torch.matmul(X, self.W_xz)
                + torch.matmul(H, self.W_hz) + self.b_z) # Calculates the update gate
            
            R = torch.sigmoid(
                torch.matmul(X, self.W_xr)
                + torch.matmul(H, self.W_hr) + self.b_r) # Calculates the reset gate
            H_tilde = torch.tanh(
                torch.matmul(X, self.W_xh)
                + torch.matmul(R * H, self.W_hh) + self.b_h) # Calculates the candidate hidden state

            # Resolve the gate coefficients
            Z_effective = Z if z_override is None else z_override(X, H, Z, t)
            if z_override is not None and Z_effective.shape != Z.shape:
                raise ValueError("z_override must return shape (B, K).")

            # Update running memory and track diagnostics
            H = Z_effective * H + (1 - Z_effective) * H_tilde
            outputs.append(H)
            if record_gates:
                gates["z_computed"].append(Z.detach())
                gates["z_effective"].append(Z_effective.detach())
                gates["r"].append(R.detach())

        # optional branch to return detached gate values for research instrumentation
        if not record_gates:
            return outputs, H
        empty_shape = (0, inputs.shape[1], self.num_hiddens)
        gates = {
            name: torch.stack(values) if values
            else inputs.new_empty(empty_shape)
            for name, values in gates.items()
        }
        return outputs, H, gates