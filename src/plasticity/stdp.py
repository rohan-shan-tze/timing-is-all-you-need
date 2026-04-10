"""
Spike-Timing-Dependent Plasticity (STDP) learning rules.

Based on Diehl & Cook (2015), using power-law weight dependence:
    Δw = η * (x_pre - x_tar) * (w_max - w)^μ

Some key features are
Presynaptic trace tracking
Weight updates only on postsynaptic spikes
Weight-dependent learning (soft bounds)
Target trace for sparsity/disconnection
"""
import torch
import math


class PowerLawSTDP:
    """
    Power-law weight dependence STDP rule.

    The learning rule uses presynaptic traces to determine which
    synapses were recently active. Weight updates occur only when
    a postsynaptic neuron fires.

    The weight change is:
        Δw = η * (x_pre - x_tar) * (w_max - w)^μ

    where:
    - η: learning rate
    - x_pre: presynaptic trace (increases on pre spike, decays otherwise)
    - x_tar: target trace value (offset to disconnect irrelevant inputs)
    - w_max: maximum weight
    - μ: weight dependence exponent
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        eta: float = 0.0001,
        x_tar: float = 0.4,
        mu: float = 0.2,
        w_max: float = 1.0,
        w_min: float = 0.0,
        tau_pre: float = 20.0,
        dt: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize the STDP rule.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            eta: Learning rate
            x_tar: Target presynaptic trace value
            mu: Weight dependence exponent (0 = additive, 1 = multiplicative)
            w_max: Maximum weight
            w_min: Minimum weight
            tau_pre: Presynaptic trace time constant in ms
            dt: Simulation timestep in ms
            device: Device to use
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.eta = eta
        self.x_tar = x_tar
        self.mu = mu
        self.w_max = w_max
        self.w_min = w_min
        self.tau_pre = tau_pre
        self.dt = dt
        self.device = device

        # Precompute decay factor
        self.pre_decay = math.exp(-dt / tau_pre)

        # Initialize presynaptic trace
        self.x_pre = torch.zeros(n_pre, device=device, dtype=torch.float32)

    def reset_traces(self):
        """Reset traces to zero (use between images)."""
        self.x_pre.zero_()

    def update_traces(self, pre_spikes: torch.Tensor):
        """
        Update presynaptic traces.

        Called every timestep. Traces decay exponentially and
        increase by 1 when a presynaptic spike occurs.

        Args:
            pre_spikes: Binary presynaptic spike tensor [n_pre]
        """
        # Decay existing traces
        self.x_pre = self.x_pre * self.pre_decay

        # Increment traces for neurons that spiked
        self.x_pre = self.x_pre + pre_spikes

    def compute_weight_updates(
        self,
        post_spikes: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weight updates based on current traces and spikes.

        Updates are computed for all synapses to postsynaptic neurons
        that spiked. The weight change depends on the presynaptic trace
        and the current weight.

        Args:
            post_spikes: Binary postsynaptic spike tensor [n_post]
            weights: Current weight matrix [n_pre, n_post]

        Returns:
            Weight delta tensor [n_pre, n_post]
        """
        # Only compute updates if there are post spikes
        if not post_spikes.any():
            return torch.zeros_like(weights)

        # Find indices of neurons that spiked
        spiking_idx = post_spikes.nonzero(as_tuple=True)[0]

        # Initialize delta matrix
        delta_w = torch.zeros_like(weights)

        # Compute weight-dependent factor: (w_max - w)^μ
        # Only for columns (postsynaptic neurons) that spiked
        w_subset = weights[:, spiking_idx]  # [n_pre, n_spiking]
        weight_factor = torch.pow(self.w_max - w_subset, self.mu)

        # Compute trace-dependent factor: (x_pre - x_tar)
        trace_factor = (self.x_pre - self.x_tar).unsqueeze(1)  # [n_pre, 1]

        # Compute delta: η * (x_pre - x_tar) * (w_max - w)^μ
        delta_subset = self.eta * trace_factor * weight_factor  # [n_pre, n_spiking]

        # Place updates in the correct columns
        delta_w[:, spiking_idx] = delta_subset

        return delta_w

    def apply_weight_update(
        self,
        weights: torch.Tensor,
        post_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute and apply weight updates in one step.

        Args:
            weights: Current weight matrix [n_pre, n_post]
            post_spikes: Binary postsynaptic spike tensor [n_post]

        Returns:
            Updated weight matrix [n_pre, n_post]
        """
        delta_w = self.compute_weight_updates(post_spikes, weights)
        weights = weights + delta_w

        # Clamp weights to valid range
        weights = torch.clamp(weights, self.w_min, self.w_max)

        return weights

    def get_trace_stats(self) -> dict:
        """Get statistics about current traces."""
        return {
            'mean': self.x_pre.mean().item(),
            'std': self.x_pre.std().item(),
            'min': self.x_pre.min().item(),
            'max': self.x_pre.max().item(),
            'above_target': (self.x_pre > self.x_tar).float().mean().item()
        }


class ExponentialSTDP:
    """
    Exponential weight dependence STDP rule (alternative from paper).

    Weight change:
        Δw = η_post * x_pre * exp(-β*w) - x_tar * exp(-β*(w_max - w))

    This rule provides smoother weight changes near the boundaries.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        eta_post: float = 0.01,
        x_tar: float = 0.4,
        beta: float = 1.0,
        w_max: float = 1.0,
        w_min: float = 0.0,
        tau_pre: float = 20.0,
        dt: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize the exponential STDP rule.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            eta_post: Learning rate for postsynaptic events
            x_tar: Target presynaptic trace value
            beta: Weight dependence strength
            w_max: Maximum weight
            w_min: Minimum weight
            tau_pre: Presynaptic trace time constant in ms
            dt: Simulation timestep in ms
            device: Device to use
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.eta_post = eta_post
        self.x_tar = x_tar
        self.beta = beta
        self.w_max = w_max
        self.w_min = w_min
        self.tau_pre = tau_pre
        self.dt = dt
        self.device = device

        # Precompute decay factor
        self.pre_decay = math.exp(-dt / tau_pre)

        # Initialize presynaptic trace
        self.x_pre = torch.zeros(n_pre, device=device, dtype=torch.float32)

    def reset_traces(self):
        """Reset traces to zero."""
        self.x_pre.zero_()

    def update_traces(self, pre_spikes: torch.Tensor):
        """Update presynaptic traces."""
        self.x_pre = self.x_pre * self.pre_decay + pre_spikes

    def compute_weight_updates(
        self,
        post_spikes: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weight updates using exponential weight dependence.

        Δw = η_post * (x_pre * exp(-β*w) - x_tar * exp(-β*(w_max - w)))
        """
        if not post_spikes.any():
            return torch.zeros_like(weights)

        spiking_idx = post_spikes.nonzero(as_tuple=True)[0]
        delta_w = torch.zeros_like(weights)

        w_subset = weights[:, spiking_idx]

        # Potentiation term: x_pre * exp(-β*w)
        pot_term = self.x_pre.unsqueeze(1) * torch.exp(-self.beta * w_subset)

        # Depression term: x_tar * exp(-β*(w_max - w))
        dep_term = self.x_tar * torch.exp(-self.beta * (self.w_max - w_subset))

        # Combined update
        delta_subset = self.eta_post * (pot_term - dep_term)
        delta_w[:, spiking_idx] = delta_subset

        return delta_w

    def apply_weight_update(
        self,
        weights: torch.Tensor,
        post_spikes: torch.Tensor
    ) -> torch.Tensor:
        """Compute and apply weight updates."""
        delta_w = self.compute_weight_updates(post_spikes, weights)
        weights = weights + delta_w
        return torch.clamp(weights, self.w_min, self.w_max)
