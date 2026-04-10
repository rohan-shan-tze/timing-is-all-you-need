"""
Conductance-based Leaky Integrate-and-Fire (LIF) neurons.

Based on Diehl & Cook (2015):
    τ * dV/dt = (E_rest - V) + g_e*(E_exc - V) + g_i*(E_inh - V)

Some key features are:
Conductance-based synapses (not current-based)
Adaptive threshold for homeostasis
Refractory period
"""
import torch
import math


class ConductanceLIFNeurons:
    """
    Population of conductance-based LIF neurons.

    The membrane potential is influenced by excitatory and inhibitory
    conductances, which drive the voltage toward their respective
    reversal potentials.
    """

    def __init__(
        self,
        n_neurons: int,
        tau_membrane: float,
        e_rest: float,
        e_exc: float,
        e_inh: float,
        v_thresh: float,
        v_reset: float,
        refrac_time: float,
        dt: float,
        tau_ge: float = 1.0,
        tau_gi: float = 2.0,
        theta_increment: float = 0.0,
        tau_theta: float = 1e7,
        device: str = "cuda"
    ):
        """
        Initialize a population of LIF neurons.

        Args:
            n_neurons: Number of neurons in the population
            tau_membrane: Membrane time constant in ms
            e_rest: Resting membrane potential in mV
            e_exc: Excitatory reversal potential in mV
            e_inh: Inhibitory reversal potential in mV
            v_thresh: Base firing threshold in mV
            v_reset: Reset potential after spike in mV
            refrac_time: Refractory period in ms
            dt: Simulation timestep in ms
            tau_ge: Excitatory conductance time constant in ms
            tau_gi: Inhibitory conductance time constant in ms
            theta_increment: Threshold increase on spike (for homeostasis)
            tau_theta: Adaptive threshold decay time constant in ms
            device: Device to use ("cuda" or "cpu")
        """
        self.n_neurons = n_neurons
        self.tau_membrane = tau_membrane
        self.e_rest = e_rest
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.refrac_time = refrac_time
        self.dt = dt
        self.tau_ge = tau_ge
        self.tau_gi = tau_gi
        self.theta_increment = theta_increment
        self.tau_theta = tau_theta
        self.device = device

        # Precompute decay factors
        self.ge_decay = math.exp(-dt / tau_ge)
        self.gi_decay = math.exp(-dt / tau_gi)
        self.theta_decay = math.exp(-dt / tau_theta)

        # Initialize state tensors
        self._init_state()

    def _init_state(self):
        """Initialize all state tensors to resting values."""
        # Membrane potential
        self.v = torch.full(
            (self.n_neurons,),
            self.e_rest,
            device=self.device,
            dtype=torch.float32
        )

        # Excitatory and inhibitory conductances
        self.g_e = torch.zeros(self.n_neurons, device=self.device, dtype=torch.float32)
        self.g_i = torch.zeros(self.n_neurons, device=self.device, dtype=torch.float32)

        # Adaptive threshold (for homeostasis)
        self.theta = torch.zeros(self.n_neurons, device=self.device, dtype=torch.float32)

        # Refractory timer (0 = not refractory)
        self.refrac_timer = torch.zeros(self.n_neurons, device=self.device, dtype=torch.float32)

    def reset_state(self):
        """
        Fully reset all state to resting values.
        Use between training runs or for initialization.
        """
        self._init_state()

    def reset_for_new_image(self):
        """
        Partial reset between images.
        Resets voltage and conductances but preserves adaptive threshold.
        """
        self.v.fill_(self.e_rest)
        self.g_e.zero_()
        self.g_i.zero_()
        self.refrac_timer.zero_()
        # Note: theta is NOT reset, it persists across images

    def step(
        self,
        g_e_input: torch.Tensor,
        g_i_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Advance the neuron population by one timestep.

        Args:
            g_e_input: Excitatory conductance input [n_neurons]
            g_i_input: Inhibitory conductance input [n_neurons]

        Returns:
            Binary spike tensor [n_neurons] (1 = spike, 0 = no spike)
        """
        # 1. Update conductances: decay + new input
        self.g_e = self.g_e * self.ge_decay + g_e_input
        self.g_i = self.g_i * self.gi_decay + g_i_input

        # 2. Compute membrane potential derivative (conductance-based LIF)
        # τ * dV/dt = (E_rest - V) + g_e*(E_exc - V) + g_i*(E_inh - V)
        dv = (
            (self.e_rest - self.v) +
            self.g_e * (self.e_exc - self.v) +
            self.g_i * (self.e_inh - self.v)
        )

        # 3. Euler integration
        self.v = self.v + (self.dt / self.tau_membrane) * dv

        # 4. Check for spikes (only neurons not in refractory period)
        # Effective threshold = base threshold + adaptive threshold
        effective_thresh = self.v_thresh + self.theta
        can_spike = self.refrac_timer <= 0
        spikes = (self.v >= effective_thresh) & can_spike

        # 5. Handle spikes
        if spikes.any():
            # Reset membrane potential for neurons that spiked
            self.v[spikes] = self.v_reset

            # Update adaptive threshold (homeostasis)
            self.theta[spikes] = self.theta[spikes] + self.theta_increment

            # Start refractory period
            self.refrac_timer[spikes] = self.refrac_time

        # 6. Decay adaptive threshold (all neurons, every timestep)
        self.theta = self.theta * self.theta_decay

        # 7. Decrement refractory timer
        self.refrac_timer = torch.clamp(self.refrac_timer - self.dt, min=0)

        return spikes.float()

    def get_effective_threshold(self) -> torch.Tensor:
        """Get the current effective threshold for each neuron."""
        return self.v_thresh + self.theta

    def get_state_dict(self) -> dict:
        """Get current state for checkpointing."""
        return {
            'v': self.v.clone(),
            'g_e': self.g_e.clone(),
            'g_i': self.g_i.clone(),
            'theta': self.theta.clone(),
            'refrac_timer': self.refrac_timer.clone()
        }

    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint."""
        self.v = state_dict['v'].to(self.device)
        self.g_e = state_dict['g_e'].to(self.device)
        self.g_i = state_dict['g_i'].to(self.device)
        self.theta = state_dict['theta'].to(self.device)
        self.refrac_timer = state_dict['refrac_timer'].to(self.device)


class ExcitatoryNeurons(ConductanceLIFNeurons):
    """
    Excitatory neuron population with paper-specific defaults.

    Uses longer membrane time constant (100ms) for better rate integration.
    """

    def __init__(
        self,
        n_neurons: int,
        dt: float = 0.5,
        device: str = "cuda",
        tau_membrane: float = 100.0,
        e_rest: float = -65.0,
        e_exc: float = 0.0,
        e_inh: float = -100.0,
        v_thresh: float = -52.0,
        v_reset: float = -65.0,
        refrac_time: float = 5.0,
        tau_ge: float = 1.0,
        tau_gi: float = 2.0,
        theta_increment: float = 0.05,
        tau_theta: float = 1e7
    ):
        super().__init__(
            n_neurons=n_neurons,
            tau_membrane=tau_membrane,
            e_rest=e_rest,
            e_exc=e_exc,
            e_inh=e_inh,
            v_thresh=v_thresh,
            v_reset=v_reset,
            refrac_time=refrac_time,
            dt=dt,
            tau_ge=tau_ge,
            tau_gi=tau_gi,
            theta_increment=theta_increment,
            tau_theta=tau_theta,
            device=device
        )


class InhibitoryNeurons(ConductanceLIFNeurons):
    """
    Inhibitory neuron population with paper-specific defaults.

    Uses standard membrane time constant (10ms).
    No adaptive threshold (theta_increment = 0).
    """

    def __init__(
        self,
        n_neurons: int,
        dt: float = 0.5,
        device: str = "cuda",
        tau_membrane: float = 10.0,
        e_rest: float = -60.0,
        e_exc: float = 0.0,
        e_inh: float = -85.0,
        v_thresh: float = -40.0,
        v_reset: float = -45.0,
        refrac_time: float = 2.0,
        tau_ge: float = 1.0,
        tau_gi: float = 2.0
    ):
        super().__init__(
            n_neurons=n_neurons,
            tau_membrane=tau_membrane,
            e_rest=e_rest,
            e_exc=e_exc,
            e_inh=e_inh,
            v_thresh=v_thresh,
            v_reset=v_reset,
            refrac_time=refrac_time,
            dt=dt,
            tau_ge=tau_ge,
            tau_gi=tau_gi,
            theta_increment=0.0,  # No homeostasis for inhibitory
            tau_theta=1e7,
            device=device
        )
