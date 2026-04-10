"""
Diehl & Cook (2015) Spiking Neural Network for MNIST classification.

Network Architecture:
- Input layer: 784 neurons (Poisson spike trains from pixels)
- Excitatory layer: n_exc neurons (e.g., 400)
- Inhibitory layer: n_exc neurons (one-to-one with excitatory)

Connections:
- Input -> Excitatory: All-to-all, plastic (STDP)
- Excitatory -> Inhibitory: One-to-one, fixed
- Inhibitory -> Excitatory: All-to-all except self, fixed (lateral inhibition)
"""
import torch
import yaml
from pathlib import Path
from typing import Optional, Tuple

from ..neurons.lif import ExcitatoryNeurons, InhibitoryNeurons
from ..plasticity.stdp import PowerLawSTDP
from ..encoding.poisson import PoissonEncoder, TTFSEncoder
from ..encoding.preprocessing import DoGPreprocessor


class DiehlNetwork:
    """
    Complete implementation of the Diehl & Cook (2015) network.

    This network learns to recognize MNIST digits through unsupervised
    STDP learning with lateral inhibition and homeostasis.
    """

    def __init__(
        self,
        n_input: int = 784,
        n_exc: int = 400,
        dt: float = 0.5,
        device: str = "cuda",
        config: dict = None,
        scale_params: bool = True,
        encoding: str = 'poisson',
        preprocessing: str = 'none'
    ):
        """
        Initialize the network.

        Args:
            n_input: Number of input neurons (784 for MNIST, 1568 for DoG ON+OFF)
            n_exc: Number of excitatory/inhibitory neurons
            dt: Simulation timestep in ms
            device: Device to use ("cuda" or "cpu")
            config: Optional configuration dictionary
            scale_params: Whether to scale w_inh_exc, theta_increment, and
                          min_spikes to match network size (default True).
                          Set False to use paper's fixed values for all sizes.
            encoding: Input encoding scheme ('poisson' or 'ttfs').
            preprocessing: Input preprocessing ('none' or 'dog').
        """
        self.n_exc = n_exc
        self.n_inh = n_exc  # One inhibitory per excitatory
        self.dt = dt
        self.device = device

        # Load config or use defaults
        if config is None:
            config = self._default_config()
        self.config = config

        # Store encoding type; TTFS uses a shorter presentation window
        self.encoding = encoding
        if encoding == 'ttfs' and 'ttfs_presentation_time' not in self.config['encoding']:
            self.config['encoding']['ttfs_presentation_time'] = 100.0

        # Initialize preprocessor and derive actual input dimension
        self.preprocessing = preprocessing

        pre_cfg = self.config.get('preprocessing', {})
        on_off = pre_cfg.get('on_off', True)
        if preprocessing == 'dog' and on_off:
            self.n_input = 1568
        else:
            self.n_input = n_input

        self.preprocessor = self._init_preprocessor()

        # Scale inhibition/homeostasis parameters to network size
        if scale_params:
            self._scale_params_for_network_size()

        # Initialize components
        self._init_neurons()
        self._init_weights()
        self._init_stdp()
        self._init_encoder()

        # State tracking
        self.inh_spikes_prev = torch.zeros(self.n_inh, device=device)

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'exc_neuron': {
                'tau_membrane': 100.0,
                'e_rest': -65.0,
                'e_exc': 0.0,
                'e_inh': -100.0,
                'v_thresh': -52.0,
                'v_reset': -65.0,
                'refrac_time': 5.0,
            },
            'inh_neuron': {
                'tau_membrane': 10.0,
                'e_rest': -60.0,
                'e_exc': 0.0,
                'e_inh': -85.0,
                'v_thresh': -40.0,
                'v_reset': -45.0,
                'refrac_time': 2.0,
            },
            'homeostasis': {
                'theta_increment': 0.05,
                'tau_theta': 1e7,
            },
            'synapse': {
                'tau_ge': 1.0,
                'tau_gi': 2.0,
                'w_max': 1.0,
                'w_init_min': 0.0,
                'w_init_max': 0.3,
                'w_exc_inh': 10.0,
                'w_inh_exc': 17.0,
            },
            'stdp': {
                'eta': 0.0001,
                'x_tar': 0.4,
                'mu': 0.2,
                'tau_pre': 20.0,
            },
            'encoding': {
                'max_rate': 63.75,
                'presentation_time': 350.0,
                'rest_time': 150.0,
                'min_spikes': 5,
                'rate_increment': 32.0,
            },
            'preprocessing': {
                'type': 'none',
                'sigma1': 1.0,
                'sigma2': 2.0,
                'on_off': True,
            },
        }

    def _scale_params_for_network_size(self, reference_n_exc: int = 400):
        """
        Scale w_inh_exc, theta_increment, and min_spikes to preserve competitive
        dynamics across network sizes.

        The paper uses fixed values calibrated for 400 neurons. For other sizes:
          - w_inh_exc is NOT scaled: per-spike suppression is independent of n_exc
          - theta_increment scales up with n_exc to keep homeostasis effective
          - min_spikes scales up with n_exc to maintain a consistent retry threshold

        Args:
            reference_n_exc: Network size the paper's parameters are calibrated for
        """
        scale = self.n_exc / reference_n_exc

        # w_inh_exc is NOT scaled: per-spike inhibition strength is independent
        # of network size (each winning spike drives exactly one inhibitory neuron,
        # which suppresses others by w_inh_exc regardless of n_exc).
        self.config['homeostasis']['theta_increment'] = (
            self.config['homeostasis']['theta_increment'] * scale
        )
        self.config['encoding']['min_spikes'] = max(
            5, int(self.n_exc / 40)
        )

    def _init_neurons(self):
        """Initialize excitatory and inhibitory neuron populations."""
        exc_cfg = self.config['exc_neuron']
        inh_cfg = self.config['inh_neuron']
        homeo_cfg = self.config['homeostasis']
        syn_cfg = self.config['synapse']

        # Excitatory neurons with homeostasis
        self.exc_neurons = ExcitatoryNeurons(
            n_neurons=self.n_exc,
            dt=self.dt,
            device=self.device,
            tau_membrane=exc_cfg['tau_membrane'],
            e_rest=exc_cfg['e_rest'],
            e_exc=exc_cfg['e_exc'],
            e_inh=exc_cfg['e_inh'],
            v_thresh=exc_cfg['v_thresh'],
            v_reset=exc_cfg['v_reset'],
            refrac_time=exc_cfg['refrac_time'],
            tau_ge=syn_cfg['tau_ge'],
            tau_gi=syn_cfg['tau_gi'],
            theta_increment=homeo_cfg['theta_increment'],
            tau_theta=homeo_cfg['tau_theta'],
        )

        # Inhibitory neurons (no homeostasis)
        self.inh_neurons = InhibitoryNeurons(
            n_neurons=self.n_inh,
            dt=self.dt,
            device=self.device,
            tau_membrane=inh_cfg['tau_membrane'],
            e_rest=inh_cfg['e_rest'],
            e_exc=inh_cfg['e_exc'],
            e_inh=inh_cfg['e_inh'],
            v_thresh=inh_cfg['v_thresh'],
            v_reset=inh_cfg['v_reset'],
            refrac_time=inh_cfg['refrac_time'],
            tau_ge=syn_cfg['tau_ge'],
            tau_gi=syn_cfg['tau_gi'],
        )

    def _init_weights(self):
        """Initialize weight matrices."""
        syn_cfg = self.config['synapse']

        # Input -> Excitatory: plastic, random initialization
        # Shape: [n_input, n_exc]
        self.W_input_exc = torch.empty(
            self.n_input, self.n_exc,
            device=self.device, dtype=torch.float32
        ).uniform_(syn_cfg['w_init_min'], syn_cfg['w_init_max'])

        # Excitatory -> Inhibitory: one-to-one, fixed
        # Shape: [n_exc, n_inh] - identity matrix scaled
        self.W_exc_inh = torch.eye(
            self.n_exc, device=self.device, dtype=torch.float32
        ) * syn_cfg['w_exc_inh']

        # Inhibitory -> Excitatory: all-to-all except self, fixed
        # Shape: [n_inh, n_exc]
        # All connections except diagonal
        self.W_inh_exc = torch.ones(
            self.n_inh, self.n_exc,
            device=self.device, dtype=torch.float32
        ) * syn_cfg['w_inh_exc']
        # Remove self-connections (diagonal)
        self.W_inh_exc.fill_diagonal_(0.0)

    def _init_preprocessor(self) -> Optional[DoGPreprocessor]:
        """Initialize DoG preprocessor if requested, else return None."""
        if self.preprocessing != 'dog':
            return None
        pre_cfg = self.config.get('preprocessing', {})
        sigma1 = pre_cfg.get('sigma1', 1.0)
        sigma2 = pre_cfg.get('sigma2', 2.0)
        on_off = pre_cfg.get('on_off', True)
        preprocessor = DoGPreprocessor(sigma1=sigma1, sigma2=sigma2, on_off=on_off, device=self.device)
        print(f"Preprocessing: dog (sigma1={sigma1}, sigma2={sigma2}, "
              f"on_off={on_off}, input_dim={self.n_input})", flush=True)
        return preprocessor

    def _init_stdp(self):
        """Initialize STDP learning rule."""
        stdp_cfg = self.config['stdp']

        self.stdp = PowerLawSTDP(
            n_pre=self.n_input,
            n_post=self.n_exc,
            eta=stdp_cfg['eta'],
            x_tar=stdp_cfg['x_tar'],
            mu=stdp_cfg['mu'],
            w_max=self.config['synapse']['w_max'],
            w_min=0.0,
            tau_pre=stdp_cfg['tau_pre'],
            dt=self.dt,
            device=self.device
        )

    def _init_encoder(self):
        """Initialize input encoder (Poisson or TTFS)."""
        enc_cfg = self.config['encoding']

        if self.encoding == 'ttfs':
            presentation_time = enc_cfg.get('ttfs_presentation_time', 100.0)
            self.encoder = TTFSEncoder(
                n_inputs=self.n_input,
                presentation_time=presentation_time,
                dt=self.dt,
                device=self.device
            )
            print(f"Encoding: ttfs (presentation_time={presentation_time}ms)", flush=True)
        else:
            self.encoder = PoissonEncoder(
                n_inputs=self.n_input,
                max_rate=enc_cfg['max_rate'],
                presentation_time=enc_cfg['presentation_time'],
                dt=self.dt,
                device=self.device
            )
            print(f"Encoding: poisson (presentation_time={enc_cfg['presentation_time']}ms)",
                  flush=True)

    def reset_for_new_image(self):
        """Reset network state between images (preserves theta)."""
        self.exc_neurons.reset_for_new_image()
        self.inh_neurons.reset_for_new_image()
        self.stdp.reset_traces()
        self.inh_spikes_prev.zero_()

    def step(
        self,
        input_spikes: torch.Tensor,
        learning: bool = True
    ) -> torch.Tensor:
        """
        Advance the network by one timestep.

        Args:
            input_spikes: Binary input spike tensor [n_input]
            learning: Whether to apply STDP learning

        Returns:
            Excitatory spike tensor [n_exc]
        """
        # 1. Compute conductance inputs to excitatory neurons
        # Input spikes contribute excitatory conductance
        g_e_to_exc = torch.matmul(input_spikes, self.W_input_exc)

        # Previous inhibitory spikes contribute inhibitory conductance
        g_i_to_exc = torch.matmul(self.inh_spikes_prev, self.W_inh_exc)

        # 2. Update excitatory neurons
        exc_spikes = self.exc_neurons.step(g_e_to_exc, g_i_to_exc)

        # 3. Compute conductance inputs to inhibitory neurons
        # Excitatory spikes drive inhibitory neurons
        g_e_to_inh = torch.matmul(exc_spikes, self.W_exc_inh)
        g_i_to_inh = torch.zeros(self.n_inh, device=self.device)

        # 4. Update inhibitory neurons
        inh_spikes = self.inh_neurons.step(g_e_to_inh, g_i_to_inh)

        # 5. STDP learning (only on input->exc weights)
        if learning:
            # Update presynaptic traces
            self.stdp.update_traces(input_spikes)

            # Apply weight updates if any excitatory neurons spiked
            if exc_spikes.any():
                self.W_input_exc = self.stdp.apply_weight_update(
                    self.W_input_exc, exc_spikes
                )

        # Store inhibitory spikes for next timestep
        self.inh_spikes_prev = inh_spikes

        return exc_spikes

    def present_image(
        self,
        image: torch.Tensor,
        learning: bool = True,
        rate_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Present a single image to the network.

        Args:
            image: MNIST image tensor [1, 28, 28] or [784], values 0-255
            learning: Whether to apply STDP learning
            rate_scale: Rate scaling factor for adaptive encoding

        Returns:
            Spike counts per excitatory neuron [n_exc]
        """
        # Flatten image if needed
        image = image.view(-1).to(self.device)

        # Apply preprocessing (DoG edge filtering) if configured
        if self.preprocessor is not None:
            image = self.preprocessor(image)

        # Reset state for new image
        self.reset_for_new_image()

        # Track spike counts
        spike_counts = torch.zeros(self.n_exc, device=self.device)

        # Present image through Poisson encoding
        for input_spikes in self.encoder.encode(image, rate_scale):
            exc_spikes = self.step(input_spikes, learning)
            spike_counts += exc_spikes

        return spike_counts

    def present_image_adaptive(
        self,
        image: torch.Tensor,
        learning: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Present image with adaptive rate increase if needed.

        If fewer than min_spikes are produced, the image is re-presented
        with increased firing rate.

        Args:
            image: MNIST image tensor
            learning: Whether to apply STDP learning

        Returns:
            Tuple of (spike counts [n_exc], number of retries)
        """
        enc_cfg = self.config['encoding']
        min_spikes = enc_cfg['min_spikes']

        retry = 0
        total_spike_counts = torch.zeros(self.n_exc, device=self.device)

        while True:
            if self.encoding == 'ttfs':
                # TTFS retry: compress spike times toward t=0 (rate_scale > 1 = earlier spikes)
                rate_scale = 1.0 + 0.5 * retry
            else:
                # Poisson retry: increase input firing rate
                rate_increment = enc_cfg['rate_increment']
                max_rate = enc_cfg['max_rate']
                rate_scale = 1.0 + (rate_increment * retry) / max_rate

            # Present image
            spike_counts = self.present_image(image, learning, rate_scale)
            total_spike_counts += spike_counts

            # Check if enough spikes
            total_spikes = total_spike_counts.sum().item()
            if total_spikes >= min_spikes:
                break

            retry += 1
            if retry > 5:  # Cap retries
                break

        return total_spike_counts, retry

    def get_weight_stats(self) -> dict:
        """Get statistics about input->exc weights."""
        w = self.W_input_exc
        return {
            'mean': w.mean().item(),
            'std': w.std().item(),
            'min': w.min().item(),
            'max': w.max().item(),
            'near_zero': (w < 0.01).float().mean().item(),
            'near_max': (w > 0.99 * self.config['synapse']['w_max']).float().mean().item(),
        }

    def get_theta_stats(self) -> dict:
        """Get statistics about adaptive thresholds."""
        theta = self.exc_neurons.theta
        return {
            'mean': theta.mean().item(),
            'std': theta.std().item(),
            'min': theta.min().item(),
            'max': theta.max().item(),
        }

    def save_checkpoint(self, path: str):
        """Save network state to file."""
        checkpoint = {
            'W_input_exc': self.W_input_exc.cpu(),
            'exc_theta': self.exc_neurons.theta.cpu(),
            'config': self.config,
            'n_input': self.n_input,
            'n_exc': self.n_exc,
            'encoding': self.encoding,
            'preprocessing': self.preprocessing,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load network state from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.W_input_exc = checkpoint['W_input_exc'].to(self.device)
        self.exc_neurons.theta = checkpoint['exc_theta'].to(self.device)
        # Restore the config the network was trained with (includes any scaling)
        self.config = checkpoint['config']
        # Restore preprocessing — always update n_input from weight matrix shape
        # and reinitialise preprocessor, encoder, and STDP to match training config
        if 'preprocessing' in checkpoint:
            self.preprocessing = checkpoint['preprocessing']
        self.preprocessor = self._init_preprocessor()
        self.n_input = self.W_input_exc.shape[0]
        # Restore encoding type
        if 'encoding' in checkpoint:
            self.encoding = checkpoint['encoding']
        self._init_encoder()
        self._init_stdp()

    @classmethod
    def from_config_file(cls, config_path: str, device: str = "cuda"):
        """Create network from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls(
            n_input=config['network']['n_input'],
            n_exc=config['network']['n_excitatory'],
            dt=config['simulation']['dt'],
            device=device,
            config=config
        )
