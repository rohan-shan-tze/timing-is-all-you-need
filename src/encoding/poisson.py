"""
Poisson spike encoding for converting images to spike trains.

Based on Diehl & Cook (2015):
- Pixel intensities are converted to firing rates
- Max rate = 63.75 Hz (for pixel value 255)
- Each image is presented for 350ms
- If fewer than 5 spikes occur, rate is increased by 32 Hz and repeated
"""
import torch
from typing import Generator, List, Tuple


class PoissonEncoder:
    """
    Converts pixel intensities to Poisson-distributed spike trains.

    The firing rate of each input neuron is proportional to the
    intensity of the corresponding pixel in the image.
    """

    def __init__(
        self,
        n_inputs: int = 784,
        max_rate: float = 63.75,
        presentation_time: float = 350.0,
        dt: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize the Poisson encoder.

        Args:
            n_inputs: Number of input neurons (784 for MNIST)
            max_rate: Maximum firing rate in Hz (default 63.75 = 255/4)
            presentation_time: Duration to present each image in ms
            dt: Simulation timestep in ms
            device: Device to use ("cuda" or "cpu")
        """
        self.n_inputs = n_inputs
        self.max_rate = max_rate
        self.presentation_time = presentation_time
        self.dt = dt
        self.device = device

        # Number of timesteps per presentation
        self.n_timesteps = int(presentation_time / dt)

    def encode(
        self,
        image: torch.Tensor,
        rate_scale: float = 1.0
    ) -> Generator[torch.Tensor, None, None]:
        """
        Generate Poisson spike trains for an image.

        Args:
            image: Flattened image tensor [784] with values 0-255
            rate_scale: Multiplier for firing rates (for adaptive increase)

        Yields:
            Binary spike tensor [n_inputs] for each timestep
        """
        # Ensure image is on correct device and flattened
        image = image.view(-1).to(self.device)

        # Convert pixel values (0-255) to firing rates (0-max_rate Hz)
        # Then apply any rate scaling
        rates = (image / 255.0) * self.max_rate * rate_scale

        # Convert rates to spike probability per timestep
        # P(spike in dt) = rate * dt / 1000 (rate in Hz, dt in ms)
        spike_prob = rates * self.dt / 1000.0

        # Clamp probability to [0, 1] (in case of high rate scaling)
        spike_prob = torch.clamp(spike_prob, 0.0, 1.0)

        # Generate spikes for each timestep
        for _ in range(self.n_timesteps):
            # Draw random numbers and compare to spike probability
            spikes = (torch.rand(self.n_inputs, device=self.device) < spike_prob).float()
            yield spikes

    def encode_with_count(
        self,
        image: torch.Tensor,
        rate_scale: float = 1.0
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Generate spike trains and return total spike count.

        Args:
            image: Flattened image tensor [784] with values 0-255
            rate_scale: Multiplier for firing rates

        Returns:
            Tuple of (list of spike tensors, total input spike count)
        """
        spike_trains = []
        total_spikes = 0

        for spikes in self.encode(image, rate_scale):
            spike_trains.append(spikes)
            total_spikes += spikes.sum().item()

        return spike_trains, int(total_spikes)

    def get_expected_spikes(self, image: torch.Tensor) -> float:
        """
        Calculate expected number of spikes for an image.

        Args:
            image: Flattened image tensor [784] with values 0-255

        Returns:
            Expected total number of input spikes
        """
        image = image.view(-1)
        rates = (image / 255.0) * self.max_rate  # Hz
        # Expected spikes = sum of (rate * time in seconds)
        expected = (rates * self.presentation_time / 1000.0).sum().item()
        return expected


class TTFSEncoder:
    """
    Time-to-First-Spike encoder.

    Each input neuron fires exactly once, with brighter pixels firing earlier.
    Zero-intensity pixels do not fire at all.

    spike_time = presentation_time * (1 - intensity / 255)

    This is deterministic: the same image always produces the same spike pattern.
    Matches the PoissonEncoder.encode() generator interface so DiehlNetwork can
    swap encoders without changing its simulation loop.
    """

    def __init__(
        self,
        n_inputs: int = 784,
        presentation_time: float = 100.0,
        dt: float = 0.5,
        threshold: int = 1,
        device: str = "cuda"
    ):
        """
        Args:
            n_inputs: Number of input neurons (784 for MNIST)
            presentation_time: Encoding window in ms (default 100ms, shorter than Poisson)
            dt: Simulation timestep in ms
            threshold: Minimum pixel intensity to generate a spike (pixels below this are silent)
            device: Device to use
        """
        self.n_inputs = n_inputs
        self.presentation_time = presentation_time
        self.dt = dt
        self.threshold = threshold
        self.device = device
        self.n_timesteps = int(presentation_time / dt)

    def encode(
        self,
        image: torch.Tensor,
        rate_scale: float = 1.0
    ) -> Generator[torch.Tensor, None, None]:
        """
        Generate TTFS spike train for an image.

        Matches PoissonEncoder.encode() interface. rate_scale compresses spike
        times toward t=0 (used by the adaptive retry mechanism).

        Args:
            image: Flattened image tensor [784] with values 0-255
            rate_scale: Time compression factor (>1 shifts spikes earlier)

        Yields:
            Binary spike tensor [n_inputs] for each timestep
        """
        image = image.view(-1).to(self.device)

        # Compute spike time for each pixel: brighter fires earlier
        # spike_time_ms in [0, presentation_time]; zero pixels get inf (never fire)
        spike_times_ms = torch.full((self.n_inputs,), float('inf'), device=self.device)
        active = image >= self.threshold
        spike_times_ms[active] = self.presentation_time * (1.0 - image[active] / 255.0)

        # rate_scale > 1 compresses spike times toward 0 (retry mechanism)
        spike_times_ms = spike_times_ms / rate_scale

        # Convert to timestep indices
        spike_timesteps = (spike_times_ms / self.dt).long()

        # Yield one binary vector per timestep
        for t in range(self.n_timesteps):
            spikes = (spike_timesteps == t).float()
            yield spikes

    def encode_with_count(
        self,
        image: torch.Tensor,
        rate_scale: float = 1.0
    ) -> Tuple[List[torch.Tensor], int]:
        """Generate spike trains and return total input spike count."""
        spike_trains = []
        total_spikes = 0
        for spikes in self.encode(image, rate_scale):
            spike_trains.append(spikes)
            total_spikes += spikes.sum().item()
        return spike_trains, int(total_spikes)


class AdaptivePoissonEncoder(PoissonEncoder):
    """
    Poisson encoder with adaptive rate increase.

    If the network doesn't produce enough spikes, the input rate
    is increased and the image is re-presented.
    """

    def __init__(
        self,
        n_inputs: int = 784,
        max_rate: float = 63.75,
        presentation_time: float = 350.0,
        dt: float = 0.5,
        min_spikes: int = 5,
        rate_increment: float = 32.0,
        device: str = "cuda"
    ):
        """
        Initialize the adaptive Poisson encoder.

        Args:
            n_inputs: Number of input neurons
            max_rate: Maximum firing rate in Hz
            presentation_time: Duration to present each image in ms
            dt: Simulation timestep in ms
            min_spikes: Minimum required excitatory spikes
            rate_increment: Rate increase per retry in Hz
            device: Device to use
        """
        super().__init__(n_inputs, max_rate, presentation_time, dt, device)
        self.min_spikes = min_spikes
        self.rate_increment = rate_increment
        self.max_rate_scale = 5.0  # Cap maximum rate scaling

    def compute_rate_scale(self, retry_count: int) -> float:
        """
        Compute the rate scaling factor based on retry count.

        Args:
            retry_count: Number of times image has been re-presented

        Returns:
            Rate scaling factor
        """
        # Each retry adds rate_increment Hz to the base max_rate
        additional_rate = self.rate_increment * retry_count
        scale = 1.0 + (additional_rate / self.max_rate)
        return min(scale, self.max_rate_scale)
