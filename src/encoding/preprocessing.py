"""
Difference-of-Gaussian (DoG) preprocessing for MNIST images.

Models retinal ganglion cell center-surround processing:
- ON-center channel: responds to bright edges (positive DoG response)
- OFF-center channel: responds to dark edges (negative DoG response)
"""
import torch
import torch.nn.functional as F
import math


class DoGPreprocessor:
    """
    Difference-of-Gaussian edge filter applied before spike encoding.

    Computes:
        on  = ReLU(blur(image, sigma1) - blur(image, sigma2))
        off = ReLU(blur(image, sigma2) - blur(image, sigma1))

    Each channel is independently rescaled to 0-255 so the encoder
    receives values in the same range as raw pixels.

    Args:
        sigma1: Inner (center) Gaussian sigma in pixels. Default 1.0.
        sigma2: Outer (surround) Gaussian sigma in pixels. Default 2.0.
        on_off: If True, return both ON and OFF channels concatenated [1568].
                If False, return only ON channel [784].
        device: Torch device for kernel tensors.
    """

    def __init__(
        self,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        on_off: bool = True,
        device: str = 'cuda'
    ):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.on_off = on_off
        self.device = device

        # Pre-compute Gaussian kernels: shape [1, 1, k, k]
        self.kernel1 = self._make_gaussian_kernel(sigma1, device)
        self.kernel2 = self._make_gaussian_kernel(sigma2, device)

    def _make_gaussian_kernel(self, sigma: float, device: str) -> torch.Tensor:
        """Build a 2D Gaussian kernel tensor [1, 1, k, k]."""
        # Kernel size: 2 * ceil(3 * sigma) + 1 (captures ±3σ)
        half = math.ceil(3 * sigma)
        size = 2 * half + 1

        # 1D Gaussian
        x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
        g1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()

        # Outer product → 2D kernel
        kernel = torch.outer(g1d, g1d)
        return kernel.view(1, 1, size, size)

    def to(self, device: str) -> 'DoGPreprocessor':
        """Move kernels to a new device."""
        self.device = device
        self.kernel1 = self.kernel1.to(device)
        self.kernel2 = self.kernel2.to(device)
        return self

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply DoG preprocessing to a flattened MNIST image.

        Args:
            image: Tensor [784] with values 0-255, on any device.

        Returns:
            Tensor [784] (on_off=False) or [1568] (on_off=True), values 0-255.
        """
        device = image.device

        # Move kernels to image device if needed
        if self.kernel1.device != device:
            self.kernel1 = self.kernel1.to(device)
            self.kernel2 = self.kernel2.to(device)

        # Reshape to [1, 1, 28, 28] for conv2d
        img2d = image.view(1, 1, 28, 28).float()

        # Padding to keep 28x28 output
        pad1 = self.kernel1.shape[-1] // 2
        pad2 = self.kernel2.shape[-1] // 2

        blur1 = F.conv2d(img2d, self.kernel1, padding=pad1)  # [1,1,28,28]
        blur2 = F.conv2d(img2d, self.kernel2, padding=pad2)  # [1,1,28,28]

        # ON-center: positive response where center > surround
        on = F.relu(blur1 - blur2).view(784)

        # Rescale to 0-255
        on = self._rescale(on)

        if not self.on_off:
            return on

        # OFF-center: positive response where surround > center
        off = F.relu(blur2 - blur1).view(784)
        off = self._rescale(off)

        return torch.cat([on, off], dim=0)  # [1568]

    @staticmethod
    def _rescale(channel: torch.Tensor) -> torch.Tensor:
        """Rescale a channel to 0-255. Returns all-zeros if channel is blank."""
        max_val = channel.max()
        if max_val < 1e-9:
            return channel
        return channel / max_val * 255.0
