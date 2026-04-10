"""
Training loop for the Diehl & Cook (2015) network.

"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import time

from ..network.diehl_network import DiehlNetwork


class Trainer:
    """
    Trainer for the STDP network.

    Handles the unsupervised training loop where images are
    presented to the network and STDP updates the weights.
    """

    def __init__(
        self,
        network: DiehlNetwork,
        train_loader: DataLoader,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 1000,
        save_interval: int = 10000
    ):
        """
        Initialize the trainer.

        Args:
            network: The DiehlNetwork to train
            train_loader: DataLoader for MNIST training set
            checkpoint_dir: Directory to save checkpoints
            log_interval: Log progress every N images
            save_interval: Save checkpoint every N images
        """
        self.network = network
        self.train_loader = train_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.total_images = 0
        self.total_spikes = 0
        self.total_retries = 0

    def train_epoch(
        self,
        epoch: int = 0,
        use_adaptive: bool = True,
        max_images: Optional[int] = None
    ) -> dict:
        """
        Train for one epoch (one pass through training set).

        Args:
            epoch: Current epoch number
            use_adaptive: Whether to use adaptive rate increase
            max_images: Optional limit on number of images this epoch

        Returns:
            Dictionary of training statistics
        """
        epoch_spikes = 0
        epoch_retries = 0
        epoch_images = 0
        start_time = time.time()

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            unit="img"
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            # Process each image in batch (usually batch_size=1)
            for img in images:
                # Optional early exit for quick trials
                if max_images is not None and epoch_images >= max_images:
                    break

                # Present image to network
                if use_adaptive:
                    spike_counts, retries = self.network.present_image_adaptive(
                        img, learning=True
                    )
                    epoch_retries += retries
                else:
                    spike_counts = self.network.present_image(img, learning=True)

                total_spikes = spike_counts.sum().item()
                epoch_spikes += total_spikes
                epoch_images += 1
                self.total_images += 1

                # Update progress bar
                pbar.set_postfix({
                    'spikes': f"{total_spikes:.0f}",
                    'avg': f"{epoch_spikes/epoch_images:.1f}"
                })

                # Logging
                if self.total_images % self.log_interval == 0:
                    self._log_progress(epoch, epoch_images, epoch_spikes, epoch_retries)

                # Checkpointing
                if self.total_images % self.save_interval == 0:
                    self._save_checkpoint(epoch)

            if max_images is not None and epoch_images >= max_images:
                break

        # End of epoch
        elapsed = time.time() - start_time
        stats = {
            'epoch': epoch,
            'images': epoch_images,
            'total_spikes': epoch_spikes,
            'avg_spikes': epoch_spikes / epoch_images,
            'retries': epoch_retries,
            'time': elapsed,
            'images_per_sec': epoch_images / elapsed
        }

        self._log_epoch_summary(stats)
        self._save_checkpoint(epoch, final=True)

        return stats

    def train(
        self,
        n_epochs: int = 3,
        use_adaptive: bool = True,
        max_images: Optional[int] = None,
        start_epoch: int = 0
    ) -> list:
        """
        Train for multiple epochs.

        Args:
            n_epochs: Number of epochs to train
            use_adaptive: Whether to use adaptive rate increase
            max_images: Optional limit on number of images per epoch
            start_epoch: Epoch number to start from (for resuming)

        Returns:
            List of per-epoch statistics
        """
        all_stats = []

        print(f"\nStarting training for {n_epochs} epochs")
        print(f"Network: {self.network.n_exc} excitatory neurons")
        print(f"Training set: {len(self.train_loader.dataset)} images")
        print("-" * 50)

        for epoch in range(start_epoch, start_epoch + n_epochs):
            stats = self.train_epoch(
                epoch,
                use_adaptive=use_adaptive,
                max_images=max_images
            )
            all_stats.append(stats)

        print("\nTraining complete!")
        return all_stats

    def _log_progress(
        self,
        epoch: int,
        images: int,
        spikes: int,
        retries: int
    ):
        """Log training progress."""
        weight_stats = self.network.get_weight_stats()
        theta_stats = self.network.get_theta_stats()

        print(f"\n[Epoch {epoch} | Image {images}]")
        print(f"  Avg spikes: {spikes/images:.1f}")
        print(f"  Retries: {retries}")
        print(f"  Weights: mean={weight_stats['mean']:.4f}, "
              f"std={weight_stats['std']:.4f}, "
              f"near_zero={weight_stats['near_zero']:.1%}")
        print(f"  Theta: mean={theta_stats['mean']:.2f}, "
              f"max={theta_stats['max']:.2f}")

    def _log_epoch_summary(self, stats: dict):
        """Log epoch summary."""
        print(f"\n{'='*50}")
        print(f"Epoch {stats['epoch']} Complete")
        print(f"  Images: {stats['images']}")
        print(f"  Avg spikes/image: {stats['avg_spikes']:.1f}")
        print(f"  Total retries: {stats['retries']}")
        print(f"  Time: {stats['time']:.1f}s ({stats['images_per_sec']:.1f} img/s)")
        print(f"{'='*50}")

    def _save_checkpoint(self, epoch: int, final: bool = False):
        """Save training checkpoint."""
        if final:
            filename = f"checkpoint_epoch{epoch}_final.pt"
        else:
            filename = f"checkpoint_epoch{epoch}_img{self.total_images}.pt"

        path = self.checkpoint_dir / filename
        self.network.save_checkpoint(str(path))
        print(f"  Saved checkpoint: {filename}")


class TrainingMonitor:
    """
    Monitor for tracking training metrics over time.
    """

    def __init__(self):
        self.metrics = {
            'images': [],
            'spikes': [],
            'weight_mean': [],
            'weight_std': [],
            'theta_mean': [],
            'theta_max': [],
        }

    def record(
        self,
        image_count: int,
        spike_count: float,
        weight_stats: dict,
        theta_stats: dict
    ):
        """Record metrics at current point."""
        self.metrics['images'].append(image_count)
        self.metrics['spikes'].append(spike_count)
        self.metrics['weight_mean'].append(weight_stats['mean'])
        self.metrics['weight_std'].append(weight_stats['std'])
        self.metrics['theta_mean'].append(theta_stats['mean'])
        self.metrics['theta_max'].append(theta_stats['max'])

    def get_dataframe(self):
        """Convert metrics to pandas DataFrame (if available)."""
        try:
            import pandas as pd
            return pd.DataFrame(self.metrics)
        except ImportError:
            return self.metrics
