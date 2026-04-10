"""
Neuron labeling after unsupervised training.

Based on Diehl & Cook (2015):
- After training, each neuron is assigned to the class (digit)
  that evokes the highest average response
- This is the only step where labels are used
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..network.diehl_network import DiehlNetwork


class NeuronLabeler:
    """
    Assigns class labels to neurons based on their responses.

    Each neuron is assigned to the digit class that produces
    the highest average spike count from that neuron.
    """

    def __init__(
        self,
        network: DiehlNetwork,
        n_classes: int = 10
    ):
        """
        Initialize the labeler.

        Args:
            network: Trained DiehlNetwork
            n_classes: Number of classes (10 for MNIST digits)
        """
        self.network = network
        self.n_classes = n_classes
        self.n_exc = network.n_exc

        # Labels assigned to each neuron (-1 = unassigned)
        self.neuron_labels = torch.full(
            (self.n_exc,), -1,
            device=network.device, dtype=torch.long
        )

        # Response matrix: [n_exc, n_classes]
        # Accumulates total spikes per neuron per class
        self.response_matrix = torch.zeros(
            self.n_exc, n_classes,
            device=network.device, dtype=torch.float32
        )

        # Count of samples per class
        self.class_counts = torch.zeros(
            n_classes, device=network.device, dtype=torch.float32
        )

    def compute_responses(self, data_loader: DataLoader, max_samples: int = None):
        """
        Compute average responses for each neuron to each class.

        Args:
            data_loader: DataLoader with labeled data
            max_samples: Maximum samples to use (None = all)
        """
        self._compute_responses_impl(data_loader, max_samples, collect_vectors=False)

    def compute_responses_with_vectors(
        self,
        data_loader: DataLoader,
        max_samples: int = None
    ):
        """
        Compute average responses and return individual spike-count vectors.

        Runs the same labeling pass as compute_responses() but also collects
        the per-image spike-count vectors needed to train a logistic regression.

        Args:
            data_loader: DataLoader with labeled data
            max_samples: Maximum samples to use (None = all)

        Returns:
            spike_vectors: numpy array [n_samples, n_exc], raw spike counts per image
            image_labels:  numpy array [n_samples], ground-truth digit label per image
        """
        return self._compute_responses_impl(data_loader, max_samples, collect_vectors=True)

    def _compute_responses_impl(
        self,
        data_loader: DataLoader,
        max_samples: int = None,
        collect_vectors: bool = False
    ):
        """Shared implementation for compute_responses and compute_responses_with_vectors."""

        # Reset accumulators
        self.response_matrix.zero_()
        self.class_counts.zero_()

        spike_vectors_list = [] if collect_vectors else None
        image_labels_list = [] if collect_vectors else None

        n_samples = 0
        pbar = tqdm(data_loader, desc="Computing responses", unit="img")

        for images, labels in pbar:
            for img, label in zip(images, labels):
                # Get network response (no learning)
                spike_counts = self.network.present_image(img, learning=False)

                # Accumulate responses
                label_idx = label.item()
                self.response_matrix[:, label_idx] += spike_counts
                self.class_counts[label_idx] += 1

                if collect_vectors:
                    spike_vectors_list.append(spike_counts.cpu().numpy())
                    image_labels_list.append(label_idx)

                n_samples += 1
                if max_samples and n_samples >= max_samples:
                    break

            if max_samples and n_samples >= max_samples:
                break

        print(f"Computed responses for {n_samples} samples")

        if collect_vectors:
            return (
                np.stack(spike_vectors_list, axis=0),
                np.array(image_labels_list, dtype=np.int64)
            )

    def assign_labels(self, method: str = 'argmax') -> torch.Tensor:
        """
        Assign each neuron to a class based on its response profile.

        Args:
            method: 'argmax' (each neuron assigned to highest-response class) or
                    'balanced' (guarantees floor(n_exc/n_classes) neurons per class).
                    Note: balanced only affects single/VFO voting. VFA uses the
                    raw response profile regardless of this setting.

        Returns:
            Tensor of neuron labels [n_exc]
        """
        # Compute average response per class
        class_counts_safe = torch.clamp(self.class_counts, min=1.0)
        avg_response = self.response_matrix / class_counts_safe.unsqueeze(0)

        if method == 'balanced':
            self.neuron_labels = self._assign_balanced(avg_response)
        else:
            self.neuron_labels = avg_response.argmax(dim=1)

        # Store normalized response profile for VFA decoding (unaffected by method)
        self._response_profile = self._compute_response_profile(avg_response)

        # Log assignment statistics
        self._log_assignment_stats(avg_response)

        return self.neuron_labels

    def _assign_balanced(self, avg_response: torch.Tensor) -> torch.Tensor:
        """
        Balanced greedy assignment guaranteeing floor(n_exc/n_classes) neurons per class.

        Weakest classes (lowest total response) get first pick of their best neurons,
        then remaining neurons are assigned by argmax.
        """
        min_per_class = self.n_exc // self.n_classes

        if min_per_class == 0:
            print(f"  Warning: n_exc ({self.n_exc}) < n_classes ({self.n_classes}), "
                  f"falling back to argmax labeling", flush=True)
            return avg_response.argmax(dim=1)

        # Work on CPU numpy for clean indexing
        R = avg_response.cpu()
        labels = torch.full((self.n_exc,), -1, dtype=torch.long)
        assigned = torch.zeros(self.n_exc, dtype=torch.bool)

        # Order classes weakest-first (lowest total response gets first pick)
        class_totals = R.sum(dim=0)
        class_order = class_totals.argsort()

        # Phase 1: guarantee min_per_class neurons per class
        for c in class_order.tolist():
            # Among unassigned neurons, find the min_per_class with highest R[:,c]
            unassigned_idx = (~assigned).nonzero(as_tuple=True)[0]
            scores = R[unassigned_idx, c]
            # argsort descending, take top min_per_class
            top_local = scores.argsort(descending=True)[:min_per_class]
            top_global = unassigned_idx[top_local]
            labels[top_global] = c
            assigned[top_global] = True

        # Phase 2: remaining neurons assigned by argmax
        remaining_idx = (~assigned).nonzero(as_tuple=True)[0]
        if len(remaining_idx) > 0:
            labels[remaining_idx] = R[remaining_idx].argmax(dim=1)

        # Log how many neurons were moved from their argmax preference
        argmax_labels = avg_response.argmax(dim=1).cpu()
        n_reassigned = (labels != argmax_labels).sum().item()
        print(f"  Labeling method: balanced (min_per_class={min_per_class})", flush=True)
        print(f"  Neurons reassigned from argmax: {n_reassigned}", flush=True)

        return labels.to(avg_response.device)

    def _compute_response_profile(self, avg_response: torch.Tensor) -> torch.Tensor:
        """
        Normalize per-neuron average responses for VFA decoding.

        Each row sums to 1. Neurons that never fired for any class
        receive a uniform 1/10 weight across all classes.
        """
        row_sums = avg_response.sum(dim=1, keepdim=True)
        uniform = torch.full_like(avg_response, 1.0 / self.n_classes)
        return torch.where(row_sums > 0, avg_response / row_sums.clamp(min=1e-9), uniform)

    def get_response_profile(self) -> torch.Tensor:
        """
        Return the normalized per-neuron response profile [n_exc, n_classes].

        Each row sums to 1. Used for VFA decoding.
        assign_labels() must be called first.
        """
        if not hasattr(self, '_response_profile'):
            raise RuntimeError("Call assign_labels() before get_response_profile()")
        return self._response_profile

    def _log_assignment_stats(self, avg_response: torch.Tensor):
        """Log statistics about label assignments."""
        labels = self.neuron_labels

        print("\nLabel Assignment Statistics:")
        print("-" * 40)

        # Count neurons per class
        for c in range(self.n_classes):
            count = (labels == c).sum().item()
            avg_resp = avg_response[labels == c, c].mean().item() if count > 0 else 0
            print(f"  Class {c}: {count:3d} neurons (avg response: {avg_resp:.2f})")

        # Neurons with zero response to their assigned class
        max_responses = avg_response.max(dim=1)[0]
        zero_response = (max_responses < 0.1).sum().item()
        print(f"\nNeurons with very low response: {zero_response}")

    def get_labels(self) -> torch.Tensor:
        """Get current neuron labels."""
        return self.neuron_labels

    def get_response_matrix(self) -> torch.Tensor:
        """Get the full response matrix."""
        return self.response_matrix

    def get_class_distribution(self) -> dict:
        """Get distribution of neurons across classes."""
        distribution = {}
        for c in range(self.n_classes):
            distribution[c] = (self.neuron_labels == c).sum().item()
        return distribution

    def save(self, path: str):
        """Save labeling results."""
        data = {
            'neuron_labels': self.neuron_labels.cpu(),
            'response_matrix': self.response_matrix.cpu(),
            'class_counts': self.class_counts.cpu(),
        }
        if hasattr(self, '_response_profile'):
            data['response_profile'] = self._response_profile.cpu()
        torch.save(data, path)

    def load(self, path: str):
        """Load labeling results."""
        data = torch.load(path, map_location=self.network.device)
        self.neuron_labels = data['neuron_labels'].to(self.network.device)
        self.response_matrix = data['response_matrix'].to(self.network.device)
        self.class_counts = data['class_counts'].to(self.network.device)
        if 'response_profile' in data:
            self._response_profile = data['response_profile'].to(self.network.device)
        else:
            # Recompute profile from response_matrix if not saved (older checkpoints)
            class_counts_safe = torch.clamp(self.class_counts, min=1.0)
            avg_response = self.response_matrix / class_counts_safe.unsqueeze(0)
            self._response_profile = self._compute_response_profile(avg_response)
