"""
Visualization utilities for the STDP network.


Visualizing learned weight matrices as receptive fields
Plotting spike rasters
Training progress plots
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_weights(
    weights: torch.Tensor,
    n_rows: int = None,
    n_cols: int = None,
    title: str = "Learned Weights",
    save_path: str = None,
    figsize: tuple = None
):
    """
    Visualize weight matrix as a grid of receptive fields.

    Each excitatory neuron's input weights are reshaped to 28x28
    to show the learned digit pattern.

    Args:
        weights: Weight matrix [784, n_exc]
        n_rows: Number of rows in grid (auto if None)
        n_cols: Number of columns in grid (auto if None)
        title: Figure title
        save_path: Path to save figure (shows if None)
        figsize: Figure size (auto if None)
    """
    # Move to CPU if needed
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()

    n_neurons = weights.shape[1]

    # Determine grid size
    if n_rows is None or n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_neurons)))
        n_rows = int(np.ceil(n_neurons / n_cols))

    # Determine figure size
    if figsize is None:
        figsize = (n_cols * 1.2, n_rows * 1.2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    # Plot each neuron's weights
    # If input_dim > 784 (e.g. DoG ON+OFF channels), show only first 784 (ON channel)
    weights_vis = weights[:784, :] if weights.shape[0] > 784 else weights
    for i in range(n_neurons):
        ax = axes[i]
        # Reshape 784 -> 28x28
        rf = weights_vis[:, i].reshape(28, 28)
        ax.imshow(rf, cmap='gray', vmin=0, vmax=weights_vis.max())
        ax.axis('off')

    # Hide unused axes
    for i in range(n_neurons, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved weights visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_weight_distribution(
    weights: torch.Tensor,
    title: str = "Weight Distribution",
    save_path: str = None
):
    """
    Plot histogram of weight values.

    Args:
        weights: Weight tensor
        title: Figure title
        save_path: Path to save figure
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(weights, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(weights.mean(), color='red', linestyle='--', label=f'Mean: {weights.mean():.4f}')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_spike_raster(
    spike_times: dict,
    duration: float,
    title: str = "Spike Raster",
    save_path: str = None
):
    """
    Plot spike raster for multiple neurons.

    Args:
        spike_times: Dict mapping neuron_id -> list of spike times
        duration: Total duration in ms
        title: Figure title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for neuron_id, times in spike_times.items():
        ax.scatter(times, [neuron_id] * len(times), marker='|', s=10, c='black')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_xlim(0, duration)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_training_progress(
    metrics: dict,
    save_path: str = None
):
    """
    Plot training progress metrics.

    Args:
        metrics: Dict with 'images', 'spikes', 'weight_mean', etc.
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Average spikes over time
    if 'images' in metrics and 'spikes' in metrics:
        ax = axes[0, 0]
        ax.plot(metrics['images'], metrics['spikes'])
        ax.set_xlabel('Images Presented')
        ax.set_ylabel('Average Spikes')
        ax.set_title('Network Activity')

    # Weight statistics
    if 'weight_mean' in metrics:
        ax = axes[0, 1]
        ax.plot(metrics['images'], metrics['weight_mean'], label='Mean')
        if 'weight_std' in metrics:
            ax.fill_between(
                metrics['images'],
                np.array(metrics['weight_mean']) - np.array(metrics['weight_std']),
                np.array(metrics['weight_mean']) + np.array(metrics['weight_std']),
                alpha=0.3
            )
        ax.set_xlabel('Images Presented')
        ax.set_ylabel('Weight Value')
        ax.set_title('Weight Statistics')
        ax.legend()

    # Theta (adaptive threshold)
    if 'theta_mean' in metrics:
        ax = axes[1, 0]
        ax.plot(metrics['images'], metrics['theta_mean'], label='Mean')
        ax.plot(metrics['images'], metrics['theta_max'], label='Max', linestyle='--')
        ax.set_xlabel('Images Presented')
        ax.set_ylabel('Theta (mV)')
        ax.set_title('Adaptive Threshold')
        ax.legend()

    # Hide unused subplot
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion: torch.Tensor,
    title: str = "Confusion Matrix",
    save_path: str = None
):
    """
    Plot confusion matrix as heatmap.

    Args:
        confusion: Confusion matrix [n_classes, n_classes]
        title: Figure title
        save_path: Path to save figure
    """
    if isinstance(confusion, torch.Tensor):
        confusion = confusion.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(confusion, cmap='Blues')
    plt.colorbar(im, ax=ax)

    # Add labels
    n_classes = confusion.shape[0]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, int(confusion[i, j]),
                          ha='center', va='center',
                          color='white' if confusion[i, j] > confusion.max()/2 else 'black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def visualize_neuron_labels(
    weights: torch.Tensor,
    labels: torch.Tensor,
    n_rows: int = None,
    n_cols: int = None,
    title: str = "Neuron Labels",
    save_path: str = None
):
    """
    Visualize weights with their assigned labels.

    Args:
        weights: Weight matrix [784, n_exc]
        labels: Neuron labels [n_exc]
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        title: Figure title
        save_path: Path to save figure
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    n_neurons = weights.shape[1]

    if n_rows is None or n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_neurons)))
        n_rows = int(np.ceil(n_neurons / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.array(axes).flatten()

    weights_vis = weights[:784, :] if weights.shape[0] > 784 else weights
    for i in range(n_neurons):
        ax = axes[i]
        rf = weights_vis[:, i].reshape(28, 28)
        ax.imshow(rf, cmap='gray', vmin=0, vmax=weights_vis.max())
        ax.set_title(f'{labels[i]}', fontsize=8)
        ax.axis('off')

    for i in range(n_neurons, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
