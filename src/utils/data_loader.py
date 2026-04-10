"""
MNIST data loading utilities.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple


def get_mnist_loaders(
    data_dir: str = "./data",
    batch_size: int = 1,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST training and test datasets.

    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for data loaders (default 1 for online learning)
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create data directory if needed
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Transform: convert to tensor (scales to [0, 1]) then scale to [0, 255]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)  # Scale to 0-255 range
    ])

    # Load datasets
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_single_image(
    loader: DataLoader,
    index: int = 0
) -> Tuple[torch.Tensor, int]:
    """
    Get a single image from a data loader by index.

    Args:
        loader: MNIST data loader
        index: Image index

    Returns:
        Tuple of (image tensor [1, 28, 28], label)
    """
    dataset = loader.dataset
    image, label = dataset[index]
    return image, label


def flatten_image(image: torch.Tensor) -> torch.Tensor:
    """
    Flatten a 28x28 MNIST image to 784 pixels.

    Args:
        image: Image tensor of shape [1, 28, 28] or [28, 28]

    Returns:
        Flattened tensor of shape [784]
    """
    return image.view(-1)
