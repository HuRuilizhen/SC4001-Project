import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple


def get_dataloaders(
    train_loader_batch_size: int = 64,
    test_loader_batch_size: int = 10000,
    root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Downloads and loads the FashionMNIST dataset.

    Parameters:
    -----------
        test_loader_batch_size: The batch size for the test DataLoader.
        train_loader_batch_size: The batch size for the train DataLoader.
        root: The root directory to download the dataset to.

    Returns:
    --------
        A tuple of two DataLoaders, one for the test set and one for the train set.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_loader_batch_size, shuffle=True
    )

    test_dataset = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_loader_batch_size, shuffle=False
    )

    return train_loader, test_loader
