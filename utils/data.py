import torch
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple


def get_dataloaders(
    train_loader_batch_size: int = 64,
    test_loader_batch_size: int = 256,
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


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixes up the data and labels according to the mixup algorithm.

    Parameters:
    -----------
        x: The input tensor.
        y: The label tensor.
        alpha: The alpha parameter for the beta distribution. It controls the ratio of the mixup.

    Returns:
    --------
        A tuple (mixed_x, y_a, y_b, lam) where mixed_x is the mixed up input tensor, y_a and y_b are the labels, and lam is the lambda value used to mix the data.

    Raises:
    -------
        ValueError: If alpha is not between 0 and 1.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")

    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.CrossEntropyLoss | nn.MSELoss,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Computes the mixup loss according to the mixup algorithm.

    Parameters:
    -----------
        criterion: The loss function to use.
        pred: The predicted tensor.
        y_a: The first label tensor.
        y_b: The second label tensor.
        lam: The lambda value used to mix the labels.

    Returns:
    --------
        The mixed loss tensor.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
