import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from typing import Tuple
from utils.data import mixup_data, mixup_criterion


def get_device() -> torch.device:
    """
    Gets the best available device for running PyTorch models.

    Returns
    -------
    torch.device
        The best available device, which is the MPS device if it is available,
        otherwise the CUDA device if available, otherwise the CPU device.
    """
    return torch.device(
        "mps"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    )


def set_seed(seed: int = 42) -> int:
    """
    Sets the random seed for NumPy and PyTorch operations.

    Parameters
    ----------
    seed : int, optional
        The seed value to set. Defaults to 42.

    Returns
    -------
    int
        The seed value that was set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam | optim.SGD | optim.RMSprop | optim.AdamW,
    epochs: int = 5,
    device: torch.device = torch.device(
        "mps"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> tuple[nn.Module, list[float]]:
    """
    Train the model with the given training data loader and optimization parameters.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    train_loader : DataLoader
        The data loader for the training data.
    criterion : nn.CrossEntropyLoss
        The loss criterion for the training.
    optimizer : optim.Adam | optim.SGD | optim.RMSprop | optim.AdamW
        The optimizer for the training.
    epochs : int, optional
        The number of epochs to train the model. Defaults to 5.
    device : torch.device, optional
        The device to train the model on. Defaults to torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu").

    Returns
    -------
    tuple[nn.Module, list[float]]
        A tuple containing the trained model and a list of the loss at each epoch.
    """

    model.to(device)
    model.train()

    lost_docs = []
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0

        len_loader = len(train_loader)
        len_split = len_loader // 10

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if (i + 1) % len_split == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len_loader}], Loss: {running_loss/len_split:.4f}"
                )
                running_loss = 0.0

        lost_docs.append(epoch_loss / len(train_loader))

    print("Finished Training")

    return model, lost_docs


def train_mixup(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss | nn.MSELoss,
    optimizer: optim.Adam | optim.SGD | optim.RMSprop | optim.AdamW,
    epochs: int = 5,
    mixup_alpha: float = 0.2,
    device: torch.device = torch.device(
        "mps"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> tuple[nn.Module, list[float]]:

    model.to(device)
    model.train()

    lost_docs = []
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0

        len_loader = len(train_loader)
        len_split = len_loader // 10

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            mixed_x, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha)
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if (i + 1) % len_split == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len_loader}], Loss: {running_loss/len_split:.4f}"
                )
                running_loss = 0.0

        lost_docs.append(epoch_loss / len(train_loader))

    print("Finished Training")

    return model, lost_docs


def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss | nn.MSELoss,
    device: torch.device = torch.device(
        "mps"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> Tuple[float, float]:
    """
    Evaluates the model on the test dataset and calculates accuracy and loss.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be evaluated.
    test_loader : DataLoader
        DataLoader for the test dataset.
    criterion : nn.CrossEntropyLoss | nn.MSELoss
        The loss function used to compute the loss.

    Returns
    -------
    tuple[float, float]
        A tuple containing the accuracy (as a percentage) and the average test loss.
    """

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    accuracy = 100 * correct / total
    test_loss /= len(test_loader)

    print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    return accuracy, test_loss
