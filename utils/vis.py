import torch
import matplotlib.pyplot as plt
import numpy as np


class CONST:
    LABELS = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    DEVICE_CPU = torch.device("cpu")


def imshow(img: torch.Tensor) -> None:
    """
    A helper function to display an image

    Parameters
    ----------
    img : torch.Tensor
        The input image tensor with shape (3, H, W)

    Returns
    -------
    None
    """
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def get_predictions_examples(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    classes=CONST.LABELS,
    num_examples: int = 6,
) -> None:
    """
    Displays a set of example predictions from a model.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model used for making predictions.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    classes : list, optional
        List of class names for labeling the predictions. Defaults to CONST.LABELS.
    num_examples : int, optional
        The number of prediction examples to display. Defaults to 6.

    Returns
    -------
    None
    """

    model.to(CONST.DEVICE_CPU)
    model.eval()

    with torch.no_grad():
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        fig = plt.figure(figsize=(12, 4))
        for idx in np.arange(num_examples):
            ax = fig.add_subplot(1, num_examples, int(idx + 1), xticks=[], yticks=[])
            imshow(images[idx])
            ax.set_title(
                f"{classes[predicted[idx]]} ({classes[labels[idx]]})",
                color=("green" if predicted[idx] == labels[idx] else "red"),
            )
        plt.show()
