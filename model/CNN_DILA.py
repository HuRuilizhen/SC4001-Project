import torch
import torch.nn as nn


class CNN_DILA(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the layers of the CNN_DILA model.

        The CNN_DILA model consists of two convolutional layers and two fully connected layers.
        The first convolutional layer takes as input a 1x28x28 image and outputs 32 feature maps.
        The second convolutional layer takes as input 32 feature maps and outputs 64 feature maps.
        The output of the second convolutional layer is flattened and passed to the first fully connected layer,
        which outputs 128 feature maps. The output of the first fully connected layer is passed to the second fully connected layer,
        which outputs 10 feature maps, corresponding to the 10 classes of the MNIST dataset.
        """
        super(CNN_DILA, self).__init__()
        # Convolution layer 1: input 1 channel (gray image), output 32 feature maps, kernel size 3x3, dilation 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=2, dilation=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling

        # Convolution layer 2: input 32 channels, output 64 feature maps, kernel size 3x3, dilation 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling

        # Fully connected layer
        self.fc1 = nn.Linear(
            64 * 7 * 7, 128
        )  # after two 2x2 pooling, 28x28 image becomes 7x7
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # output layer, 10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN_DILA model.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor with shape (batch_size, channels, height, width).

        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, num_classes), representing
            the class scores for each input image.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # flatten feature maps to one-dimensional vector
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
