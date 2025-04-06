import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolution layer 1: input 1 channel (gray image), output 32 feature maps, kernel size 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling

        # Convolution layer 2: input 32 channels, output 64 feature maps, kernel size 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling

        # Fully connected layer
        self.fc1 = nn.Linear(
            64 * 7 * 7, 128
        )  # after two 2x2 pooling, 28x28 image becomes 7x7
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # output layer, 10 classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # flatten feature maps to one-dimensional vector
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
