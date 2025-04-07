import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import torch


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super(DeformableConv2d, self).__init__()

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # 2 offsets for each point in kernel
            kernel_size=kernel_size,
            padding=padding,
        )

        self.modulator_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,  # 1 modulator for each point in kernel
            kernel_size=kernel_size,
            padding=padding,
        )

        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        # Initialize weights for offset and modulator convs
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)  # type: ignore
        nn.init.constant_(self.modulator_conv.weight, 0)
        nn.init.constant_(self.modulator_conv.bias, 0)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate offsets and modulators
        offsets = self.offset_conv(x)
        modulators = 2.0 * torch.sigmoid(self.modulator_conv(x))

        # Get the weight from regular conv
        weight = self.regular_conv.weight

        # Apply deformable convolution
        x = deform_conv2d(
            x,
            offset=offsets,
            weight=weight,
            bias=self.regular_conv.bias,
            padding=self.regular_conv.padding,  # type: ignore
            mask=modulators,
        )
        return x


class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        # Deformable Convolution layer 1: input 1 channel, output 32 feature maps
        self.conv1 = DeformableConv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling

        # Deformable Convolution layer 2: input 32 channels, output 64 feature maps
        self.conv2 = DeformableConv2d(32, 64, kernel_size=3, padding=1)
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
