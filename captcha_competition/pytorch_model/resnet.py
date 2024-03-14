import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 10


class ResNet(nn.Module):

    def __init__(self, initial_filters: int = 64, multiplier: float = 2.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, initial_filters, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_filters)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dynamically create blocks based on initial_filters and multiplier
        filters = initial_filters
        self.blocks = nn.Sequential(
            ResNetBlock(filters, filters, stride=2),
            ResNetBlock(filters, filters, stride=1),
        )
        previous_filters = filters
        filters = int(round(filters * multiplier))
        self.blocks.add_module(
            "block3", ResNetBlock(previous_filters, filters, stride=2)
        )
        self.blocks.add_module(
            "block4", ResNetBlock(filters, filters, stride=1)
        )
        previous_filters = filters
        filters = int(round(filters * multiplier))

        self.blocks.add_module(
            "block5", ResNetBlock(previous_filters, filters, stride=2)
        )
        self.blocks.add_module(
            "block6", ResNetBlock(filters, filters, stride=1)
        )

        self.conv2d = nn.Conv2d(
            in_channels=filters,
            out_channels=NUM_CLASSES,
            kernel_size=(2, 1),
            stride=(1, 1),
            # padding="same",
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.blocks(x)

        x = self.conv2d(x)
        x = x.squeeze(2)
        return self.softmax(x)


class ResNetBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=(3, 3), stride=2
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


if __name__ == "__main__":
    from torchsummary import summary  # type: ignore

    model = ResNet()
    summary(model, (3, 64, 192))
