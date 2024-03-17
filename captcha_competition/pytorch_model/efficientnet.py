import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 10
INPUT_SHAPE = (3, 80, 210)


class EfficientNet(nn.Module):
    def __init__(
        self,
        initial_filters: int = 32,
        multiplier: float = 1.5,
        in_channels: int = 3,
    ):
        super().__init__()
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=initial_filters,
            kernel_size=3,
            stride=2,
            bias=False,
        )
        self.initial_bn = nn.BatchNorm2d(initial_filters)

        # Setup MBConv blocks dynamically
        current_filters = round(initial_filters * multiplier)
        print(f"{current_filters=}")
        self.mb_conv_blocks = nn.Sequential(
            MBConvBlock(
                initial_filters,
                current_filters,
                kernel_size=3,
                stride=2,
            ),
        )
        for i in range(2):
            next_filters = round(current_filters * multiplier)
            self.mb_conv_blocks.add_module(
                f"mb_conv_block_{i + 2}",
                MBConvBlock(
                    in_channels=current_filters,
                    out_channels=next_filters,
                    kernel_size=3,
                    stride=2,
                ),
            )
            current_filters = next_filters

        self.conv2d = nn.Conv2d(
            in_channels=current_filters,
            out_channels=NUM_CLASSES,
            kernel_size=(4, 2),
            stride=(1, 2),
        )
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.mb_conv_blocks(x)
        x = self.conv2d(x)
        x = x.squeeze(2)
        return self.softmax(x)


class MBConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size, stride
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.dw_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            # padding="same",
            groups=out_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.dw_conv(x)))
        x = self.bn3(self.conv2(x))
        return x


if __name__ == "__main__":
    from torchsummary import summary  # type: ignore

    model = EfficientNet()
    summary(model, INPUT_SHAPE, device="cpu")
