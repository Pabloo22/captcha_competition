import torch
from torch import nn
import torch.nn.functional as F

from captcha_competition.pytorch_model.resnet import ResNetBlock

NUM_CLASSES = 10
NUM_DIGITS = 6


class ResNetTransformer(nn.Module):

    def __init__(
        self,
        initial_filters: int = 64,
        multiplier: float = 2.0,
        in_channels: int = 3,
        embedding_size: int = 128,
        n_head: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_first: bool = False,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(
            in_channels,
            initial_filters,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
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
            out_channels=embedding_size,
            kernel_size=(2, 1),
            stride=(1, 1),
        )

        # Replace the last conv 2d layer with a EncoderTransformer
        self.positional_encoding = nn.Parameter(
            torch.randn(NUM_DIGITS, embedding_size)
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=n_head,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.fc = nn.Linear(embedding_size, num_classes)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.blocks(x)
        x = self.conv2d(x)
        # Remove the height dimension, now shape: [batch_size, emb_size, 6]
        x = x.squeeze(2)
        x = x.transpose(1, 2)  # Transpose to get shape [-1, 6, emb_size]
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.fc(x)  # x has shape [-1, 6, 10]
        return self.softmax(x)


if __name__ == "__main__":
    from torchinfo import summary  # type: ignore

    model = ResNetTransformer()
    summary(model, (1, 3, 64, 192), device="cpu")
    # print(model)
    # dummy_input = torch.randn(1, 3, 64, 192)
    # out = model(dummy_input)
    # print(out.shape)
