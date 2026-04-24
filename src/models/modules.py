import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
    ):
        super().__init__()

        layers = [nn.Linear(in_features, out_features)]

        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
