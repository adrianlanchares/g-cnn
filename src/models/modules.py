import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn

GSPACES = {
    "Z2": lambda: gspaces.flip2dOnR2(),  # reflection only,  group size 2
    "p4": lambda: gspaces.rot2dOnR2(N=4),  # 4 rotations,      group size 4
    "p4m": lambda: gspaces.flipRot2dOnR2(N=4),  # rotations+flips,  group size 8
}


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batchnorm: bool = False,
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
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class GECNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        group: str,
        batchnorm: bool = False,
    ):
        super().__init__()

        in_type = enn.FieldType(
            GSPACES[group](), in_channels * [GSPACES[group]().regular_repr]
        )
        out_type = enn.FieldType(
            GSPACES[group](), out_channels * [GSPACES[group]().regular_repr]
        )

        layers = [
            enn.R2Conv(
                in_type=in_type,
                out_type=out_type,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # standard practice with equivariant convs
            ),
            enn.ReLU(out_type),  # equivariant ReLU
        ]
        if batchnorm:
            layers.append(enn.InnerBatchNorm(out_type))

        self.block = enn.SequentialModule(*layers)

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
