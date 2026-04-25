import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn

from src.models.modules import GECNNBlock, MLPBlock

GSPACES = {
    "Z2": lambda: gspaces.flip2dOnR2(),  # reflection only,  group size 2
    "p4": lambda: gspaces.rot2dOnR2(N=4),  # 4 rotations,      group size 4
    "p4m": lambda: gspaces.flipRot2dOnR2(N=4),  # rotations+flips,  group size 8
}


class GECNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        padding: list[int],
        batchnorm: bool,
        linear_hidden_features: list[int],
        linear_out_features: int = 1,
        group: str = "Z2",
        final_tanh: bool = False,
    ):
        """Group-Equivariant CNN using escnn.

        Architecture mirrors BaseCNN exactly:
            - (R2Conv -> Activation -> (PointwiseMaxPool)?) * N
            - R2Conv (1x1 projection)
            - GroupPooling (invariant aggregation over group dimension)
            - AdaptiveAvgPool2d(1,1)
            - Flatten
            - Linear

        The group is controlled by cfg.group. Feature maps are GeometricTensors
        carrying both data and transformation behaviour under the group.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: List of hidden channel sizes.
            kernel_sizes: List of kernel sizes for each convolutional layer.
            strides: List of strides for each convolutional layer.
            padding: List of padding for each convolutional layer.
            linear_out_features: Number of output features for the final linear layer.

        Raises:
            ValueError: If hidden_channels and kernel_sizes have different lengths.
            ValueError: If cfg.group is not one of: Z2, p4, p4m.
        """
        super().__init__()

        if len(hidden_channels) != len(kernel_sizes):
            raise ValueError(
                "Length of hidden_channels and kernel_sizes must be the same."
            )
        if group not in GSPACES:
            raise ValueError(
                f"Unknown group '{group}'. Choose from: {list(GSPACES.keys())}"
            )

        gspace = GSPACES[group]()
        self.in_type = enn.FieldType(gspace, in_channels * [gspace.regular_repr])

        blocks = []

        for hidden_channel, kernel_size, stride, pad in zip(
            hidden_channels, kernel_sizes, strides, padding
        ):
            blocks.append(
                GECNNBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    group=group,
                    batchnorm=batchnorm,
                )
            )
            in_channels = hidden_channel  # for next layer

        blocks.append(
            GECNNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
                group=group,
            )
        )

        self.eq_layers = enn.SequentialModule(*blocks)

        self.flatten = nn.Flatten()

        # Calculate the number of features after the convolutional layers to determine the input size for the linear layer
        linear_in_features = out_channels
        linear_layers = []
        for hidden_feature in linear_hidden_features:
            linear_layers.append(MLPBlock(linear_in_features, hidden_feature))
            linear_in_features = hidden_feature

        linear_layers.append(nn.Linear(linear_in_features, linear_out_features))
        if final_tanh:
            linear_layers.append(nn.Tanh())

        self.head = nn.Sequential(*linear_layers)

    def _get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # wrap input tensor into a GeometricTensor so escnn can track transformations
        x = enn.GeometricTensor(x, self.in_type)

        x = self.eq_layers(x)
        x = x.tensor

        x = self.flatten(x)
        x = self.head(x)
        return x
