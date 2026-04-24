import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn

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
        linear_out_features: int = 1,
        group: str = "Z2",
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

        self.gspace = GSPACES[group]()
        repr = self.gspace.regular_repr

        self.in_type = enn.FieldType(
            self.gspace, in_channels * [self.gspace.trivial_repr]
        )

        blocks = []
        current_type = self.in_type

        for hidden_channel, kernel_size, stride, pad in zip(
            hidden_channels, kernel_sizes, strides, padding
        ):
            out_type = enn.FieldType(self.gspace, hidden_channel * [repr])
            blocks.append(
                enn.R2Conv(
                    current_type,
                    out_type,
                    kernel_size=kernel_size,
                    padding=pad,
                    stride=stride,
                    bias=False,  # standard practice with equivariant convs
                )
            )
            blocks.append(enn.ReLU(out_type))  # equivariant ReLU

            current_type = out_type

        # 1x1 projection conv (matches BaseCNN structure)
        proj_type = enn.FieldType(self.gspace, out_channels * [repr])
        blocks.append(
            enn.R2Conv(
                current_type,
                proj_type,
                kernel_size=1,
                bias=False,
            )
        )

        # GroupPooling: averages over the group dimension → makes output invariant
        # output is a plain tensor after this, not a GeometricTensor
        blocks.append(enn.GroupPooling(proj_type))

        self.eq_layers = enn.SequentialModule(*blocks)

        # these are plain nn layers — GeometricTensor is unwrapped after GroupPooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(out_channels, linear_out_features)

    def _get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # wrap input tensor into a GeometricTensor so escnn can track transformations
        x = enn.GeometricTensor(x, self.in_type)

        # equivariant layers — x stays a GeometricTensor throughout
        x = self.eq_layers(x)

        # unwrap: GroupPooling already converted to plain tensor
        x = x.tensor

        # standard layers from here
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)
