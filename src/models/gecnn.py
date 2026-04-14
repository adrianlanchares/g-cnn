import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn

from src.config import GECNNConfig

GSPACES = {
    "Z2": lambda: gspaces.flip2dOnR2(),  # reflection only,  group size 2
    "p4": lambda: gspaces.rot2dOnR2(N=4),  # 4 rotations,      group size 4
    "p4m": lambda: gspaces.flipRot2dOnR2(N=4),  # rotations+flips,  group size 8
}


class GECNN(nn.Module):
    def __init__(self, cfg: GECNNConfig):
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
            cfg: GECNNConfig instance specifying architecture and symmetry group.

        Raises:
            ValueError: If hidden_channels and kernel_sizes have different lengths.
            ValueError: If cfg.group is not one of: Z2, p4, p4m.
        """
        super().__init__()

        if len(cfg.hidden_channels) != len(cfg.kernel_sizes):
            raise ValueError(
                "Length of hidden_channels and kernel_sizes must be the same."
            )
        if cfg.group not in GSPACES:
            raise ValueError(
                f"Unknown group '{cfg.group}'. Choose from: {list(GSPACES.keys())}"
            )

        self.gspace = GSPACES[cfg.group]()
        repr = self.gspace.regular_repr

        self.in_type = enn.FieldType(
            self.gspace, cfg.in_channels * [self.gspace.trivial_repr]
        )

        blocks = []
        current_type = self.in_type

        for hidden_channel, kernel_size in zip(cfg.hidden_channels, cfg.kernel_sizes):
            out_type = enn.FieldType(self.gspace, hidden_channel * [repr])
            blocks.append(
                enn.R2Conv(
                    current_type,
                    out_type,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,  # standard practice with equivariant convs
                )
            )
            blocks.append(enn.ReLU(out_type))  # equivariant ReLU

            if cfg.use_pooling:
                blocks.append(
                    enn.PointwiseMaxPool(
                        out_type,
                        kernel_size=cfg.pool_kernel_size,
                        stride=cfg.pool_stride,
                    )
                )
            current_type = out_type

        # 1x1 projection conv (matches BaseCNN structure)
        proj_type = enn.FieldType(self.gspace, cfg.out_channels * [repr])
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
        self.head = nn.Linear(cfg.out_channels, cfg.linear_out_features)

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
