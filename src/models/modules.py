from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_spatial_transform(
    tensor: torch.Tensor,
    rotation_k: int,
    mirrored: bool,
) -> torch.Tensor:
    transformed = tensor
    if mirrored:
        transformed = torch.flip(transformed, dims=(-1,))
    if rotation_k:
        transformed = torch.rot90(transformed, k=rotation_k, dims=(-2, -1))
    return transformed


class GroupSpec:
    def __init__(self, rotation_order: int, include_reflections: bool):
        if rotation_order < 1:
            raise ValueError("rotation_order must be >= 1")

        self.rotation_order = rotation_order
        if include_reflections:
            self.elements = tuple(
                (rotation, mirror)
                for mirror in (0, 1)
                for rotation in range(rotation_order)
            )
        else:
            self.elements = tuple((rotation, 0) for rotation in range(rotation_order))

        self._index_by_element = {
            element: idx for idx, element in enumerate(self.elements)
        }

    @property
    def order(self) -> int:
        return len(self.elements)

    def compose_index(self, lhs_idx: int, rhs_idx: int) -> int:
        lhs = self.elements[lhs_idx]
        rhs = self.elements[rhs_idx]

        lhs_rotation, lhs_mirror = lhs
        rhs_rotation, rhs_mirror = rhs

        sign = -1 if lhs_mirror else 1
        new_rotation = (lhs_rotation + sign * rhs_rotation) % self.rotation_order
        new_mirror = lhs_mirror ^ rhs_mirror

        return self._index_by_element[(new_rotation, new_mirror)]

    def inverse_index(self, index: int) -> int:
        rotation, mirror = self.elements[index]

        if mirror:
            inverse_rotation = rotation % self.rotation_order
        else:
            inverse_rotation = (-rotation) % self.rotation_order

        return self._index_by_element[(inverse_rotation, mirror)]

    def relative_index(self, output_group_idx: int, input_group_idx: int) -> int:
        return self.compose_index(
            self.inverse_index(output_group_idx),
            input_group_idx,
        )

    def transform_spatial(self, tensor: torch.Tensor, element_idx: int) -> torch.Tensor:
        rotation, mirror = self.elements[element_idx]
        return _apply_spatial_transform(
            tensor,
            rotation_k=rotation,
            mirrored=bool(mirror),
        )


GROUP_SPECS = {
    "Z2": GroupSpec(rotation_order=1, include_reflections=True),
    "p4": GroupSpec(rotation_order=4, include_reflections=False),
    "p4m": GroupSpec(rotation_order=4, include_reflections=True),
}


def get_group_spec(group: str) -> GroupSpec:
    if group not in GROUP_SPECS:
        raise ValueError(
            f"Unknown group '{group}'. Choose from: {list(GROUP_SPECS.keys())}"
        )
    return GROUP_SPECS[group]


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
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LiftingConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        group: str,
        bias: bool = False,
    ):
        super().__init__()

        self.group_spec = get_group_spec(group)
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_per_group_element = []

        for group_idx in range(self.group_spec.order):
            transformed_weight = self.group_spec.transform_spatial(self.weight, group_idx)
            output_per_group_element.append(
                F.conv2d(
                    x,
                    transformed_weight,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                )
            )

        output = torch.stack(output_per_group_element, dim=2)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output


class GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        group: str,
        bias: bool = False,
    ):
        super().__init__()

        self.group_spec = get_group_spec(group)
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                self.group_spec.order,
                kernel_size,
                kernel_size,
            )
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, input_group_size, _, _ = x.shape
        if input_group_size != self.group_spec.order:
            raise ValueError(
                "Input group dimension does not match configured group order: "
                f"expected {self.group_spec.order}, got {input_group_size}."
            )

        output_per_group_element = []

        for output_group_idx in range(self.group_spec.order):
            transformed_weight = self.group_spec.transform_spatial(
                self.weight,
                output_group_idx,
            )

            output = None
            for input_group_idx in range(self.group_spec.order):
                relative_idx = self.group_spec.relative_index(
                    output_group_idx,
                    input_group_idx,
                )
                current = F.conv2d(
                    x[:, :, input_group_idx],
                    transformed_weight[:, :, relative_idx],
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                )
                output = current if output is None else output + current

            if self.bias is not None:
                output = output + self.bias.view(1, -1, 1, 1)

            output_per_group_element.append(output)

        return torch.stack(output_per_group_element, dim=2)


class GroupBatchNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(x)


class GECNNLiftBlock(nn.Module):
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

        layers: list[nn.Module] = [
            LiftingConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                group=group,
                bias=False,
            ),
        ]
        if batchnorm:
            layers.append(GroupBatchNorm(out_channels))

        layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        layers: list[nn.Module] = [
            GroupConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                group=group,
                bias=False,
            )
        ]
        if batchnorm:
            layers.append(GroupBatchNorm(out_channels))

        layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
