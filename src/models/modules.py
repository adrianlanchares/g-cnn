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


def _stack_rotations(tensor: torch.Tensor, rotation_order: int) -> torch.Tensor:
    if rotation_order == 1:
        return tensor.unsqueeze(0)
    if rotation_order == 4:
        return torch.stack(
            (
                tensor,
                torch.rot90(tensor, k=1, dims=(-2, -1)),
                torch.rot90(tensor, k=2, dims=(-2, -1)),
                torch.rot90(tensor, k=3, dims=(-2, -1)),
            ),
            dim=0,
        )
    raise ValueError(f"Unsupported rotation order: {rotation_order}")


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
        self.group_order = self.group_spec.order
        self.stride = stride
        self.padding = padding

        rotation_indices = torch.tensor(
            [rotation for rotation, _ in self.group_spec.elements],
            dtype=torch.long,
        )
        mirror_indices = torch.tensor(
            [mirror for _, mirror in self.group_spec.elements],
            dtype=torch.long,
        )
        self.register_buffer("_rotation_indices", rotation_indices, persistent=False)
        self.register_buffer("_mirror_indices", mirror_indices, persistent=False)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def _transform_all_weights(self, weight: torch.Tensor) -> torch.Tensor:
        rotated = _stack_rotations(weight, self.group_spec.rotation_order)
        if self.group_spec.elements[-1][1] == 0:
            all_transforms = rotated.unsqueeze(0)
        else:
            mirrored = torch.flip(weight, dims=(-1,))
            mirrored_rotated = _stack_rotations(mirrored, self.group_spec.rotation_order)
            all_transforms = torch.stack((rotated, mirrored_rotated), dim=0)

        return all_transforms[self._mirror_indices, self._rotation_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        transformed_weight = self._transform_all_weights(self.weight)
        transformed_weight = transformed_weight.reshape(
            self.group_order * transformed_weight.shape[1],
            transformed_weight.shape[2],
            transformed_weight.shape[3],
            transformed_weight.shape[4],
        )

        output = F.conv2d(
            x,
            transformed_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )

        output = output.view(
            batch_size,
            self.group_order,
            self.weight.shape[0],
            output.shape[-2],
            output.shape[-1],
        ).permute(0, 2, 1, 3, 4)

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
        self.group_order = self.group_spec.order
        self.stride = stride
        self.padding = padding

        rotation_indices = torch.tensor(
            [rotation for rotation, _ in self.group_spec.elements],
            dtype=torch.long,
        )
        mirror_indices = torch.tensor(
            [mirror for _, mirror in self.group_spec.elements],
            dtype=torch.long,
        )
        relative_indices = torch.tensor(
            [
                [self.group_spec.relative_index(g_out, g_in) for g_in in range(self.group_order)]
                for g_out in range(self.group_order)
            ],
            dtype=torch.long,
        )
        self.register_buffer("_rotation_indices", rotation_indices, persistent=False)
        self.register_buffer("_mirror_indices", mirror_indices, persistent=False)
        self.register_buffer("_relative_indices", relative_indices, persistent=False)

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

    def _transform_all_weights(self, weight: torch.Tensor) -> torch.Tensor:
        rotated = _stack_rotations(weight, self.group_spec.rotation_order)
        if self.group_spec.elements[-1][1] == 0:
            all_transforms = rotated.unsqueeze(0)
        else:
            mirrored = torch.flip(weight, dims=(-1,))
            mirrored_rotated = _stack_rotations(mirrored, self.group_spec.rotation_order)
            all_transforms = torch.stack((rotated, mirrored_rotated), dim=0)

        return all_transforms[self._mirror_indices, self._rotation_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, input_group_size, _, _ = x.shape
        if input_group_size != self.group_order:
            raise ValueError(
                "Input group dimension does not match configured group order: "
                f"expected {self.group_order}, got {input_group_size}."
            )

        batch_size, in_channels, _, height, width = x.shape
        out_channels = self.weight.shape[0]

        transformed_weight = self._transform_all_weights(self.weight)
        gather_index = self._relative_indices.view(
            self.group_order,
            1,
            1,
            self.group_order,
            1,
            1,
        ).expand(
            self.group_order,
            out_channels,
            in_channels,
            self.group_order,
            self.weight.shape[-2],
            self.weight.shape[-1],
        )

        transformed_weight = torch.gather(
            transformed_weight,
            dim=3,
            index=gather_index,
        )

        transformed_weight = transformed_weight.permute(0, 1, 3, 2, 4, 5).reshape(
            self.group_order * out_channels,
            self.group_order * in_channels,
            self.weight.shape[-2],
            self.weight.shape[-1],
        )

        x = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            self.group_order * in_channels,
            height,
            width,
        )

        bias = None
        if self.bias is not None:
            bias = self.bias.repeat(self.group_order)

        output = F.conv2d(
            x,
            transformed_weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
        )

        return output.view(
            batch_size,
            self.group_order,
            out_channels,
            output.shape[-2],
            output.shape[-1],
        ).permute(0, 2, 1, 3, 4)


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
