import torch
import torch.nn as nn

from src.models.modules import GECNNBlock, GECNNLiftBlock, MLPBlock, get_group_spec


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
        """Group-Equivariant CNN implemented with native PyTorch.

        Architecture mirrors BaseCNN:
            - LiftingConv2d -> Activation
            - (GroupConv2d -> Activation) * N
            - GroupConv2d projection
            - Mean pooling over group and spatial dimensions
            - Flatten
            - Linear

        The group is controlled by cfg.group. Internally, the tensor shape is
        [B, C, |G|, H, W], where |G| is the group order.

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
            ValueError: If group is not one of: Z2, p4, p4m.
        """
        super().__init__()

        if len(hidden_channels) != len(kernel_sizes):
            raise ValueError(
                "Length of hidden_channels and kernel_sizes must be the same."
            )
        get_group_spec(group)

        layers = []
        current_in_channels = in_channels

        first_hidden = hidden_channels[0] if hidden_channels else out_channels
        first_kernel = kernel_sizes[0] if kernel_sizes else 3
        first_stride = strides[0] if strides else 1
        first_padding = padding[0] if padding else 1

        layers.append(
            GECNNLiftBlock(
                in_channels=current_in_channels,
                out_channels=first_hidden,
                kernel_size=first_kernel,
                stride=first_stride,
                padding=first_padding,
                group=group,
                batchnorm=batchnorm,
            )
        )
        current_in_channels = first_hidden

        for hidden_channel, kernel_size, stride, pad in zip(
            hidden_channels[1:], kernel_sizes[1:], strides[1:], padding[1:]
        ):
            layers.append(
                GECNNBlock(
                    in_channels=current_in_channels,
                    out_channels=hidden_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    group=group,
                    batchnorm=batchnorm,
                )
            )
            current_in_channels = hidden_channel

        layers.append(
            GECNNBlock(
                in_channels=current_in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                group=group,
            )
        )

        self.eq_layers = nn.Sequential(*layers)

        self.invariant_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

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
        x = self.eq_layers(x)

        # invariant pooling over group dimension
        x = x.mean(dim=2)
        x = self.invariant_pool(x)

        x = self.flatten(x)
        x = self.head(x)
        return x
