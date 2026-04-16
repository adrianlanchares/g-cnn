import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        use_pooling: bool = False,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
        linear_out_features: int = 1,
    ):
        """Base CNN model for the project. It has the following architecture:
            - (Conv2d -> Activation -> (MaxPool2d)?) * N
            - Conv2d
            - (AdaptiveAvgPool2d)?
            - Flatten
            - Linear

           Pooling layers are optional and can be controlled by the `use_pooling` flag in the config.
           The number of convolutional blocks is determined by the length of `hidden_channels` and `kernel_sizes` in the config.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: List of hidden channel sizes.
            kernel_sizes: List of kernel sizes for each convolutional layer.
            use_pooling: Whether to use pooling layers.
            pool_kernel_size: Kernel size for pooling layers.
            pool_stride: Stride for pooling layers.
            linear_out_features: Number of output features for the final linear layer.

        Raises:
            ValueError: If the length of hidden_channels and kernel_sizes is not the same.
            ValueError: If the specified activation function is not supported.
        """
        super().__init__()

        if len(hidden_channels) != len(kernel_sizes):
            raise ValueError(
                "Length of hidden_channels and kernel_sizes must be the same."
            )

        layers = []
        in_channels = in_channels
        for hidden_channel, kernel_size, stride in zip(
            hidden_channels, kernel_sizes, strides
        ):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())

            if use_pooling:
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_kernel_size,
                        stride=pool_stride,
                    )
                )
            in_channels = hidden_channel

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        )

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(out_channels, linear_out_features))

        self.layers = nn.Sequential(*layers)

    def _get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
