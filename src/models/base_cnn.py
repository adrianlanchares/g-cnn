import torch
import torch.nn as nn

from src.models.modules import CNNBlock, MLPBlock


class BaseCNN(nn.Module):
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
        final_tanh: bool = False,
    ):
        """Base CNN model for the project. It has the following architecture:
            - (Conv2d -> Activation -> (MaxPool2d)?) * N
            - Conv2d
            - (AdaptiveAvgPool2d)?
            - Flatten
            - Linear

           The number of convolutional blocks is determined by the length of `hidden_channels` and `kernel_sizes` in the config.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: List of hidden channel sizes.
            kernel_sizes: List of kernel sizes for each convolutional layer.
            strides: List of strides for each convolutional layer.
            padding: List of padding for each convolutional layer.
            linear_hidden_features: List of hidden feature sizes for the linear layers.
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
        current_in_channels = in_channels

        for hidden_channel, kernel_size, stride, pad in zip(
            hidden_channels, kernel_sizes, strides, padding
        ):
            layers.append(
                CNNBlock(
                    in_channels=current_in_channels,
                    out_channels=hidden_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    batchnorm=batchnorm,
                )
            )
            current_in_channels = hidden_channel

        layers.append(
            CNNBlock(
                in_channels=current_in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        self.conv_layers = nn.Sequential(*layers)

        self.flatten = nn.Flatten()

        # Calculate the number of features after the convolutional layers to determine the input size for the linear layer
        dummy_input = torch.zeros(1, in_channels, 8, 8)
        with torch.no_grad():
            dummy_output = self.flatten(self.conv_layers(dummy_input))
        linear_in_features = dummy_output.shape[1]

        head_layers = []
        for feature_size in linear_hidden_features:
            head_layers.append(MLPBlock(linear_in_features, feature_size))
            linear_in_features = feature_size

        head_layers.append(nn.Linear(linear_in_features, linear_out_features))
        if final_tanh:
            head_layers.append(nn.Tanh())

        self.head = nn.Sequential(*head_layers)

    def _get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
