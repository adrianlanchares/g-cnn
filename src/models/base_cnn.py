import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseCNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        """Base CNN model for the project. It has the following architecture:
            - (Conv2d -> Activation -> (MaxPool2d)?) * N
            - Conv2d
            - (AdaptiveAvgPool2d)?
            - Flatten
            - Linear

           Pooling layers are optional and can be controlled by the `use_pooling` flag in the config.
           The number of convolutional blocks is determined by the length of `hidden_channels` and `kernel_sizes` in the config.

        Args:
            cfg: Config to use for building the model.

        Raises:
            ValueError: If the length of hidden_channels and kernel_sizes is not the same.
            ValueError: If the specified activation function is not supported.
        """
        super().__init__()

        if len(cfg.hidden_channels) != len(cfg.kernel_sizes):
            raise ValueError(
                "Length of hidden_channels and kernel_sizes must be the same."
            )

        layers = []
        in_channels = cfg.in_channels
        for hidden_channel, kernel_size in zip(cfg.hidden_channels, cfg.kernel_sizes):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channel,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())

            if cfg.use_pooling:
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=cfg.pool_kernel_size,
                        stride=cfg.pool_stride,
                    )
                )
            in_channels = hidden_channel

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=cfg.out_channels,
                kernel_size=1,
            )
        )

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(cfg.out_channels, cfg.linear_out_features))

        self.layers = nn.Sequential(*layers)

    def _get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
