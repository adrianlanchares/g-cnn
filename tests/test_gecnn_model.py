import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from src.models.gecnn import GECNN


def test_gecnn_forward_shape_default_config() -> None:
    cfg = OmegaConf.load("config/model/gecnn.yaml")
    model = instantiate(cfg)

    x = torch.randn(4, cfg.in_channels, 8, 8)
    y = model(x)

    assert y.shape == (4, cfg.linear_out_features)


def test_gecnn_backward_runs() -> None:
    model = GECNN(
        in_channels=12,
        out_channels=32,
        hidden_channels=[16, 16],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        padding=[1, 1],
        batchnorm=False,
        linear_hidden_features=[32],
        linear_out_features=1,
        group="p4m",
    )

    x = torch.randn(2, 12, 8, 8)
    y = model(x)
    loss = y.mean()
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)
    assert all(
        (grad is None) or torch.isfinite(grad).all().item() for grad in grads
    )


def test_invalid_group_raises() -> None:
    with pytest.raises(ValueError, match="Unknown group"):
        GECNN(
            in_channels=12,
            out_channels=16,
            hidden_channels=[16],
            kernel_sizes=[3],
            strides=[1],
            padding=[1],
            batchnorm=False,
            linear_hidden_features=[16],
            linear_out_features=1,
            group="bad_group",
        )
