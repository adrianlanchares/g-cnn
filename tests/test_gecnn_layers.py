import torch

from src.models.modules import GroupConv2d, LiftingConv2d, get_group_spec


def test_lifting_conv_output_shape() -> None:
    group = "p4"
    group_order = get_group_spec(group).order

    layer = LiftingConv2d(
        in_channels=12,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        group=group,
    )

    x = torch.randn(3, 12, 8, 8)
    y = layer(x)

    assert y.shape == (3, 16, group_order, 8, 8)


def test_group_conv_output_shape() -> None:
    group = "p4m"
    group_order = get_group_spec(group).order

    layer = GroupConv2d(
        in_channels=10,
        out_channels=20,
        kernel_size=3,
        stride=1,
        padding=1,
        group=group,
    )

    x = torch.randn(2, 10, group_order, 8, 8)
    y = layer(x)

    assert y.shape == (2, 20, group_order, 8, 8)


def test_group_conv_backward_runs() -> None:
    group = "C2"
    group_order = get_group_spec(group).order

    layer = GroupConv2d(
        in_channels=8,
        out_channels=12,
        kernel_size=3,
        stride=1,
        padding=1,
        group=group,
    )

    x = torch.randn(2, 8, group_order, 8, 8, requires_grad=True)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all().item()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all().item()
