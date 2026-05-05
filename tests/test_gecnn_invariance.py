import torch

from src.models.gecnn import GECNN


def _build_small_gecnn(group: str) -> GECNN:
    return GECNN(
        in_channels=12,
        out_channels=16,
        hidden_channels=[16, 16],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        padding=[1, 1],
        batchnorm=False,
        linear_hidden_features=[16],
        linear_out_features=1,
        group=group,
    )


@torch.no_grad()
def test_c2_flip_invariance_after_group_pooling() -> None:
    model = _build_small_gecnn("C2")
    model.eval()

    x = torch.randn(2, 12, 8, 8)
    x_transformed = torch.flip(x, dims=(-1,))

    y = model(x)
    y_transformed = model(x_transformed)

    assert torch.allclose(y, y_transformed, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_p4_rotation_invariance_after_group_pooling() -> None:
    model = _build_small_gecnn("p4")
    model.eval()

    x = torch.randn(2, 12, 8, 8)
    x_transformed = torch.rot90(x, k=1, dims=(-2, -1))

    y = model(x)
    y_transformed = model(x_transformed)

    assert torch.allclose(y, y_transformed, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_p4m_flip_rotation_invariance_after_group_pooling() -> None:
    model = _build_small_gecnn("p4m")
    model.eval()

    x = torch.randn(2, 12, 8, 8)
    x_transformed = torch.rot90(torch.flip(x, dims=(-1,)), k=1, dims=(-2, -1))

    y = model(x)
    y_transformed = model(x_transformed)

    assert torch.allclose(y, y_transformed, atol=1e-5, rtol=1e-5)
