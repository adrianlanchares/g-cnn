"""
Pytest suite for the Group-Equivariant CNN (G-CNN) implementation.

Tests are organised around the theoretical requirements from:
  Cohen & Welling, "Group Equivariant Convolutional Networks", ICML 2016.
  https://arxiv.org/abs/1602.07576

Coverage
--------
1. GroupSpec  – group algebra (composition, inverse, closure, associativity)
2. LiftingConv2d – equivariance from Z² → G  (paper eq. 10 / 12)
3. GroupConv2d   – equivariance on G → G     (paper eq. 11 / 12)
4. GroupBatchNorm – equivariance of BN (paper sec. 6.1 note on BN)
5. GECNNLiftBlock / GECNNBlock – block-level equivariance
6. GECNN (full model) – invariance after mean-pooling over group dim
7. Weight-sharing – transformed weights are genuine spatial transforms
8. Shape contracts  – output tensors have the expected dimensions
9. Bias equivariance – one bias per G-feature map preserves equivariance
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Adjust these imports to match the actual package layout in your project.
# ---------------------------------------------------------------------------
from src.models.modules import (
    GROUP_SPECS,
    GroupBatchNorm,
    GroupConv2d,
    GroupSpec,
    GECNNBlock,
    GECNNLiftBlock,
    LiftingConv2d,
    get_group_spec,
)
from src.models.gecnn import GECNN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GROUPS = ["Z2", "C2", "p4", "p4m"]
GROUP_ORDERS = {"Z2": 1, "C2": 2, "p4": 4, "p4m": 8}

ATOL = 1e-5  # absolute tolerance for floating-point comparisons


def _apply_group_transform_to_lifting_output(
    output: torch.Tensor,
    group_spec: GroupSpec,
    element_idx: int,
) -> torch.Tensor:
    """
    Transform a LiftingConv2d output tensor [B, C, |G|, H, W] by group element
    `element_idx`.

    Transformation of a G-feature map by u:
      - permute the group axis:  new_g = u · g   (i.e., g -> u·g)
      - spatially transform the spatial patch corresponding to each group slot
    """
    B, C, G, H, W = output.shape
    result = torch.zeros_like(output)

    for g_in in range(G):
        g_out = group_spec.compose_index(element_idx, g_in)
        spatial_patch = output[:, :, g_in, :, :]  # [B, C, H, W]
        transformed_patch = group_spec.transform_spatial(spatial_patch, element_idx)
        result[:, :, g_out, :, :] = transformed_patch

    return result


def _rotate_input_90(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Rotate a [B, C, H, W] image tensor by 90*k degrees."""
    return torch.rot90(x, k=k, dims=(-2, -1))


def _reflect_input(x: torch.Tensor) -> torch.Tensor:
    """Horizontal mirror a [B, C, H, W] tensor."""
    return torch.flip(x, dims=(-1,))


# ---------------------------------------------------------------------------
# 1. GroupSpec – group algebra
# ---------------------------------------------------------------------------


class TestGroupSpec:
    """Unit-tests for the GroupSpec algebra."""

    @pytest.mark.parametrize("group", GROUPS)
    def test_order(self, group: str) -> None:
        """Group has the expected number of elements."""
        spec = get_group_spec(group)
        assert spec.order == GROUP_ORDERS[group]

    @pytest.mark.parametrize("group", GROUPS)
    def test_identity_element(self, group: str) -> None:
        """The first element (rotation=0, mirror=0) must be the identity."""
        spec = get_group_spec(group)
        identity_idx = spec._index_by_element[(0, 0)]
        for g in range(spec.order):
            # identity ∘ g == g
            assert spec.compose_index(identity_idx, g) == g
            # g ∘ identity == g
            assert spec.compose_index(g, identity_idx) == g

    @pytest.mark.parametrize("group", GROUPS)
    def test_inverse(self, group: str) -> None:
        """g ∘ g⁻¹ == identity for every element."""
        spec = get_group_spec(group)
        identity_idx = spec._index_by_element[(0, 0)]
        for g in range(spec.order):
            inv_g = spec.inverse_index(g)
            assert spec.compose_index(g, inv_g) == identity_idx
            assert spec.compose_index(inv_g, g) == identity_idx

    @pytest.mark.parametrize("group", GROUPS)
    def test_closure(self, group: str) -> None:
        """Composing any two elements yields an element in the group."""
        spec = get_group_spec(group)
        for a in range(spec.order):
            for b in range(spec.order):
                result = spec.compose_index(a, b)
                assert 0 <= result < spec.order

    @pytest.mark.parametrize("group", GROUPS)
    def test_associativity(self, group: str) -> None:
        """(a ∘ b) ∘ c == a ∘ (b ∘ c)."""
        spec = get_group_spec(group)
        for a in range(spec.order):
            for b in range(spec.order):
                for c in range(spec.order):
                    lhs = spec.compose_index(spec.compose_index(a, b), c)
                    rhs = spec.compose_index(a, spec.compose_index(b, c))
                    assert lhs == rhs

    @pytest.mark.parametrize("group", GROUPS)
    def test_relative_index_definition(self, group: str) -> None:
        """relative_index(g_out, g_in) == g_out⁻¹ ∘ g_in."""
        spec = get_group_spec(group)
        for g_out in range(spec.order):
            for g_in in range(spec.order):
                expected = spec.compose_index(spec.inverse_index(g_out), g_in)
                assert spec.relative_index(g_out, g_in) == expected

    @pytest.mark.parametrize("group", GROUPS)
    def test_spatial_transform_identity(self, group: str) -> None:
        """Applying the identity element must return the same tensor."""
        spec = get_group_spec(group)
        identity_idx = spec._index_by_element[(0, 0)]
        tensor = torch.randn(3, 5, 5)
        transformed = spec.transform_spatial(tensor, identity_idx)
        assert torch.allclose(tensor, transformed)

    @pytest.mark.parametrize("group", GROUPS)
    def test_spatial_transform_inverse(self, group: str) -> None:
        """Applying g then g⁻¹ must recover the original tensor."""
        spec = get_group_spec(group)
        tensor = torch.randn(3, 5, 5)
        for g in range(spec.order):
            inv_g = spec.inverse_index(g)
            recovered = spec.transform_spatial(spec.transform_spatial(tensor, g), inv_g)
            assert torch.allclose(tensor, recovered, atol=ATOL)


# ---------------------------------------------------------------------------
# 2. LiftingConv2d – equivariance  Z² → G
# ---------------------------------------------------------------------------


class TestLiftingConv2dEquivariance:
    """
    The lifting layer must satisfy (paper eq. 10 / 12):

        LiftConv(T_g x) == L_g LiftConv(x)

    where T_g is a spatial transformation of the input image, and L_g is the
    corresponding action on G-feature maps (permute group axis + spatial
    transform of each patch).

    We test for every group with both rotation and reflection generators.
    """

    @pytest.fixture(params=["p4", "p4m"])
    def layer_and_spec(self, request):
        group = request.param
        torch.manual_seed(0)
        layer = LiftingConv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            group=group,
            bias=False,
        )
        layer.eval()
        return layer, get_group_spec(group)

    def _check_equivariance(
        self,
        layer: LiftingConv2d,
        spec: GroupSpec,
        x: torch.Tensor,
        element_idx: int,
    ) -> None:
        rotation, mirror = spec.elements[element_idx]

        # Build the transformed input
        x_transformed = x.clone()
        if mirror:
            x_transformed = torch.flip(x_transformed, dims=(-1,))
        if rotation:
            x_transformed = torch.rot90(x_transformed, k=rotation, dims=(-2, -1))

        with torch.no_grad():
            # Path 1: transform input first, then lift
            out_transform_first = layer(x_transformed)
            # Path 2: lift first, then transform output
            out_lift_first = layer(x)
            out_lift_transformed = _apply_group_transform_to_lifting_output(
                out_lift_first, spec, element_idx
            )

        assert out_transform_first.shape == out_lift_transformed.shape, (
            f"Shape mismatch for group={spec.rotation_order}, element_idx={element_idx}"
        )
        assert torch.allclose(out_transform_first, out_lift_transformed, atol=ATOL), (
            f"Equivariance violated for element_idx={element_idx}, "
            f"element={spec.elements[element_idx]}, "
            f"max_diff={(out_transform_first - out_lift_transformed).abs().max().item():.2e}"
        )

    def test_equivariance_all_elements(self, layer_and_spec) -> None:
        """Check equivariance for every group element."""
        layer, spec = layer_and_spec
        torch.manual_seed(42)
        x = torch.randn(2, 2, 8, 8)
        for element_idx in range(spec.order):
            self._check_equivariance(layer, spec, x, element_idx)

    def test_equivariance_p4_rotation_90(self) -> None:
        """Explicit 90° rotation test for p4 (the canonical G-CNN group)."""
        torch.manual_seed(1)
        layer = LiftingConv2d(2, 4, 3, 1, 1, group="p4", bias=False)
        layer.eval()
        spec = get_group_spec("p4")
        x = torch.randn(1, 2, 8, 8)
        # element 1 == rotation by 90°
        self._check_equivariance(layer, spec, x, element_idx=1)

    def test_equivariance_p4m_reflection(self) -> None:
        """Explicit reflection test for p4m."""
        torch.manual_seed(2)
        layer = LiftingConv2d(2, 4, 3, 1, 1, group="p4m", bias=False)
        layer.eval()
        spec = get_group_spec("p4m")
        x = torch.randn(1, 2, 8, 8)
        # Find a mirror element
        mirror_idx = next(
            idx for idx, (r, m) in enumerate(spec.elements) if m == 1 and r == 0
        )
        self._check_equivariance(layer, spec, x, element_idx=mirror_idx)


# ---------------------------------------------------------------------------
# 3. GroupConv2d – equivariance  G → G
# ---------------------------------------------------------------------------


class TestGroupConv2dEquivariance:
    """
    The G-convolution layer must satisfy (paper eq. 11 / 12):

        GroupConv(L_g f) == L_g GroupConv(f)

    where both input and output are G-feature maps.
    """

    @pytest.fixture(params=["p4", "p4m"])
    def layer_and_spec(self, request):
        group = request.param
        torch.manual_seed(0)
        layer = GroupConv2d(
            in_channels=3,
            out_channels=5,
            kernel_size=3,
            stride=1,
            padding=1,
            group=group,
            bias=False,
        )
        layer.eval()
        return layer, get_group_spec(group)

    def _make_g_feature_map(
        self, batch: int, channels: int, group_order: int, h: int, w: int
    ) -> torch.Tensor:
        """Random G-feature map of shape [B, C, |G|, H, W]."""
        return torch.randn(batch, channels, group_order, h, w)

    def _apply_g_transform(
        self, f: torch.Tensor, spec: GroupSpec, element_idx: int
    ) -> torch.Tensor:
        """L_u applied to a G-feature map [B, C, |G|, H, W]."""
        return _apply_group_transform_to_lifting_output(f, spec, element_idx)

    def test_equivariance_all_elements(self, layer_and_spec) -> None:
        layer, spec = layer_and_spec
        torch.manual_seed(7)
        f = self._make_g_feature_map(2, 3, spec.order, 8, 8)

        for element_idx in range(spec.order):
            f_transformed = self._apply_g_transform(f, spec, element_idx)

            with torch.no_grad():
                # Path 1: transform then convolve
                out_tf = layer(f_transformed)
                # Path 2: convolve then transform
                out_ft = self._apply_g_transform(layer(f), spec, element_idx)

            assert torch.allclose(out_tf, out_ft, atol=ATOL), (
                f"GroupConv2d equivariance violated for group={spec.rotation_order}, "
                f"element_idx={element_idx}, "
                f"max_diff={(out_tf - out_ft).abs().max().item():.2e}"
            )

    def test_input_validation_wrong_ndim(self, layer_and_spec) -> None:
        """GroupConv2d must reject 4-D input."""
        layer, spec = layer_and_spec
        x = torch.randn(2, 3, 8, 8)
        with pytest.raises(ValueError, match="5 dimensions"):
            layer(x)

    def test_input_validation_wrong_group_size(self, layer_and_spec) -> None:
        """GroupConv2d must reject input whose group dim doesn't match."""
        layer, spec = layer_and_spec
        wrong_g = spec.order + 1
        x = torch.randn(2, 3, wrong_g, 8, 8)
        with pytest.raises(ValueError, match="group dimension"):
            layer(x)


# ---------------------------------------------------------------------------
# 4. GroupBatchNorm – equivariance
# ---------------------------------------------------------------------------


class TestGroupBatchNormEquivariance:
    """
    BatchNorm with a single scale/bias per G-feature-map channel must remain
    equivariant (paper sec. 6.1).
    """

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_bn_equivariance(self, group: str) -> None:
        spec = get_group_spec(group)
        torch.manual_seed(3)
        bn = GroupBatchNorm(channels=4)
        bn.train()  # running stats need multiple elements → train mode

        x = torch.randn(4, 4, spec.order, 6, 6)

        for element_idx in range(spec.order):
            x_transformed = _apply_group_transform_to_lifting_output(
                x, spec, element_idx
            )
            with torch.no_grad():
                out_tf = bn(x_transformed)
                out_ft = _apply_group_transform_to_lifting_output(
                    bn(x), spec, element_idx
                )

            assert torch.allclose(out_tf, out_ft, atol=1e-4), (
                f"GroupBatchNorm equivariance violated for {group}, "
                f"element_idx={element_idx}, "
                f"max_diff={(out_tf - out_ft).abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# 5. Block-level equivariance
# ---------------------------------------------------------------------------


class TestBlockEquivariance:
    """GECNNLiftBlock and GECNNBlock must each be equivariant."""

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_lift_block_equivariance(self, group: str) -> None:
        spec = get_group_spec(group)
        torch.manual_seed(4)
        block = GECNNLiftBlock(2, 4, 3, 1, 1, group=group, batchnorm=False)
        block.eval()
        x = torch.randn(2, 2, 8, 8)

        for element_idx in range(spec.order):
            rotation, mirror = spec.elements[element_idx]
            x_t = x.clone()
            if mirror:
                x_t = torch.flip(x_t, dims=(-1,))
            if rotation:
                x_t = torch.rot90(x_t, k=rotation, dims=(-2, -1))

            with torch.no_grad():
                out_tf = block(x_t)
                out_ft = _apply_group_transform_to_lifting_output(
                    block(x), spec, element_idx
                )

            assert torch.allclose(out_tf, out_ft, atol=ATOL), (
                f"GECNNLiftBlock equivariance failed for {group}, "
                f"element_idx={element_idx}"
            )

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_gecnn_block_equivariance(self, group: str) -> None:
        spec = get_group_spec(group)
        torch.manual_seed(5)
        block = GECNNBlock(3, 5, 3, 1, 1, group=group, batchnorm=False)
        block.eval()
        f = torch.randn(2, 3, spec.order, 8, 8)

        for element_idx in range(spec.order):
            f_t = _apply_group_transform_to_lifting_output(f, spec, element_idx)
            with torch.no_grad():
                out_tf = block(f_t)
                out_ft = _apply_group_transform_to_lifting_output(
                    block(f), spec, element_idx
                )

            assert torch.allclose(out_tf, out_ft, atol=ATOL), (
                f"GECNNBlock equivariance failed for {group}, element_idx={element_idx}"
            )


# ---------------------------------------------------------------------------
# 6. Full GECNN – invariance
# ---------------------------------------------------------------------------


class TestGECNNInvariance:
    """
    After mean-pooling over the group dimension the full model must be
    invariant: GECNN(T_g x) == GECNN(x) for all g in G.

    This is the key property that makes G-CNNs useful for classification.
    """

    def _make_model(self, group: str) -> GECNN:
        return GECNN(
            in_channels=3,
            out_channels=8,
            hidden_channels=[4, 8],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            batchnorm=False,
            linear_hidden_features=[16],
            linear_out_features=4,
            group=group,
            final_tanh=False,
        )

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_invariance_all_rotations(self, group: str) -> None:
        """Output must be the same for every rotation of the input."""
        torch.manual_seed(10)
        model = self._make_model(group)
        model.eval()

        x = torch.randn(1, 3, 12, 12)
        spec = get_group_spec(group)

        with torch.no_grad():
            out_base = model(x)

        for element_idx in range(spec.order):
            rotation, mirror = spec.elements[element_idx]
            x_t = x.clone()
            if mirror:
                x_t = torch.flip(x_t, dims=(-1,))
            if rotation:
                x_t = torch.rot90(x_t, k=rotation, dims=(-2, -1))

            with torch.no_grad():
                out_transformed = model(x_t)

            assert torch.allclose(out_base, out_transformed, atol=1e-4), (
                f"GECNN invariance violated for {group}, "
                f"element_idx={element_idx}, "
                f"element={spec.elements[element_idx]}, "
                f"max_diff={(out_base - out_transformed).abs().max().item():.2e}"
            )

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_invariance_with_batchnorm(self, group: str) -> None:
        """Invariance must hold with GroupBatchNorm in eval mode."""
        torch.manual_seed(11)
        model = GECNN(
            in_channels=3,
            out_channels=8,
            hidden_channels=[4, 8],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            batchnorm=True,
            linear_hidden_features=[16],
            linear_out_features=4,
            group=group,
        )
        # Warm up running stats
        model.train()
        with torch.no_grad():
            for _ in range(10):
                model(torch.randn(4, 3, 12, 12))

        model.eval()
        spec = get_group_spec(group)
        x = torch.randn(1, 3, 12, 12)

        with torch.no_grad():
            out_base = model(x)

        for element_idx in range(spec.order):
            rotation, mirror = spec.elements[element_idx]
            x_t = x.clone()
            if mirror:
                x_t = torch.flip(x_t, dims=(-1,))
            if rotation:
                x_t = torch.rot90(x_t, k=rotation, dims=(-2, -1))

            with torch.no_grad():
                out_t = model(x_t)

            assert torch.allclose(out_base, out_t, atol=1e-4), (
                f"GECNN+BN invariance violated for {group}, "
                f"element_idx={element_idx}, "
                f"max_diff={(out_base - out_t).abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# 7. Weight-sharing – transformed weights are genuine spatial transforms
# ---------------------------------------------------------------------------


class TestWeightSharing:
    """
    In a G-CNN, weight-sharing means the filter for group element g is a
    deterministic spatial transform of the canonical filter (the filter for
    the identity element). This is the core parameter-efficiency claim.
    """

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_lifting_weight_transform_consistency(self, group: str) -> None:
        """
        For LiftingConv2d: transformed_weight[g] must equal the spatially-
        transformed version of transformed_weight[0] (the identity filter).
        """
        torch.manual_seed(20)
        spec = get_group_spec(group)
        layer = LiftingConv2d(2, 4, 3, 1, 1, group=group, bias=False)
        layer.eval()

        with torch.no_grad():
            # Shape: [|G|, C_out, C_in, K, K]
            tw = layer._transform_all_weights(layer.weight)

        identity_idx = spec._index_by_element[(0, 0)]
        base_filter = tw[identity_idx]  # [C_out, C_in, K, K]

        for g in range(spec.order):
            rotation, mirror = spec.elements[g]
            expected = base_filter.clone()
            if mirror:
                expected = torch.flip(expected, dims=(-1,))
            if rotation:
                expected = torch.rot90(expected, k=rotation, dims=(-2, -1))

            assert torch.allclose(tw[g], expected, atol=ATOL), (
                f"LiftingConv2d weight transform inconsistent at g={g} "
                f"({spec.elements[g]}) for group={group}"
            )

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_group_conv_weight_transform_consistency(self, group: str) -> None:
        torch.manual_seed(21)
        spec = get_group_spec(group)
        layer = GroupConv2d(3, 5, 3, 1, 1, group=group, bias=False)
        layer.eval()

        with torch.no_grad():
            tw = layer._transform_all_weights(
                layer.weight
            )  # [|G_out|, C_out, C_in, |G_in|, K, K]

        identity_idx = spec._index_by_element[(0, 0)]
        base_slice = tw[
            identity_idx
        ]  # [C_out, C_in, |G|, K, K] — weight at identity (no rotation, no permutation)

        for g_out in range(spec.order):
            rotation, mirror = spec.elements[g_out]
            for g_in in range(spec.order):
                rel = spec.relative_index(g_out, g_in)
                # Spatial patch for (g_out, g_in) = spatial transform of base at slot relative(g_out, g_in)
                expected = base_slice[:, :, rel, :, :].clone()  # [C_out, C_in, K, K]
                if mirror:
                    expected = torch.flip(expected, dims=(-1,))
                if rotation:
                    expected = torch.rot90(expected, k=rotation, dims=(-2, -1))

                assert torch.allclose(tw[g_out, :, :, g_in], expected, atol=ATOL), (
                    f"GroupConv2d weight inconsistent at g_out={g_out}, g_in={g_in} for group={group}, "
                    f"max_diff={(tw[g_out, :, :, g_in] - expected).abs().max().item():.2e}"
                )


# ---------------------------------------------------------------------------
# 8. Shape contracts
# ---------------------------------------------------------------------------


class TestShapeContracts:
    """Verify that all modules produce tensors of the expected shape."""

    @pytest.mark.parametrize("group", GROUPS)
    def test_lifting_conv_output_shape(self, group: str) -> None:
        spec = get_group_spec(group)
        layer = LiftingConv2d(3, 8, 3, 1, 1, group=group, bias=False)
        x = torch.randn(2, 3, 10, 10)
        out = layer(x)
        # Expected: [B, C_out, |G|, H, W]
        assert out.shape == (2, 8, spec.order, 10, 10)

    @pytest.mark.parametrize("group", GROUPS)
    def test_group_conv_output_shape(self, group: str) -> None:
        spec = get_group_spec(group)
        layer = GroupConv2d(4, 6, 3, 1, 1, group=group, bias=False)
        x = torch.randn(2, 4, spec.order, 10, 10)
        out = layer(x)
        assert out.shape == (2, 6, spec.order, 10, 10)

    @pytest.mark.parametrize("group", GROUPS)
    def test_lifting_conv_stride_shape(self, group: str) -> None:
        spec = get_group_spec(group)
        layer = LiftingConv2d(3, 4, 3, stride=2, padding=1, group=group, bias=False)
        x = torch.randn(1, 3, 16, 16)
        out = layer(x)
        assert out.shape == (1, 4, spec.order, 8, 8)

    @pytest.mark.parametrize("group", GROUPS)
    def test_gecnn_output_shape(self, group: str) -> None:
        model = GECNN(
            in_channels=3,
            out_channels=8,
            hidden_channels=[4, 8],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            batchnorm=False,
            linear_hidden_features=[16],
            linear_out_features=5,
            group=group,
        )
        x = torch.randn(2, 3, 16, 16)
        out = model(x)
        assert out.shape == (2, 5)

    @pytest.mark.parametrize("group", GROUPS)
    def test_gecnn_batch_independence(self, group: str) -> None:
        """Output for one sample must be identical regardless of batch size."""
        model = GECNN(
            in_channels=3,
            out_channels=8,
            hidden_channels=[4],
            kernel_sizes=[3],
            strides=[1],
            padding=[1],
            batchnorm=False,
            linear_hidden_features=[16],
            linear_out_features=3,
            group=group,
        )
        model.eval()
        torch.manual_seed(99)
        x = torch.randn(1, 3, 8, 8)

        with torch.no_grad():
            out_single = model(x)
            # Same sample, but wrapped in a bigger batch
            x_batch = x.expand(4, -1, -1, -1)
            out_batch = model(x_batch)

        assert torch.allclose(out_single, out_batch[:1], atol=ATOL)


# ---------------------------------------------------------------------------
# 9. Bias equivariance
# ---------------------------------------------------------------------------


class TestBiasEquivariance:
    """
    Per-channel (not per-spatial-slot) bias must not break equivariance.
    Paper sec. 6.1: "there is only one bias per G-feature map."
    """

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_lifting_conv_bias_equivariance(self, group: str) -> None:
        spec = get_group_spec(group)
        torch.manual_seed(30)
        layer = LiftingConv2d(2, 4, 3, 1, 1, group=group, bias=True)
        layer.eval()
        x = torch.randn(2, 2, 8, 8)

        for element_idx in range(spec.order):
            rotation, mirror = spec.elements[element_idx]
            x_t = x.clone()
            if mirror:
                x_t = torch.flip(x_t, dims=(-1,))
            if rotation:
                x_t = torch.rot90(x_t, k=rotation, dims=(-2, -1))

            with torch.no_grad():
                out_tf = layer(x_t)
                out_ft = _apply_group_transform_to_lifting_output(
                    layer(x), spec, element_idx
                )

            assert torch.allclose(out_tf, out_ft, atol=ATOL), (
                f"Bias breaks LiftingConv2d equivariance for {group}, "
                f"element_idx={element_idx}"
            )

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_group_conv_bias_equivariance(self, group: str) -> None:
        spec = get_group_spec(group)
        torch.manual_seed(31)
        layer = GroupConv2d(3, 5, 3, 1, 1, group=group, bias=True)
        layer.eval()
        f = torch.randn(2, 3, spec.order, 8, 8)

        for element_idx in range(spec.order):
            f_t = _apply_group_transform_to_lifting_output(f, spec, element_idx)
            with torch.no_grad():
                out_tf = layer(f_t)
                out_ft = _apply_group_transform_to_lifting_output(
                    layer(f), spec, element_idx
                )

            assert torch.allclose(out_tf, out_ft, atol=ATOL), (
                f"Bias breaks GroupConv2d equivariance for {group}, "
                f"element_idx={element_idx}"
            )


# ---------------------------------------------------------------------------
# 10. Gradient flow sanity check
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Ensure gradients flow back to all parameters (no dead graph edges)."""

    @pytest.mark.parametrize("group", ["p4", "p4m"])
    def test_gecnn_gradients_flow(self, group: str) -> None:
        model = GECNN(
            in_channels=3,
            out_channels=8,
            hidden_channels=[4],
            kernel_sizes=[3],
            strides=[1],
            padding=[1],
            batchnorm=False,
            linear_hidden_features=[8],
            linear_out_features=2,
            group=group,
        )
        x = torch.randn(2, 3, 8, 8)
        target = torch.zeros(2, 2)
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter '{name}'"
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient for parameter '{name}'"
                )
