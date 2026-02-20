"""Stress tests for flash deformable attention: edge shapes, invalid inputs, numerics, gradients."""

from __future__ import absolute_import, division, print_function

import pytest
import torch
import torch.nn.functional as F

from tests.flash_deform_attn import (
    FlashDeformAttnFunction,
    ms_deform_attn_core_pytorch_3d,
)
from tests.conftest import OPS3D_AVAILABLE


def _make_inputs(N, M, D, Lq, L, K, shapes, device, dtype=torch.float32, seed=42):
    """Build value, sampling_locations, attention_weights, sampling_loc_attn, level_start_index."""
    if isinstance(shapes, torch.Tensor):
        shapes_t = shapes
    else:
        shapes_t = torch.tensor(shapes, dtype=torch.long, device=device)
    level_start_index = torch.cat(
        (shapes_t.new_zeros(1), shapes_t.prod(1).cumsum(0)[:-1])
    )
    S = int(shapes_t.prod(1).sum())

    torch.manual_seed(seed)
    value = torch.rand(N, S, M, D, device=device, dtype=dtype) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, K, 3, device=device, dtype=dtype)
    attention_weights = torch.rand(N, Lq, M, L, K, device=device, dtype=dtype) + 1e-5
    attn_soft = F.softmax(attention_weights.flatten(-2, -1), dim=-1).unflatten(
        -1, (L, K)
    )
    sampling_loc_attn = torch.cat(
        [
            sampling_locations.reshape(N, Lq, M, L * K * 3),
            attention_weights.reshape(N, Lq, M, L * K),
        ],
        dim=-1,
    )
    return value, sampling_locations, attn_soft, sampling_loc_attn, shapes_t, level_start_index


# ---------------------------------------------------------------------------
# Edge shapes (parametrized)
# ---------------------------------------------------------------------------

EDGE_SHAPE_CASES = [
    (1, (1, 1, 1), 1),
    (1, (4, 4, 4), 1),
    (1, (7, 7, 7), 1),
    (1, (8, 8, 8), 1),
    (1, (16, 16, 16), 1),
    (2, (4, 4, 4), 1),
    (2, (8, 8, 8), 1),
    (3, (4, 4, 4), 1),
    (3, (16, 16, 16), 1),
    (4, (8, 8, 8), 2),
    (5, (4, 4, 4), 1),
    (1, (32, 16, 8), 1),
    (2, (16, 8, 4), 1),
    (1, (8, 4, 2), 1),
    (2, (32, 16, 8), 1),
    (1, (8, 8, 8), 64),
]


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
@pytest.mark.parametrize("L,spatial_single,B", EDGE_SHAPE_CASES)
@pytest.mark.parametrize("K", [4, 8])
@torch.no_grad()
def test_forward_edge_shapes(L, spatial_single, B, K, device):
    """Kernel matches reference across edge shapes (isotropic and anisotropic)."""
    M, D = 4, 32
    d, h, w = spatial_single
    Lq = min(64, d * h * w)
    if Lq < 1:
        Lq = 1
    im2col_step = B
    shapes = [list(spatial_single)] * L
    shapes_t = torch.tensor(shapes, dtype=torch.long, device=device)

    value, sampling_locations, attn_soft, sampling_loc_attn, _, level_start_index = _make_inputs(
        B, M, D, Lq, L, K, shapes_t, device
    )

    out_cuda = FlashDeformAttnFunction.apply(
        value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
    )
    out_ref = ms_deform_attn_core_pytorch_3d(
        value, shapes_t, sampling_locations, attn_soft
    )

    assert not torch.isnan(out_cuda).any() and not torch.isinf(out_cuda).any()
    assert not torch.isnan(out_ref).any() and not torch.isinf(out_ref).any()
    assert torch.allclose(out_cuda, out_ref, rtol=1e-2, atol=1e-3), (
        f"L={L}, spatial={spatial_single}, B={B}, K={K}"
    )


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
@torch.no_grad()
def test_forward_uneven_levels(device):
    """Multilevel with very different spatial sizes."""
    shapes = torch.tensor([[1, 1, 1], [64, 64, 64]], dtype=torch.long, device=device)
    N, M, D, Lq, L, K = 1, 4, 32, 64, 2, 4
    im2col_step = 1

    value, sampling_locations, attn_soft, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, shapes, device
    )

    out_cuda = FlashDeformAttnFunction.apply(
        value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
    )
    out_ref = ms_deform_attn_core_pytorch_3d(
        value, shapes_t, sampling_locations, attn_soft
    )

    assert torch.allclose(out_cuda, out_ref, rtol=1e-2, atol=1e-3)


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
@torch.no_grad()
def test_forward_anisotropic_multilevel(device):
    """Anisotropic multi-level: [[32,16,8], [16,8,4]]."""
    shapes = torch.tensor(
        [[32, 16, 8], [16, 8, 4]], dtype=torch.long, device=device
    )
    N, M, D, Lq, L, K = 1, 4, 32, 64, 2, 4
    im2col_step = 1

    value, sampling_locations, attn_soft, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, shapes, device
    )

    out_cuda = FlashDeformAttnFunction.apply(
        value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
    )
    out_ref = ms_deform_attn_core_pytorch_3d(
        value, shapes_t, sampling_locations, attn_soft
    )

    assert torch.allclose(out_cuda, out_ref, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
def test_invalid_batch_not_divisible_by_im2col_step(device):
    """batch % im2col_step != 0 raises (kernel uses im2col_step_=min(batch,im2col_step))."""
    N, M, D, Lq, L, K = 5, 4, 32, 64, 2, 4
    im2col_step = 2
    shapes = torch.tensor([[4, 4, 4], [4, 4, 4]], dtype=torch.long, device=device)

    value, _, _, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, shapes, device
    )

    with pytest.raises((RuntimeError, AssertionError)):
        FlashDeformAttnFunction.apply(
            value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
        )


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
def test_invalid_L6(device):
    """L=6 (unsupported) raises."""
    N, M, D, Lq, L, K = 1, 4, 32, 64, 6, 4
    im2col_step = 1
    shapes = torch.tensor(
        [[4, 4, 4]] * 6, dtype=torch.long, device=device
    )

    value, _, _, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, shapes, device
    )

    with pytest.raises((RuntimeError, ValueError, Exception)):
        FlashDeformAttnFunction.apply(
            value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
        )


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
def test_invalid_cpu_tensors(device):
    """CPU tensors raise."""
    N, M, D, Lq, L, K = 1, 4, 32, 64, 2, 4
    im2col_step = 1
    shapes = torch.tensor([[4, 4, 4], [4, 4, 4]], dtype=torch.long, device="cpu")

    value, _, _, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, shapes, "cpu"
    )

    with pytest.raises((RuntimeError, AssertionError)):
        FlashDeformAttnFunction.apply(
            value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
        )


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
def test_invalid_shape_mismatch(device):
    """sampling_loc_attn level count mismatch vs value raises ValueError."""
    N, M, D, Lq, L, K = 1, 4, 32, 64, 2, 4
    im2col_step = 1
    shapes = torch.tensor([[4, 4, 4], [4, 4, 4]], dtype=torch.long, device=device)

    value, _, _, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, shapes, device
    )
    L_wrong = 1
    sampling_loc_attn_wrong = torch.cat(
        [
            torch.rand(N, Lq, M, L_wrong * K * 3, device=device),
            torch.rand(N, Lq, M, L_wrong * K, device=device) + 1e-5,
        ],
        dim=-1,
    )

    with pytest.raises(ValueError, match="does not match"):
        FlashDeformAttnFunction.apply(
            value,
            shapes_t,
            level_start_index,
            sampling_loc_attn_wrong,
            im2col_step,
            K,
            True,
        )


# ---------------------------------------------------------------------------
# Numeric stress
# ---------------------------------------------------------------------------


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
@torch.no_grad()
def test_numeric_zeros(device, stress_params):
    """value=0, uniform attn, locs=0.5 -> allclose vs ref, no NaN/Inf."""
    p = stress_params
    N, M, D, Lq, L, K = p["N"], p["M"], p["D"], p["Lq"], p["L"], p["K"]
    shapes = p["shapes"]
    level_start_index = p["level_start_index"]
    im2col_step = p["im2col_step"]
    S = p["S"]

    value = torch.zeros(N, S, M, D, device=device)
    sampling_locations = torch.full((N, Lq, M, L, K, 3), 0.5, device=device)
    attn_soft = torch.full((N, Lq, M, L, K), 1.0 / (L * K), device=device)
    sampling_loc_attn = torch.cat(
        [
            sampling_locations.reshape(N, Lq, M, L * K * 3),
            attn_soft.reshape(N, Lq, M, L * K),
        ],
        dim=-1,
    )

    out_cuda = FlashDeformAttnFunction.apply(
        value, shapes, level_start_index, sampling_loc_attn, im2col_step, K, True
    )
    out_ref = ms_deform_attn_core_pytorch_3d(
        value, shapes, sampling_locations, attn_soft
    )

    assert not torch.isnan(out_cuda).any() and not torch.isinf(out_cuda).any()
    assert torch.allclose(out_cuda, out_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
@torch.no_grad()
def test_numeric_small_values(device, stress_params):
    """Small values (1e-6 scale) -> allclose vs ref."""
    p = stress_params
    N, M, D, Lq, L, K = p["N"], p["M"], p["D"], p["Lq"], p["L"], p["K"]
    im2col_step = p["im2col_step"]

    value, sampling_locations, attn_soft, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, p["shapes"], device
    )
    value = value * 1e-6
    sampling_locations = sampling_locations * 1e-6 + 0.5
    sampling_loc_attn = torch.cat(
        [
            sampling_locations.reshape(N, Lq, M, L * K * 3),
            attn_soft.reshape(N, Lq, M, L * K),
        ],
        dim=-1,
    )

    out_cuda = FlashDeformAttnFunction.apply(
        value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
    )
    out_ref = ms_deform_attn_core_pytorch_3d(
        value, shapes_t, sampling_locations, attn_soft
    )

    assert torch.allclose(out_cuda, out_ref, rtol=1e-2, atol=1e-4)


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
@torch.no_grad()
def test_numeric_large_values_fp32(device, stress_params):
    """Large value scale (1e4) in fp32 -> no crash, allclose vs ref."""
    p = stress_params
    N, M, D, Lq, L, K = p["N"], p["M"], p["D"], p["Lq"], p["L"], p["K"]
    im2col_step = p["im2col_step"]

    value, sampling_locations, attn_soft, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, p["shapes"], device
    )
    value = value * 1e4

    out_cuda = FlashDeformAttnFunction.apply(
        value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
    )
    out_ref = ms_deform_attn_core_pytorch_3d(
        value, shapes_t, sampling_locations, attn_soft
    )

    assert torch.allclose(out_cuda, out_ref, rtol=1e-2, atol=1e-1)


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
@torch.no_grad()
def test_numeric_fp16(device, stress_params):
    """float16 -> allclose vs ref with relaxed tol."""
    p = stress_params
    N, M, D, Lq, L, K = p["N"], p["M"], p["D"], p["Lq"], p["L"], p["K"]
    im2col_step = p["im2col_step"]

    value, sampling_locations, attn_soft, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, p["shapes"], device, dtype=torch.float16
    )

    out_cuda = FlashDeformAttnFunction.apply(
        value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
    )
    value_f32 = value.float()
    sampling_locations_f32 = sampling_locations.float()
    attn_soft_f32 = attn_soft.float()
    out_ref = ms_deform_attn_core_pytorch_3d(
        value_f32, shapes_t, sampling_locations_f32, attn_soft_f32
    )

    assert torch.allclose(out_cuda.float(), out_ref, rtol=1e-2, atol=1e-3)


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
@torch.no_grad()
def test_numeric_bfloat16(device, stress_params):
    """bfloat16 -> allclose vs ref if supported."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported on this GPU")

    p = stress_params
    N, M, D, Lq, L, K = p["N"], p["M"], p["D"], p["Lq"], p["L"], p["K"]
    im2col_step = p["im2col_step"]

    value, sampling_locations, attn_soft, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, p["shapes"], device, dtype=torch.bfloat16
    )

    out_cuda = FlashDeformAttnFunction.apply(
        value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
    )
    value_f32 = value.float()
    sampling_locations_f32 = sampling_locations.float()
    attn_soft_f32 = attn_soft.float()
    out_ref = ms_deform_attn_core_pytorch_3d(
        value_f32, shapes_t, sampling_locations_f32, attn_soft_f32
    )

    assert torch.allclose(out_cuda.float(), out_ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Gradient stress
# ---------------------------------------------------------------------------


@pytest.mark.gpu_stress
@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
def test_gradient_zero_grad_output(device):
    """Zero grad_output -> backward yields zero gradients."""
    N, M, D, Lq, L, K = 1, 4, 32, 64, 2, 4
    im2col_step = 1
    shapes = torch.tensor([[4, 4, 4], [4, 4, 4]], dtype=torch.long, device=device)

    value, sampling_locations, attn_soft, sampling_loc_attn, shapes_t, level_start_index = _make_inputs(
        N, M, D, Lq, L, K, shapes, device
    )
    value = value.requires_grad_(True)
    sampling_loc_attn = sampling_loc_attn.requires_grad_(True)

    out = FlashDeformAttnFunction.apply(
        value, shapes_t, level_start_index, sampling_loc_attn, im2col_step, K, True
    )
    grad_output = torch.zeros_like(out)
    out.backward(grad_output)

    assert value.grad is not None and value.grad.abs().max() == 0
    assert sampling_loc_attn.grad is not None and sampling_loc_attn.grad.abs().max() == 0
