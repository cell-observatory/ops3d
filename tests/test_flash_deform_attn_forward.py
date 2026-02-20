"""Forward correctness: CUDA kernel vs PyTorch reference."""

from __future__ import absolute_import, division, print_function

import pytest
import torch

from tests.flash_deform_attn import (
    FlashDeformAttnFunction,
    ms_deform_attn_core_pytorch_3d,
)

from tests.conftest import OPS3D_AVAILABLE


@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="ops3d._C not compiled; run pip install -e .",
)
@torch.no_grad()
def test_forward_equal_with_pytorch_half(device, default_forward_params):
    """Kernel output matches PyTorch reference (float16)."""
    p = default_forward_params
    N, M, D = p["N"], p["M"], p["D"]
    Lq, L, K = p["Lq"], p["L"], p["K"]
    S = p["S"]
    shapes = p["shapes"]
    level_start_index = p["level_start_index"]
    im2col_step = p["im2col_step"]

    torch.manual_seed(42)
    value = torch.rand(N, S, M, D, device=device) * 0.01

    sampling_locations = torch.rand(N, Lq, M, L, K, 3, device=device)
    attention_weights = torch.rand(N, Lq, M, L, K, device=device) + 1e-5
    sampling_loc_attn = torch.cat(
        [
            sampling_locations.reshape(N, Lq, M, L * K * 3),
            attention_weights.reshape(N, Lq, M, L * K),
        ],
        dim=-1,
    )
    attention_weights = torch.nn.functional.softmax(
        attention_weights.flatten(-2, -1), dim=-1
    ).unflatten(-1, (L, K))

    output_cuda = (
        FlashDeformAttnFunction.apply(
            value.half(),
            shapes,
            level_start_index,
            sampling_loc_attn.half(),
            im2col_step,
            K,
            True,
        )
        .detach()
        .cpu()
        .double()
    )

    output_pytorch = (
        ms_deform_attn_core_pytorch_3d(
            value,
            shapes,
            sampling_locations,
            attention_weights,
        )
        .detach()
        .double()
        .cpu()
    )

    if torch.isnan(output_pytorch).any() or torch.isinf(output_pytorch).any():
        pytest.fail(
            "Output from ms_deform_attn_core_pytorch_3d contains NaN or Inf"
        )
    if torch.isnan(output_cuda).any() or torch.isinf(output_cuda).any():
        pytest.fail(
            "Output from FlashDeformAttnFunction contains NaN or Inf"
        )

    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = (
        (output_cuda - output_pytorch).abs() / (output_pytorch.abs() + 1e-8)
    ).max()

    assert torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3), (
        f"forward mismatch; max abs: {max_abs_err}, max rel: {max_rel_err}"
    )
