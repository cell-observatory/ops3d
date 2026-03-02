"""Forward correctness: CUDA kernel vs PyTorch reference."""

from __future__ import absolute_import, division, print_function

import pytest

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ops3d import _C as _C
from ops3d.flash_deform_attn import findspec, findspec_bwd, ms_deform_attn_core_pytorch_3d


class FlashDeformAttnFunctionWrapper(Function):
    @staticmethod
    # @torch.autocast("cuda", enabled=True, dtype=torch.float16)
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_loc_attn,
        im2col_step,  # partitions batch into bs/im2col calls
        K=8,  # num points sampled
        use_reg=True,  # use warp based implementation
    ):
        if _C is None:
            raise RuntimeError("ops3d._C extension not built; FlashDeformAttnFunction requires it.")
        ctx.K = K
        ctx.im2col_step = im2col_step

        # findspec(Batch Size, Queries, Num_heads = G, Channels per Group = C)
        # determine number of channels per thread inside group = d_stride and total number of threads in block = multiplier * G * C / d_stride
        # where we partition threads into blocks of dimension Z=multiplier, Y=G, X=C/d_stride
        d_stride, blockthread = findspec(value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3])
        d_stride_backward, blockthread_backward = findspec_bwd(
            value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3]
        )

        ctx.d_stride_backward = d_stride_backward
        ctx.blockthread_backward = blockthread_backward

        output = _C.flash_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            ctx.im2col_step,
            K,
            d_stride,
            blockthread,
            use_reg,
        )

        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_loc_attn)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_loc_attn = ctx.saved_tensors

        grad_value, grad_sampling_loc_attn = _C.flash_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            grad_output.contiguous(),
            ctx.im2col_step,
            ctx.K,
            ctx.d_stride_backward,
            ctx.blockthread_backward,
        )

        return grad_value, None, None, grad_sampling_loc_attn, None, None, None
    

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
        FlashDeformAttnFunctionWrapper.apply(
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
            "Output from FlashDeformAttnFunctionWrapper contains NaN or Inf"
        )

    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = (
        (output_cuda - output_pytorch).abs() / (output_pytorch.abs() + 1e-8)
    ).max()

    assert torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3), (
        f"forward mismatch; max abs: {max_abs_err}, max rel: {max_rel_err}"
    )