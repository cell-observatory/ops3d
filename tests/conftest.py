"""Pytest configuration and shared fixtures for ops3d flash deformable attention tests."""

from __future__ import absolute_import, division, print_function

import pytest
import torch

# Skip entire test module if CUDA unavailable
pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

# Skip if ops3d._C not compiled
try:
    import ops3d._C as _C  # noqa: F401
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False


def pytest_configure(config):
    if not OPS3D_AVAILABLE:
        config.addinivalue_line(
            "markers",
            "skip_no_ops3d: skip when ops3d._C is not compiled (run pip install -e .)",
        )


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="session")
def default_forward_params():
    """Params matching cell_observatory test_flash_deform_attn_fwd."""
    N = 1
    M = 8
    D = 288
    Lq = 32 * 32 * 32
    L = 4
    K = 8
    im2col_step = 128
    shapes = torch.tensor(
        [[64, 64, 64], [32, 32, 32], [16, 16, 16], [8, 8, 8]],
        dtype=torch.long,
        device="cuda",
    )
    level_start_index = torch.cat(
        (shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1])
    )
    S = int(shapes.prod(1).sum())
    return {
        "N": N,
        "M": M,
        "D": D,
        "Lq": Lq,
        "L": L,
        "K": K,
        "im2col_step": im2col_step,
        "shapes": shapes,
        "level_start_index": level_start_index,
        "S": S,
    }


@pytest.fixture(scope="session")
def default_backward_params():
    """Params matching cell_observatory test_flash_deform_attn_bwd (smaller for speed)."""
    N, M, D = 1, 8, 48
    Lq = 64 * 64 * 64 + 32 * 32 * 32
    L, K = 2, 4
    im2col_step = 64
    shapes = torch.tensor(
        [[64, 64, 64], [32, 32, 32]],
        dtype=torch.long,
        device="cuda",
    )
    level_start_index = torch.cat(
        (shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1])
    )
    S = int(shapes.prod(1).sum())
    return {
        "N": N,
        "M": M,
        "D": D,
        "Lq": Lq,
        "L": L,
        "K": K,
        "im2col_step": im2col_step,
        "shapes": shapes,
        "level_start_index": level_start_index,
        "S": S,
    }
