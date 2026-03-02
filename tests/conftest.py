"""Pytest configuration and shared fixtures for ops3d flash deformable attention tests."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path

# Ensure project root is on path so "tests" package is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
import torch

# Skip entire test module if CUDA unavailable
pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


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


@pytest.fixture(scope="session")
def stress_params():
    """Small config for stress tests (fast)."""
    N, M, D = 1, 4, 32
    Lq = 64
    L, K = 1, 4
    im2col_step = 1
    shapes = torch.tensor([[4, 4, 4]], dtype=torch.long, device="cuda")
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
