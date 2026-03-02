#!/usr/bin/env python
"""
Benchmark: MSDeformAttn vs Flash SDPA — head-to-head comparison.

Compares:
  - MSDeformAttn (CUDA kernel): sparse attention, L×K samples per query
  - MSDeformAttn (PyTorch ref): naive implementation of same
  - Flash SDPA: dense cross-attention over all S positions

Run from ops3d root:
  python -m tests.benchmark_flash_deform_attn
  python -m tests.benchmark_flash_deform_attn --dtype float32 --warmup 5 --repeats 20
"""

from __future__ import absolute_import, division, print_function

import argparse
import fnmatch
import math
import sys
import time
from math import prod
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ops3d import (
    FlashDeformAttnFunction,
    ms_deform_attn_core_pytorch_3d,
)


def _flash_sdpa_available(device, dtype):
    try:
        q = torch.randn(1, 1, 2, 4, device=device, dtype=dtype)
        k = torch.randn(1, 1, 2, 4, device=device, dtype=dtype)
        v = torch.randn(1, 1, 2, 4, device=device, dtype=dtype)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            F.scaled_dot_product_attention(q, k, v)
        return True
    except RuntimeError:
        return False


COP_LARGE_DEFAULTS = {
    "num_heads": 8,
    "embed_dim": 256,
    "num_points": 4,
    "num_queries": 200,
    "batch_size": 8,
}


def parse_config(cfg: dict) -> dict:
    """Parse human-readable config into N, M, D, Lq, L, K, shapes."""
    if "sample_size" not in cfg:
        return cfg

    sample_size = tuple(cfg["sample_size"])
    strides = tuple(cfg["strides"])
    num_heads = cfg["num_heads"]
    embed_dim = cfg["embed_dim"]
    num_points = cfg["num_points"]
    mode = cfg.get("mode", "self")
    batch_size = cfg.get("batch_size", 1)

    shapes = [[sample_size[i] // s for i in range(3)] for s in strides]
    S = sum(prod(s) for s in shapes)
    L = len(strides)

    N = batch_size
    M = num_heads
    D = embed_dim // num_heads
    K = num_points
    Lq = S if mode == "self" else cfg["num_queries"]

    return {
        "name": cfg.get("name", "unnamed"),
        "N": N,
        "M": M,
        "D": D,
        "Lq": Lq,
        "L": L,
        "K": K,
        "shapes": shapes,
        "S": S,
        "sample_size": sample_size,
        "strides": strides,
        "mode": mode,
        "num_heads": num_heads,
        "embed_dim": embed_dim,
        "num_points": num_points,
        "num_queries": cfg.get("num_queries") if mode == "cross" else None,
        "batch_size": batch_size,
    }


INPUT_STRIDE_CONFIGS = [
    ("hypercube", (128, 256, 512), (8, 16, 32)),
    ("hypercube", (128, 256, 512), (16, 32)),
    ("hypercube", (128, 256, 512), (8, 16)),
    ("hypercube", (128, 256, 512), (8,)),
    ("hypercube", (128, 256, 512), (16,)),
    ("tile", (256, 512, 2048), (8, 16, 32)),
    ("tile", (256, 512, 2048), (16, 32)),
    ("tile", (256, 512, 2048), (8, 16)),
    ("tile", (256, 512, 2048), (8,)),
    ("tile", (256, 512, 2048), (16,)),
]


def build_named_configs():
    configs = []
    for input_name, sample_size, strides in INPUT_STRIDE_CONFIGS:
        strides_str = "_".join(map(str, strides))
        for mode in ("self", "cross"):
            cfg = {
                "name": f"{mode}_{input_name}_strides_{strides_str}",
                "mode": mode,
                "sample_size": sample_size,
                "strides": strides,
                **COP_LARGE_DEFAULTS,
            }
            configs.append(parse_config(cfg))
    return configs


BENCHMARK_CONFIGS = build_named_configs()


def _format_bytes(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.2f} KB"
    return f"{n:.0f} B"


def _format_time(sec):
    if sec >= 1:
        return f"{sec:.3f} s"
    if sec >= 1e-3:
        return f"{sec * 1e3:.2f} ms"
    return f"{sec * 1e6:.1f} us"


def _bar(frac, width=20, filled="█", empty="░"):
    n = int(frac * width)
    return filled * n + empty * (width - n)


def run_benchmark(
    config,
    device,
    dtype,
    warmup,
    repeats,
    im2col_step=64,
    flash_available=True,
):
    """Run kernel and reference, return timing and memory stats."""
    shapes = torch.tensor(config["shapes"], dtype=torch.long, device=device)
    level_start_index = torch.cat(
        (shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1])
    )
    S = int(shapes.prod(1).sum())
    N, M, D = config["N"], config["M"], config["D"]
    Lq, L, K = config["Lq"], config["L"], config["K"]

    torch.manual_seed(42)
    value = torch.rand(N, S, M, D, device=device) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, K, 3, device=device)
    attention_weights = torch.rand(N, Lq, M, L, K, device=device) + 1e-5
    attn_soft = torch.nn.functional.softmax(
        attention_weights.flatten(-2, -1), dim=-1
    ).unflatten(-1, (L, K))
    sampling_loc_attn = torch.cat(
        [
            sampling_locations.reshape(N, Lq, M, L * K * 3),
            attention_weights.reshape(N, Lq, M, L * K),
        ],
        dim=-1,
    )

    value = value.to(dtype)
    sampling_loc_attn = sampling_loc_attn.to(dtype)
    sampling_locations = sampling_locations.to(dtype)
    attn_soft = attn_soft.to(dtype)

    # Ensure batch divides im2col_step
    im2col_step = min(im2col_step, N)
    if N % im2col_step != 0:
        im2col_step = N

    def run_kernel():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start = time.perf_counter()
        with torch.no_grad():
            FlashDeformAttnFunction.apply(
                value,
                shapes,
                level_start_index,
                sampling_loc_attn,
                im2col_step,
                K,
                True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        mem_peak = torch.cuda.max_memory_allocated()
        mem_allocated = torch.cuda.memory_allocated()
        return elapsed, mem_allocated, mem_peak

    def run_reference():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start = time.perf_counter()
        with torch.no_grad():
            ms_deform_attn_core_pytorch_3d(
                value,
                shapes,
                sampling_locations,
                attn_soft,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        mem_peak = torch.cuda.max_memory_allocated()
        mem_allocated = torch.cuda.memory_allocated()
        return elapsed, mem_allocated, mem_peak

    def run_flash():
        """Flash SDPA: Lq queries attend over all S positions (dense cross-attention)."""
        query = value[:, :Lq, :, :].reshape(N, Lq, M * D)
        keys = value.reshape(N, S, M * D)
        values = value.reshape(N, S, M * D)
        q = query.view(N, Lq, M, D).transpose(1, 2)  # (N, M, Lq, D)
        k = keys.view(N, S, M, D).transpose(1, 2)  # (N, M, S, D)
        v = values.view(N, S, M, D).transpose(1, 2)  # (N, M, S, D)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start = time.perf_counter()
        with torch.no_grad():
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION]
            ):
                out = F.scaled_dot_product_attention(q, k, v)
                
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        mem_peak = torch.cuda.max_memory_allocated()
        mem_allocated = torch.cuda.memory_allocated()
        return elapsed, mem_allocated, mem_peak

    # Warmup
    for _ in range(warmup):
        run_kernel()
        run_reference()
        if flash_available:
            run_flash()

    # Timed runs
    kernel_times = []
    ref_times = []
    flash_times = []
    kernel_mem_peak = 0
    ref_mem_peak = 0
    flash_mem_peak = 0
    for _ in range(repeats):
        t, _, mp = run_kernel()
        kernel_times.append(t)
        kernel_mem_peak = max(kernel_mem_peak, mp)
        t, _, mp = run_reference()
        ref_times.append(t)
        ref_mem_peak = max(ref_mem_peak, mp)
        if flash_available:
            t, _, mp = run_flash()
            flash_times.append(t)
            flash_mem_peak = max(flash_mem_peak, mp)

    n_elements = N * Lq * M * D
    kernel_mean = sum(kernel_times) / len(kernel_times)
    kernel_std = (sum((t - kernel_mean) ** 2 for t in kernel_times) / len(kernel_times)) ** 0.5
    ref_mean = sum(ref_times) / len(ref_times)
    ref_std = (sum((t - ref_mean) ** 2 for t in ref_times) / len(ref_times)) ** 0.5
    if flash_available and flash_times:
        flash_mean = sum(flash_times) / len(flash_times)
        flash_std = (sum((t - flash_mean) ** 2 for t in flash_times) / len(flash_times)) ** 0.5
        speedup_vs_flash = flash_mean / kernel_mean
        flash_throughput = n_elements / flash_mean
    else:
        flash_mean = float("nan")
        flash_std = float("nan")
        speedup_vs_flash = float("nan")
        flash_throughput = float("nan")
        flash_mem_peak = 0

    kernel_throughput = n_elements / kernel_mean
    ref_throughput = n_elements / ref_mean
    speedup_vs_ref = ref_mean / kernel_mean

    return {
        "config_name": config["name"],
        "sample_size": config.get("sample_size"),
        "strides": config.get("strides"),
        "mode": config.get("mode"),
        "num_heads": config.get("num_heads"),
        "embed_dim": config.get("embed_dim"),
        "num_points": config.get("num_points"),
        "num_queries": config.get("num_queries"),
        "batch_size": config.get("batch_size"),
        "shapes": config["shapes"],
        "N": N,
        "M": M,
        "D": D,
        "Lq": Lq,
        "L": L,
        "K": K,
        "S": S,
        "n_elements": n_elements,
        "dtype": str(dtype).split(".")[-1],
        "kernel_mean_ms": kernel_mean * 1000,
        "kernel_std_ms": kernel_std * 1000,
        "ref_mean_ms": ref_mean * 1000,
        "ref_std_ms": ref_std * 1000,
        "flash_mean_ms": flash_mean * 1000,
        "flash_std_ms": flash_std * 1000,
        "speedup_vs_ref": speedup_vs_ref,
        "speedup_vs_flash": speedup_vs_flash,
        "kernel_throughput_Mel_s": kernel_throughput / 1e6,
        "ref_throughput_Mel_s": ref_throughput / 1e6,
        "flash_throughput_Mel_s": flash_throughput / 1e6,
        "kernel_mem_peak": kernel_mem_peak,
        "ref_mem_peak": ref_mem_peak,
        "flash_mem_peak": flash_mem_peak,
    }


def print_results(results):
    """Pretty-print benchmark results to CLI."""
    mw, cw, sw = 24, 18, 22  # metric, data column, improvement column widths
    w = 2 + mw + 3 * (cw + 1) + (sw + 1)
    sep = "  " + "-" * mw + " " + "-" * cw + " " + "-" * cw + " " + "-" * cw + " " + "-" * sw
    print()
    print("=" * w)
    print("  MSDeformAttn vs Flash SDPA — Head-to-Head Comparison")
    print("=" * w)
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
    print("=" * w)
    print("  MSDeformAttn: sparse (L×K samples/query) | Flash SDPA: dense (all S positions)")
    print("=" * w)

    for r in results:
        name = r["config_name"]
        print()
        print(f"  [{name}] ({r['dtype']})")
        print("-" * w)
        if r.get("sample_size") is not None:
            print("  Config:")
            print(f"    sample_size={r['sample_size']}  strides={r.get('strides')}  mode={r.get('mode')}")
            parts = [
                f"num_heads={r.get('num_heads')}",
                f"embed_dim={r.get('embed_dim')}",
                f"num_queries={r.get('num_queries') or r.get('Lq')}",
                f"num_points={r.get('num_points')}",
                f"batch_size={r.get('batch_size')}",
            ]
            print(f"    {'  '.join(str(p) for p in parts)}")
        print(
            "  Parsed: N={} Lq={} M={} D={} L={} K={} S={}  shapes={}".format(
                r["N"], r["Lq"], r["M"], r["D"], r["L"], r["K"], r["S"], r.get("shapes", [])
            )
        )
        print("-" * w)

        if r.get("oom"):
            oom_msg = r.get("oom_message", "CUDA out of memory")
            oom_val = "OOM (crashed)"
            print(f"  *** {oom_msg} ***")
            print()

            def row(metric, k_val, r_val, f_val, imp_val):
                return f"  {metric:<{mw}} {k_val:>{cw}} {r_val:>{cw}} {f_val:>{cw}} {imp_val:>{sw}}"

            print(row("Metric", "MSDeform", "PyTorch Ref", "Flash SDPA", "MSDeform Improvement"))
            print(sep)
            print(row("Time (mean ± std)", oom_val, oom_val, oom_val, "—"))
            print(row("Throughput (M el/s)", oom_val, oom_val, oom_val, "—"))
            print(row("Peak GPU memory", oom_val, oom_val, oom_val, "—"))
            print()
            continue

        k_mean = r["kernel_mean_ms"]
        k_std = r["kernel_std_ms"]
        ref_mean = r["ref_mean_ms"]
        ref_std = r["ref_std_ms"]
        flash_mean = r["flash_mean_ms"]
        flash_std = r["flash_std_ms"]
        flash_na = math.isnan(flash_mean)

        k_time = f"{k_mean:.2f} ± {k_std:.2f} ms"
        r_time = f"{ref_mean:.2f} ± {ref_std:.2f} ms"
        f_time = "N/A" if flash_na else f"{flash_mean:.2f} ± {flash_std:.2f} ms"
        if flash_na:
            speedup_str = f"{r['speedup_vs_ref']:.2f}x / N/A"
        else:
            speedup_str = f"{r['speedup_vs_ref']:.2f}x / {r['speedup_vs_flash']:.2f}x"

        def row(metric, k_val, r_val, f_val, imp_val):
            return f"  {metric:<{mw}} {k_val:>{cw}} {r_val:>{cw}} {f_val:>{cw}} {imp_val:>{sw}}"

        print(row("Metric", "MSDeform", "PyTorch Ref", "Flash SDPA", "MSDeform Improvement"))
        print(sep)
        print(row("Time (mean ± std)", k_time, r_time, f_time, speedup_str))
        flash_throughput_str = "N/A" if flash_na else f"{r['flash_throughput_Mel_s']:.2f}"
        print(
            row(
                "Throughput (M el/s)",
                f"{r['kernel_throughput_Mel_s']:.2f}",
                f"{r['ref_throughput_Mel_s']:.2f}",
                flash_throughput_str,
                speedup_str,
            )
        )
        ref_mem_ratio = r["ref_mem_peak"] / r["kernel_mem_peak"] if r["kernel_mem_peak"] > 0 else 0
        flash_mem_ratio = (
            r["flash_mem_peak"] / r["kernel_mem_peak"] if r["kernel_mem_peak"] > 0 and not flash_na else 0
        )
        mem_speedup = f"{ref_mem_ratio:.2f}x / {flash_mem_ratio:.2f}x" if not flash_na else f"{ref_mem_ratio:.2f}x / N/A"
        flash_mem_str = "N/A" if flash_na else _format_bytes(r["flash_mem_peak"])
        print(
            row(
                "Peak GPU memory",
                _format_bytes(r["kernel_mem_peak"]),
                _format_bytes(r["ref_mem_peak"]),
                flash_mem_str,
                mem_speedup,
            )
        )

        # ASCII bar: all three (shorter bar = faster), or two when Flash N/A
        max_time = max(k_mean, ref_mean, flash_mean if not flash_na else 0)
        if max_time > 0:
            k_bar = k_mean / max_time
            ref_bar = ref_mean / max_time
            flash_bar = flash_mean / max_time if not flash_na else 0
        else:
            k_bar = ref_bar = flash_bar = 0
        print()
        print(f"  Relative time (shorter bar = faster):")
        print(f"    MSDeform   {_bar(k_bar)} {_format_time(k_mean / 1000)}")
        print(f"    PyTorch   {_bar(ref_bar)} {_format_time(ref_mean / 1000)}")
        if flash_na:
            print(f"    Flash     N/A (kernel not available on this GPU)")
        else:
            print(f"    Flash     {_bar(flash_bar)} {_format_time(flash_mean / 1000)}")
        print()

    print("=" * w)
    print("  MSDeform Improvement = other_time / MSDeform_time (Ref / Flash, higher = faster)")
    print("=" * w)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark flash deformable 3D attention kernel vs PyTorch"
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Data type (default: bfloat16)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Timed repeats (default: 20)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config name or fnmatch pattern (e.g. self_hypercube_strides_16_32 or 'self_hypercube*'). Default: all",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Aborting.")
        sys.exit(1)

    try:
        import ops3d._C  # noqa: F401
    except ImportError:
        print("ops3d._C not found. Run: pip install -e .")
        sys.exit(1)

    _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = _dtype_map[args.dtype]
    device = torch.device("cuda")

    configs = BENCHMARK_CONFIGS
    if args.config:
        configs = [
            c
            for c in configs
            if c["name"] == args.config or fnmatch.fnmatch(c["name"], args.config)
        ]
        if not configs:
            print(f"No configs match: {args.config}")
            sys.exit(1)

    flash_available = _flash_sdpa_available(device, dtype)
    if not flash_available:
        print("Note: Flash SDPA not available on this GPU (e.g. requires sm80+). Flash SDPA column will show N/A.")

    results = []
    for cfg in configs:
        try:
            r = run_benchmark(
                cfg, device, dtype, args.warmup, args.repeats, flash_available=flash_available
            )
            results.append(r)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__:
                torch.cuda.empty_cache()
                oom_result = {
                    "config_name": cfg["name"],
                    "sample_size": cfg.get("sample_size"),
                    "strides": cfg.get("strides"),
                    "mode": cfg.get("mode"),
                    "num_heads": cfg.get("num_heads"),
                    "embed_dim": cfg.get("embed_dim"),
                    "num_points": cfg.get("num_points"),
                    "num_queries": cfg.get("num_queries"),
                    "batch_size": cfg.get("batch_size"),
                    "shapes": cfg["shapes"],
                    "N": cfg["N"],
                    "M": cfg["M"],
                    "D": cfg["D"],
                    "Lq": cfg["Lq"],
                    "L": cfg["L"],
                    "K": cfg["K"],
                    "S": cfg["S"],
                    "n_elements": cfg["N"] * cfg["Lq"] * cfg["M"] * cfg["D"],
                    "dtype": str(dtype).split(".")[-1],
                    "oom": True,
                    "oom_message": str(e)[:80],
                    "kernel_mean_ms": float("nan"),
                    "kernel_std_ms": float("nan"),
                    "ref_mean_ms": float("nan"),
                    "ref_std_ms": float("nan"),
                    "flash_mean_ms": float("nan"),
                    "flash_std_ms": float("nan"),
                    "speedup_vs_ref": float("nan"),
                    "speedup_vs_flash": float("nan"),
                    "kernel_throughput_Mel_s": float("nan"),
                    "ref_throughput_Mel_s": float("nan"),
                    "flash_throughput_Mel_s": float("nan"),
                    "kernel_mem_peak": 0,
                    "ref_mem_peak": 0,
                    "flash_mem_peak": 0,
                }
                results.append(oom_result)
            else:
                raise

    print_results(results)


if __name__ == "__main__":
    main()
