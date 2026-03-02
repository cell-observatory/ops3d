# ops3d – High-Performance 3D CUDA Kernels
Specialised CUDA/CPP ops for volumetric computer-vision tasks: Flash Deformable 3D Attention, Deformable 3D Convolution V4, 3D NMS and 3D ROI-Align tuned for petabyte-scale Light-Sheet microscopy workloads.

## Table of Contents

- [Installation](#installation)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Kernels](#kernels)
  - [Flash Deformable 3D Attention](#flash-deformable-3d-attention)
  - [Deformable 3D Convolution Version 4](#deformable-3d-convolution-version-4)
  - [NMS 3D](#nms-3d)
  - [ROI-Align 3D](#roi-align-3d)
  
## Installation

This package has been tested on **CUDA 13.1** with **PyTorch 2.10** and **Python 3.12**. 

### Docker

```bash
docker pull ghcr.io/cell-observatory/ops3d:main_torch_26_01
docker run --network host -u 1000 --privileged -v $(pwd):/workspace/ops3d -w /workspace/ops3d --env PYTHONUNBUFFERED=1 --pull missing -it --rm  --ipc host --gpus all ghcr.io/cell-observatory/ops3d:main_torch_26_01 bash
```

### Running docker image on a cluster via apptainer
```shell
apptainer pull --arch amd64 --force ops3d_26_01.sif docker://ghcr.io/cell-observatory/ops3d:main_torch_26_01
```

### Built distribution
```bash
git clone https://github.com/cell-observatory/ops3d.git
cd ops3d

pip install /dist/ops3d-0.1.0-cp312-cp312-linux_x86_64.whl
```

### From source

```bash
git clone https://github.com/cell-observatory/ops3d.git
cd ops3d

# Install in editable mode
pip install -e .
```


## Testing

Run the flash deformable attention tests (kernel vs PyTorch reference):

```bash
pytest tests/ -v
# or
python -m pytest tests/ -v
```

Tests require CUDA and a compiled ops3d package (`pip install -e .`).

Run stress tests (edge shapes, invalid inputs, numerics, gradients):

```bash
pytest tests/ -m gpu_stress -v
```

## Benchmarking

Benchmark MSDeformAttn vs Flash SDPA (head-to-head comparison):

```bash
python -m tests.benchmark_flash_deform_attn
```

Compares:
- **MSDeformAttn** (CUDA kernel): sparse attention, L×K samples per query
- **MSDeformAttn** (PyTorch ref): naive implementation
- **Flash SDPA**: dense cross-attention over all S positions

### Config modes

Configs mirror [CellObservatoryPlatform](https://github.com/cell-observatory/ops3d) settings:

- **Self-attention** (`self_*`): queries = keys = values (e.g. MaskDINO encoder). Each spatial token attends to L×K sampled locations.
- **Cross-attention** (`cross_*`): object queries attend over spatial tokens (e.g. MaskDINO decoder). Lq = 200 queries.

### Config naming

Configs are named `{mode}_{input}_{strides}`:

- **Input**: `hypercube` (128×256×512), `tile` (256×512×2048)
- **Strides**: `8`, `16`, `32` or combinations like `8_16`, `16_32`, `8_16_32`

Examples: `self_hypercube_strides_16_32`, `cross_tile_strides_8_16_32`

### CLI options

| Option | Default | Description |
|--------|---------|--------------|
| `--dtype` | bfloat16 | Data type: `bfloat16`, `float16`, `float32` |
| `--warmup` | 5 | Warmup iterations before timing |
| `--repeats` | 20 | Timed repeats per config |
| `--config` | all | Config name or fnmatch pattern (e.g. `self_hypercube*`) |

### Examples

```bash
# Run all configs (default)
python -m tests.benchmark_flash_deform_attn

# Run a single config
python -m tests.benchmark_flash_deform_attn --config self_hypercube_strides_16_32

# Run all hypercube self-attention configs
python -m tests.benchmark_flash_deform_attn --config "self_hypercube*"

# Float32 with fewer repeats for quick checks
python -m tests.benchmark_flash_deform_attn --dtype float32 --warmup 2 --repeats 5
```

### Output

Each config prints:
- **Config**: sample_size, strides, mode, num_heads, embed_dim, num_points, num_queries
- **Parsed**: N, Lq, M, D, L, K, S, shapes
- **Metrics**: wall time (mean ± std), throughput (M el/s), peak GPU memory, speedup vs PyTorch ref and Flash SDPA
- **ASCII bar**: relative timing (shorter = faster)

### OOM handling

If a config runs out of GPU memory, the benchmark catches the error, clears the cache, and continues. The failed config is reported with `OOM (crashed)` in the metrics table.

## Kernels

### Flash Deformable 3D Attention

A 3-D generalisation of Flash Deformable Attention from the DCNv4 paper that lifts the original 2-D idea to volumetric feature maps.
Two fully-custom kernels are provided:

#### Register & Warp-Level Pipeline  <a id="alg-reg"></a>

The register path (Algorithm&nbsp;1) incorporates three key optimisations:

- **Channel fusion per thread** – each thread accumulates  
 $d_{\text{stride}} \in \{1,2,4,8,16\}$ contiguous channels that share a common offset, eliminating redundant reads.
- **Vectorised memory transactions** – packed loads/stores collapse memory instructions, cutting global-memory traffic.

<summary><strong>Algorithm 1 — Register/Warp Pipeline</strong></summary>

```text
Input :  value  [B,N,G,D]               // feature tensor
         offset [B,N,G,L,K,4]           // (x,y,z,α) offsets
         shape  [L,3]                   // (Sd,Sh,Sw) per level
         lvl_start[L]

Thread  : (b,q,g,d0)  // slice = d0 : d0+d_stride-1
-----------------------------------------------------------------
1) Load L×K logits into registers
   mask[L·K] ← offset[b,q,g,:, :,3]
   mask      ← softmax(mask)
   p         ← 0  // register accumulator

2) Accumulate per thread
   for ℓ = 0 … L-1:
       (Sd,Sh,Sw) ← shape[ℓ]
       ptr ← value + (lvl_start[ℓ] + b·N) · (G·D)
       for k = 0 … K-1:
           (u,v,w,α) ← offset[b,q,g,ℓ,k,0:4]
           (z,y,x)   ← (u·Sd-0.5, v·Sh-0.5, w·Sw-0.5)
           if (z,y,x) inside volume:
               p += α · TriInterp(ptr,(z,y,x), g·D+d0, d_stride)

3) Vectorised register write-back
   VectorStore(out[b,q,g,d0:], p)
```


#### Shared-Memory Pipeline  <a id="alg-smp"></a>

The shared-memory path (Algorithm 2) incorporates four key optimisations:

- **Co-operative mask staging** – threads in a block collectively load the L × K logits for all G heads into shared memory.  
- **Single soft-max pass** – each head’s weights are normalised once in shared memory and reused by all threads.  
- **Minimised barriers** – only two block-wide barriers (before/after the soft-max); inner loops are barrier-free.  
- **Vectorised write-back** – identical vector store to the register kernel.  

<summary><strong>Algorithm&nbsp;2 — Shared-Memory Pipeline</strong></summary>

```text
Input :  value  [B,N,G,D]
         offset [B,N,G,L,K,4]
         shape  [L,3]
         lvl_start[L]
Output:  out    [B,N,G,D]

-----------------------------------------------------------------
Shared memory : maskSH[G_blk, L, K]  // logits for this block

1) Cooperative mask staging
   forall threads t in block:
       copy strided chunk of offset[b,q,g,:, :,3] → maskSH
   __syncthreads()

2) One-time soft-max per head
   if threadIdx.x == 0:
       forall g in block:
           maskSH[g,:,:] ← softmax(maskSH[g,:,:])
   __syncthreads()

3) Per-thread accumulation
   forall threads t = (g,d0) in block:
       acc ← 0
       for ℓ = 0 … L-1:
           (Sd,Sh,Sw) ← shape[ℓ]
           ptr ← value + (lvl_start[ℓ] + b·N) · (G·D)
           for k = 0 … K-1:
               (u,v,w) ← offset[b,q,g,ℓ,k,0:3]
               α       ← maskSH[g,ℓ,k]
               (z,y,x) ← (u·Sd-0.5, v·Sh-0.5, w·Sw-0.5)
               if (z,y,x) inside volume:
                   acc += α · TriInterp(ptr,(z,y,x), g·D+d0, d_stride)

       VectorStore(out[b,q,g,d0:], acc)
```

### Deformable 3D Convolution V4

A **3-D adaptation of DCNv4 deformable convolution** that samples a  $K = k_d \times k_h \times k_w$ grid of offsets (default **`3` × `3` × `3` = `27`**) per voxel. Two fully-custom kernels mirror the attention implementation.

#### Register & Warp-Level Pipeline  <a id="alg-dcn-reg"></a>

Key optimisations:

- **Channel fusion per thread** – each thread integrates $d_{\text{stride}} \in \{1,2,4,8,16\}$ contiguous channels sharing the same offset triplet.  
- **Vectorised memory transactions** – packed loads/stores eliminate scalar memory ops.  
- **Register-resident mask, soft-max & accumulation** – all K weights plus partial sums stay entirely in registers.  

<summary><strong>Algorithm&nbsp;3 — Register/Warp Pipeline</strong></summary>

```text
Input :  value    [B,N,G,D]            // N = D_in·H_in·W_in voxels
         offset   [B,N,G,K,3]          // (Δx,Δy,Δz)
         mask     [B,N,G,K]            // attention logits or weights
         params   kernel/stride/pad/dil, remove_center, offset_scale
Thread : (b,n,g,d0)  // slice = d0 : d0+d_stride-1
--------------------------------------------------------------------
1) Load K logits → mask[K];   mask ← softmax(mask)  (optional)

2) Decode voxel coords (x,y,z) from n; pre-compute base (p0_x,p0_y,p0_z)

3) Accumulate per sampling point
   for each (kz,ky,kx) in kernel grid:
       if remove_center && (kx,ky,kz)==(0,0,0):  continue
       (Δx,Δy,Δz) ← offset[k]
       (wx,wy,wz) ← p0 + (kx,ky,kz)·dil + (Δx,Δy,Δz)·offset_scale
       if in-bounds:
           val  ← TriInterp(value, (wx,wy,wz), g·D+d0, d_stride)
           acc += mask[k] · val

4) Vectorised register write-back
   VectorStore(out[b,n,g,d0:], acc)
```

#### Shared-Memory Pipeline  <a id="alg-dcn-smp"></a>

Key optimisations:

- **Co-operative mask staging** – all threads in a block collectively copy K logits into shared memory.  
- **Single soft-max pass** – one normalisation per head, reused by every thread.  
- **Minimal barriers** – only two block-wide syncs protect the shared soft-max; inner loops are barrier-free.  
- **Vectorised write-back** – identical packed store to the register kernel.  

<summary><strong>Algorithm&nbsp;4 — Shared-Memory Pipeline</strong></summary>

```text
Input :  value    [B,N,G,D]
         offset   [B,N,G,K,3]
         mask     [B,N,G,K]
Output:  out      [B,N,G,D]
Shared : maskSH[G_blk,K]              // logits for this block
--------------------------------------------------------------------
1) Cooperative mask staging
   forall threads t in block:
       copy strided chunk of mask[b,n,g,:] → maskSH
   __syncthreads()

2) Single soft-max per head
   if threadIdx.x == 0:
       forall g in block:  maskSH[g,:] ← softmax(maskSH[g,:])
   __syncthreads()

3) Per-thread accumulation
   forall threads t = (g,d0):
       acc ← 0
       decode (x,y,z) from n; pre-compute p0_x,y,z
       for each kernel point (kx,ky,kz):
           if remove_center && centre: continue
           (Δx,Δy,Δz) ← offset[b,n,g,k]
           (wx,wy,wz) ← p0 + (kx,ky,kz)·dil + (Δx,Δy,Δz)·offset_scale
           if in-bounds:
               val  ← TriInterp(value,(wx,wy,wz), g·D+d0, d_stride)
               acc += maskSH[g,k] · val

       VectorStore(out[b,n,g,d0:], acc)
```

### NMS 3D  <a id="nms-3d"></a>

3-D non-maximum suppression CUDA/CPP kernel from https://github.com/TimothyZero/MedVision.

### ROI-Align 3D  <a id="roi-align-3d"></a>

3-D ROI-Align CUDA/CPP kernel from https://github.com/TimothyZero/MedVision.


# License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

   [Apache License 2.0](LICENSE)

Copyright 2025 Cell Observatory.
