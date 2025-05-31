# ops3d – High-Performance 3D CUDA Kernels
Specialised CUDA/CPP ops for volumetric computer-vision tasks: Flash Deformable 3D Attention, Deformable 3D Convolution V4, 3D NMS and 3D ROI-Align tuned for petabyte-scale Light-Sheet microscopy workloads.

## Table of Contents

- [Installation](#installation)
- [Kernels](#kernels)
  - [Flash Deformable 3D Attention](#flash-deformable-3d-attention)
  - [Deformable 3D Convolution Version 4](#deformable-3d-convolution-version-4)
  - [NMS 3D](#nms-3d)
  - [ROI-Align 3D](#roi-align-3d)
  
## Installation

This package has been tested on **CUDA 12.4** with **PyTorch 2.4.1** and **Python 3.10**. We recommend using a dedicated `conda` environment.

```bash
# 1. Create & activate conda environment
conda create -n ops3d python=3.10
conda activate ops3d

# 2. Clone the repository
https://github.com/cell-observatory/ops3d.git
cd ops3d

# 3. Install package in editable mode
pip install -e . 
```

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

A **3-D adaptation of DCNv4 deformable convolution** that samples a  $K = k_d \times k_h \times k_w$ grid of offsets (default **`3` × `3` × `3` = `27`**) per voxel.  
Two fully-custom kernels mirror the attention implementation.

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
