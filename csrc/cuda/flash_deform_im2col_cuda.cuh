//  Modified to 3D from:
//  https://github.com/OpenGVLab/DCNv4/blob/main/DCNv4_op/src/cuda/flash_deform_im2col_cuda.cuh
//  MIT License

//  Copyright (c) 2022 OpenGVLab

//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:

//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.

//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.


#include <algorithm>
#include <cstdio>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.h"


// shared mem. version
template <typename scalar_t, int d_stride, typename transfer_t, int L, int K>
__global__ void forward_kernel(const scalar_t *p_value, const int64_t *data_spatial_shapes,
                const int64_t *data_level_start_index, const scalar_t *p_offset,
                scalar_t *p_output, const int N, const int G, const int D,
                const int Q, const int block_multiplier) {

    // allocate shared memory 
    extern __shared__ char _s[];

    // get (bi,qi,gi,di_s) indices (based on block partitioning logic explained below)
    const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
    const int &bi = blockIdx.x * block_multiplier / Q;
    const int &di_s = threadIdx.x * d_stride;
    const int &gi = threadIdx.y;

    // allocate d_stride memory for thread for out channel output
    opmath_t p_out_shm[d_stride] = {0.};

    // for given (qi, gi), we will load L*K values into shared memory
    // for threads that attend to a given group head 
    // shared memory lives on block level, so we need not concern ourselves with
    // block level indexing (for block we are moving along z,y axis and loading for all x)
    // which amounts to moving along threadidx.z (some query) and head idx (gi)
    opmath_t *const p_mask_shm = (opmath_t *)(_s) + (threadIdx.z * G + gi) * L * K;

    // move offset ptr to (bi,qi,gi,L,offsets) where offsets are (x,y,z,attn)
    const scalar_t *p_offset_ptr = p_offset + (((bi * Q + qi) * G + gi) * L) * K * 4;

    // get number of k values each thread will process
    // each thread processes some chunk of L*K values for given (bi,qi,gi)
    // that it belongs to 
    const int mask_length = L * K;
    const int num_thread = (D / d_stride);
    const int num_iter = mask_length / num_thread;
    const int remainder = mask_length - num_iter * num_thread;

  // write mask values (for offsets=(x,y,z,mask)) to shared memory 
  // partitioned across threads with strided writes by num_thread
    for (int i = 0; i < num_iter; i++) {
        *(p_mask_shm + num_thread * i + threadIdx.x) =
            *(scalar_t *)(p_offset_ptr + L * K * 3 + num_thread * i + threadIdx.x);
    }
  
    if (remainder > 0 && threadIdx.x < remainder) {
        *(p_mask_shm + num_thread * num_iter + threadIdx.x) =
            *(scalar_t *)(p_offset_ptr + L * K * 3 + num_thread * num_iter +
                            threadIdx.x);
    }

    __syncthreads();

    // Calculate softmax over L and K
    // TODO: can this be simplified?
    if (threadIdx.x == 0) { // di = 0
        opmath_t softmax_max = -1e100;
        opmath_t softmax_sum = 0.0;

        // get max
        for (int j = 0; j < L * K; j++) {
            softmax_max = max(softmax_max, p_mask_shm[j]);
        }

        // get sumexp
        for (int j = 0; j < L * K; j++) {
            opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
            p_mask_shm[j] = exp_results;
            softmax_sum += exp_results;
        }

        // normalize
        for (int j = 0; j < L * K; j++) {
            p_mask_shm[j] /= softmax_sum;
        }
    }

    __syncthreads();

    int offset_idx = 0;
    int mask_idx = 0;

    // w=x stride is across all groups and channels
    // for a given (qi,gi) representing some location (x,y,z)
    const int w_stride = G * D;
    // base_ptr for given group and channel set (gi,di_s)
    // added to p_value_ptr below to index into specific
    // gi, di_s location inside p_value
    const int base_ptr = gi * D + di_s;

    for (int li = 0; li < L; li++) {
        // get spatial shapes
        const int spatial_d = data_spatial_shapes[li * 3];
        const int spatial_h = data_spatial_shapes[li * 3 + 1];
        const int spatial_w = data_spatial_shapes[li * 3 + 2];
        const int level_start_id = data_level_start_index[li];

        // move p_value to block (bi,qi)
        const scalar_t *p_value_ptr = p_value + (bi * N + level_start_id) * G * D;

        for (int ki = 0; ki < K; ki++) {
            // one-time reads for offsets => no point in storing in 
            // shared memory since no reuse per groups of threads
            const opmath_t loc_w = p_offset_ptr[offset_idx];
            const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
            const opmath_t loc_d = p_offset_ptr[offset_idx+ 2];
            const opmath_t attn = p_mask_shm[mask_idx];

            const opmath_t d_im = loc_d * spatial_d - 0.5;
            const opmath_t h_im = loc_h * spatial_h - 0.5;
            const opmath_t w_im = loc_w * spatial_w - 0.5;
        
        if (d_im > -1 && h_im > -1 && w_im > -1 && d_im < spatial_d && h_im < spatial_h && w_im < spatial_w) {
            // trilinear interpolation with learned attn weights (see common.h)
            ms_deform_attn_im2col_trilinear<scalar_t, transfer_t, d_stride>(
                p_out_shm, p_value_ptr, 
                spatial_d, spatial_h, spatial_w, 
                d_im, h_im, w_im, 
                attn, w_stride, base_ptr);
        }
        // update offset and mask index
        offset_idx += 3;
        mask_idx += 1;
    }
  }

    // out_idx at (bi,qi,gi,di_s)
    int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

    // alias cast to fp16 and cast in copy
    // then save in one go 
    scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
    #pragma unroll
    for (int ds = 0; ds < d_stride; ds++) {
        fp16_regs[ds] = p_out_shm[ds];
    }

    *(transfer_t *)(p_output + out_idx) = *(transfer_t *)(p_out_shm);
}


// register based version 
template <typename scalar_t, int d_stride, typename transfer_t, int L, int K>
__global__ void forward_kernel_reg(const scalar_t *p_value, const int64_t *data_spatial_shapes,
                const int64_t *data_level_start_index, const scalar_t *p_offset,
                scalar_t *p_output, const int N, const int G, const int D,
                const int Q, const int block_multiplier) {

    // get (bi,qi,gi,di_s) indices based on block partitioning logic
    const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
    const int &bi = blockIdx.x * block_multiplier / Q;
    const int &di_s = threadIdx.x * d_stride;
    const int &gi = threadIdx.y;

    // allocate register level memory  
    // we need d_stride values for output over channels
    // we need L*K values for storing mask values for given (bi,qi,gi)
    // thus we incur more memory ops here than above but data lives in registers
    // vs shared memory and we don't need to synchronize threads
    // with larger L*K the memory footprint should increase and make shared memory
    // version more efficient, but for small L*K register version may be faster
    opmath_t p_out_shm[d_stride] = {0.};
    opmath_t p_mask_shm[L*K] = {0.};

    // move ptr of offset to (bi,qi,gi,L,offsets) where offsets
    // are (x,y,z,attn)
    const scalar_t *p_offset_ptr = p_offset + (((bi * Q + qi) * G + gi) * L) * K * 4;

    // set values in memory for mask/attention 
    for (int i=0; i < L*K; i++){
        p_mask_shm[i] = *(p_offset_ptr + L * K * 3 + i);
    }

    // calculate softmax over L and K
    opmath_t softmax_max = -1e100;
    opmath_t softmax_sum = 0.0;

    // get max
    for (int j = 0; j < L * K; j++) {
        softmax_max = max(softmax_max, p_mask_shm[j]);
    }

    // get sumexp
    for (int j = 0; j < L * K; j++) {
        opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
        p_mask_shm[j] = exp_results;
        softmax_sum += exp_results;
    }

    // normalize
    for (int j = 0; j < L * K; j++) {
        p_mask_shm[j] /= softmax_sum;
    }

    int offset_idx = 0;
    int mask_idx = 0;

    // stride in x=w over all channels and groups
    // for a given (qi,gi) representing some location (x,y,z)
    const int w_stride = G * D;
    // base_ptr to p_value start
    // for given group and channel set (gi,di_s)
    const int base_ptr = gi * D + di_s;

    for (int li = 0; li < L; li++) {
        const int spatial_d = data_spatial_shapes[li * 3];
        const int spatial_h = data_spatial_shapes[li * 3 + 1];
        const int spatial_w = data_spatial_shapes[li * 3 + 2];

        const int level_start_id = data_level_start_index[li];

        // move p_value to start of block (bi,qi)
        const scalar_t *p_value_ptr = p_value + (bi * N + level_start_id) * G * D;

        for (int ki = 0; ki < K; ki++) {
            const opmath_t loc_w = p_offset_ptr[offset_idx];
            const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
            const opmath_t loc_d = p_offset_ptr[offset_idx + 2];
            const opmath_t attn = p_mask_shm[mask_idx];

            const opmath_t d_im = loc_d * spatial_d - 0.5;
            const opmath_t h_im = loc_h * spatial_h - 0.5;
            const opmath_t w_im = loc_w * spatial_w - 0.5;
            if (d_im > -1 && h_im > -1 && w_im > -1 && d_im < spatial_d && h_im < spatial_h && w_im < spatial_w) {
                ms_deform_attn_im2col_trilinear<scalar_t, transfer_t, d_stride>(
                    p_out_shm, p_value_ptr, 
                    spatial_d, spatial_h, spatial_w, 
                    d_im, h_im, w_im, 
                    attn, w_stride, base_ptr);
            }
            offset_idx += 3;
            mask_idx += 1;
        }
    }

    // we save to out_idx at (bi,qi,gi,di_s)
    int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

    // alias cast to fp16 and cast in copy
    // then save in one go 
    scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
    #pragma unroll
    for (int ds = 0; ds < d_stride; ds++) {
        fp16_regs[ds] = p_out_shm[ds];
    }

    *(transfer_t *)(p_output + out_idx) = *(transfer_t *)(p_out_shm);
}


template <typename scalar_t, typename stride_type, int K, int d_stride>
void _flash_deformable_im2col_cuda(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 3
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 4
    scalar_t *output,                      // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, const int block_thread, 
    const bool _use_reg) {

    assert(D % d_stride == 0);

    // blockdim.z is given by total_nr_threads/ (blockdim.y * blockdim.x)
    // where blockdim.y = G and blockdim.x = D/d_stride
    // we want this to be some multiple of all queries (B*Q) 
    // i.e. since each query attends to G groups and D channels
    // we want blockdim.z to cover a slice B*Q/Blockdim.z = N queries
    // this will allow us to launch N blocks to cover all elements
    const int block_multiplier = block_thread / (D / d_stride) / G;
    assert((B*Q) % block_multiplier == 0);
    dim3 num_blocks(B*Q / block_multiplier);
    dim3 num_threads(D / d_stride, G, block_multiplier);

    // each block attends to N queries each with G groups and D channels
    // each thread attends to D/d_stride channels for a given group and query
    size_t shm_size = block_multiplier * G * L * K * sizeof(opmath_t);

    auto kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 1, K>;

    if (_use_reg){
        kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 1, K>;
    } else {
        kernel = forward_kernel<scalar_t, d_stride, stride_type, 1, K>;
    }

    switch (L) {
        case 1:
            if (_use_reg){
                kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 1, K>;
            } else {
                kernel = forward_kernel<scalar_t, d_stride, stride_type, 1, K>;
            }
            // kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 1, K>;
            break;
        case 2:
            if (_use_reg){
                kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 2, K>;
            } else {
                kernel = forward_kernel<scalar_t, d_stride, stride_type, 2, K>;
            }
            // kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 2, K>;
            break;
            case 3:
            if (_use_reg){
                kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 3, K>;
            } else {
                kernel = forward_kernel<scalar_t, d_stride, stride_type, 3, K>;
            }
            // kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 3, K>;
            break;
        case 4:
            if (_use_reg){
                kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 4, K>;
            } else {
                kernel = forward_kernel<scalar_t, d_stride, stride_type, 4, K>;
            }
            // kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 4, K>;
            break;
        case 5:
            if (_use_reg){
                kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 5, K>;
            } else {
                kernel = forward_kernel<scalar_t, d_stride, stride_type, 5, K>;
            }
            // kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 5, K>;
            break;
        default:
            printf("L=%ld\n", L);
            throw std::invalid_argument("invalid number of scales");
        }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                        shm_size);

    kernel<<<num_blocks, num_threads, shm_size, stream>>>(
        value, 
        data_spatial_shapes, data_level_start_index, 
        offset, output, 
        N, G, D, Q, block_multiplier);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in flash_deformable_im2col_cuda: %s\n",
                cudaGetErrorString(err));
        printf("launch arguments: gridDim=(%d, %d, %d), blockDim=(%d, %d, %d), "
                "shm_size=%d, Q=%d\n\n",
                num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x,
                num_threads.y, num_threads.z, shm_size, Q);
        AT_ASSERTM(false, "kernel launch error");
    }
}


template <typename scalar_t, int K>
void flash_deformable_im2col_cuda_inner(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 3
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 4
    scalar_t *output,                      // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, const int d_stride, 
    const int block_thread,
    const bool _use_reg) {
    assert(D % d_stride == 0);

    // stratify by scalar_t type
    if(sizeof(scalar_t) == 2) {
        // fp16 or bf16
        // adjust stride_dtype based on scalar_t type s.t.
        // sizeof(stride_dtype) = d_stride in scalar_t elements
        // uint = 4 bytes, uint2 = 8 bytes, uint4 = 16 bytes, ulonglong4 = 32 bytes
        switch(d_stride) {
            case 1:
                _flash_deformable_im2col_cuda<scalar_t, scalar_t, K, 1>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    output,                 // B, N, G, D
                    B, N, G, D, L, Q,
                    block_thread,
                    _use_reg);
                break;
            case 2:
                _flash_deformable_im2col_cuda<scalar_t, uint, K, 2>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    output,                 // B, N, G, D
                    B, N, G, D, L, Q,
                    block_thread,
                    _use_reg);
                break;
            case 4:
                _flash_deformable_im2col_cuda<scalar_t, uint2, K, 4>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    output,                 // B, N, G, D
                    B, N, G, D, L, Q,
                    block_thread,
                    _use_reg);
                break;
            case 8:
                _flash_deformable_im2col_cuda<scalar_t, uint4, K, 8>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    output,                 // B, N, G, D
                    B, N, G, D, L, Q,
                    block_thread,
                    _use_reg);
                break;
            case 16:
                _flash_deformable_im2col_cuda<scalar_t, ulonglong4, K, 16>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    output,                 // B, N, G, D
                    B, N, G, D, L, Q,
                    block_thread,
                    _use_reg);
                break;
        default:
            printf("d_stride > 16 for fp16 not supported ");
            throw std::invalid_argument("invalid d_stride");
        }
    } else {
        assert(sizeof(scalar_t) == 4);
        // fp32
        switch(d_stride) {
            case 1:
                _flash_deformable_im2col_cuda<scalar_t, scalar_t, K, 1>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    output,                 // B, N, G, D
                    B, N, G, D, L, Q,
                    block_thread,
                    _use_reg);
                break;
            case 2:
                _flash_deformable_im2col_cuda<scalar_t, uint2, K, 2>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    output,                 // B, N, G, D
                    B, N, G, D, L, Q,
                    block_thread,
                    _use_reg);
                break;
            case 4:
                _flash_deformable_im2col_cuda<scalar_t, uint4, K, 4>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    output,                 // B, N, G, D
                    B, N, G, D, L, Q,
                    block_thread,
                    _use_reg);
                break;
            case 8:
                _flash_deformable_im2col_cuda<scalar_t, ulonglong4, K, 8>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    output,                 // B, N, G, D
                    B, N, G, D, L, Q,
                    block_thread,
                    _use_reg);
                break;
            default:
                printf("d_stride > 8 for fp32 not supported");
                throw std::invalid_argument("invalid d_stride");
            }
        }
    }


template <typename scalar_t>
void flash_deformable_im2col_cuda(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 3
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 4
    scalar_t *output,                      // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, const int K, const int d_stride, 
    const int block_thread,
    const bool _use_reg) {
    // stratify by nr. of sampling points K
    switch (K) {
        case 4:
            flash_deformable_im2col_cuda_inner<scalar_t, 4>(
                stream,
                value,                  // B, N, G, D
                data_spatial_shapes,    // L * 3
                data_level_start_index, // L
                offset,                 // B, N, G, L, K, 4
                output,                 // B, N, G, D
                B, N, G, D, L, 
                Q, d_stride, 
                block_thread, 
                _use_reg);
            break;
        case 8:
            flash_deformable_im2col_cuda_inner<scalar_t, 8>(
                stream,
                value,                  // B, N, G, D
                data_spatial_shapes,    // L * 3
                data_level_start_index, // L
                offset,                 // B, N, G, L, K, 4
                output,                 // B, N, G, D
                B, N, G, D, L,
                Q, d_stride,
                block_thread, 
                _use_reg);
            break;
        default:
            // TODO: include more k values 
            printf("not supported for K not in [4, 8]");
            throw std::invalid_argument("invalid K");
    }
}