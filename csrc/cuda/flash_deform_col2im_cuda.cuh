//  Modified to 3D from:
//  https://github.com/OpenGVLab/DCNv4/blob/main/DCNv4_op/src/cuda/flash_deform_col2im_cuda.cuh
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


// shared memory version
template <typename scalar_t, int d_stride, typename transfer_t, int L, int K>
__global__ void backward_kernel(const scalar_t *p_value, 
                const int64_t *data_spatial_shapes,const int64_t *data_level_start_index, 
                const scalar_t *p_offset,
                const scalar_t *grad_output, 
                const int N, const int G, const int D, const int Q, 
                const int block_multiplier, 
                opmath_t *grad_im, 
                opmath_t *grad_offset) {
    // allocate dynamic shared memory
    extern __shared__ char _s[];

    // get (bi,qi,gi,di_s) indicies based on thread block logic
    const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
    const int &bi = blockIdx.x * block_multiplier / Q;
    const int &di_s = threadIdx.x * d_stride;
    const int &gi = threadIdx.y;

    // g_mask storage: block_multiplier*G*L*K
    // since each (bi,qi,gi) block attends to L*K locations
    // and each block has blockdim.Z * blockDim.y * blockDim.x threads
    // attending to block_multiplier=blockdim.z * G (bi,qi,gi) pairs
    opmath_t *cache_g_mask_before_softmax = (opmath_t *)(_s); 
    // grad_offsets storage: block_multiplier*G*D/d_stride*4
    // blockdim.z * blockDim.y * D/d_stride * 4 offset gradients stored
    // for each (bi,qi,gi) block since blockdim.z= block_multiplier,
    // blockdim.y=G, and blockdim.x=D/d_stride each with 4 offsets
    opmath_t *cache_grad_offset =
        (opmath_t *)(cache_g_mask_before_softmax + block_multiplier * G * L *K);
    // p_mask_shm: G*block_multiplier * L * K
    opmath_t *const p_mask_shm =
        ((opmath_t *)(cache_grad_offset + block_multiplier * G * D / d_stride * 4)) + (threadIdx.z * G + gi) * L * K; 

    // move offset ptr to (bi,qi,gi) block each with L levels, K points with 
    // 4 = (x,y,z,attention) offset values
    const scalar_t *p_offset_ptr = p_offset + (((bi * Q + qi) * G + gi) * L) * K * 4;
    
    // load mask attention values into shared memory, 
    // each thread loads total mask / num_thread of total
    // for given block 
    const int mask_length = L * K;
    const int num_thread = (D / d_stride);
    const int num_iter = mask_length / num_thread;
    const int remainder = mask_length - num_iter * num_thread;

    // move top_grad ptr to (bi,qi,gi,di_s) block
    const scalar_t *top_grad = grad_output + ((bi * Q + qi) * G + gi) * D + di_s;

    // move past offsets to load mask attention values
    // into shared memory for (bi,qi,gi) block
    // strided storage of mask attention values
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

    // calculate softmax over L and K
    if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
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

    // w_stride=x_stride number of elements between x values
    // each with G groups and D channels
    const int w_stride = G * D;
    // base_ptr is the offset from value_ptr for given gi, di_s
    const int base_ptr = gi * D + di_s;

    for (int li = 0; li < L; li++) {
        // get spatial shapes for current level
        const int spatial_d = data_spatial_shapes[li * 3];
        const int spatial_h = data_spatial_shapes[li * 3 + 1];
        const int spatial_w = data_spatial_shapes[li * 3 + 2];
        const int level_start_id = data_level_start_index[li];
        
        // load p_value_ptr and grad_im_ptr to (bi,qi) block
        // level_start_id is number of queries in previous levels
        const long long elem_offset =(static_cast<long long>(bi) * static_cast<long long>(N) + level_start_id) * static_cast<long long>(G) * D;
        const scalar_t* p_value_ptr = p_value + elem_offset;
        // const scalar_t *p_value_ptr = p_value + (bi * N + level_start_id) * G * D;

        opmath_t *grad_im_ptr = grad_im + (bi * N + level_start_id) * G * D;

        // move cache_grad_offset pointer for block to grad offset for given (qi, gi, di_s) 
        int cache_grad_off_idx =((threadIdx.z * G + threadIdx.y) * blockDim.x + threadIdx.x) * 4;
        for (int ki = 0; ki < K; ki++) {
            const opmath_t loc_w = p_offset_ptr[offset_idx];
            const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
            const opmath_t loc_d = p_offset_ptr[offset_idx + 2];
            const opmath_t attn = p_mask_shm[mask_idx];
            
            const opmath_t d_im = loc_d * spatial_d - 0.5;
            const opmath_t h_im = loc_h * spatial_h - 0.5;
            const opmath_t w_im = loc_w * spatial_w - 0.5;
        
            cache_grad_offset[cache_grad_off_idx] = 0;
            cache_grad_offset[cache_grad_off_idx + 1] = 0;
            cache_grad_offset[cache_grad_off_idx + 2] = 0;
            cache_grad_offset[cache_grad_off_idx + 3] = 0;

            if (d_im > -1 && h_im > -1 && w_im > -1 && d_im < spatial_d && h_im < spatial_h && w_im < spatial_w) {
                // trilinear interpolation with learned attention weights backward (see common.h)
                ms_deform_attn_col2im_trilinear<scalar_t, transfer_t, d_stride>(
                    p_value_ptr, 
                    spatial_d, spatial_h, spatial_w, 
                    d_im, h_im, w_im, 
                    attn, w_stride, base_ptr, 
                    spatial_d, spatial_h, spatial_w, 
                    top_grad, grad_im_ptr, cache_grad_offset + cache_grad_off_idx);
            }
            
            __syncthreads();
            
            // aggregate across different channel for offset
            if (threadIdx.x == 0) {
                // same offset as cache_grad_off_idx but with threadIdx.x=0
                int _didx = (threadIdx.z * G + threadIdx.y) * blockDim.x * 4;
                // update grad offset values for x,y,z and attention weight
                opmath_t _grad_w = cache_grad_offset[_didx];
                opmath_t _grad_h = cache_grad_offset[_didx + 1];
                opmath_t _grad_d = cache_grad_offset[_didx + 2];
                opmath_t _grad_a = cache_grad_offset[_didx + 3];
                
                // sum contribution for all threads in block 
                for (int c_id = 1; c_id < blockDim.x; ++c_id) {
                    _grad_w += cache_grad_offset[_didx + 4 * c_id];
                    _grad_h += cache_grad_offset[_didx + 4 * c_id + 1];
                    _grad_d += cache_grad_offset[_didx + 4 * c_id + 2];
                    _grad_a += cache_grad_offset[_didx + 4 * c_id + 3];
                }
                
                // store grad_w/h/d/a to final idx in grad offset matrix for idx (bi,qi,gi,li,ki)
                grad_offset[((bi * Q + qi) * G + gi) * L * K * 4 + li * K * 3 + ki * 3] = _grad_w;
                grad_offset[((bi * Q + qi) * G + gi) * L * K * 4 + li * K * 3 + ki * 3 + 1] = _grad_h;
                grad_offset[((bi * Q + qi) * G + gi) * L * K * 4 + li * K * 3 + ki * 3 + 2] = _grad_d;
                // store attention weight gradient to shared memory for block idx given by (gi, li, ki)
                cache_g_mask_before_softmax[((threadIdx.z * G + threadIdx.y) * L + li) * K + ki] = _grad_a;
            }

            __syncthreads();

            // update offset/mask increments
            offset_idx += 3;
            mask_idx += 1;
        }
    }
    
    // backward for softmax, see dcnv4 kernel for more details
    if (threadIdx.x == 0) {
        for (int i = 0; i < L * K; ++i) {
            opmath_t grad_i = 0.;
            const opmath_t *group_g_mask = cache_g_mask_before_softmax + (threadIdx.y + threadIdx.z * G) * L * K;
                for (int j = 0; j < L * K; ++j) {
                    if (i != j) {
                        grad_i -= group_g_mask[j] * p_mask_shm[i] * p_mask_shm[j];
                    } else {
                        grad_i += group_g_mask[i] * p_mask_shm[i] * (1 - p_mask_shm[i]);
                    }
            }
            // store attn weight gradient to final grad offset matrix
            grad_offset[((bi * Q + qi) * G + gi) * L * K * 4 + L * K * 3 + i] = grad_i;
        }
    }
    __syncthreads();
}


// register/warp version
template <typename scalar_t, int d_stride, typename transfer_t, int L, int K>
__global__ void backward_kernel_warp_primitive(const scalar_t *p_value, 
            const int64_t *data_spatial_shapes,
            const int64_t *data_level_start_index, 
            const scalar_t *p_offset,
            const scalar_t *grad_output, const int N, const int G,
            const int D, const int Q, 
            const int block_multiplier, 
            opmath_t *grad_im, 
            opmath_t *grad_offset) {
    
    extern __shared__ char _s[];

    // get (bi,qi,gi,di_s) indicies based on thread block logic
    const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
    const int &bi = blockIdx.x * block_multiplier / Q;
    const int &di_s = threadIdx.x * d_stride;
    const int &gi = threadIdx.y;

    // full thread id for 3d block 
    const int tid = (threadIdx.z * blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
    // lane_id (id within warp, warp size = 32)
    const int lane_id = tid % kWarpSize;
    // find the position of current group in the current warp
    // blockDim.x gives number of thread groups over channel dim 
    const int group_per_warp = kWarpSize / blockDim.x;
    const int group_in_warp_id = (threadIdx.z * G + threadIdx.y) % group_per_warp;
    const unsigned lane_mask = ((1 << blockDim.x) - 1) << (group_in_warp_id * blockDim.x);

    // g_mask storage: block_multiplier*G*L*K
    // since each (bi,qi,gi) block attends to L*K locations
    // and each block has blockdim.Z * blockDim.y * blockDim.x threads
    // attending to block_multiplier=blockdim.z * G (bi,qi,gi) pairs
    opmath_t *cache_g_mask_before_softmax = (opmath_t *)(_s); 

    // mask storage: G*block_multiplier * L * K
    // each block attends to block_multiplier=blockdim.z * G (qi,gi) pairs each with L*K values
    // for given block is given by threadIdx.z and gi=threadIdx.y
    opmath_t *const p_mask_shm = ((opmath_t *)(cache_g_mask_before_softmax + block_multiplier * G * L * K)) + (threadIdx.z * G + gi) * L * K; 

    // offset for (bi,qi,gi) block each with L levels, K points with 4 values (x,y,z,attention)
    const scalar_t *p_offset_ptr = p_offset + (((bi * Q + qi) * G + gi) * L) * K * 4;
    
    // load mask attention values into shared memory,
    // each thread loads total mask / num_thread of total
    // for given block with strided storage
    const int mask_length = L * K;
    const int num_thread = (D / d_stride);
    const int num_iter = mask_length / num_thread;
    const int remainder = mask_length - num_iter * num_thread;

    // move top_grad ptr to (bi,qi,gi,di_s) block
    const scalar_t *top_grad = grad_output + ((bi * Q + qi) * G + gi) * D + di_s;

    for (int i = 0; i < num_iter; i++) {
    *(p_mask_shm + num_thread * i + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + L * K * 3 + num_thread * i + threadIdx.x);
    }

    if (remainder > 0 && threadIdx.x < remainder) {
    *(p_mask_shm + num_thread * num_iter + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + L * K * 3 + num_thread * num_iter + threadIdx.x);
    }

    __syncthreads();

    // Calculate softmax over L and K
    if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
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

    // w_stride=x number of elements between x values
    const int w_stride = G * D;
    // base_ptr is the offset from value_ptr for given gi, di_s
    const int base_ptr = gi * D + di_s;

    for (int li = 0; li < L; li++) {
        // get spatial shapes for current level
        const int spatial_d = data_spatial_shapes[li * 3];
        const int spatial_h = data_spatial_shapes[li * 3 + 1];
        const int spatial_w = data_spatial_shapes[li * 3 + 2];
        const int level_start_id = data_level_start_index[li];

        // load p_value_ptr and grad_im_ptr to (bi,qi) block
        const scalar_t *p_value_ptr = p_value + (bi * N + level_start_id) * G * D;
        opmath_t *grad_im_ptr = grad_im + (bi * N + level_start_id) * G * D;

        // register level offset gradients array for h,w,d,attn_value
        opmath_t reg_grad_offset[4] = {0.};
        
        for (int ki = 0; ki < K; ki++) {
            const opmath_t loc_w = p_offset_ptr[offset_idx];
            const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
            const opmath_t loc_d = p_offset_ptr[offset_idx + 2];
            const opmath_t attn = p_mask_shm[mask_idx];
            
            const opmath_t d_im = loc_d * spatial_d - 0.5;
            const opmath_t h_im = loc_h * spatial_h - 0.5;
            const opmath_t w_im = loc_w * spatial_w - 0.5;
            
            reg_grad_offset[0] = 0;
            reg_grad_offset[1] = 0;
            reg_grad_offset[2] = 0;
            reg_grad_offset[3] = 0;

            if (d_im > -1 && h_im > -1 && w_im > -1 && d_im < spatial_d && h_im < spatial_h && w_im < spatial_w) {
                ms_deform_attn_col2im_trilinear<scalar_t, transfer_t, d_stride>(
                    p_value_ptr, 
                    spatial_d, spatial_h, spatial_w, 
                    d_im, h_im, w_im, 
                    attn, w_stride, base_ptr, 
                    spatial_d, spatial_h, spatial_w, 
                    top_grad, grad_im_ptr, reg_grad_offset);
            }
            
            __syncthreads();

            // warp shuffle reduction, in each iteration warps with idx1=idx2*2 will
            // accumulate the values of idx1 into idx2 until all values are accumulated
            // in threadIdx.x=0, offset >>= 1 => 2x offset reduction
            for (uint32_t offset = blockDim.x>>1; offset > 0; offset >>= 1){
                reg_grad_offset[0] += __shfl_down_sync(lane_mask, reg_grad_offset[0], offset);
                reg_grad_offset[1] += __shfl_down_sync(lane_mask, reg_grad_offset[1], offset);
                reg_grad_offset[2] += __shfl_down_sync(lane_mask, reg_grad_offset[2], offset);
                reg_grad_offset[3] += __shfl_down_sync(lane_mask, reg_grad_offset[3], offset);
            }

            if (threadIdx.x == 0) {
                // store grad_w/h/d/a to final idx in grad offset matrix for idx (bi,qi,gi,li,ki)
                grad_offset[((bi * Q + qi) * G + gi) * L * K * 4 + li * K * 3 + ki * 3] = reg_grad_offset[0];
                grad_offset[((bi * Q + qi) * G + gi) * L * K * 4 + li * K * 3 + ki * 3 + 1] = reg_grad_offset[1];
                grad_offset[((bi * Q + qi) * G + gi) * L * K * 4 + li * K * 3 + ki * 3 + 2] = reg_grad_offset[2];
                cache_g_mask_before_softmax[((threadIdx.y + threadIdx.z * G) * L + li) * K + ki] = reg_grad_offset[3];
            }
        
            __syncthreads();

            offset_idx += 3;
            mask_idx += 1;
        }
    }

    // backward for softmax, see dcnv4 kernel for more details
    if (threadIdx.x == 0) {
        for (int i = 0; i < L * K; ++i) {
            opmath_t grad_i = 0.;
            const opmath_t *group_g_mask = cache_g_mask_before_softmax + (threadIdx.y + threadIdx.z * G) * L * K;
            for (int j = 0; j < L * K; ++j) {
                if (i != j) {
                    grad_i -= group_g_mask[j] * p_mask_shm[i] * p_mask_shm[j];
                } else {
                    grad_i += group_g_mask[i] * p_mask_shm[i] * (1 - p_mask_shm[i]);
                }
            }
            grad_offset[((bi * Q + qi) * G + gi) * L * K * 4 + L * K * 3 + i] = grad_i;
        }
    }
    __syncthreads();
}


template <typename scalar_t, typename stride_type, int K, int d_stride>
void _flash_deformable_col2im_cuda(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 3
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 4
    const scalar_t *grad_output,           // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, opmath_t *grad_im, opmath_t *grad_offset,
    const int block_thread) 
{
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

    int shm_size;

    // check_backward_warpp checks if num. threads per 
    // channel block D is less than warpsize of 32 and that
    // that the num. threads is divisible by warpsize
    // this lets us use warp primitives since a given warp
    // can perform computations on all channels in a given block 
    if(check_backward_warpp(d_stride, D)){
        // shared memory size for grad mask before softmax and mask
        shm_size =
            sizeof(opmath_t) * (block_multiplier * G * L * K) +
            sizeof(opmath_t) * (G * block_multiplier * L * K);
    }
    else{
        // shared memory size for g mask before softmax mask
        // and offset gradients for a given block
        shm_size =
            sizeof(opmath_t) * (block_multiplier * G * L * K) +
            sizeof(opmath_t) * (G * block_multiplier * L * K) + 
            sizeof(opmath_t) * (G * block_multiplier * D / d_stride * 4);
    }

    auto kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 1, K>;

    switch (L) {
        case 1:
            if(check_backward_warpp(d_stride, D)){
                kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 1, K>;
            } else {
                kernel = backward_kernel<scalar_t, d_stride, stride_type, 1, K>;
            }
            break;
        case 2:
            if(check_backward_warpp(d_stride, D)){
                kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 2, K>;
            } else {
                kernel = backward_kernel<scalar_t, d_stride, stride_type, 2, K>;
            }
            break;
        case 3:
            if(check_backward_warpp(d_stride, D)){
                kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 3, K>;
            } else {
                kernel = backward_kernel<scalar_t, d_stride, stride_type, 3, K>;
            }
            break;
        case 4:
            if(check_backward_warpp(d_stride, D)){
                kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 4, K>;
            } else {
                kernel = backward_kernel<scalar_t, d_stride, stride_type, 4, K>;
            }
            break;
        case 5:
            if(check_backward_warpp(d_stride, D)){
                kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 5, K>;
            } else {
                kernel = backward_kernel<scalar_t, d_stride, stride_type, 5, K>;
            }
            break;
        default:
            printf("L=%ld\n", L);
            throw std::invalid_argument("invalid number of scales");
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                        shm_size);

    kernel<<<num_blocks, num_threads, shm_size, stream>>>(
        value, data_spatial_shapes, data_level_start_index, 
        offset, grad_output,
        N, G, D, Q, block_multiplier, 
        grad_im, grad_offset);

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
void flash_deformable_col2im_cuda_inner(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 3
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 4
    const scalar_t *grad_output,           // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, opmath_t *grad_im, opmath_t *grad_offset, 
    const int d_stride, const int block_thread) {
    assert(D % d_stride == 0);

    // stratify by scalar_t type
    if(sizeof(scalar_t) == 2) {
        // fp16 or bf16
        // adjust stride_dtype based on scalar_t type s.t.
        // sizeof(stride_dtype) = d_stride in scalar_t elements
        // uint = 4 bytes, uint2 = 8 bytes, uint4 = 16 bytes, ulonglong4 = 32 bytes
        switch(d_stride) {
            case 1:
                _flash_deformable_col2im_cuda<scalar_t, scalar_t, K, 1>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    grad_output,            // B, N, G, D
                    B, N, G, D, L, Q, 
                    grad_im, grad_offset,
                    block_thread);
                break;
            case 2:
                _flash_deformable_col2im_cuda<scalar_t, uint, K, 2>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    grad_output,            // B, N, G, D
                    B, N, G, D, L, Q, 
                    grad_im, grad_offset,
                    block_thread);
                break;
            case 4:
                _flash_deformable_col2im_cuda<scalar_t, uint2, K, 4>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    grad_output,            // B, N, G, D
                    B, N, G, D, L, Q, grad_im, grad_offset,
                    block_thread);
                break;
            case 8:
                _flash_deformable_col2im_cuda<scalar_t, uint4, K, 8>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    grad_output,            // B, N, G, D
                    B, N, G, D, L, Q, 
                    grad_im, grad_offset,
                    block_thread);
                break;
            case 16:
                _flash_deformable_col2im_cuda<scalar_t, ulonglong4, K, 16>(
                    stream,
                    value,                  // B, N, G, D
                    data_spatial_shapes,    // L * 3
                    data_level_start_index, // L
                    offset,                 // B, N, G, L, K, 4
                    grad_output,            // B, N, G, D
                    B, N, G, D, L, Q, 
                    grad_im, grad_offset,
                    block_thread);
                break;
            default:
                printf("not supported for d_stride > 16 for fp16");
                throw std::invalid_argument("invalid d_stride");
            }
        } else {
            // fp32
            assert(sizeof(scalar_t) == 4);
            switch(d_stride) {
            case 1:  
                _flash_deformable_col2im_cuda<scalar_t, scalar_t, K, 1>(
                stream,
                value,                  // B, N, G, D
                data_spatial_shapes,    // L * 3
                data_level_start_index, // L
                offset,                 // B, N, G, L, K, 4
                grad_output,            // B, N, G, D
                B, N, G, D, L, Q, 
                grad_im, grad_offset,
                block_thread);
                break;
            case 2:  
                _flash_deformable_col2im_cuda<scalar_t, uint2, K, 2>(
                stream,
                value,                  // B, N, G, D
                data_spatial_shapes,    // L * 3
                data_level_start_index, // L
                offset,                 // B, N, G, L, K, 4
                grad_output,            // B, N, G, D
                B, N, G, D, L, Q, 
                grad_im, grad_offset,
                block_thread);
                break;
            case 4:  
                _flash_deformable_col2im_cuda<scalar_t, uint4, K, 4>(
                stream,
                value,                  // B, N, G, D
                data_spatial_shapes,    // L * 3
                data_level_start_index, // L
                offset,                 // B, N, G, L, K, 4
                grad_output,            // B, N, G, D
                B, N, G, D, L, Q, 
                grad_im, grad_offset,
                block_thread);
                break;
            case 8:  
                _flash_deformable_col2im_cuda<scalar_t, ulonglong4, K, 8>(
                stream,
                value,                  // B, N, G, D
                data_spatial_shapes,    // L * 3
                data_level_start_index, // L
                offset,                 // B, N, G, L, K, 4
                grad_output,            // B, N, G, D
                B, N, G, D, L, Q, 
                grad_im, grad_offset,
                block_thread);
                break;
            default:
                printf("not supported for d_stride > 8 for fp32");
                throw std::invalid_argument("invalid d_stride");
            }
        }
    }


template <typename scalar_t>
void flash_deformable_col2im_cuda(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 3
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 4
    const scalar_t *grad_output,           // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, const int K, 
    opmath_t *grad_im, opmath_t *grad_offset,
    const int d_stride, const int block_thread) {
    // stratify by nr. of sampling points K
    switch (K) {
        // TODO: support more K > 8
        case 4:
            flash_deformable_col2im_cuda_inner<scalar_t, 4>(
                stream,
                value,                  // B, N, G, D
                data_spatial_shapes,    // L * 3
                data_level_start_index, // L
                offset,                 // B, N, G, L, K, 4
                grad_output,            // B, N, G, D
                B, N, G, D, L, Q,
                grad_im, grad_offset,
                d_stride, block_thread);
            break;
        case 8:
            flash_deformable_col2im_cuda_inner<scalar_t, 8>(
                stream,
                value,                  // B, N, G, D
                data_spatial_shapes,    // L * 3
                data_level_start_index, // L
                offset,                 // B, N, G, L, K, 4
                grad_output,            // B, N, G, D
                B, N, G, D, L, Q, 
                grad_im, grad_offset,
                d_stride, block_thread);
            break;
        default:
            printf("not supported for K not in [4, 8]");
            throw std::invalid_argument("invalid K");
        }
    }