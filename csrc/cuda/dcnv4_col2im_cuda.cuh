// Modified to 3D from:
//  https://github.com/OpenGVLab/DCNv4/blob/main/DCNv4_op/src/cuda/flash_deform_col2im_cuda.cuh
//
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


template <typename scalar_t, int d_stride, typename transfer_t, int L, int K, bool softmax>
__global__ void backward_kernel_dcn(
    const scalar_t *value_ptr, const scalar_t *offset_ptr, const scalar_t *grad_output, 
    const int G, const int D, const int Q, 
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int depth_in, const int height_in, const int width_in,
    const int depth_out, const int height_out, const int width_out,
    const opmath_t offset_scale, const int remove_center, 
    const int block_multiplier, 
    opmath_t *grad_im, opmath_t *grad_offset,
    const int padded_offset_dim) {     

    // fwd: o_{b,q,g,d} = SUM_{i=0..D} a_{b,q,g,i} * trilinear(x_{b,q,g,i}, y_{b,q,g,i}, z_{b,q,g,i})[d]
    // where each thread handles d in {ds_i, ds_i+1, ..., ds_i+N}. 
    // Thus, bkwd:
    // dL/dx_{b,q,g,i} = SUM_{d=0..D} dL/do_{b,q,g,d} * do_{b,q,g,d}/dx_{b,q,g,i} 
    //                 = SUM_{d=0..D} g_{b,q,g,d} * do_{b,q,g,d}/dx_{b,q,g,i}
    // and similarly for y and z. 

    // allocate dynamic shared memory 
    extern __shared__ char _s[];

    // get (bi,qi,gi,di_s) block from threadIdx values 
    // for the given thread
    const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
    const int &bi = blockIdx.x * block_multiplier / Q;
    const int &di_s = threadIdx.x * d_stride;
    const int &gi = threadIdx.y;
    const int li = 0;

    // shared memory layout
    // Region 1: mask‐before‐softmax: 
    // size = block_multiplier*G*K
    // holds: g_i * a_i for each of the M=block_multiplier queries x G heads x K sampling points
    // Region 2: per-thread offset grads
    // size = block_multiplier*G*blockDim.x*4
    // holds each thread's 4 partials (∂x,∂y,∂z,∂a) for every (q,g,ds_i)
    // Region 3: mask values
    // size = block_multiplier*G*K
    // holds: raw or softmaxed a_i values for each (q,g,k)

    // cast shared memory to opmath_t, start of shared memory buffer
    // region 1 = attention weights gradients 
    opmath_t *const cache_g_mask_before_softmax = (opmath_t *)(_s); 
    // region 2 = offset gradients 
    opmath_t *const cache_grad_offset = 
        (opmath_t *)(cache_g_mask_before_softmax + block_multiplier * G * K);
    // region 3 = mask values
    opmath_t *const p_mask_shm = 
    (opmath_t *)(cache_grad_offset + block_multiplier * G * blockDim.x * 4) + 
    (threadIdx.z * G + gi) * K;
    
    // move offset ptr to start of (bi,qi,gi) block 
    const scalar_t *p_offset_ptr = offset_ptr + (bi*Q + qi)*padded_offset_dim + gi*K*4;
    
    // for a given group we need to load K mask attention values 
    // we have num_thread thread that all use same mask/attention values
    // so we make each tread load K/num_thread values and store them in
    // shared memory 
    const int mask_length = K;
    const int num_thread = (D / d_stride);
    const int num_iter = mask_length / num_thread;
    const int remainder = mask_length - num_iter * num_thread;

    __syncthreads();

    // get attention weights for a given (bi,qi,gi) block
    // and store them in shared memory
    for (int i = 0; i < num_iter; i++){
        *(p_mask_shm + num_thread * i + threadIdx.x) = 
            *(scalar_t *)(p_offset_ptr + K * 3 + num_thread * i + threadIdx.x);
    }
    if (remainder > 0 && threadIdx.x < remainder) {
        *(p_mask_shm + num_thread * num_iter + threadIdx.x) = 
            *(scalar_t *)(p_offset_ptr + K * 3 + num_thread * num_iter + threadIdx.x);
    }

    if(softmax){
        // synchronize threads to ensure all threads
        // have loaded their mask values into shared memory
        __syncthreads();

        if (threadIdx.x == 0){
            opmath_t softmax_max = -1e100;
            opmath_t softmax_sum = 0.0;
            
            // get max
            for(int i = 0; i < K; i++){
                softmax_max = max(softmax_max, p_mask_shm[i]);
            }

            // get sumexp
            for (int i = 0; i < K; i++){
                opmath_t exp_results = exp(p_mask_shm[i] - softmax_max);
                p_mask_shm[i] = exp_results;
                softmax_sum += exp_results;
            }

            // normalize
            for (int i = 0; i < K; i++){
                p_mask_shm[i] /= softmax_sum;
            }
        }
    __syncthreads();
    }

    int offset_idx = 0;
    int mask_idx = 0;

    // value_ptr: [B,Q,G,D] where Q = D*H*W
    // stride (d,h,w) -> (d,h,w+1) is G*D
    const int w_stride = G * D;
    // used in trilinear interpolation
    const int base_ptr = gi * D + di_s;

    // value pointer (b,q=(D*H*W),g,d) 
    const scalar_t *p_value_ptr = 
        value_ptr + (bi * (depth_in * height_in * width_in)) * (G * D);

    // grad_im pointer (b,q=(D*H*W),g,d) 
    opmath_t *grad_im_ptr = 
        grad_im + (bi * (depth_in * height_in * width_in)) * (G * D);

    // get current (x,y,z) pos. given qi 
    int x = qi % width_out;
    int yi = qi / width_out;
    int y = yi % height_out;
    int z = yi / height_out;
    
    // (dilation_w * (kernel_w-1))/2 is half-span of kernel_w 
    // we then shift back padding pad_w and find correct starting
    // width position for the kernel given the stride
    const opmath_t p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + x*stride_w; 
    const opmath_t p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + y*stride_h;
    const opmath_t p0_d = ((dilation_d * (kernel_d - 1)) >> 1) - pad_d + z*stride_d;
    
    // to skip subtracting that same half‐span
    // i.e. to avoid: 
    // p0_w + ((i * dilation_w + Δ)>>1)*offset_scale - ((dilation_w * (kernel_w-1))>>1)*offset_scale
    // we pre-subtract it once and in the inner loop we just do:
    // w_im = p0_w_ + (i*dilation_w + Δx)*offset_scale (and similar for y,z)
    const opmath_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
    const opmath_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;
    const opmath_t p0_d_ = p0_d - ((dilation_d * (kernel_d - 1)) >> 1) * offset_scale;

    const int center_d = kernel_d / 2;
    const int center_h = kernel_h / 2;
    const int center_w = kernel_w / 2;

    // grad ptrs

    // (bi,qi,gi,K*4) offset for output gradients
    grad_offset += (bi * Q + qi) * padded_offset_dim + gi * K * 4;

    // grad offset for mask for output gradients
    opmath_t *grad_offset_softmax = grad_offset + K * 3;

    // position in shared memory for offset values
    int cache_grad_off_idx = 
        ((threadIdx.z * G + threadIdx.y) * blockDim.x + threadIdx.x) * 4;

    // (bi,qi,gi,di_s) top grad
    const scalar_t *top_grad = grad_output + ((bi * Q + qi)*G + gi)*D + di_s;    
    
    for (int i=0; i < kernel_w; i++){
        for (int j=0; j < kernel_h; j++){
            for (int k=0; k < kernel_d; ++k){
                if (i != center_w || j!= center_h || k != center_d || !remove_center){
                    // get offset values for (x,y,z) position and mask attention value
                    const opmath_t w_im = p0_w_ + (i * dilation_w + (opmath_t)p_offset_ptr[offset_idx])*offset_scale; 
                    const opmath_t h_im = p0_h_ + (j * dilation_h + (opmath_t)p_offset_ptr[offset_idx + 1])*offset_scale;
                    const opmath_t d_im = p0_d_ + (k * dilation_d + (opmath_t)p_offset_ptr[offset_idx + 2])*offset_scale;
                    const opmath_t attn = p_mask_shm[mask_idx];

                    cache_grad_offset[cache_grad_off_idx] = 0;
                    cache_grad_offset[cache_grad_off_idx + 1] = 0;
                    cache_grad_offset[cache_grad_off_idx + 2] = 0;
                    cache_grad_offset[cache_grad_off_idx + 3] = 0;
                    
                    if (d_im > -1 && h_im > -1 && w_im > -1 && d_im < depth_in && h_im < height_in && w_im < width_in){
                        // perform trilinear interpolation backwards op. (see common.h) 
                        ms_deform_attn_col2im_trilinear<scalar_t, transfer_t, d_stride>(
                            p_value_ptr, 
                            depth_in, height_in, width_in,
                            d_im, h_im, w_im, 
                            attn, w_stride, 
                            base_ptr, offset_scale, offset_scale, offset_scale,
                            top_grad, grad_im_ptr, cache_grad_offset + cache_grad_off_idx 
                        );
                    }
                    
                    __syncthreads();

                    if (threadIdx.x == 0){
                        // for (q,g) within block we have 4 floats per channel slice
                        // similar as cache_grad_off_idx except ofc threadIdx.x is 0
                        int _didx = (threadIdx.z * G + threadIdx.y) * blockDim.x * 4;
                        // thread 0 grad contributions 
                        opmath_t _grad_w = cache_grad_offset[_didx];
                        opmath_t _grad_h = cache_grad_offset[_didx + 1];
                        opmath_t _grad_d = cache_grad_offset[_didx + 2];
                        opmath_t _grad_a = cache_grad_offset[_didx + 3];
                        
                        // for each thread in block, we write gradients for offset 
                        // and attention mask
                        for (int c_id = 1; c_id < blockDim.x; ++c_id){
                            _grad_w += cache_grad_offset[_didx + 4 * c_id];
                            _grad_h += cache_grad_offset[_didx + 4 * c_id + 1];
                            _grad_d += cache_grad_offset[_didx + 4 * c_id + 2];
                            _grad_a += cache_grad_offset[_didx + 4 * c_id + 3];
                        }

                        // write gradients to global memory
                        *(grad_offset) = _grad_w;
                        *(grad_offset + 1) = _grad_h;
                        *(grad_offset + 2) = _grad_d;
                        
                        // if softmax is not used then, then your attention weight is just
                        // ai=zi => dL/dzi = dL/dai = gi
                        // else if ai = e^zi / sum(e^zj) and so
                        // dL/dz_i = sum_{j=0..K-1} (dL/da_j) * (da_j/dz_i)
                        // da_j/dz_i = {  a_i*(1 - a_i)    if j == i
                        //             { -a_j * a_i       if j != i 
                        // => sum_j g_j * (da_j/dz_i) = a_i * (g_i - sum_j g_j * a_j).
                        // hence:
                        if (softmax){
                            // here we store g_i * a_i in shared memory for later 
                            // recall size of cache_g_mask_before_softmax = block_multiplier * G * K
                            // threadIdx.y designates which group we are in and threadIdx.z which
                            // query slice we are in and we have K points per entry 
                            cache_g_mask_before_softmax[(threadIdx.z * G + threadIdx.y)* K + mask_idx] = _grad_a * attn;
                        } else{
                            grad_offset_softmax[mask_idx] = _grad_a;
                        }
                    }
                    __syncthreads();

                    offset_idx += 3;
                    mask_idx += 1;
                    grad_offset += 3;
                }
            }
        }
    }

    // backward for softmax
    if(softmax){
        if(threadIdx.x == 0){
            // acess stored g_i * a_i values in shared memory (see above)
            const opmath_t* group_g_mask = cache_g_mask_before_softmax + (threadIdx.z*G + threadIdx.y)*K;
            #pragma unroll
            for (int i = 0; i < K; ++i){
                opmath_t sum = 0.;
                for (int j = 0; j < K; ++j){
                    // compute sum_j g_j * a_j
                    sum += group_g_mask[j];
                }
                // performs computation: a_i*g_i - g_i * SUM_j g_j * a_j
                // i.e. backward for attention weights with softmax
                *(grad_offset_softmax) = group_g_mask[i] - p_mask_shm[i] * sum;
                grad_offset_softmax += 1;
            }
        }
        __syncthreads();
    }
}


template <typename scalar_t, int d_stride, typename transfer_t, int L, int K, bool softmax>
__global__ void backward_kernel_dcn_warp_primitive(
    const scalar_t *p_value, const scalar_t *p_offset,
    const scalar_t *grad_output, const int G, const int D, const int Q,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w, 
    const int pad_d, const int pad_h, const int pad_w, 
    const int dilation_d, const int dilation_h, const int dilation_w, 
    const int depth_in, const int height_in, const int width_in,
    const int depth_out, const int height_out, const int width_out, 
    const opmath_t offset_scale, const int remove_center, const int block_multiplier, 
    opmath_t *grad_im, opmath_t *grad_offset, const int padded_offset_dim) {

    // allocate shared memory
    extern __shared__ char _s[];

    // get (bi,qi,gi,di_s) indices
    const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
    const int &bi = blockIdx.x * block_multiplier / Q;
    const int &di_s = threadIdx.x * d_stride;
    const int &gi = threadIdx.y;
    constexpr int li = 0;

    // full thread id 
    const int tid = (threadIdx.z * blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;

    // lane_id (id within warp, warp size = 32)
    const int lane_id = tid % kWarpSize;

    // find the position of current group in the current warp
    // blockDim.x gives number of thread groups over channel dim 
    const int group_per_warp = kWarpSize / blockDim.x;
    const int group_in_warp_id = (threadIdx.z * G + threadIdx.y) % group_per_warp;
    const unsigned lane_mask = ((1 << blockDim.x) - 1) << (group_in_warp_id * blockDim.x);

    // we need BlockDim.Z * G * K shared memory space for masks for a given block
    // move ptr to start of (thread_idx_in_block, group_head_id) for this mem. region
    opmath_t *const p_mask_shm = (opmath_t *)(_s) + (threadIdx.z * G + gi) * K;
    // move ptr to start of (thread_idx_in_block, group_head_id) past dynamic shm allocated for 
    // mask to cache mask gradient before softmax 
    opmath_t *cache_g_mask_before_softmax = (opmath_t *)((opmath_t *)(_s) + block_multiplier * G * K) +
                                            (threadIdx.z*G+gi)*K; 
  
    // move offset ptr to start of (bi,qi,gi) block 
    const scalar_t *p_offset_ptr = p_offset + (bi*Q + qi)*padded_offset_dim + gi*K*4;

    // each thread loads K/total_num_threads_for_each_channel_slice 
    // number of points 
    const int mask_length = K;
    const int num_thread = (D / d_stride);
    const int num_iter = mask_length / num_thread;
    const int remainder = mask_length - num_iter * num_thread;

    __syncthreads();

    // put mask values in shared memory for given (bi,qi,gi) block
    for (int i = 0; i < num_iter; i++) {
        *(p_mask_shm + num_thread * i + threadIdx.x) =
            *(scalar_t *)(p_offset_ptr + K * 3 + num_thread * i + threadIdx.x);
    }
    if (remainder > 0 && threadIdx.x < remainder) {
        *(p_mask_shm + num_thread * num_iter + threadIdx.x) = *(
            scalar_t *)(p_offset_ptr + K * 3 + num_thread * num_iter + threadIdx.x);
    }

    if (softmax) {
        // synchronize threads to ensure all threads
        // have loaded their mask values into shared memory
        __syncthreads();
        
        // calculate softmax over L and K
        if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
            opmath_t softmax_max = -1e100;
            opmath_t softmax_sum = 0.0;

            // get max
            for (int j = 0; j < K; j++) {
                softmax_max = max(softmax_max, p_mask_shm[j]);
            }

            // get sumexp
            for (int j = 0; j < K; j++) {
                opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
                p_mask_shm[j] = exp_results;
                softmax_sum += exp_results;
            }

            // normalize
            for (int j = 0; j < K; j++) {
                p_mask_shm[j] /= softmax_sum;
            }
        }
    __syncthreads();
  }

    int offset_idx = 0;
    int mask_idx = 0;

    // value_ptr: [B,Q,G,D] where Q = D*H*W
    // stride (d,h,w) -> (d,h,w+1) is G*D
    const int w_stride = G * D;
    const int base_ptr = gi * D + di_s;

    const scalar_t *p_value_ptr = p_value + (bi * (depth_in * height_in * width_in)) * (G * D);

    // get current (x,y,z) pos. given qi 
    int x = qi % width_out;
    int yi = qi / width_out;
    int y = yi % height_out;
    int z = yi / height_out;

    // move to start of kernel in x,y,z
    const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + x*stride_w;
    const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + y*stride_h;
    const int p0_d = ((dilation_d * (kernel_d - 1)) >> 1) - pad_d + z*stride_d;

    // pre-subtract half-span of kernel (in offset scale space)
    const opmath_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
    const opmath_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;
    const opmath_t p0_d_ = p0_d - ((dilation_d * (kernel_d - 1)) >> 1) * offset_scale;

    // get centre of kernel in x,y,z
    const int center_d = kernel_d / 2;
    const int center_h = kernel_h / 2;
    const int center_w = kernel_w / 2;

    // grad ptrs

    // grad output (bi,qi,gi,di_s)
    const scalar_t *top_grad = grad_output + ((bi * Q + qi) * G + gi) * D + di_s;

    // grad_im pointer (b,q=(D*H*W),g,d)
    opmath_t *grad_im_ptr = grad_im + (bi * (depth_in * height_in * width_in)) * (G * D);

    // grad_offset pointer (b,q=(D*H*W),g,K*4)
    grad_offset += (bi * Q + qi)*padded_offset_dim + gi*K*4;

    // grad_offset_sofmtax ptr (bi,qi,gi) past offsets
    opmath_t *grad_offset_softmax = grad_offset + K * 3;

    // allocate registers to store gradients for offsets 
    opmath_t reg_grad_offset[4] = {0.};

    for (int i = 0; i < kernel_w; ++i) {
        for (int j = 0; j < kernel_h; ++j) {
            for (int k = 0; k < kernel_d; ++k){
                if (i != center_w || j != center_h || k!= center_d || !remove_center) {
                    // get offset values for (x,y,z) position and mask attention value
                    const opmath_t w_im = p0_w_ + (i * dilation_w + (opmath_t)p_offset_ptr[offset_idx]) *offset_scale;
                    const opmath_t h_im = p0_h_ + (j * dilation_h + (opmath_t)p_offset_ptr[offset_idx + 1]) *offset_scale;
                    const opmath_t d_im = p0_d_ + (k * dilation_d + (opmath_t)p_offset_ptr[offset_idx + 2]) * offset_scale;
                    const opmath_t attn = p_mask_shm[mask_idx];
                
                    // zero out the register cache for gradients
                    reg_grad_offset[0] = 0;
                    reg_grad_offset[1] = 0;
                    reg_grad_offset[2] = 0;
                    reg_grad_offset[3] = 0;

                    if (d_im > -1 && h_im > -1 && w_im > -1 && d_im < depth_in && h_im < height_in && w_im < width_in) {
                        // perform trilinear interpolation backwards op. (see common.h)
                        ms_deform_attn_col2im_trilinear<scalar_t, transfer_t, d_stride>(
                            p_value_ptr, 
                            depth_in, height_in, width_in, 
                            d_im, h_im, w_im, 
                            attn, w_stride,
                            base_ptr, 
                            offset_scale, offset_scale, offset_scale,
                            top_grad, grad_im_ptr,
                            reg_grad_offset);
                    }

                    // aggregated across different channel using warp shuffle
                    // reduction that ends up accumulating thread gradients on thread 0
                    // start offset at nr of threads groups for channels / 2
                    // then reduce by half until we reach 1
                    // each time thread gradients that are offset idx apart will be added
                    for (uint32_t offset = blockDim.x>>1; offset > 0; offset >>= 1){
                        reg_grad_offset[0] += __shfl_down_sync(lane_mask, reg_grad_offset[0], offset);
                        reg_grad_offset[1] += __shfl_down_sync(lane_mask, reg_grad_offset[1], offset);
                        reg_grad_offset[2] += __shfl_down_sync(lane_mask, reg_grad_offset[2], offset);
                        reg_grad_offset[3] += __shfl_down_sync(lane_mask, reg_grad_offset[3], offset);
                    }

                    if (threadIdx.x == 0) { 
                        // store shared memory cache for gradient offsets and mask values
                        // to global memory (B x D x H x W x G x L x K x 3)
                        *(grad_offset) = reg_grad_offset[0];     
                        *(grad_offset + 1) = reg_grad_offset[1]; 
                        *(grad_offset + 2) = reg_grad_offset[2]; 
                        if (softmax) {
                            // here we store g_i * a_i in shared memory for later 
                            // if softmax is used
                            cache_g_mask_before_softmax[mask_idx] = reg_grad_offset[3] * attn;
                        }
                        else{
                            grad_offset_softmax[mask_idx] = reg_grad_offset[3];
                        }
                    }
                    // move to next offset and mask value
                    offset_idx += 3;
                    mask_idx += 1;
                    grad_offset += 3;
                }
            }
        }
        // backward for softmax
        if(softmax){
            if (threadIdx.x == 0) {
                opmath_t sum = 0.;
                #pragma unroll
                for (int i=0; i < K; ++i){
                    // get sum of g_i * a_i
                    sum += cache_g_mask_before_softmax[i];
                }
                #pragma unroll
                for (int i = 0; i < K; ++i) {
                    // perform computation: a_i*g_i - g_i * SUM_j g_j * a_j and store to global memory
                    // i.e. backward for attention weights with softmax
                    *(grad_offset_softmax) = cache_g_mask_before_softmax[i] - p_mask_shm[i] * sum;
                    grad_offset_softmax += 1;
                }
            }
        }
    }
}


template <typename scalar_t, typename stride_type, int d_stride>
void _dcnv4_col2im_cuda(
  cudaStream_t stream,
  const scalar_t *value,      // B, D * H * W, (G * D)
  const scalar_t *p_offset,  // B, D * H * W, (G*K*3)
  const scalar_t *grad_output, // B, D_out*H_out*W_out, G * D
  const int kernel_d, const int kernel_h, const int kernel_w,
  const int stride_d, const int stride_h, const int stride_w, 
  const int pad_d, const int pad_h, const int pad_w,
  const int dilation_d, const int dilation_h, const int dilation_w, 
  const int G, const int D, const int B, 
  const int depth_in, const int height_in, const int width_in, 
  const int depth_out, const int height_out, const int width_out,
  const opmath_t offset_scale, const int remove_center,
  opmath_t *grad_im, opmath_t *grad_offset, const int block_thread,
  const bool softmax, const int padded_offset_dim) {

    constexpr int L = 1;

    auto kernel = backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, L, 27, false>;

    int Q = depth_out * height_out * width_out;
    int K = kernel_d * kernel_h * kernel_w;

    if (remove_center) {
        K -= 1;
    }

    if (softmax) {
        switch (K) {
            case 27:
                if(check_backward_warpp(d_stride, D)){
                    kernel = backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, L, 27, true>;
                } else {
                    kernel = backward_kernel_dcn<scalar_t, d_stride, stride_type, L, 27, true>;
                }
                break;
            case 26:
                if(check_backward_warpp(d_stride, D)){
                    kernel = backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, L, 26, true>;
                } else {
                    kernel = backward_kernel_dcn<scalar_t, d_stride, stride_type, L, 26, true>;
                }
                break;
            default:
                    printf("K=%ld\n", K);
                    throw std::invalid_argument("invalid kernel shape");
        }
    } else {
        switch (K){
            case 27:
                if(check_backward_warpp(d_stride, D)){
                    kernel = backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, L, 27, false>;
                } else {
                    kernel = backward_kernel_dcn<scalar_t, d_stride, stride_type, L, 27, false>;
                }
                break;
            case 26:
                if(check_backward_warpp(d_stride, D)){
                    kernel = backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, L, 26, false>;
                } else {
                    kernel = backward_kernel_dcn<scalar_t, d_stride, stride_type, L, 26, false>;
                }
                break;
            default:
                printf("K=%ld\n", K);
                throw std::invalid_argument("invalid kernel shape");
            }
        }

    const int block_multiplier = block_thread / (D / d_stride) / G;
    assert((B*Q) % block_multiplier == 0);

    dim3 num_blocks(B*Q / block_multiplier);
    dim3 num_threads(D / d_stride, G, block_multiplier);

    const int blockdimX = D / d_stride;

    int shm_size = sizeof(opmath_t) * (G * block_multiplier * K)*3;
    if (!check_backward_warpp(d_stride, D)) {
        // shared memory siez for non primitive kernel is split into 3 regions each of which is
        // block_multiplier * G * K for region 1 and 3 (attention values/weights) and 
        // block_multiplier * G * blockdimX * 4 for region 2 (offset gradients)
        shm_size = sizeof(opmath_t) * ((G * block_multiplier * K) * 3 + G * block_multiplier * blockdimX * 4);
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    kernel<<<num_blocks, num_threads, shm_size, stream>>>(
        value, p_offset, grad_output, G, D, Q, 
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w, 
        pad_d, pad_h, pad_w, 
        dilation_d, dilation_h, dilation_w, 
        depth_in, height_in, width_in,
        depth_out, height_out, width_out, 
        offset_scale, remove_center,
        block_multiplier, 
        grad_im, grad_offset, padded_offset_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in dcnv4_col2im_cuda: %s\n", cudaGetErrorString(err));
        printf("Launch arguments: gridDim=(%d,%d,%d), blockDim=(%d,%d,%d), "
                "shm_size=%d\n\n",
                num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x,
                num_threads.y, num_threads.z, shm_size
        );
        AT_ASSERT(false, "kernel launch error");
    }
}

template <typename scalar_t> void dcnv4_col2im_cuda(
    cudaStream_t stream, 
    const scalar_t *value,     // B, D * H * W, (G * D)
    const scalar_t *p_offset, // B, D * H * W, (G*K*3)
    const scalar_t *grad_output, // B, D_out*H_out*W_out, G * D
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int G, const int D, const int B,
    const int depth_in, const int height_in, const int width_in,
    const int depth_out, const int height_out, const int width_out,
    const opmath_t offset_scale, const int remove_center,
    opmath_t *grad_im, 
    opmath_t *grad_offset, 
    const int d_stride,
    const int block_thread, 
    const bool softmax, 
    const int padded_offset_dim){
    assert(D%d_stride ==0);

    const int size_scalar = sizeof(scalar_t);
    // recall template: <scalar_t, stride_type, d_stride>
    // cast output dtype to: d_stride * sizeof(scalar_t)
    // uint = 4 bytes, uint2 = 8 bytes, uint4 = 16 bytes, ulonglong4 = 32 bytes
    if (size_scalar == 2){
        switch (d_stride) {
            case 1:
                _dcnv4_col2im_cuda<scalar_t, scalar_t, 1>(
                    stream, value, p_offset, grad_output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, depth_in, height_in, width_in, depth_out, height_out, width_out,
                    offset_scale, remove_center, grad_im, grad_offset, block_thread, 
                    softmax, padded_offset_dim);
                break;
            case 2: 
                _dcnv4_col2im_cuda<scalar_t, uint2, 2>(
                    stream, value, p_offset, grad_output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, depth_in, height_in, width_in, depth_out, height_out, width_out,
                    offset_scale, remove_center, grad_im, grad_offset, block_thread, 
                    softmax, padded_offset_dim);
                break;
            case 4:
                _dcnv4_col2im_cuda<scalar_t, uint4, 4>(
                    stream, value, p_offset, grad_output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, depth_in, height_in, width_in, depth_out, height_out, width_out,
                    offset_scale, remove_center, grad_im, grad_offset, block_thread, 
                    softmax, padded_offset_dim);
                break;
            case 8:
                _dcnv4_col2im_cuda<scalar_t, uint4, 8>(
                    stream, value, p_offset, grad_output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, depth_in, height_in, width_in, depth_out, height_out, width_out,
                    offset_scale, remove_center, grad_im, grad_offset, block_thread, 
                    softmax, padded_offset_dim);
                break;
            case 16:
                _dcnv4_col2im_cuda<scalar_t, ulonglong4, 16>(
                    stream, value, p_offset, grad_output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, depth_in, height_in, width_in, depth_out, height_out, width_out,
                    offset_scale, remove_center, grad_im, grad_offset, block_thread, 
                    softmax, padded_offset_dim);
                break;
        }
    } else{
        // fp32
        assert(size_scalar == 4);
        switch(d_stride){
            case 1:
            _dcnv4_col2im_cuda<scalar_t, uint2, 2>(
                stream, value, p_offset, grad_output, 
                kernel_d, kernel_h, kernel_w, 
                stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                dilation_d, dilation_h, dilation_w, 
                G, D, B, depth_in, height_in, width_in, depth_out, height_out, width_out,
                offset_scale, remove_center, grad_im, grad_offset, block_thread, 
                softmax, padded_offset_dim);
                break;
            case 2:
            _dcnv4_col2im_cuda<scalar_t, uint2, 2>(
                stream, value, p_offset, grad_output, 
                kernel_d, kernel_h, kernel_w, 
                stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                dilation_d, dilation_h, dilation_w, 
                G, D, B, depth_in, height_in, width_in, depth_out, height_out, width_out,
                offset_scale, remove_center, grad_im, grad_offset, block_thread, 
                softmax, padded_offset_dim);
            break;
            case 4:
            _dcnv4_col2im_cuda<scalar_t, uint4, 4>(
                stream, value, p_offset, grad_output, 
                kernel_d, kernel_h, kernel_w, 
                stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                dilation_d, dilation_h, dilation_w, 
                G, D, B, depth_in, height_in, width_in, depth_out, height_out, width_out,
                offset_scale, remove_center, grad_im, grad_offset, block_thread, 
                softmax, padded_offset_dim);
            break;
            case 8:
            _dcnv4_col2im_cuda<scalar_t, ulonglong4, 8>(
                stream, value, p_offset, grad_output, 
                kernel_d, kernel_h, kernel_w, 
                stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, 
                dilation_d, dilation_h, dilation_w, 
                G, D, B, depth_in, height_in, width_in, depth_out, height_out, width_out,
                offset_scale, remove_center, grad_im, grad_offset, block_thread, 
                softmax, padded_offset_dim);
            break;
        }
    }
}