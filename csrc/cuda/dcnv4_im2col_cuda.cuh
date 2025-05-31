//  Modified to 3D from:
//  https://github.com/OpenGVLab/DCNv4/blob/main/DCNv4_op/src/cuda/dcnv4_im2col_cuda.cuh
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


// TODO: not used, need further testing to determine when to use register-based implementation
//       vs shared memory implementation
template <typename scalar_t, int d_stride, typename transfer_t, int L, int K, bool softmax>
__global__ void forward_kernel_dcn(
    const scalar_t *value_ptr, 
    const scalar_t* offset_ptr, 
    scalar_t *output_ptr,
    const int G, const int D, const int Q,
    const int kernel_d, const int kernel_h, const int kernel_w, 
    const int stride_d, const int stride_h, const int stride_w, 
    const int pad_d, const int pad_h, const int pad_w, 
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int depth_in, const int height_in, const int width_in, 
    const int depth_out, const int height_out, const int width_out, 
    const opmath_t offset_scale, const int remove_center, 
    const int block_multiplier, const int padded_offset_dim){

    // input: [B, Q, G, D] where Q = D * H * W 
    // we launch kernel s.t. GridDim.x * blockDim.z == B*Q 
    // blockDim.x = d_stride (channels per thread), blockDim.y = G

    // get query and batch index
    const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
    const int &bi = blockIdx.x * block_multiplier / Q;

    // get group and channel index
    // group idx is directly mapped to threadIdx.y
    // each threadIdx.x attends to d_stride channels
    // we set li = 0 for DCN in contrast to ms_deform_attn
    const int &di_s = threadIdx.x * d_stride;
    const int &gi = threadIdx.y;
    constexpr int li = 0;
    
    // declare block of dynamic shared memory
    // shared memory lives on each SM and can be accessed 
    // by all threads in the thread block 
    // NOTE: char[] is most flexible way to index by byte
    //       we can then cast into whatever types needed
    extern __shared__ char _s[];

    // load K attention/mask values
    // each thread loads K/N values 
    // into shared memory that each thread will then use to 
    // compute the trilinear interpolation for that (b,q,g)

    // Equation is: Y_{b,q,g,d} = SUM a_{b,q,g,k}*triilinear(x,d_{b,q,g,k},h_{b,q,g,k},w_{b,q,g,k})[d]
    // where a_{b,q,g,k} is the mask value for (b,q,g,k) 
    // d,h,w are our learned offests for the kernel points
    // each thread will attend to: d_s = threadIdx.x * d_stride
    // i.e. d in { d_s, d_s+1, …, d_s + d_stride - 1 }.
    // and compute a partial output vector: 
    // Y[b,q,g,d_s:d_s+d_stride−1]=p_out_shm[0:d_stride−1]
    // once computed for all threads we get output O[b,q,g] value

    // thread will access attn/mask weights (b_i,q_i,g_i)
    // where each b_i,q_i,g_i attends to K points (L=1) 
    opmath_t *const p_mask_shm = 
        (opmath_t *)(_s) + ((threadIdx.z * G + gi) * L + li) * K;
    
    // initialize output d_stride len. array
    opmath_t p_out_shm[d_stride] = {0.};
    
    // Note: size of one (b,q) block with padding is padded_offset_dim
    //       thread will access offsets for (bi,qi,gi)
    const scalar_t *p_offset_ptr = offset_ptr + (bi*Q + qi)*padded_offset_dim + gi*K*4;

    // we predict K attention/mask values per head and query
    // for each query, i.e. we have a_{b,q,g,k}Trilinear(X,d_0+l_k+Δz_k, h_0+j_k+Δy_k, w0+i_k+Δx_k) 
    // where K=kernel_d*kernel_h*kernel_w and l_k,j_k,i_k are the offsets for the k-th sampling point  
    const int mask_length = K;
    // each thread attends to d_stride of total number of channels per head group
    // equivalently we could have set num_thread = BlockDim.x but this way
    // num_thread is a compile-time constant and we can unroll the loop below
    const int num_thread = (D/d_stride);
    // number of full strided loads for each thread
    const int num_iter = mask_length / num_thread;
    // the remainder after last full strided load
    const int remainder = mask_length - num_iter * num_thread;

    // load mask/attention values into shared memory
    // we have num_thread threads and we need to load K*num_channel attention values
    for (int i=0; i<num_iter; i++){
        // move beyond K*3 offset ptr values and then access atten value at
        // 4K + i*T + t and store in shared memory at i*T + t
        *(p_mask_shm + num_thread*i + threadIdx.x) = 
            *(scalar_t *)(p_offset_ptr + K*3 + num_thread*i + threadIdx.x);
    }
    if (remainder > 0 && threadIdx.x < remainder){
        // for iteration num_iter we load threads such that 
        // threadIdx.x < remainder [mask_length % num_thread threads in total] 
        *(p_mask_shm + num_thread*num_iter + threadIdx.x) = 
            *(scalar_t *)(p_offset_ptr + K*3 + num_thread*num_iter + threadIdx.x);
    }

    if (softmax){
        // synchronize threads to ensure all threads have loaded
        // the mask values before we start calculating softmax
        __syncthreads();
        // softmax over L and K but L=1 for DCN
        // standard subtract by max value for numerical stability
        if (threadIdx.x == 0){
            opmath_t softmax_max = -1e100;
            opmath_t softmax_sum = 0.0;
            
            // get max across K
            for (int j=0; j<K; j++){
                softmax_max = max(softmax_max, p_mask_shm[j]);
            }
            
            // x = exp(x - max(x)) for numerical stability
            for (int j=0; j<K; j++){
                opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
                p_mask_shm[j] = exp_results;
                softmax_sum += exp_results;
            }

            // x = x / sum(x)
            for  (int j = 0; j < K; j++){
                p_mask_shm[j] /= softmax_sum;
            }
        }
    
        __syncthreads();
    }

    // idx in shared memory for mask 
    // and global memory for offset pointer
    int offset_idx = 0;
    int mask_idx = 0;

    // value_ptr: [B,Q,G,D] where Q = D*H*W
    // stride (d,h,w) -> (d,h,w+1) given by G*D
    const int w_stride = G*D;
    // move to (bi,qi) block, used in trilinear interpolation op.
    const int base_ptr = gi*D + di_s;
    // value_ptr to (bi,qi,gi) block 
    const scalar_t *p_value_ptr = 
        value_ptr + (bi * (depth_in * height_in * width_in)) * (G * D);
    
    // get current (x,y,z) pos. given qi 
    int x = qi % width_out;
    int yi = qi / width_out;
    int y = yi % height_out;
    int z = yi / height_out;

    // (dilation_w * (kernel_w-1))/2 is half-span of kernel_w 
    // we then shift back padding pad_w and find correct starting
    // width position for the kernel given the stride
    const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + x*stride_w;
    const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + y*stride_h;
    const int p0_d = ((dilation_d * (kernel_d - 1)) >> 1) - pad_d + z*stride_d;
    
    // to skip subtracting that same half‐span
    // i.e. to avoid: 
    // p0_w + ((i * dilation_w + Δ)>>1)*offset_scale - ((dilation_w * (kernel_w-1))>>1)*offset_scale
    // we pre-subtract it once and in the inner loop we just do:
    // w_im = p0_w_ + (i*dilation_w + Δx)*offset_scale (and similar for y,z)
    const opmath_t p0_w_ = p0_w - (dilation_w * (kernel_w - 1) >> 1) * offset_scale;
    const opmath_t p0_h_ = p0_h - (dilation_h * (kernel_h - 1) >> 1) * offset_scale;
    const opmath_t p0_d_ = p0_d - (dilation_d * (kernel_d - 1) >> 1) * offset_scale;

    const int center_d = kernel_d / 2;
    const int center_h = kernel_h / 2;
    const int center_w = kernel_w / 2;

    int out_idx = ((bi * Q + qi) * G + gi) * D + di_s; 
    for (int i = 0; i < kernel_w; ++i){
        for (int j = 0; j < kernel_h; ++j){
            for (int k = 0; k < kernel_d; ++k){
                // if remove_center flag is set we remove center kernel point 
                // since this is just the identity point and may not be needed
                if (i != center_w || j != center_h || k != center_d || !remove_center){
                    // add learned offset to the kernel position including dilation offset
                    const opmath_t w_im = 
                        p0_w_ + (i*dilation_w + (opmath_t)p_offset_ptr[offset_idx])*offset_scale;
                    const opmath_t h_im = 
                        p0_h_ + (j*dilation_h + (opmath_t)p_offset_ptr[offset_idx+1])*offset_scale;
                    const opmath_t d_im = 
                        p0_d_ + (k*dilation_d + (opmath_t)p_offset_ptr[offset_idx+2])*offset_scale;
                    const opmath_t attn = p_mask_shm[mask_idx];
                    
                    if (d_im > -1 && h_im > -1 && w_im > -1 && d_im < depth_in && h_im < height_in && w_im < width_in){
                        // perform trilinear interpolation of value tensor (see common.h)
                        ms_deform_attn_im2col_trilinear<scalar_t, transfer_t, d_stride>(
                            p_out_shm, p_value_ptr, 
                            depth_in, height_in, width_in, 
                            d_im, h_im, w_im, 
                            attn, w_stride, base_ptr);
                    }

                    offset_idx += 3;
                    mask_idx += 1;
                }
            }
        }
    }
    // alias the start of p_out_shm[] as if it were an 
    // array of scalar_t
    scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
    #pragma unroll
    for (int ds = 0; ds < d_stride; ds++){
        // implicitly cast p_out_shm[] to scalar_t
        fp16_regs[ds] = p_out_shm[ds];
    }
    // store (bi,qi,gi,di_s) block in output_ptr tensor
    // one wide transfer_t store into global memory
    *(transfer_t *)(output_ptr + out_idx) = *(transfer_t *)(p_out_shm);
}


// register‐based implementation of above 
// i.e. we keep the entire K-length mask in thread-local registers
template <typename scalar_t, int d_stride, typename transfer_t, int L, int K, bool softmax>
__global__ void forward_kernel_dcn_reg(
    const scalar_t *value_ptr, const scalar_t* offset_ptr, scalar_t *output_ptr,
    const int G, const int D, const int Q, 
    const int kernel_d, const int kernel_h, const int kernel_w, 
    const int stride_d, const int stride_h, const int stride_w, 
    const int pad_d, const int pad_h, const int pad_w, 
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int depth_in, const int height_in, const int width_in, 
    const int depth_out, const int height_out, const int width_out, 
    const opmath_t offset_scale, const int remove_center,
    const int block_multiplier, const int padded_offset_dim) {

    // get query and batch index
    const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
    const int &bi = blockIdx.x * block_multiplier / Q;

    // get channel slice and group index (li=0 for DCN)
    const int &di_s = threadIdx.x * d_stride;
    const int &gi = threadIdx.y;
    constexpr int li = 0;
    
    // set up K opmath_t registers to hold mask values
    // also set up d_stride opmath_t registers to hold output values
    // shm bit of a misnomer here
    opmath_t p_mask_shm[K] = {0.};
    opmath_t p_out_shm[d_stride] = {0.};

    const scalar_t *p_offset_ptr = offset_ptr + (bi*Q + qi)*padded_offset_dim + gi*K*4;

    for (int i=0; i < K; i++){
        // move past K*3 offset ptr values and then access 
        // mask/atten values and put in registers
        p_mask_shm[i] = *(p_offset_ptr + K*3 + i);
    }

    if (softmax) {
        // Calculate softmax over L and K
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

    // indices to update to get next set of 
    // offset and mask values
    int offset_idx = 0;
    int mask_idx = 0;

    // value_ptr: [B,Q,G,D] where Q = D*H*W
    // stride (d,h,w) -> (d,h,w+1) is G*D
    const int w_stride = G * D;
    const int base_ptr = gi * D + di_s;
    const scalar_t *p_value_ptr =
        value_ptr + (bi * (depth_in * height_in * width_in)) * (G * D);

    // get current (x,y,z) pos. given qi 
    int x = qi % width_out;
    int yi = qi / width_out;
    int y = yi % height_out;
    int z = yi / height_out;

    // (dilation_w * (kernel_w-1))/2 is half-span of kernel_w 
    // we then shift back padding pad_w and find correct starting
    // width position for the kernel given the stride
    const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + x*stride_w;
    const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + y*stride_h;
    const int p0_d = ((dilation_d * (kernel_d - 1)) >> 1) - pad_d + z*stride_d;
    
    // to skip subtracting that same half‐span
    // i.e. to avoid: 
    // p0_w + ((i * dilation_w + Δ)>>1)*offset_scale - ((dilation_w * (kernel_w-1))>>1)*offset_scale
    // we pre-subtract it once and in the inner loop we just do:
    // w_im = p0_w_ + (i*dilation_w + Δx)*offset_scale (and similar for y,z)
    const opmath_t p0_w_ = p0_w - (dilation_w * (kernel_w - 1) >> 1) * offset_scale;
    const opmath_t p0_h_ = p0_h - (dilation_h * (kernel_h - 1) >> 1) * offset_scale;
    const opmath_t p0_d_ = p0_d - (dilation_d * (kernel_d - 1) >> 1) * offset_scale;

    // get center of kernel in x,y,z
    const int center_d = kernel_d / 2;
    const int center_h = kernel_h / 2;
    const int center_w = kernel_w / 2;

    int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

    for (int i = 0; i < kernel_w; ++i) {
        for (int j = 0; j < kernel_h; ++j) {
            for (int k = 0; k < kernel_d; ++k) {
                if (i != center_w || j != center_h || k != center_d || !remove_center) {
                    const opmath_t w_im =
                        p0_w_ + (i * dilation_w + (opmath_t)p_offset_ptr[offset_idx]) *
                                    offset_scale;
                    const opmath_t h_im =
                        p0_h_ + (j * dilation_h + (opmath_t)p_offset_ptr[offset_idx + 1]) *
                                    offset_scale;
                    const opmath_t d_im = 
                        p0_d_ + (k * dilation_d + (opmath_t)p_offset_ptr[offset_idx + 2]) *
                                    offset_scale;
                    const opmath_t attn = p_mask_shm[mask_idx];

                    if (d_im > -1 && h_im > -1 && w_im > -1 && d_im < depth_in && h_im < height_in && w_im < width_in) {
                    ms_deform_attn_im2col_trilinear<scalar_t, transfer_t, d_stride>(
                        p_out_shm, p_value_ptr, 
                        depth_in, height_in, width_in,
                        d_im, h_im, w_im, 
                        attn,
                        w_stride, base_ptr);
                    }
                    offset_idx += 3;
                    mask_idx += 1;
                }
            }
        }
    }
    scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
    #pragma unroll
    for (int ds = 0; ds < d_stride; ds++) {
        fp16_regs[ds] = p_out_shm[ds];
    }
    // store (bi,qi,gi,di_s) block in p_output tensor 
    // one wide transfer_t store into global memory
    *(transfer_t *)(output_ptr + out_idx) = *(transfer_t *)(p_out_shm);
}


template <typename scalar_t, typename stride_type, int d_stride>
void _dcnv4_im2col_cuda(cudaStream_t stream, 
                            const scalar_t *value,   // B, D * H * W, G * D
                            const scalar_t *p_offset, // B, D * H * W, G * K * 3
                            scalar_t *output,       // B, D_out*H_out*W_out, G * D
                            const int kernel_d, const int kernel_h, const int kernel_w,
                            const int stride_d, const int stride_h, const int stride_w,
                            const int pad_d, const int pad_h, const int pad_w, 
                            const int dilation_d, const int dilation_h, const int dilation_w,
                            const int G, const int D, const int B,
                            const int depth_in, const int height_in, const int width_in, 
                            const int depth_out, const int height_out, const int width_out,
                            const opmath_t offset_scale, 
                            const int remove_center, const int block_thread, 
                            const int softmax, 
                            const int padded_offset_dim) {
    constexpr int L = 1;

    // recall template: <scalar_t, d_stride, stride_type, L, K, softmax>
    // kernel is 3x3x3=27 
    auto kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 27, true>;
    
    int Q = depth_out * height_out * width_out;
    int K = kernel_d * kernel_h * kernel_w;
    
    // remove center kernel point if needed
    if (remove_center){
        K -= 1;
    }

    // set kernel based on softmax flag and if remove_center or not
    if(softmax) {
        switch (K){
            case 27:
                kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 27, true>;
                break;
            case 26:
                kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 26, true>;
                break;
            default:
                printf("K=%d\n", K);
                throw std::invalid_argument("invalid kernel shape");
        }
    } else {
        switch (K){
            case 27:
                kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 27, false>;
                break;
            case 26:
                kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 26, false>;
                break;
            default:
                printf("K=%d\n", K);
                throw std::invalid_argument("invalid kernel shape");
        }
    }

    // blockDim.x must be D/d_stride
    // blockDim.y must be G
    // blockDim.z must be block_multiplier such that B*Q % block_multiplier == 0
    // after we launch GridDim.x * blockDim.z threads
    const int block_multiplier = block_thread / (D / d_stride) / G;
    assert(((B*Q) % block_multiplier) == 0);

    // get number of blocks which must be B*Q/block_multiplier
    dim3 num_blocks(B*Q / block_multiplier);
    // define size of each block based on wanting each thread to 
    // attend to d_stride channels and one G group and 
    // block_multiplier=blockdim.z (batch*query) queries 
    dim3 num_threads(D / d_stride, G, block_multiplier);

    // we launch with no dynamic shared memory
    // since we are using registers to store the mask values
    // i.e. all shared‐mem usage in kernel will come from 
    // statically‐declared arrays
    int shm_size = 0;

    // tell CUDA driver that kernel wont need more than shm_size bytes
    // of dynamic shared memory, i.e. no dynamically allocated shared memory
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_size);

    // launch kernel
    kernel<<<num_blocks, num_threads, shm_size, stream>>>(
        value, p_offset, output, 
        G, D, Q,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        depth_in, height_in, width_in, 
        depth_out, height_out, width_out, 
        offset_scale, remove_center,
        block_multiplier, padded_offset_dim);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in dcnv4_im2col_cudaL %s\n", cudaGetErrorString(err));
        printf("launch arguments: gridDim=(%d, %d, %d), blockDim=(%d,%d,%d), "
               "shm_size=%d\n\n",
               num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x,
               num_threads.y, num_threads.z, shm_size);
        AT_ASSERTM(false, "kernel launch error");
    }
}

template <typename scalar_t>
void dcnv4_im2col_cuda(
    cudaStream_t stream, 
    const scalar_t *value,
    const scalar_t *p_offset,
    scalar_t *output,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int G, const int D, const int B,
    const int depth_in, const int height_in, const int width_in,
    const int depth_out, const int height_out, const int width_out,
    const opmath_t offset_scale, 
    const int remove_center, const int d_stride, const int block_thread, const int softmax,
    const int padded_offset_dim){
    // ensure that each thread attends
    // to a number of channels that
    // is a multiple of total
    assert (D % d_stride == 0);
    // if half-precision
    if (sizeof(scalar_t) == 2){
        // recall template: <scalar_t, stride_type, d_stride>
        // cast output dtype to: d_stride * sizeof(scalar_t)
        // uint = 4 bytes, uint2 = 8 bytes, uint4 = 16 bytes, ulonglong4 = 32 bytes
        switch (d_stride){
            case 1:
                _dcnv4_im2col_cuda<scalar_t, scalar_t, 1>(
                    stream, value, p_offset, output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, 
                    pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, 
                    depth_in, height_in, width_in, 
                    depth_out, height_out, width_out, 
                    offset_scale, remove_center,
                    block_thread, softmax, 
                    padded_offset_dim);
                break;
            case 2:
                _dcnv4_im2col_cuda<scalar_t, uint, 2>(
                    stream, value, p_offset, output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, 
                    pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, 
                    depth_in, height_in, width_in, 
                    depth_out, height_out, width_out, 
                    offset_scale, remove_center,
                    block_thread, softmax, 
                    padded_offset_dim);
                break;
            case 4:
                _dcnv4_im2col_cuda<scalar_t, uint2, 4>(
                    stream, value, p_offset, output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, 
                    pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, 
                    depth_in, height_in, width_in, 
                    depth_out, height_out, width_out, 
                    offset_scale, remove_center,
                    block_thread, softmax, 
                    padded_offset_dim);
                break;
            case 8:
                _dcnv4_im2col_cuda<scalar_t, uint4, 8>(
                    stream, value, p_offset, output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, 
                    pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, 
                    depth_in, height_in, width_in, 
                    depth_out, height_out, width_out, 
                    offset_scale, remove_center,
                    block_thread, softmax, 
                    padded_offset_dim);
                break;
            case 16:
                _dcnv4_im2col_cuda<scalar_t, ulonglong4, 16>(
                    stream, value, p_offset, output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, 
                    pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, 
                    depth_in, height_in, width_in, 
                    depth_out, height_out, width_out, 
                    offset_scale, remove_center,
                    block_thread, softmax, 
                    padded_offset_dim);
                break;
        }
    } else {
        // for fp32
        assert (sizeof(scalar_t) == 4);
        switch (d_stride){
            case 1:
                _dcnv4_im2col_cuda<scalar_t, uint, 1>(
                    stream, value, p_offset, output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, 
                    pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, 
                    depth_in, height_in, width_in, 
                    depth_out, height_out, width_out, 
                    offset_scale, remove_center,
                    block_thread, softmax, 
                    padded_offset_dim);
                break;
            case 2:
                _dcnv4_im2col_cuda<scalar_t, uint2, 2>(
                    stream, value, p_offset, output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, 
                    pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, 
                    depth_in, height_in, width_in, 
                    depth_out, height_out, width_out, 
                    offset_scale, remove_center,
                    block_thread, softmax, 
                    padded_offset_dim);
                break;
            case 4:
                _dcnv4_im2col_cuda<scalar_t, uint4, 4>(
                    stream, value, p_offset, output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, 
                    pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, 
                    depth_in, height_in, width_in, 
                    depth_out, height_out, width_out, 
                    offset_scale, remove_center,
                    block_thread, softmax, 
                    padded_offset_dim);
                break;
            case 8:
                _dcnv4_im2col_cuda<scalar_t, ulonglong4, 8>(
                    stream, value, p_offset, output, 
                    kernel_d, kernel_h, kernel_w, 
                    stride_d, stride_h, stride_w, 
                    pad_d, pad_h, pad_w, 
                    dilation_d, dilation_h, dilation_w, 
                    G, D, B, 
                    depth_in, height_in, width_in, 
                    depth_out, height_out, width_out, 
                    offset_scale, remove_center,
                    block_thread, softmax, 
                    padded_offset_dim);
                break;
        default:
            printf("d_stride > 8 not supported for fp32");
            throw std::invalid_argument("invalid d_stride");
        }
    }
}