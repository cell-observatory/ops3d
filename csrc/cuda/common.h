#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>


using at::Half;
using at::Tensor;
using phalf = at::Half;

constexpr int kWarpSize = 32;
const int CUDA_NUM_THREADS = 512;
const int THREADS_PER_BLOCK = CUDA_NUM_THREADS;

#define opmath_t at::opmath_type<scalar_t>

#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

inline bool check_backward_warpp(int d_stride, int D){
  int n_group_threads = D / d_stride;
  return (n_group_threads <= kWarpSize) && (kWarpSize % n_group_threads == 0);
}

inline int GET_BLOCKS(const int N){
    int optimal_block_num = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    int max_block_num = 4096;
    return min(optimal_block_num, max_block_num);
}


template <typename scalar_t, typename transfer_t, int c_per_thread>
__device__ void ms_deform_attn_im2col_trilinear(
    opmath_t out_reg_array[], 
    const scalar_t *&p_value, 
    const int &depth, const int &height, const int &width, 
    const opmath_t &d, const opmath_t &h, const opmath_t &w, 
    const opmath_t &attn_weight, 
    const int &w_stride, 
    const int &base_ptr) {
    
    // get -1 indices from the fractional point
    const int d_low = floor(d);
    const int h_low = floor(h);
    const int w_low = floor(w);
  
    // get +1 indices from the fractional point
    const int d_high = d_low + 1;
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    // get distance to -1 indices from fractional point
    // (effectively interpolation weights)
    const opmath_t dz = d - d_low;
    const opmath_t dy = h - h_low;
    const opmath_t dx = w - w_low;

    // get distance to the +1 indices from fractional point
    // (effectively interpolation weights)
    const opmath_t dz1 = 1 - dz;
    const opmath_t dy1 = 1 - dy;
    const opmath_t dx1 = 1 - dx;

    // strides in flattened layout (D, H, W, G, C)
    const int h_stride = width * w_stride;
    const int d_stride = height * h_stride; 

    // NOTE: c_per_thread = d_c = x-block dim
    scalar_t v000_array[c_per_thread] = {0.};
    scalar_t v001_array[c_per_thread] = {0.};
    scalar_t v010_array[c_per_thread] = {0.};
    scalar_t v011_array[c_per_thread] = {0.};
    scalar_t v100_array[c_per_thread] = {0.};
    scalar_t v101_array[c_per_thread] = {0.};
    scalar_t v110_array[c_per_thread] = {0.};
    scalar_t v111_array[c_per_thread] = {0.};

    // safe load of the eight corner values
    auto safe_grid_access = [&](int dz, int dy, int dx, scalar_t *array) -> void {
      if (dz>=0 && dz<depth && dy>=0 && dy<height && dx>=0 && dx<width)
        {
            int64_t offset = dz * d_stride + dy * h_stride + dx * w_stride + base_ptr;
            // TODO: should be possible to do wide store here
            #pragma unroll
            for (int c = 0; c < c_per_thread; ++c){
                array[c] = *(p_value + offset + c);
            }
        }
    };

    // safe load of the eight corner values
    safe_grid_access(d_low , h_low , w_low, v000_array);
    safe_grid_access(d_low , h_low , w_high, v001_array);
    safe_grid_access(d_low , h_high, w_low, v010_array);
    safe_grid_access(d_low , h_high, w_high, v011_array);
    safe_grid_access(d_high, h_low , w_low, v100_array);
    safe_grid_access(d_high, h_low , w_high, v101_array);
    safe_grid_access(d_high, h_high, w_low, v110_array);
    safe_grid_access(d_high, h_high, w_high, v111_array);

    // compute trilinear weights
    opmath_t w000 = dz1 * dy1 * dx1;
    opmath_t w001 = dz1 * dy1 *  dx;
    opmath_t w010 = dz1 *  dy * dx1;
    opmath_t w011 = dz1 *  dy *  dx;
    opmath_t w100 = dz * dy1 * dx1;
    opmath_t w101 = dz * dy1 *  dx;
    opmath_t w110 = dz *  dy * dx1;
    opmath_t w111 = dz *  dy *  dx;

    // compute the final weighted sum
    #pragma unroll
    for (int i = 0; i < c_per_thread; i++) {
        out_reg_array[i] += (opmath_t)attn_weight * 
        (w000*v000_array[i] + w001*v001_array[i] + w010*v010_array[i] + w011*v011_array[i] +
            w100*v100_array[i] + w101*v101_array[i] + w110*v110_array[i] + w111*v111_array[i]);
    }
}


template <typename scalar_t, typename transfer_t, int c_per_thread>
__device__ void ms_deform_attn_col2im_trilinear(
    const scalar_t *&p_value, 
    const int &depth, const int &height, const int &width,
    const opmath_t &d, const opmath_t &h, const opmath_t &w, 
    const opmath_t &attn_weight, 
    const int &x_stride, 
    const int &base_ptr, 
    const opmath_t offset_scale_d, const opmath_t offset_scale_h, const opmath_t offset_scale_w, 
    const scalar_t *&top_grad, 
    opmath_t *&grad_value, 
    opmath_t *grad_sampling_loc) 
{
    // Variables:
    // out[b,q,m,c] = SUM_{l,p} V(sample_xyz) * w(l,p)
    // top_grad = ∂L/∂out[b,q,m,c]
    // grad_attn_weight = ∂L/∂w(l,p)
    // grad_value = ∂L/∂V(sample_xyz)  
    // grad_sampling_loc = ∂L/∂sample_xyz

    // Computation graph:
    //   V_000 ... V_111                (feature-map voxels)
    //   │          
    //   │ w_ijk (interp. weights)     
    //   ▼             
    // sample_lp  (= v = SUM w_ijk*V_ijk)       (one (l,p) sample)
    //        │
    //        │  w^{attn}_{lp}
    //        ▼
    // out_{b,q,m,c}  (= SUM_{l,p} w^{attn}_{lp}*sample_lp)
    //        │
    //        ▼
    //    L  (loss)


    // mixing x,y,z with d,h,w with i,j,k
    // mentally set: x = w = k, y = h = j, z = d = i

    // identical computations to forward pass:

    // get -1 indices from the fractional point
    const int z0 = floor(d);
    const int y0 = floor(h);
    const int x0 = floor(w);

    // get +1 indices from the fractional point
    const int z1 = z0 + 1;
    const int y1 = y0 + 1;
    const int x1 = x0 + 1;

    // get distance to -1 indices from fractional point
    // (effectively interpolation weights)
    const opmath_t dz = d - z0;
    const opmath_t dy = h - y0;
    const opmath_t dx = w - x0;

    // get distance to the +1 indices from fractional point
    // (effectively interpolation weights)
    const opmath_t dz1 = 1 - dz;
    const opmath_t dy1 = 1 - dy;
    const opmath_t dx1 = 1 - dx;

    // strides in flattened layout (D, H, W, G, C)
    const int y_stride = width * x_stride;
    const int z_stride = height * y_stride;

    // register level array for top_grad values
    // for c_per_thread channels
    scalar_t _top_grad_array[c_per_thread] = {0.};

    // read from memory into registers
    *(transfer_t *)(_top_grad_array) = *(transfer_t *)(top_grad);

    opmath_t top_grad_array[c_per_thread] = {0.};

    for (int i = 0; i < c_per_thread; ++i) {
        top_grad_array[i] = (opmath_t)(_top_grad_array[i]);
    }

    // initialize arrays for voxel values in registers
    scalar_t v000_array[c_per_thread] = {0.};
    scalar_t v001_array[c_per_thread] = {0.};
    scalar_t v010_array[c_per_thread] = {0.};
    scalar_t v011_array[c_per_thread] = {0.};
    scalar_t v100_array[c_per_thread] = {0.};
    scalar_t v101_array[c_per_thread] = {0.};
    scalar_t v110_array[c_per_thread] = {0.};
    scalar_t v111_array[c_per_thread] = {0.};

    // initialize arrays for gradients wrt values in registers
    opmath_t grad_dz_weight[c_per_thread] = {0.};
    opmath_t grad_dy_weight[c_per_thread] = {0.};
    opmath_t grad_dx_weight[c_per_thread] = {0.};

    // safe load + scatter + grad wrt coords
    auto process_voxel = [&](int zz, int yy, int xx,
                                opmath_t wght,
                                opmath_t *g_dx, opmath_t *g_dy, opmath_t *g_dz,
                                scalar_t *value_array) -> void
    {
        if (zz < 0 || zz >= depth  ||
            yy < 0 || yy >= height ||
            xx < 0 || xx >= width)
            return;

        auto ptr = zz*z_stride + yy*y_stride + xx*x_stride + base_ptr;
        *(transfer_t *)(value_array) = *(transfer_t *)(p_value + ptr);

        // recall: 
        // w_ijk=(i?dz:1−dz)(j?dy:1−dy)(k?dx:1−dx)
        // v_ijk is value of the voxel at (i,j,k) weighted by trilinear weights
        // thus we get interpolation v = SUM(wijk*vijk)

        // sample_lp = SUM_ijk w_ijk * v_ijk (i,j,k) ∈ [0,1] for each (l,p)
        // out_bqmc = SUM_lp w^{attn}_{lp} * sample_lp
        // Hence: ∂L/∂V_ijk = ∂L/∂out_bqmc * ∂out_bqmc/∂v * ∂v/∂V_ijk 
        //              = ∂L/∂out_bqmc * w^{attn}_{lp} * w_ijk
        //              = top_grad * attn_weight * w_ijk 
        
        #pragma unroll
        for (int i = 0; i < c_per_thread; ++i) {
            atomicAdd(grad_value + ptr + i, wght * top_grad_array[i] * attn_weight);
            
            // accumulate ∂w_ijk/∂x = (i?dz:1−dz)(j?dy:1−dy)∂/∂x[(k?dx:1−dx)]. Then since
            // since ∂/∂x[(k?dx:1−dx)] = +1 if k=1 and -1 if k=0 => ∂w_ijk/∂x = C * ((xx==x1) ? +1 : -1)
            // for C = (i?dz:1−dz)(j?dy:1−dy) 
            // and similarly for y and z
            // Hence, ∂L/∂x = ∂L/∂v * ∂v/∂x = ∂L/∂out_bqmc * w^{attn}_{lp} * ∂v/∂x
            // where ∂v/∂x = ∂V_ijk/∂x = ∂w_ijk/∂x * v_ijk = v_ijk * ((xx==x1)?+1:-1)(i?dz:1−dz)(j?dy:1−dy) 
            // Thus, ∂L/∂x = g * w^{attn}_{lp} * (v_ijk * ((xx==x1)?+1:-1)(i?dz:1−dz)(j?dy:1−dy))
            // where g = ∂L/∂out_bqmc 
            // grad_dx_weight etc. computes the inner sum, i.e. omitting g * w^{attn}_{lp}
            // which is computed later (see below)

            opmath_t fz = (zz == z1) ? dz  : dz1;   
            opmath_t fy = (yy == y1) ? dy  : dy1;   
            opmath_t fx = (xx == x1) ? dx  : dx1;   

            grad_dx_weight[i] += fz * fy * ((xx==x1) ? +1 : -1) * value_array[i];
            grad_dy_weight[i] += fz * fx * ((yy==y1) ? +1 : -1) * value_array[i];
            grad_dz_weight[i] += fy * fx * ((zz==z1) ? +1 : -1) * value_array[i];
        }
    };
  
    // iterate over 8 neighbours and add contribution
    // to the gradients
    opmath_t w000 = dz1*dy1*dx1;   // (z0,y0,x0)
    opmath_t w001 = dz1*dy1*dx;    // (z0,y0,x1)
    opmath_t w010 = dz1*dy *dx1;   // (z0,y1,x0)
    opmath_t w011 = dz1*dy *dx;    // (z0,y1,x1)
    opmath_t w100 = dz *dy1*dx1;   // (z1,y0,x0)
    opmath_t w101 = dz *dy1*dx;    // (z1,y0,x1)
    opmath_t w110 = dz *dy *dx1;   // (z1,y1,x0)
    opmath_t w111 = dz *dy *dx;    // (z1,y1,x1)

    process_voxel(z0,y0,x0,w000,grad_dx_weight,grad_dy_weight,grad_dz_weight, v000_array);
    process_voxel(z0,y0,x1,w001,grad_dx_weight,grad_dy_weight,grad_dz_weight, v001_array);
    process_voxel(z0,y1,x0,w010,grad_dx_weight,grad_dy_weight,grad_dz_weight, v010_array);
    process_voxel(z0,y1,x1,w011,grad_dx_weight,grad_dy_weight,grad_dz_weight, v011_array);
    process_voxel(z1,y0,x0,w100,grad_dx_weight,grad_dy_weight,grad_dz_weight, v100_array);
    process_voxel(z1,y0,x1,w101,grad_dx_weight,grad_dy_weight,grad_dz_weight, v101_array);
    process_voxel(z1,y1,x0,w110,grad_dx_weight,grad_dy_weight,grad_dz_weight, v110_array);
    process_voxel(z1,y1,x1,w111,grad_dx_weight,grad_dy_weight,grad_dz_weight, v111_array);

    // as described above, ∂L/∂x = ∂L/∂v * ∂v/∂x = ∂L/∂out_bqmc * w^{attn}_{lp} * ∂v/∂x
    // ∂L/∂x = g * w^{attn}_{lp} * (v_ijk * w_ijk * (k?+1:−1) * 1/(k?dx:1−dx))
    // ∂L/∂x = g * w^{attn}_{lp} * g_dx
    // lastly we convert to normalized grads by multiplying with scale factor

    //accumulate contributions to the gradient 
    // wrt sampling locations
    opmath_t _grad_offset_x = 0;
    opmath_t _grad_offset_y = 0;
    opmath_t _grad_offset_z = 0;

    // see above for equation derivations

    #pragma unroll
    for (int i = 0; i < c_per_thread; ++i) {
        _grad_offset_x +=
            grad_dx_weight[i] * top_grad_array[i]; 
        _grad_offset_y +=
            grad_dy_weight[i] * top_grad_array[i]; 
        _grad_offset_z +=
            grad_dz_weight[i] * top_grad_array[i];
    }

    _grad_offset_x *= (offset_scale_w * attn_weight); 
    _grad_offset_y *= (offset_scale_h * attn_weight);
    _grad_offset_z *= (offset_scale_d * attn_weight);

    // store the gradients wrt sampling locations 
    // to memory
    *(grad_sampling_loc) = _grad_offset_x;
    *(grad_sampling_loc + 1) = _grad_offset_y;
    *(grad_sampling_loc + 2) = _grad_offset_z;

    opmath_t current_val;
    opmath_t _grad_offset_attn = 0;
    #pragma unroll
    for (int i = 0; i < c_per_thread; i++) {
        current_val = (opmath_t)(
            w000 * v000_array[i] + w001 * v001_array[i] +
            w010 * v010_array[i] + w011 * v011_array[i] +
            w100 * v100_array[i] + w101 * v101_array[i] +
            w110 * v110_array[i] + w111 * v111_array[i]
        );
        _grad_offset_attn += current_val * top_grad_array[i];
    }

    // ∂L/∂grad_attn = ∂L/∂out[b,q,m,c] * ∂out[b,q,m,c]/∂attn_weight
    // where ∂out[b,q,m,c]/∂attn_weight = sample_lp i.e. v returned by function
    *(grad_sampling_loc + 3) = _grad_offset_attn;
}