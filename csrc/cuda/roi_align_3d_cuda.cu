//  https://github.com/TimothyZero/MedVision/blob/main/medvision/csrc/cuda/roi_align_3d_cuda.cu
//
//  Apache License
//  Version 2.0, January 2004
//  http://www.apache.org/licenses/

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.


#include "roi_align_3d_cuda_kernel.cuh"

using namespace at;


void ROIAlign3DForwardCUDAKernelLauncher(
    const at::Tensor features, const at::Tensor rois, const float spatial_scale,
    const int sampling_ratio, const int order,
    const int channels,
    const int depth, const int height, const int width,
    const int num_rois,
    const int pooled_depth, const int pooled_height, const int pooled_width,
    at::Tensor output) {
    const int output_size = num_rois * pooled_depth * pooled_height * pooled_width * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.scalar_type(), "ROIAlign3DLauncherForward", ([&] {
        const scalar_t *bottom_data = features.contiguous().data<scalar_t>();
        const scalar_t *rois_data = rois.contiguous().data<scalar_t>();
        scalar_t *top_data = output.contiguous().data<scalar_t>();

        roi_align_3d_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sampling_ratio, order, channels,
                depth, height, width,
                pooled_depth, pooled_height, pooled_width,
                top_data);
        }));

    AT_CUDA_CHECK(cudaGetLastError());
}

void ROIAlign3DBackwardCUDAKernelLauncher(
    const at::Tensor top_grad, const at::Tensor rois, const float spatial_scale,
    const int sampling_ratio, const int order,
    const int channels,
    const int depth, const int height, const int width,
    const int num_rois,
    const int pooled_depth, const int pooled_height, const int pooled_width,
    at::Tensor bottom_grad) {
    const int output_size = num_rois * pooled_depth * pooled_height * pooled_width * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        top_grad.scalar_type(), "ROIAlign3DLauncherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.contiguous().data<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
        roi_align_3d_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, spatial_scale, sampling_ratio,
                order, channels,
                depth, height, width,
                pooled_depth, pooled_height, pooled_width,
                bottom_diff);
        }));
    AT_CUDA_CHECK(cudaGetLastError());
    }