//  https://github.com/TimothyZero/MedVision/blob/main/medvision/csrc/roi_align_3d.h

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


#pragma once
#include <torch/extension.h>

using namespace at;


#ifdef WITH_CUDA
void ROIAlign3DForwardCUDAKernelLauncher(
    const at::Tensor features, const at::Tensor rois, const float spatial_scale,
    const int sample_ratio, const int order,
    const int channels,
    const int depth, const int height, const int width,
    const int num_rois,
    const int aligned_depth, const int aligned_height, const int aligned_width,
    at::Tensor output);

void ROIAlign3DBackwardCUDAKernelLauncher(
    const at::Tensor top_grad, const at::Tensor rois, const float spatial_scale,
    const int sample_ratio, const int order,
    const int channels,
    const int depth, const int height, const int width,
    const int num_rois,
    const int aligned_depth, const int aligned_height, const int aligned_width,
    at::Tensor bottom_grad);

void roi_align_3d_forward_cuda(
    Tensor features, Tensor rois, Tensor output,
    int aligned_depth, int aligned_height, int aligned_width,
    float spatial_scale, int sample_ratio,
    int order) {
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);

    if (size_rois != 7) {
        AT_ERROR("wrong roi size! size_rois should be 7");
    }

    int num_channels = features.size(1);
    int data_depth = features.size(2);
    int data_height = features.size(3);
    int data_width = features.size(4);
    ROIAlign3DForwardCUDAKernelLauncher(
        features, rois, spatial_scale, sample_ratio, order,
        num_channels, data_depth, data_height, data_width, num_rois, aligned_depth, aligned_height,
        aligned_width, output);
}

void roi_align_3d_backward_cuda(
    Tensor top_grad, Tensor rois, Tensor bottom_grad,
    int aligned_depth, int aligned_height, int aligned_width,
    float spatial_scale, int sample_ratio, int order) {
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 7) {
        AT_ERROR("wrong roi size! size_rois should be 7");
    }

    int num_channels = bottom_grad.size(1);
    int data_depth = bottom_grad.size(2);
    int data_height = bottom_grad.size(3);
    int data_width = bottom_grad.size(4);
    ROIAlign3DBackwardCUDAKernelLauncher(
        top_grad, rois, spatial_scale, sample_ratio, order,
        num_channels, data_depth, data_height, data_width, num_rois, aligned_depth, aligned_height,
        aligned_width, bottom_grad);
}
#endif


void roi_align_3d_forward(Tensor input, Tensor rois, Tensor output,
                               int aligned_depth, int aligned_height, int aligned_width,
                               float spatial_scale, int sampling_ratio,
                               int order) {
    if (input.device().is_cuda()) {
#ifdef WITH_CUDA
//    CHECK_CUDA_INPUT(input);
//    CHECK_CUDA_INPUT(rois);
//    CHECK_CUDA_INPUT(output);

        roi_align_3d_forward_cuda(input, rois, output, aligned_depth, aligned_height,
                                        aligned_width, spatial_scale, sampling_ratio,
                                        order);
#else
    AT_ERROR("RoIAlign is not compiled with GPU support");
#endif
  } else {
//    CHECK_CPU_INPUT(input);
//    CHECK_CPU_INPUT(rois);
//    CHECK_CPU_INPUT(output);
//    roi_alignforward_cpu(input, rois, output, aligned_height,
//                                  aligned_width, spatial_scale, sampling_ratio,
//                                  aligned, clockwise);
    AT_ERROR("RoIAlign is not implemented on CPU");
  }
}

void roi_align_3d_backward(Tensor top_grad, Tensor rois,
                                Tensor bottom_grad, int aligned_depth, int aligned_height,
                                int aligned_width, float spatial_scale,
                                int sampling_ratio, int order) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
//    CHECK_CUDA_INPUT(top_grad);
//    CHECK_CUDA_INPUT(rois);
//    CHECK_CUDA_INPUT(bottom_grad);

    roi_align_3d_backward_cuda(top_grad, rois, bottom_grad, aligned_depth, aligned_height,
                                    aligned_width, spatial_scale,
                                    sampling_ratio, order);
#else
    AT_ERROR("RoIAlign is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("RoIAlign is not implemented on CPU");
  }
}