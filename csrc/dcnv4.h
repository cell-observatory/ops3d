#pragma once

#ifdef WITH_CUDA
#include "cuda/dcnv4_cuda.h"
#endif

at::Tensor dcnv4_forward(
    const at::Tensor &value,
    const at::Tensor &p_offset,
    const int kernel_d, const int kernel_h, const int kernel_w, 
    const int stride_d, const int stride_h, const int stride_w, 
    const int pad_d, const int pad_h, const int pad_w, 
    const int dilation_d, const int dilation_h, const int dilation_w, 
    const int group, const int group_channels,
    const float offset_scale, const int im2col_step, const int remove_center,
    const int d_stride, const int block_thread, const bool softmax) {
    if (value.device().is_cuda()) {
#ifdef WITH_CUDA
    return dcnv4_cuda_forward(
        value, p_offset, 
        kernel_d, kernel_h, kernel_w, 
        stride_d, stride_h, stride_w, 
        pad_d, pad_h, pad_w, 
        dilation_d, dilation_h, dilation_w, 
        group, group_channels, offset_scale,
        im2col_step, remove_center, d_stride, block_thread, softmax);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> dcnv4_backward(
    const at::Tensor &value, 
    const at::Tensor &p_offset, 
    const int kernel_d, const int kernel_h, const int kernel_w, 
    const int stride_d, const int stride_h, const int stride_w, 
    const int pad_d, const int pad_h, const int pad_w, 
    const int dilation_d, const int dilation_h, const int dilation_w, 
    const int group, const int group_channels,
    const float offset_scale, const int im2col_step, const at::Tensor &grad_output, 
    const int remove_center, const int d_stride, const int block_thread,
    const bool softmax){
    if (value.device().is_cuda()) {
#ifdef WITH_CUDA
    return dcnv4_cuda_backward(
        value, p_offset, 
        kernel_d, kernel_h, kernel_w, 
        stride_d, stride_h, stride_w, 
        pad_d, pad_h, pad_w, 
        dilation_d, dilation_h, dilation_w, 
        group, group_channels, offset_scale,
        im2col_step, grad_output, remove_center, d_stride, block_thread,
        softmax);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}