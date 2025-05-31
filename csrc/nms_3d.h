//  https://github.com/TimothyZero/MedVision/blob/main/medvision/csrc/nms_3d.h

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
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


at::Tensor nms_3d_cuda(const at::Tensor boxes, float nms_overlap_thresh);


at::Tensor nms_3d(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float threshold) {
    if (dets.device().is_cuda()) {

        if (dets.numel() == 0) {
            at::cuda::CUDAGuard device_guard(dets.device());
            return at::empty({0}, dets.options().dtype(at::kLong));
        }
        auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
        return nms_3d_cuda(b, threshold);

    }
    AT_ERROR("Not compiled with CPU support");
//  at::Tensor result = nms_cpu(dets, scores, threshold);
//  return result;
}