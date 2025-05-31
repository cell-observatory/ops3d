#include "nms_3d.h"
#include "roi_align_3d.h"
#include "flash_deform_attn.h"
#include "dcnv4.h"

// DEPRECATED:
// #include "ms_deform_attn_v1.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_3d", &nms_3d, "non-maximum suppression for 3d");
    
    m.def("roi_align_3d_forward", &roi_align_3d_forward, "roi_align_3d_forward");
    m.def("roi_align_3d_backward", &roi_align_3d_backward, "roi_align_3d_backward");

    m.def("dcnv4_forward", &dcnv4_forward, "dcnv4_forward");
    m.def("dcnv4_backward", &dcnv4_backward, "dcnv4_backward");

    m.def("flash_deform_attn_forward", &flash_deform_attn_forward, "flash_deform_attn_forward");
    m.def("flash_deform_attn_backward", &flash_deform_attn_backward, "flash_deform_attn_backward");

    // DEPRECATED:
    // m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
    // m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}