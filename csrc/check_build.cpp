/**
 * Minimal template extension used to verify the build pipeline (compiler,
 * includes, PyTorch linkage) before running the full CUDA build.
 * Compiles in seconds; no CUDA code.
 */
#include <torch/extension.h>

bool check_build_ok() {
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("check_build_ok", &check_build_ok, "Verify build pipeline (template)");
}
