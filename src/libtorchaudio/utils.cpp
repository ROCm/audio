#include <libtorchaudio/utils.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

namespace torchaudio {

bool is_align_available() {
#ifdef INCLUDE_ALIGN
  return true;
#else
  return false;
#endif
}

std::optional<int64_t> cuda_version() {
#if defined(TORCH_HIP_VERSION)
  // TORCH_HIP_VERSION = {ROCM_VERSION[0] * 100 + ROCM_VERSION[1]}
  return static_cast<int64_t>(TORCH_HIP_VERSION);
#elif defined(USE_CUDA)
  return CUDA_VERSION;
#else
  return {};
#endif
}

} // namespace torchaudio
