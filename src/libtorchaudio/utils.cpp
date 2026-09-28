#include <libtorchaudio/utils.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

#ifdef USE_ROCM
#include <rocm-core/rocm_version.h>
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
#if defined(USE_ROCM)
  return static_cast<int64_t>(ROCM_VERSION_MAJOR * 100 + ROCM_VERSION_MINOR);
#elif defined(USE_CUDA)
  return CUDA_VERSION;
#else
  return {};
#endif
}

} // namespace torchaudio
