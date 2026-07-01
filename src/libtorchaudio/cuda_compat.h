#pragma once

// CUDA/HIP compatibility shim for building libtorchaudio GPU sources under ROCm.
//
// The torch CUDAExtension HIPIFY path only rewrites source files (.cu/.cpp
// listed as extension sources), not headers. CUDA runtime symbols that appear
// in headers therefore survive un-hipified, so map them to their HIP
// equivalents explicitly here. HIP natively supports the triple-chevron kernel
// launch syntax and the warp shuffle intrinsics used elsewhere; only the
// runtime type/enum/function names need aliasing.
#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

using cudaStream_t = hipStream_t;
using cudaError_t = hipError_t;
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif
