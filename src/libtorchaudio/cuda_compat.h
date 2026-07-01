#pragma once

// CUDA/HIP compatibility shim for building libtorchaudio GPU sources under ROCm.
//
// The torch CUDAExtension HIPIFY path does not reliably rewrite every CUDA
// symbol in torchaudio's sources/headers (torchaudio has no whole-tree
// build_amd.py hipify step). Map the CUDA runtime names used by libtorchaudio
// to their HIP equivalents under USE_ROCM so the sources compile with the HIP
// toolchain regardless of HIPIFY coverage. HIP natively supports the <<<>>>
// launch syntax and the warp shuffle intrinsics already guarded in the kernels.
#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// Types
using cudaStream_t = hipStream_t;
using cudaError_t = hipError_t;

// Enums / values
#define cudaSuccess hipSuccess
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Error helpers
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName

// Device / stream / memory runtime API
#define cudaSetDevice hipSetDevice
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif
