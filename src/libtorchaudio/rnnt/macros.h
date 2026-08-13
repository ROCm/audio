#pragma once

#if defined(USE_CUDA) || defined(USE_ROCM)
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define REDUCE_THREADS 256
#define HOST_AND_DEVICE __host__ __device__
#define FORCE_INLINE __forceinline__
#if defined(USE_ROCM)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#else
#define HOST_AND_DEVICE
#define FORCE_INLINE inline
#endif // USE_CUDA || USE_ROCM
