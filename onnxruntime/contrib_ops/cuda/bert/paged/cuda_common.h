// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdio>

// TODO: split cuda and hip

#if !defined(__HIP_PLATFORM_AMD__)
#include <cuda_runtime.h>
#include "contrib_ops/cuda/bert/paged/config.h"
#else
#include <hip/hip_runtime.h>
#include "contrib_ops/rocm/bert/paged/config.h"
#endif

#if !defined(__HIP_PLATFORM_AMD__)
#define CUDA_CHECK(expr)                                                                               \
  do {                                                                                                 \
    cudaError_t err = (expr);                                                                          \
    if (err != cudaSuccess) {                                                                          \
      fprintf(stderr, "CUDA Error on %s:%d\n", __FILE__, __LINE__);                                    \
      fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", err, cudaGetErrorString(err)); \
      exit(err);                                                                                       \
    }                                                                                                  \
  } while (0)
#else
#define HIP_CHECK(expr)                                                                              \
  do {                                                                                               \
    hipError_t err = (expr);                                                                         \
    if (err != hipSuccess) {                                                                         \
      fprintf(stderr, "HIP Error on %s:%d\n", __FILE__, __LINE__);                                   \
      fprintf(stderr, "HIP Error Code  : %d\n     Error String: %s\n", err, hipGetErrorString(err)); \
      exit(err);                                                                                     \
    }                                                                                                \
  } while (0)
#define CUDA_CHECK(expr) HIP_CHECK(expr)
#endif

#if defined(__HIPCC__)
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMemsetAsync hipMemsetAsync
#endif

namespace onnxruntime::contrib::paged {

template <typename... Ts>
constexpr bool always_false = false;

#if !defined(__HIP_PLATFORM_AMD__)
using stream_t = cudaStream_t;
#else
using stream_t = hipStream_t;
#endif

#if !defined(__HIP_PLATFORM_AMD__)
using dev_props_ptr = const cudaDeviceProp*;
#else
using dev_props_ptr = const hipDeviceProp*;
#endif


}  // namespace onnxruntime::contrib::paged
