// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

// Shape/layout eligibility of LaunchTinyGemm2: 1 <= M <= 64, N * K <= 32M elements, N and K multiples of
// 8 (TMA row pitch), and 16-byte aligned A and B.
bool CanUseTinyGemm2(int64_t m, int64_t n, int64_t k, const void* a, const void* b);

// True when the current device can run the kernel: SM 9.0+ code is present in the binary for this
// device and the ~195 KiB of shared memory per block fits.
bool IsTinyGemm2Supported(const cudaDeviceProp& device_prop);

// C[m, n] = A[m, k] * B[k, n] (row-major, T is half or nv_bfloat16) with the TMA/warp-specialized
// tinygemm2 kernel from TensorRT-LLM, reading B in place. fp32 accumulation, deterministic.
// Launched with programmatic dependent launch so the weight loads overlap the preceding kernel.
template <typename T>
Status LaunchTinyGemm2(cudaStream_t stream, const T* a, const T* b, T* c, int m, int n, int k);

}  // namespace cuda
}  // namespace onnxruntime
