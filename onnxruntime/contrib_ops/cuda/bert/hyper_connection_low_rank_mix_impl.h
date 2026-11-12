// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "core/common/status.h"

namespace onnxruntime::contrib::cuda {

struct HyperConnectionLowRankMixConfig {
  int64_t rows = 0;
  int branches = 0;  // C
  int hidden = 0;    // H, width of one branch; the streams have C * H features
  int rank = 0;      // R, width of the low-rank bottleneck
  float epsilon = 1e-5f;
  float reduction_scale = 1.0f;
  float silu_alpha = 1.0f;
  float gate_div = 1.0f;
  float gate_mul = 1.0f;
};

template <typename T>
struct HyperConnectionLowRankMixTensors {
  const T* streams = nullptr;
  const T* norm_scale = nullptr;
  const T* down_weight = nullptr;
  const T* up_weight = nullptr;
  const T* inject_weight = nullptr;
  const T* prev_block_output = nullptr;  // prologue inputs; both null or both non-null
  const T* prev_post_gate = nullptr;
  T* pre_output = nullptr;
  T* post_gate = nullptr;
  T* streams_out = nullptr;  // optional on the unfused path; required on the fused path when the prologue is present
};

// Rows up to this value use the two-kernel fused path; larger inputs use the cuBLAS based path.
constexpr int64_t kHyperConnectionLowRankMixMaxFusedRows = 8;

// True when the fused two-kernel path supports this problem on the given device.
template <typename T>
bool CanUseFusedHyperConnectionLowRankMix(const cudaDeviceProp& device_prop,
                                          const HyperConnectionLowRankMixConfig& config);

// Number of float elements of scratch memory the fused path needs.
size_t HyperConnectionLowRankMixFusedScratchElements(const HyperConnectionLowRankMixConfig& config);

// Programmatic dependent launch is enabled on sm_90+ only when the weight inputs are constant initializers.
template <typename T>
Status LaunchHyperConnectionLowRankMixFused(const cudaDeviceProp& device_prop, cudaStream_t stream,
                                            const HyperConnectionLowRankMixConfig& config,
                                            const HyperConnectionLowRankMixTensors<T>& tensors,
                                            float* scratch, bool weights_are_constant = false);

// General path built from cuBLAS GEMMs and elementwise kernels. Scratch buffers (in elements of T):
// hn [rows, C * H], down [rows, R], up [rows, C * H].
template <typename T>
Status LaunchHyperConnectionLowRankMixUnfused(cublasHandle_t cublas, cudaStream_t stream,
                                              const HyperConnectionLowRankMixConfig& config,
                                              const HyperConnectionLowRankMixTensors<T>& tensors,
                                              T* hn, T* down, T* up);

}  // namespace onnxruntime::contrib::cuda
