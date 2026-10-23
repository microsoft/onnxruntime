// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "core/common/status.h"

namespace onnxruntime::contrib::cuda {

constexpr size_t kHyperConnectionPackedWeightBytes = 160 * 2560 * 16;
// Exact CTA shared-memory image for C4/H2560/R320. Does not change arithmetic order.
Status LaunchPackHyperConnectionWeight(void* packed, const void* source, bool up, cudaStream_t stream);

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
  // Early weight reads are legal only when all weight inputs are immutable/already ready.
  bool weights_ready = false;
};

template <typename T>
struct HyperConnectionLowRankMixTensors {
  const T* streams = nullptr;
  const T* norm_scale = nullptr;
  const T* down_weight = nullptr;
  const T* up_weight = nullptr;
  const T* inject_weight = nullptr;
  const T* packed_down_weight = nullptr;
  const T* packed_up_weight = nullptr;
  const T* prev_block_output = nullptr;  // prologue inputs; both null or both non-null
  const T* prev_post_gate = nullptr;
  T* pre_output = nullptr;
  T* post_gate = nullptr;
  T* streams_out = nullptr;  // required with prologue; fused paths also write optional pass-through output
};

// Rows up to this value use the fused path; larger inputs use the cuBLAS based path.
constexpr int64_t kHyperConnectionLowRankMixMaxFusedRows = 8;

// True when the fused path supports this problem on the given device. Rows 1/2 use one cooperative
// kernel if supported and fully resident; otherwise they (and rows 3..8) use two kernels.
template <typename T>
bool CanUseFusedHyperConnectionLowRankMix(const cudaDeviceProp& device_prop,
                                          const HyperConnectionLowRankMixConfig& config);

// Number of float elements of scratch memory the fused path needs.
size_t HyperConnectionLowRankMixFusedScratchElements(const HyperConnectionLowRankMixConfig& config);

template <typename T>
Status LaunchHyperConnectionLowRankMixFused(const cudaDeviceProp& device_prop, cudaStream_t stream,
                                            const HyperConnectionLowRankMixConfig& config,
                                            const HyperConnectionLowRankMixTensors<T>& tensors,
                                            float* scratch);

// General path built from cuBLAS GEMMs and elementwise kernels. Scratch buffers (in elements of T):
// hn [rows, C * H], down [rows, R], up [rows, C * H], inject [rows, C].
template <typename T>
Status LaunchHyperConnectionLowRankMixUnfused(cublasHandle_t cublas, cudaStream_t stream,
                                              const HyperConnectionLowRankMixConfig& config,
                                              const HyperConnectionLowRankMixTensors<T>& tensors,
                                              T* hn, T* down, T* up, T* inject);

}  // namespace onnxruntime::contrib::cuda
