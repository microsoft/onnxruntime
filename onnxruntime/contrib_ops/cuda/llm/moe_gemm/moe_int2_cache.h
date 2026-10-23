// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "contrib_ops/cuda/llm/moe_gemm/common.h"

namespace onnxruntime::llm::kernels::moe_gemv {

struct ExpertCacheView {
  const int* slots = nullptr;
  const uint8_t* weights = nullptr;
};

// Metadata contains expert->slot, slot->expert and slot fill requests, in that order.
void PrepareInt2ExpertCache(const uint8_t* fc1, const uint8_t* fc2,
                           uint8_t* cache, int* metadata, const int64_t* expert_offsets,
                           int experts, int slots, size_t fc1_bytes, size_t fc2_bytes,
                           cudaStream_t stream);

struct CachedInt2GemvParams {
  bool bf16 = false;
  bool swiglu = false;
  const void* input = nullptr;
  const uint8_t* weights = nullptr;
  const void* scales = nullptr;
  const void* bias = nullptr;
  void* output = nullptr;
  const int64_t* expert_offsets = nullptr;
  const int* row_experts = nullptr;
  const int* source_rows = nullptr;
  int experts = 0;
  int64_t expanded_rows = 0;
  int64_t rows = 0;
  int64_t n = 0;
  int64_t k = 0;
  int group_size = 0;
  cutlass_kernels::ActivationParams activation;
  ExpertCacheView cache;
  cudaStream_t stream = nullptr;
};

void LaunchCachedInt2Gemv(const CachedInt2GemvParams& params);

}  // namespace onnxruntime::llm::kernels::moe_gemv
