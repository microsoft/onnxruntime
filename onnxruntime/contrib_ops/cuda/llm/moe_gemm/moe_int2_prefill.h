// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace onnxruntime::llm::kernels::cutlass_kernels {

struct Int2MoePrefillParams {
  const void* input = nullptr;
  const uint8_t* fc1_weights = nullptr;
  const uint8_t* fc2_weights = nullptr;
  const void* fc1_scales = nullptr;
  const void* fc2_scales = nullptr;
  const void* fc1_bias = nullptr;
  const void* fc2_bias = nullptr;
  const int* selected_experts = nullptr;
  const float* routing_weights = nullptr;
  int* unpermuted_to_permuted = nullptr;
  void* output = nullptr;
  bool is_bf16 = false;
  bool fake_int4_activations = false;
  bool is_nvfp4 = false;
  bool fuse_nvfp4_swiglu = false;
  bool reuse_nvfp4_routing_scratch = false;
  bool tile_nvfp4_down_output = false;
  const float* fc1_global_scales = nullptr;
  const float* fc2_global_scales = nullptr;
  int64_t num_rows = 0;
  int hidden_size = 0;
  int inter_size = 0;
  int block_size = 0;
  int fc1_weight_bits = 0;
  int fc2_weight_bits = 0;
  int num_experts = 0;
  int top_k = 0;
  int sm = 0;
  int multiprocessor_count = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = 0.0f;
  cudaStream_t stream = nullptr;
  int cache_capacity = 0;
  int* cache_metadata = nullptr;
  uint8_t* cache_weights = nullptr;
};

size_t GetInt2MoePrefillWorkspaceSize(const Int2MoePrefillParams& params);
void RunInt2MoePrefill(const Int2MoePrefillParams& params, void* workspace);

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
