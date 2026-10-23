// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace onnxruntime::contrib::cuda {

struct Int3Projection {
  const uint8_t* weights;
  const void* scales;
  const uint8_t* zeros;
  const void* bias;
  int bits;
  int layout_version = 0;
};

struct Int3MoeParams {
  const void* input;
  void* output;
  void* activation;
  const int* experts;
  const float* routing_scales;
  const void* router_weights = nullptr;
  int num_experts = 0;
  Int3Projection fc1;
  Int3Projection fc2;
  int rows;
  int hidden;
  int intermediate;
  int top_k;
  int block_size;
  float alpha;
  float beta;
  float limit;
  bool is_bfloat16;
};

cudaError_t LaunchInt3Moe(const Int3MoeParams& params, cudaStream_t stream);
cudaError_t PackInt3MoeWeights(const uint8_t* source, uint8_t* destination,
                               size_t rows, int logical_k, int bits, cudaStream_t stream);
cudaError_t ValidateInt3MoeContent(const Int3Projection& projection, size_t rows,
                                   int logical_k, int block_size, bool is_bfloat16,
                                   int* invalid, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
