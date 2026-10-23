// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace onnxruntime::contrib::cuda {

void LaunchUpdateMoeExpertCounters(const int* ids, const float* scales, size_t route_count,
                                   double* counters, float* selected_probabilities, int64_t* first_routes, size_t expert_count,
                                   double alpha, double probability_weight, cudaStream_t stream);
void LaunchUpdateMoeExpertCounters(const int* ids, const float* scales, size_t route_count,
                                   double* counters, float* selected_probabilities, int64_t* first_routes,
                                   size_t expert_count, size_t route_offset, bool finalize,
                                   double alpha, double probability_weight, cudaStream_t stream);

void LaunchPackMoeCpuRows(const half* input, const int* ids, const float* scales, const int* expert_map,
                          half* packed_input, int* packed_ids, float* packed_scales,
                          int* rows, uint32_t* row_count, uint32_t* completion, size_t num_rows, size_t hidden_size,
                          size_t k, cudaStream_t stream);
void LaunchPackMoeCpuRows(const __nv_bfloat16* input, const int* ids, const float* scales, const int* expert_map,
                          __nv_bfloat16* packed_input, int* packed_ids, float* packed_scales,
                          int* rows, uint32_t* row_count, uint32_t* completion, size_t num_rows, size_t hidden_size,
                          size_t k, cudaStream_t stream);

void LaunchScatterAddMoeCpuOutput(const half* contribution, half* output,
                                  const int* rows, const uint32_t* row_count, size_t num_rows,
                                  size_t hidden_size, cudaStream_t stream);
void LaunchScatterAddMoeCpuOutput(const __nv_bfloat16* contribution, __nv_bfloat16* output,
                                  const int* rows, const uint32_t* row_count, size_t num_rows,
                                  size_t hidden_size, cudaStream_t stream);

void LaunchTransposeMoeExpertWeights(const half* input, half* output,
                                     size_t rows, size_t columns, cudaStream_t stream);
void LaunchTransposeMoeExpertWeights(const __nv_bfloat16* input, __nv_bfloat16* output,
                                     size_t rows, size_t columns, cudaStream_t stream);

void LaunchPackMlasQ4Weights(const uint8_t* input, uint8_t* output,
                             size_t rows, size_t columns, size_t block_size,
                             size_t sub_block_size, bool cutlass_layout, cudaStream_t stream);

void LaunchRemapMoeExpertIndices(const int* expert_indices, int* remapped_expert_indices,
                                 const int* expert_map, size_t count, cudaStream_t stream);

void LaunchAddMoeFp16Output(half* output, const half* contribution,
                            size_t count, cudaStream_t stream);
void LaunchAddMoeBf16Output(__nv_bfloat16* output, const __nv_bfloat16* contribution,
                            size_t count, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
