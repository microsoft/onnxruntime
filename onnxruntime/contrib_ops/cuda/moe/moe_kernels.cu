// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_kernels.h"
#include "core/providers/cuda/cuda_common.h"

#include <algorithm>
#include <type_traits>

namespace onnxruntime::contrib::cuda {
namespace {

__global__ void UpdateMoeExpertCountersKernel(const int* ids, const float* scales, size_t route_count,
                                              double* counters, float* selected_probabilities, int64_t* first_routes,
                                              size_t expert_count,
                                              double alpha, double probability_weight) {
  const size_t expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= expert_count) {
    return;
  }
  float probability = -1.0f;
  int64_t first_route = -1;
  for (size_t route = 0; route < route_count; ++route) {
    if (ids[route] == static_cast<int>(expert)) {
      if (first_route < 0) {
        first_route = static_cast<int64_t>(route);
      }
      if (probability_weight > 0.0 && (!isfinite(scales[route]) || scales[route] < 0.0f || scales[route] > 1.0f)) {
        probability = nanf("");
        break;
      }
      probability = fmaxf(probability, scales[route]);
      probability = fmaxf(probability, 0.0f);
    }
  }
  selected_probabilities[expert] = probability;
  first_routes[expert] = first_route;
  const double decayed = __dmul_rn(counters[expert], alpha);
  const double contribution = probability >= 0.0f
                                  ? __dadd_rn(1.0 - probability_weight,
                                              __dmul_rn(probability_weight, static_cast<double>(probability)))
                                  : 0.0;
  counters[expert] = __dadd_rn(decayed, __dmul_rn(1.0 - alpha, contribution));
}

template <typename T>
__global__ void PackMoeCpuRowsKernel(const T* input, const int* ids, const float* scales, const int* expert_map,
                                     T* packed_input, int* packed_ids, float* packed_scales,
                                     int* rows, uint32_t* row_count, uint32_t* completion,
                                     size_t num_rows, size_t hidden_size, size_t k) {
  __shared__ int slot;
  for (size_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
    if (threadIdx.x == 0) {
      bool selected = false;
      for (size_t route = row * k; route < (row + 1) * k; ++route) {
        selected |= scales[route] > 0.0f && (expert_map == nullptr || expert_map[ids[route]] < 0);
      }
      slot = selected ? static_cast<int>(atomicAdd(row_count, 1U)) : -1;
      if (slot < 0 && num_rows == 1) {
        // Decode without CPU routes must not wait for a host notification.
        *reinterpret_cast<volatile uint32_t*>(completion) = 1U;
        __threadfence_system();
      }
      if (slot >= 0) {
        rows[slot] = static_cast<int>(row);
        for (size_t i = 0; i < k; ++i) {
          const size_t route = row * k + i;
          const bool cpu = expert_map == nullptr || expert_map[ids[route]] < 0;
          packed_ids[static_cast<size_t>(slot) * k + i] = cpu ? ids[route] : -1;
          packed_scales[static_cast<size_t>(slot) * k + i] = cpu ? scales[route] : 0.0f;
        }
      }
    }
    __syncthreads();
    if (slot >= 0) {
      for (size_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        packed_input[static_cast<size_t>(slot) * hidden_size + column] = input[row * hidden_size + column];
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void ScatterAddMoeCpuOutputKernel(const T* contribution, T* output,
                                             const int* rows, const uint32_t* row_count,
                                             const uint32_t* completion, size_t num_rows, size_t hidden_size) {
  if (*completion != 1U) {
    return;
  }
  const size_t count = static_cast<size_t>(*row_count) * hidden_size;
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
    const size_t row = index / hidden_size;
    const size_t destination = static_cast<size_t>(rows[row]) * hidden_size + index % hidden_size;
    if constexpr (std::is_same_v<T, half>) {
      output[destination] = __hadd(output[destination], contribution[index]);
    } else {
      output[destination] = __float2bfloat16(__bfloat162float(output[destination]) +
                                             __bfloat162float(contribution[index]));
    }
  }
}

template <typename T>
void PackMoeCpuRows(const T* input, const int* ids, const float* scales, const int* expert_map,
                    T* packed_input, int* packed_ids, float* packed_scales, int* rows, uint32_t* row_count,
                    uint32_t* completion,
                    size_t num_rows, size_t hidden_size, size_t k, cudaStream_t stream) {
  CUDA_CALL_THROW(cudaMemsetAsync(row_count, 0, sizeof(uint32_t), stream));
  if (num_rows == 0) {
    return;
  }
  PackMoeCpuRowsKernel<<<static_cast<unsigned int>(std::min<size_t>(num_rows, 65535)), 256, 0, stream>>>(
      input, ids, scales, expert_map, packed_input, packed_ids, packed_scales, rows, row_count,
      completion, num_rows, hidden_size, k);
  CUDA_CALL_THROW(cudaGetLastError());
}

template <typename T>
void ScatterAddMoeCpuOutput(const T* contribution, T* output, const int* rows, const uint32_t* row_count,
                            const uint32_t* completion, size_t num_rows, size_t hidden_size, cudaStream_t stream) {
  const size_t blocks = (num_rows * hidden_size + 255) / 256;
  if (blocks == 0) {
    return;
  }
  ScatterAddMoeCpuOutputKernel<<<static_cast<unsigned int>(std::min<size_t>(blocks, 65535)), 256, 0, stream>>>(
      contribution, output, rows, row_count, completion, num_rows, hidden_size);
  CUDA_CALL_THROW(cudaGetLastError());
}

template <typename T>
__global__ void TransposeMoeExpertWeightsKernel(const T* input, T* output, size_t rows, size_t columns) {
  __shared__ T tile[32][33];
  for (size_t tile_row = blockIdx.y * 32; tile_row < rows; tile_row += gridDim.y * 32) {
    for (size_t tile_column = blockIdx.x * 32; tile_column < columns; tile_column += gridDim.x * 32) {
      const size_t column = tile_column + threadIdx.x;
      for (size_t offset = threadIdx.y; offset < 32; offset += blockDim.y) {
        if (tile_row + offset < rows && column < columns) {
          tile[offset][threadIdx.x] = input[(tile_row + offset) * columns + column];
        }
      }
      __syncthreads();
      const size_t row = tile_row + threadIdx.x;
      for (size_t offset = threadIdx.y; offset < 32; offset += blockDim.y) {
        if (row < rows && tile_column + offset < columns) {
          output[(tile_column + offset) * rows + row] = tile[threadIdx.x][offset];
        }
      }
      __syncthreads();
    }
  }
}

template <typename T>
void TransposeMoeExpertWeights(const T* input, T* output, size_t rows, size_t columns, cudaStream_t stream) {
  ORT_ENFORCE(rows > 0 && columns > 0, "MoE expert weights must have non-empty dimensions.");
  const dim3 grid(static_cast<unsigned int>(std::min<size_t>((columns - 1) / 32 + 1, 65535)),
                  static_cast<unsigned int>(std::min<size_t>((rows - 1) / 32 + 1, 65535)));
  TransposeMoeExpertWeightsKernel<<<grid, dim3(32, 8), 0, stream>>>(input, output, rows, columns);
  CUDA_CALL_THROW(cudaGetLastError());
}

__global__ void RemapMoeExpertIndicesKernel(const int* expert_indices, int* remapped_expert_indices,
                                            const int* expert_map, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const int expert = expert_indices[index];
    remapped_expert_indices[index] = expert >= 0 ? expert_map[expert] : -1;
  }
}

__global__ void AddMoeFp16OutputKernel(half* output, const half* contribution, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __hadd(output[index], contribution[index]);
  }
}

__global__ void AddMoeBf16OutputKernel(__nv_bfloat16* output, const __nv_bfloat16* contribution, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __float2bfloat16(__bfloat162float(output[index]) + __bfloat162float(contribution[index]));
  }
}

}  // namespace

void LaunchUpdateMoeExpertCounters(const int* ids, const float* scales, size_t route_count,
                                   double* counters, float* selected_probabilities, int64_t* first_routes,
                                   size_t expert_count,
                                   double alpha, double probability_weight, cudaStream_t stream) {
  UpdateMoeExpertCountersKernel<<<static_cast<unsigned int>((expert_count + 255) / 256), 256, 0, stream>>>(
      ids, scales, route_count, counters, selected_probabilities, first_routes, expert_count, alpha, probability_weight);
  CUDA_CALL_THROW(cudaGetLastError());
}

#define DEFINE_MOE_CPU_ROUTING(T)                                                                                      \
  void LaunchPackMoeCpuRows(const T* input, const int* ids, const float* scales, const int* expert_map,                \
                            T* packed_input, int* packed_ids, float* packed_scales,                                    \
                            int* rows, uint32_t* row_count, uint32_t* completion, size_t num_rows, size_t hidden_size, \
                            size_t k, cudaStream_t stream) {                                                           \
    PackMoeCpuRows(input, ids, scales, expert_map, packed_input, packed_ids, packed_scales,                            \
                   rows, row_count, completion, num_rows, hidden_size, k, stream);                                     \
  }                                                                                                                    \
  void LaunchScatterAddMoeCpuOutput(const T* contribution, T* output,                                                  \
                                    const int* rows, const uint32_t* row_count, const uint32_t* completion,            \
                                    size_t num_rows, size_t hidden_size, cudaStream_t stream) {                        \
    ScatterAddMoeCpuOutput(contribution, output, rows, row_count, completion, num_rows, hidden_size, stream);          \
  }

DEFINE_MOE_CPU_ROUTING(half)
DEFINE_MOE_CPU_ROUTING(__nv_bfloat16)
#undef DEFINE_MOE_CPU_ROUTING

void LaunchTransposeMoeExpertWeights(const half* input, half* output,
                                     size_t rows, size_t columns, cudaStream_t stream) {
  TransposeMoeExpertWeights(input, output, rows, columns, stream);
}

void LaunchTransposeMoeExpertWeights(const __nv_bfloat16* input, __nv_bfloat16* output,
                                     size_t rows, size_t columns, cudaStream_t stream) {
  TransposeMoeExpertWeights(input, output, rows, columns, stream);
}

void LaunchRemapMoeExpertIndices(const int* expert_indices, int* remapped_expert_indices,
                                 const int* expert_map, size_t count, cudaStream_t stream) {
  constexpr int threads = 256;
  RemapMoeExpertIndicesKernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0, stream>>>(
      expert_indices, remapped_expert_indices, expert_map, count);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchAddMoeFp16Output(half* output, const half* contribution,
                            size_t count, cudaStream_t stream) {
  constexpr int threads = 256;
  AddMoeFp16OutputKernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0, stream>>>(
      output, contribution, count);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchAddMoeBf16Output(__nv_bfloat16* output, const __nv_bfloat16* contribution,
                            size_t count, cudaStream_t stream) {
  constexpr int threads = 256;
  AddMoeBf16OutputKernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0, stream>>>(
      output, contribution, count);
  CUDA_CALL_THROW(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
