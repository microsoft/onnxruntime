// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/qmoe_int3.h"
#include "contrib_ops/cuda/moe/packed_code.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace onnxruntime::contrib::cuda {
namespace {

__device__ float WarpSum(float value) {
  for (int delta = 16; delta > 0; delta /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  return value;
}

template <typename T, int Bits>
__device__ float Weight(const Int3Projection& projection, size_t row, int logical_k,
                        int block_size, int feature) {
  const size_t row_bytes = (static_cast<size_t>(logical_k) * Bits + 7) / 8;
  const size_t blocks = logical_k / block_size;
  unsigned int code = 0;
  if (projection.layout_version == 1) {
    const auto* planes = reinterpret_cast<const uint32_t*>(projection.weights) +
                         (row * (logical_k / 32) + feature / 32) * Bits;
    for (int bit = 0; bit < Bits; ++bit) {
      const uint32_t plane = planes[bit];
      code |= ((plane >> (feature % 32)) & 1u) << bit;
    }
  } else {
    code = LoadRowLocalPackedCode<Bits>(projection.weights + row * row_bytes, row_bytes, feature);
  }
  int zero = 1 << (Bits - 1);
  if (projection.zeros) {
    const size_t zero_bytes = (blocks * Bits + 7) / 8;
    zero = static_cast<int>(LoadRowLocalPackedCode<Bits>(
        projection.zeros + row * zero_bytes, zero_bytes, feature / block_size));
  }
  const float scale = static_cast<float>(static_cast<const T*>(projection.scales)[row * blocks + feature / block_size]);
  return static_cast<float>(static_cast<T>((static_cast<int>(code) - zero) * scale));
}

template <typename T, int Bits>
__global__ void FusedFc1(Int3MoeParams params) {
  const size_t item = static_cast<size_t>(blockIdx.x) * 4 + threadIdx.x / 32;
  const size_t count = static_cast<size_t>(params.rows) * params.top_k * params.intermediate;
  if (item >= count) return;
  const int lane = threadIdx.x % 32;
  const size_t route = item / params.intermediate;
  const int column = item % params.intermediate;
  const int expert = params.experts[route];
  const size_t weight_row = static_cast<size_t>(expert) * (2 * params.intermediate) + 2 * column;
  const auto* input = static_cast<const T*>(params.input) + (route / params.top_k) * params.hidden;
  float gate = 0;
  float up = 0;
  for (int feature = lane; feature < params.hidden; feature += 32) {
    const float value = static_cast<float>(input[feature]);
    gate += value * Weight<T, Bits>(params.fc1, weight_row, params.hidden, params.block_size, feature);
    up += value * Weight<T, Bits>(params.fc1, weight_row + 1, params.hidden, params.block_size, feature);
  }
  gate = WarpSum(gate);
  up = WarpSum(up);
  if (lane == 0) {
    if (params.fc1.bias) {
      gate += static_cast<float>(static_cast<const T*>(params.fc1.bias)[weight_row]);
      up += static_cast<float>(static_cast<const T*>(params.fc1.bias)[weight_row + 1]);
    }
    gate = fminf(gate, params.limit);
    up = fminf(fmaxf(up, -params.limit), params.limit);
    static_cast<T*>(params.activation)[item] = static_cast<T>(
        (gate / (1.0f + expf(-params.alpha * gate))) * (up + params.beta));
  }
}

template <typename T, int Bits>
__global__ void Fc2AndFinalize(Int3MoeParams params) {
  const size_t item = static_cast<size_t>(blockIdx.x) * 4 + threadIdx.x / 32;
  if (item >= static_cast<size_t>(params.rows) * params.hidden) return;
  const int lane = threadIdx.x % 32;
  const size_t token = item / params.hidden;
  const int column = item % params.hidden;
  float output = 0;
  for (int selected = 0; selected < params.top_k; ++selected) {
    const size_t route = token * params.top_k + selected;
    const size_t weight_row = static_cast<size_t>(params.experts[route]) * params.hidden + column;
    const auto* input = static_cast<const T*>(params.activation) + route * params.intermediate;
    float value = 0;
    for (int feature = lane; feature < params.intermediate; feature += 32) {
      value += static_cast<float>(input[feature]) *
               Weight<T, Bits>(params.fc2, weight_row, params.intermediate, params.block_size, feature);
    }
    value = WarpSum(value);
    if (lane == 0) {
      if (params.fc2.bias) value += static_cast<float>(static_cast<const T*>(params.fc2.bias)[weight_row]);
      value = static_cast<float>(static_cast<T>(value));
      const float scale = params.router_weights
                              ? static_cast<float>(static_cast<const T*>(params.router_weights)[token * params.num_experts + params.experts[route]])
                              : params.routing_scales[route];
      output += value * scale;
    }
  }
  if (lane == 0) static_cast<T*>(params.output)[item] = static_cast<T>(output);
}

template <typename T>
__global__ void CheckContent(Int3Projection projection, size_t rows, int logical_k,
                             int block_size, int* invalid) {
  const size_t blocks = logical_k / block_size;
  const size_t count = rows * blocks;
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    if (projection.scales) {
      const float scale = static_cast<float>(static_cast<const T*>(projection.scales)[index]);
      if (!isfinite(scale) || scale < 0) atomicExch(invalid, 1);
    }
    if (projection.zeros && projection.bits == 3) {
      const size_t bytes = (blocks * 3 + 7) / 8;
      const auto* row = projection.zeros + (index / blocks) * bytes;
      if (LoadRowLocalPackedCode<3>(row, bytes, index % blocks) != 4) atomicExch(invalid, 1);
      const int used_bits = (blocks * 3) % 8;
      if (index % blocks == 0 && used_bits && (row[bytes - 1] >> used_bits) != 0) {
        atomicExch(invalid, 1);
      }
    }
  }
}

template <int Bits>
__global__ void PackBitplanes(const uint8_t* source, uint32_t* destination, size_t tiles, int logical_k) {
  const int lane = threadIdx.x % 32;
  for (size_t tile = static_cast<size_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
       tile < tiles; tile += static_cast<size_t>(gridDim.x) * (blockDim.x / 32)) {
    const size_t row_bytes = static_cast<size_t>(logical_k) * Bits / 8;
    const size_t row = tile / (logical_k / 32);
    const size_t feature = (tile % (logical_k / 32)) * 32 + lane;
    const unsigned int code = LoadRowLocalPackedCode<Bits>(source + row * row_bytes, row_bytes, feature);
    for (int bit = 0; bit < Bits; ++bit) {
      const uint32_t plane = __ballot_sync(0xffffffffu, (code >> bit) & 1u);
      if (lane == 0) destination[tile * Bits + bit] = plane;
    }
  }
}

template <typename T>
cudaError_t LaunchTyped(const Int3MoeParams& params, cudaStream_t stream) {
  const size_t fc1_count = static_cast<size_t>(params.rows) * params.top_k * params.intermediate;
  const size_t fc2_count = static_cast<size_t>(params.rows) * params.hidden;
  const unsigned int fc1_blocks = static_cast<unsigned int>((fc1_count + 3) / 4);
  const unsigned int fc2_blocks = static_cast<unsigned int>((fc2_count + 3) / 4);
  if (params.fc1.bits == 3)
    FusedFc1<T, 3><<<fc1_blocks, 128, 0, stream>>>(params);
  else
    FusedFc1<T, 4><<<fc1_blocks, 128, 0, stream>>>(params);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  if (params.fc2.bits == 3)
    Fc2AndFinalize<T, 3><<<fc2_blocks, 128, 0, stream>>>(params);
  else
    Fc2AndFinalize<T, 4><<<fc2_blocks, 128, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace

cudaError_t PackInt3MoeWeights(const uint8_t* source, uint8_t* destination,
                               size_t rows, int logical_k, int bits, cudaStream_t stream) {
  if (!source || !destination || !rows || logical_k <= 0 || logical_k % 32 ||
      (bits != 3 && bits != 4)) return cudaErrorInvalidValue;
  const size_t tiles = rows * (logical_k / 32);
  if (bits == 3)
    PackBitplanes<3><<<256, 128, 0, stream>>>(source, reinterpret_cast<uint32_t*>(destination), tiles, logical_k);
  else
    PackBitplanes<4><<<256, 128, 0, stream>>>(source, reinterpret_cast<uint32_t*>(destination), tiles, logical_k);
  return cudaGetLastError();
}

cudaError_t LaunchInt3Moe(const Int3MoeParams& params, cudaStream_t stream) {
  if (params.rows == 0) return cudaSuccess;
  if ((params.fc1.bits != 3 && params.fc1.bits != 4) ||
      (params.fc2.bits != 3 && params.fc2.bits != 4) || params.hidden <= 0 ||
      params.intermediate <= 0 || params.top_k <= 0 || params.block_size <= 0 ||
      params.fc1.layout_version < 0 || params.fc1.layout_version > 1 ||
      params.fc2.layout_version < 0 || params.fc2.layout_version > 1 ||
      ((params.fc1.layout_version || params.fc2.layout_version) &&
       (params.hidden % 32 || params.intermediate % 32))) {
    return cudaErrorInvalidValue;
  }
  return params.is_bfloat16 ? LaunchTyped<__nv_bfloat16>(params, stream) : LaunchTyped<half>(params, stream);
}

cudaError_t ValidateInt3MoeContent(const Int3Projection& projection, size_t rows,
                                   int logical_k, int block_size, bool is_bfloat16,
                                   int* invalid, cudaStream_t stream) {
  if (is_bfloat16)
    CheckContent<__nv_bfloat16><<<256, 256, 0, stream>>>(projection, rows, logical_k, block_size, invalid);
  else
    CheckContent<half><<<256, 256, 0, stream>>>(projection, rows, logical_k, block_size, invalid);
  return cudaGetLastError();
}

}  // namespace onnxruntime::contrib::cuda
