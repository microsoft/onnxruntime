// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/mtp_moe.h"

#include <cuda_fp8.h>

#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace mtp_moe {
namespace {

constexpr int kChunk = 8;
constexpr int kGateUpRows = 2 * kFfnSize;
constexpr int kWarpsPerBlock = 8;
constexpr int kKChunks = kHiddenSize / kChunk;
constexpr int kPerLane = kKChunks / 32;
constexpr int kDownChunks = kFfnSize / kChunk;
constexpr int kDownPerLane = (kDownChunks + 31) / 32;

__device__ __forceinline__ float RoundBf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

__device__ __forceinline__ float Sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ void UnpackBf16x8(const uint4& packed, float* values) {
  const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[2 * i] = __uint_as_float(words[i] << 16);
    values[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ float DecodeE4M3(uint8_t bits) {
  const __half_raw half_bits = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(bits), __NV_E4M3);
  return __half2float(*reinterpret_cast<const __half*>(&half_bits));
}

__device__ __forceinline__ float DecodeE2M1(uint32_t code) {
  const uint32_t magnitude = code & 7u;
  const uint32_t exponent_mantissa =
      magnitude == 0u ? 0u : ((magnitude == 1u ? 252u : (magnitude + 252u)) << 22);
  return __uint_as_float(((code & 8u) << 28) | exponent_mantissa);
}

__device__ __forceinline__ void UnpackNvfp4(uint32_t packed, uint8_t scale, float* values) {
  const float decoded_scale = DecodeE4M3(scale);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    values[i] = DecodeE2M1((packed >> (4 * i)) & 0xfu) * decoded_scale;
  }
}

__device__ __forceinline__ void MergeTop(float& best, int& best_index,
                                         float candidate, int candidate_index) {
  if (candidate > best || (candidate == best && candidate_index < best_index)) {
    best = candidate;
    best_index = candidate_index;
  }
}

// Mirrors the reference custom op's FP32 lane-wise softmax and stable iterative top-10.
__global__ void RouterKernel(const __nv_bfloat16* __restrict__ logits,
                             float* __restrict__ scratch) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  constexpr int kIterations = kExpertCount / 32;
  float logits_per_lane[kIterations];
  float max_logit = -INFINITY;
#pragma unroll
  for (int i = 0; i < kIterations; ++i) {
    logits_per_lane[i] = __bfloat162float(logits[static_cast<int64_t>(row) * kExpertCount + i * 32 + lane]);
    max_logit = fmaxf(max_logit, logits_per_lane[i]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_logit = fmaxf(max_logit, __shfl_xor_sync(0xffffffffu, max_logit, offset));
  }

  float exponentials[kIterations];
  float denominator = 0.0f;
#pragma unroll
  for (int i = 0; i < kIterations; ++i) {
    exponentials[i] = expf(logits_per_lane[i] - max_logit);
    denominator += exponentials[i];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    denominator += __shfl_xor_sync(0xffffffffu, denominator, offset);
  }

  float probabilities[kIterations];
#pragma unroll
  for (int i = 0; i < kIterations; ++i) {
    probabilities[i] = exponentials[i] / denominator;
  }

  float top_values[kTopK];
  int top_indices[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best = -1.0f;
    int best_index = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < kIterations; ++i) {
      if (probabilities[i] > best) {
        best = probabilities[i];
        best_index = i * 32 + lane;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_value = __shfl_xor_sync(0xffffffffu, best, offset);
      const int other_index = __shfl_xor_sync(0xffffffffu, best_index, offset);
      MergeTop(best, best_index, other_value, other_index);
    }
    top_values[k] = best;
    top_indices[k] = best_index;
#pragma unroll
    for (int i = 0; i < kIterations; ++i) {
      if (best_index == i * 32 + lane) {
        probabilities[i] = -2.0f;
      }
    }
  }

  if (lane == 0) {
    float top_sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      top_sum += top_values[k];
    }
    float* row_scratch = scratch + static_cast<int64_t>(row) * kScratchStride;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      row_scratch[k] = __int_as_float(top_indices[k]);
      row_scratch[kTopK + k] = RoundBf16(top_values[k] / top_sum);
    }
  }
}

__global__ void GateUpKernel(const __nv_bfloat16* __restrict__ input,
                             const uint8_t* __restrict__ gate_up,
                             float* __restrict__ scratch,
                             float global_scale) {
  const int row = blockIdx.z;
  const int slot = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  float* row_scratch = scratch + static_cast<int64_t>(row) * kScratchStride;
  const int expert = __float_as_int(row_scratch[slot]);
  const int output_row = blockIdx.x * kWarpsPerBlock + warp;
  constexpr int kPackedRowBytes = kHiddenSize * 9 / 16;
  const uint4* input_vector = reinterpret_cast<const uint4*>(input + static_cast<int64_t>(row) * kHiddenSize);
  const uint8_t* gate_row = gate_up + (static_cast<int64_t>(expert) * kGateUpRows + output_row) * kPackedRowBytes;
  const uint8_t* up_row =
      gate_up + (static_cast<int64_t>(expert) * kGateUpRows + kFfnSize + output_row) * kPackedRowBytes;

  uint32_t packed_gate[kPerLane];
  uint32_t packed_up[kPerLane];
  uint8_t gate_scales[kPerLane];
  uint8_t up_scales[kPerLane];
  uint4 packed_input[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    const int column_chunk = i * 32 + lane;
    packed_gate[i] = __ldg(reinterpret_cast<const uint32_t*>(gate_row) + column_chunk);
    packed_up[i] = __ldg(reinterpret_cast<const uint32_t*>(up_row) + column_chunk);
    gate_scales[i] = __ldg(gate_row + kHiddenSize / 2 + (column_chunk >> 1));
    up_scales[i] = __ldg(up_row + kHiddenSize / 2 + (column_chunk >> 1));
    packed_input[i] = __ldg(input_vector + column_chunk);
  }

  float gate_accumulator = 0.0f;
  float up_accumulator = 0.0f;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    float gate_values[8], up_values[8], input_values[8];
    UnpackNvfp4(packed_gate[i], gate_scales[i], gate_values);
    UnpackNvfp4(packed_up[i], up_scales[i], up_values);
    UnpackBf16x8(packed_input[i], input_values);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      gate_accumulator = fmaf(gate_values[k], input_values[k], gate_accumulator);
      up_accumulator = fmaf(up_values[k], input_values[k], up_accumulator);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate_accumulator += __shfl_xor_sync(0xffffffffu, gate_accumulator, offset);
    up_accumulator += __shfl_xor_sync(0xffffffffu, up_accumulator, offset);
  }

  if (lane == 0) {
    gate_accumulator *= global_scale;
    up_accumulator *= global_scale;
    const float gate = RoundBf16(gate_accumulator);
    const float up = RoundBf16(up_accumulator);
    const float silu = RoundBf16(gate * Sigmoid(gate));
    auto* hidden = reinterpret_cast<__nv_bfloat16*>(row_scratch + 32);
    hidden[slot * kFfnSize + output_row] = __float2bfloat16(silu * up);
  }
}

__global__ void DownKernel(const uint8_t* __restrict__ down,
                           const float* __restrict__ scratch,
                           __nv_bfloat16* __restrict__ output,
                           float global_scale) {
  const int row = blockIdx.y;
  const int slot = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int output_column = blockIdx.x;
  const float* row_scratch = scratch + static_cast<int64_t>(row) * kScratchStride;
  const int expert = __float_as_int(row_scratch[slot]);
  constexpr int kPackedRowBytes = kFfnSize * 9 / 16;
  const uint8_t* weight_row =
      down + (static_cast<int64_t>(expert) * kHiddenSize + output_column) * kPackedRowBytes;
  const auto* hidden = reinterpret_cast<const __nv_bfloat16*>(row_scratch + 32) + slot * kFfnSize;
  const uint4* packed_hidden = reinterpret_cast<const uint4*>(hidden);
  float accumulator = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < kDownPerLane; ++chunk) {
    const int column_chunk = chunk * 32 + lane;
    if (column_chunk < kDownChunks) {
      const uint32_t packed_weight = __ldg(reinterpret_cast<const uint32_t*>(weight_row) + column_chunk);
      const uint8_t scale = __ldg(weight_row + kFfnSize / 2 + (column_chunk >> 1));
      const uint4 hidden_chunk = __ldg(packed_hidden + column_chunk);
      float weights[8], activations[8];
      UnpackNvfp4(packed_weight, scale, weights);
      UnpackBf16x8(hidden_chunk, activations);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accumulator = fmaf(weights[i], activations[i], accumulator);
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    accumulator += __shfl_xor_sync(0xffffffffu, accumulator, offset);
  }
  __shared__ float weighted[kTopK];
  if (lane == 0) {
    accumulator *= global_scale;
    weighted[slot] = RoundBf16(RoundBf16(accumulator) * row_scratch[kTopK + slot]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int expert_slot = 0; expert_slot < kTopK; ++expert_slot) {
      total += weighted[expert_slot];
    }
    output[static_cast<int64_t>(row) * kHiddenSize + output_column] = __float2bfloat16(total);
  }
}

}  // namespace

Status LaunchRouter(const __nv_bfloat16* router_logits, float* scratch, int rows, cudaStream_t stream) {
  if (rows <= 0) {
    return Status(common::StatusCategory::ONNXRUNTIME, common::StatusCode::INVALID_ARGUMENT,
                  "MtpMoe requires at least one row");
  }
  RouterKernel<<<rows, 32, 0, stream>>>(router_logits, scratch);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::OK()
             : Status(common::StatusCategory::SYSTEM, static_cast<int>(error), cudaGetErrorString(error));
}

Status LaunchNvfp4MtpMoe(const __nv_bfloat16* input, const __nv_bfloat16* router_logits,
                         const uint8_t* gate_up_weights, const uint8_t* down_weights,
                         float gate_up_global_scale, float down_global_scale,
                         float* scratch, __nv_bfloat16* output, int rows, cudaStream_t stream) {
  if (rows <= 0) {
    return Status(common::StatusCategory::ONNXRUNTIME, common::StatusCode::INVALID_ARGUMENT,
                  "MtpMoe requires at least one row");
  }
  RouterKernel<<<rows, 32, 0, stream>>>(router_logits, scratch);
  GateUpKernel<<<dim3(kFfnSize / kWarpsPerBlock, kTopK, rows), kWarpsPerBlock * 32, 0, stream>>>(
      input, gate_up_weights, scratch, gate_up_global_scale);
  DownKernel<<<dim3(kHiddenSize, rows), kTopK * 32, 0, stream>>>(down_weights, scratch, output, down_global_scale);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::OK()
             : Status(common::StatusCategory::SYSTEM, static_cast<int>(error), cudaGetErrorString(error));
}

}  // namespace mtp_moe
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
