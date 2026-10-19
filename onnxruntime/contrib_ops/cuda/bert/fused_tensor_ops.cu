// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/fused_tensor_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <limits>

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime::contrib::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kArgmaxThreads = 256;

__device__ __forceinline__ float ToFloat(float value) { return value; }
__device__ __forceinline__ float ToFloat(half value) { return __half2float(value); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T>
__global__ void ArgmaxRowsKernel(const T* input, const int64_t* input_indices,
                                 int64_t width, int64_t partitions, float* partial_values,
                                 int64_t* output) {
  __shared__ float values[kArgmaxThreads];
  __shared__ int64_t indices[kArgmaxThreads];
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x / partitions;
  const int64_t partition = blockIdx.x % partitions;
  const int64_t end = partitions == 1 ? width : min(width, (partition + 1) * 4096);
  float best_value = -CUDART_INF_F;
  int64_t best_index = INT64_MAX;
  for (int64_t col = partition * 4096 + tid; col < end; col += blockDim.x) {
    const float value = ToFloat(input[row * width + col]);
    const int64_t index = input_indices == nullptr ? col : input_indices[row * width + col];
    if (value > best_value || (value == best_value && index < best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_value = values[tid + stride];
      const int64_t other_index = indices[tid + stride];
      if (other_value > values[tid] ||
          (other_value == values[tid] && other_index < indices[tid])) {
        values[tid] = other_value;
        indices[tid] = other_index;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = indices[0];
    if (partial_values != nullptr) partial_values[blockIdx.x] = values[0];
  }
}

template <typename T>
Status LaunchTypedArgmax(cudaStream_t stream, const T* input, int64_t rows,
                         int64_t width, int64_t* output,
                         float* partial_values, int64_t* partial_indices) {
  if (rows == 0) return Status::OK();
  const int64_t partitions = GetArgMaxLastAxisPartitions(width);
  ORT_RETURN_IF_NOT(rows <= std::numeric_limits<unsigned>::max() / partitions,
                    "ArgMaxLastAxis has too many rows for a CUDA grid");
  ArgmaxRowsKernel<<<static_cast<unsigned>(rows * partitions), kArgmaxThreads, 0, stream>>>(
      input, nullptr, width, partitions, partitions == 1 ? nullptr : partial_values,
      partitions == 1 ? output : partial_indices);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  if (partitions > 1) {
    ArgmaxRowsKernel<<<static_cast<unsigned>(rows), kArgmaxThreads, 0, stream>>>(
        partial_values, partial_indices, partitions, 1, nullptr, output);
  }
  return CUDA_CALL(cudaGetLastError());
}

__global__ void EmbedNormKernel(const int64_t* ids, const uint16_t* table,
                                const float* weight, float epsilon,
                                uint16_t* output, int64_t rows, int64_t hidden_size) {
  __shared__ float sums[512];
  const int64_t row_index = blockIdx.x;
  const int64_t token_id = ids[row_index];
  const uint16_t* embedding = table + token_id * hidden_size;
  float sum = 0.0f;
  for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    const float value = __uint_as_float(static_cast<uint32_t>(embedding[col]) << 16);
    sum += value * value;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float scale = 1.0f / sqrtf(sums[0] / static_cast<float>(hidden_size) + epsilon);
  for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    const float value = __uint_as_float(static_cast<uint32_t>(embedding[col]) << 16);
    const float normalized = (value * scale) * weight[col];
    output[row_index * hidden_size + col] =
        __bfloat16_as_ushort(__float2bfloat16_rn(normalized));
  }
  (void)rows;
}

__global__ void RowSelectKernel(const int64_t* selection,
                                const uint16_t* input0, const uint16_t* input1,
                                const uint16_t* input2, uint16_t* output0,
                                uint16_t* output1, uint16_t* output2,
                                int64_t row_size0, int64_t row_size1,
                                int64_t row_size2) {
  const int which = blockIdx.y;
  const uint16_t* input = which == 0 ? input0 : (which == 1 ? input1 : input2);
  uint16_t* output = which == 0 ? output0 : (which == 1 ? output1 : output2);
  const int64_t row_size = which == 0 ? row_size0 : (which == 1 ? row_size1 : row_size2);
  const int64_t selected_row = selection[0] != 0 ? 1 : 0;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < row_size; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    output[i] = input[selected_row * row_size + i];
  }
}

__device__ __forceinline__ float StableSigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + expf(-value));
  }
  const float exp_value = expf(value);
  return exp_value / (1.0f + exp_value);
}

template <bool BroadcastGate, bool HasResidual, bool SiluGate = false>
__global__ void SigmoidGateMulKernel(const __nv_bfloat16* attention,
                                     const __nv_bfloat16* projection,
                                     __nv_bfloat16* output, int64_t count,
                                     int head_dim, int heads, int q_stride,
                                     int64_t token_stride, int64_t q_offset,
                                     int64_t gate_row_width, const __nv_bfloat16* residual,
                                     bool sigmoid_in_fp32) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int64_t gate_index;
  if constexpr (SiluGate) {
    gate_index = i;
  } else if constexpr (BroadcastGate) {
    gate_index = i / gate_row_width;
  } else {
    const int64_t row = i / head_dim;
    const int64_t token = row / heads;
    const int head = static_cast<int>(row % heads);
    const int dim = static_cast<int>(i % head_dim);
    gate_index = token * token_stride + q_offset +
                 static_cast<int64_t>(head) * q_stride + head_dim + dim;
  }
  const float gate = __bfloat162float(projection[gate_index]);
  __nv_bfloat16 sigmoid_bf16;
  if (sigmoid_in_fp32) {
    sigmoid_bf16 = __float2bfloat16_rn(StableSigmoid(gate));
  } else {
    using namespace onnxruntime::cuda;
    const BFloat16 a(gate);
    // Match CUDA Sigmoid's input-type intermediates, including its negative-input subtraction.
    const BFloat16 value = a > BFloat16(0.0f)
                               ? BFloat16(1.0f) / (BFloat16(1.0f) + _Exp(-a))
                               : BFloat16(1.0f) - BFloat16(1.0f) / (BFloat16(1.0f) + _Exp(a));
    sigmoid_bf16 = __float2bfloat16_rn(static_cast<float>(value));
  }
  if constexpr (SiluGate) {
    sigmoid_bf16 = __float2bfloat16_rn(gate * __bfloat162float(sigmoid_bf16));
  }
  __nv_bfloat16 value = __float2bfloat16_rn(
      __bfloat162float(attention[i]) * __bfloat162float(sigmoid_bf16));
  if constexpr (HasResidual) {
    value = __float2bfloat16_rn(__bfloat162float(value) + __bfloat162float(residual[i]));
  }
  output[i] = value;
}

__device__ __forceinline__ float DecodeE4M3FN(uint8_t value) {
  const float sign = (value & 0x80u) != 0 ? -1.0f : 1.0f;
  const int exponent = (value >> 3) & 0x0f;
  const int mantissa = value & 0x07;
  if (exponent == 0) {
    return sign * ldexpf(static_cast<float>(mantissa), -9);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return CUDART_NAN_F;
  }
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                       exponent - 7);
}

__global__ void DequantizeLinearFp8Bf16Kernel(const uint8_t* input,
                                              const uint8_t* zero_point,
                                              const float* scales, uint16_t* output,
                                              int64_t count, int64_t axis_size,
                                              int64_t inner_size,
                                              bool scalar_zero_point) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float value = DecodeE4M3FN(input[index]);
  const int64_t scale_index = axis_size == 1 ? 0 : (index / inner_size) % axis_size;
  if (zero_point != nullptr) {
    value -= DecodeE4M3FN(zero_point[scalar_zero_point ? 0 : scale_index]);
  }
  value *= scales[scale_index];
  output[index] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ void UntileHcBf16Kernel(const uint16_t* input, uint16_t* output, int kind) {
  constexpr int64_t kVectorCount = 204800;
  constexpr int64_t kElementCount = kVectorCount * 16;
  const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= kElementCount) return;
  const int64_t vector_index = element / 16;
  const int64_t lane = element % 16;
  int64_t destination_vector;
  if (kind == 0) {
    const int64_t rp = vector_index & 127;
    const int64_t t1 = vector_index >> 7;
    const int64_t i = t1 % 10;
    const int64_t t2 = t1 / 10;
    const int64_t j = t2 % 20;
    const int64_t s = t2 / 20;
    destination_vector = (((s * 10 + i) * 128 + rp) * 20 + j);
  } else {
    const int64_t c = vector_index & 3;
    const int64_t t1 = vector_index >> 2;
    const int64_t rp = t1 & 31;
    const int64_t t2 = t1 >> 5;
    const int64_t i = t2 % 10;
    const int64_t hb = t2 / 10;
    destination_vector = (((i * 32 + rp) * 4 + c) * 160 + hb);
  }
  output[destination_vector * 16 + lane] = input[element];
}

}  // namespace

Status LaunchArgMaxLastAxis(cudaStream_t stream, const void* input, int input_type,
                            int64_t rows, int64_t width, int64_t* output,
                            float* partial_values, int64_t* partial_indices) {
  ORT_RETURN_IF_NOT(rows >= 0 && width > 0,
                    "ArgMaxLastAxis requires a non-negative row count and positive vocabulary dimension");
  if (rows == 0) return Status::OK();
  switch (input_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return LaunchTypedArgmax(stream, static_cast<const float*>(input), rows, width, output,
                               partial_values, partial_indices);
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return LaunchTypedArgmax(stream, static_cast<const half*>(input), rows, width, output,
                               partial_values, partial_indices);
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return LaunchTypedArgmax(stream, static_cast<const __nv_bfloat16*>(input), rows, width, output,
                               partial_values, partial_indices);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ArgMaxLastAxis input type is not supported");
  }
}

Status LaunchEmbeddingRMSNorm(cudaStream_t stream, const int64_t* ids, const uint16_t* table,
                              const float* weight, float epsilon, uint16_t* output,
                              int64_t rows, int64_t vocab_size, int64_t hidden_size) {
  ORT_RETURN_IF_NOT(hidden_size > 0 && vocab_size > 0,
                    "EmbeddingRMSNorm requires non-empty table dimensions");
  ORT_RETURN_IF_NOT(rows <= std::numeric_limits<unsigned>::max(),
                    "EmbeddingRMSNorm has too many rows for a CUDA grid");
  if (rows == 0) return Status::OK();
  EmbedNormKernel<<<static_cast<unsigned>(rows), 512, 0, stream>>>(
      ids, table, weight, epsilon, output, rows, hidden_size);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchRowSelect(cudaStream_t stream, const int64_t* selection,
                       const uint16_t* const* inputs, uint16_t* const* outputs,
                       const int64_t* row_sizes) {
  const dim3 grid(8, 3);
  RowSelectKernel<<<grid, kThreads, 0, stream>>>(
      selection, inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], outputs[2],
      row_sizes[0], row_sizes[1], row_sizes[2]);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchSigmoidGateMul(cudaStream_t stream, const uint16_t* attention,
                            const uint16_t* projection, uint16_t* output, int64_t count,
                            int head_dim, int heads, int q_stride,
                            int64_t token_stride, int64_t q_offset,
                            int64_t gate_row_width, const uint16_t* residual, bool sigmoid_in_fp32,
                            bool silu_gate) {
  if (count == 0) return Status::OK();
  const int64_t block_count = (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(block_count <= std::numeric_limits<int>::max(),
                    "SigmoidGateMul requires too many CUDA blocks");
#define LAUNCH_GATE(BROADCAST, RESIDUAL, SILU)                                                             \
  SigmoidGateMulKernel<BROADCAST, RESIDUAL, SILU><<<static_cast<int>(block_count), kThreads, 0, stream>>>( \
      reinterpret_cast<const __nv_bfloat16*>(attention),                                                   \
      reinterpret_cast<const __nv_bfloat16*>(projection),                                                  \
      reinterpret_cast<__nv_bfloat16*>(output), count, head_dim, heads, q_stride,                          \
      token_stride, q_offset, gate_row_width, reinterpret_cast<const __nv_bfloat16*>(residual), sigmoid_in_fp32)
  if (silu_gate) {
    if (residual != nullptr) {
      LAUNCH_GATE(false, true, true);
    } else {
      LAUNCH_GATE(false, false, true);
    }
  } else if (gate_row_width > 0) {
    if (residual != nullptr) {
      LAUNCH_GATE(true, true, false);
    } else {
      LAUNCH_GATE(true, false, false);
    }
  } else if (residual != nullptr) {
    LAUNCH_GATE(false, true, false);
  } else {
    LAUNCH_GATE(false, false, false);
  }
#undef LAUNCH_GATE
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchDequantizeLinearFp8Bf16(cudaStream_t stream, const uint8_t* input,
                                     const uint8_t* zero_point, const float* scales,
                                     uint16_t* output, int64_t count,
                                     int64_t axis_size, int64_t inner_size,
                                     bool scalar_zero_point) {
  if (count == 0) return Status::OK();
  const int64_t block_count = (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(block_count <= std::numeric_limits<int>::max(),
                    "DequantizeLinear FP8/BF16 requires too many CUDA blocks");
  DequantizeLinearFp8Bf16Kernel<<<static_cast<int>(block_count), kThreads, 0, stream>>>(
      input, zero_point, scales, output, count, axis_size, inner_size, scalar_zero_point);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchUntileHyperConnectionWeights(cudaStream_t stream, const uint16_t* input,
                                          uint16_t* output, int kind) {
  constexpr int64_t kElementCount = 204800ll * 16;
  constexpr int threads = 256;
  constexpr int blocks = static_cast<int>((kElementCount + threads - 1) / threads);
  UntileHcBf16Kernel<<<blocks, threads, 0, stream>>>(input, output, kind);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
