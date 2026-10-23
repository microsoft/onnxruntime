// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/fused_tensor_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>
#include <cmath>
#include <limits>

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/mtp_moe_kernels.cuh"

namespace onnxruntime::contrib::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kArgmaxThreads = 256;

__global__ void MtpProjectionAddKernel(const __nv_bfloat16* hidden, const __nv_bfloat16* embedding,
                                       __nv_bfloat16* output, int64_t count, int64_t width, int64_t branches) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const int64_t embedding_index = i / (branches * width) * width + i % width;
    output[i] = __float2bfloat16_rn(__bfloat162float(hidden[i]) + __bfloat162float(embedding[embedding_index]));
  }
}

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
                                int64_t row_size2, int64_t rows) {
  const int which = blockIdx.y;
  const uint16_t* input = which == 0 ? input0 : (which == 1 ? input1 : input2);
  uint16_t* output = which == 0 ? output0 : (which == 1 ? output1 : output2);
  const int64_t row_size = which == 0 ? row_size0 : (which == 1 ? row_size1 : row_size2);
  const int64_t index = selection[0];
  const int64_t selected_row = rows == 2 ? (index != 0 ? 1 : 0)
                                        : (index < 0 ? 0 : (index >= rows ? rows - 1 : index));
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

__global__ void DotProductGateKernel(
    const __nv_bfloat16* key, const __nv_bfloat16* query, const __nv_bfloat16* value,
    __nv_bfloat16* output, int64_t width, int64_t value_repeat, float divisor, float epsilon) {
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage reduction;
  __shared__ float gate;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * width;
  float dot = 0.0f;
  for (int64_t k = threadIdx.x; k < width; k += kThreads) {
    dot += __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(key[base + k]) * __bfloat162float(query[base + k])));
  }
  const float total = Reduce(reduction).Sum(dot);
  if (threadIdx.x == 0) {
    const auto round = [](float x) { return __bfloat162float(__float2bfloat16_rn(x)); };
    const float scaled = round(round(total) / round(divisor));
    const float sign = scaled > 0 ? 1.0f : (scaled < 0 ? -1.0f : scaled);
    const float root = round(sqrtf(fmaxf(fabsf(scaled), round(epsilon))));
    using namespace onnxruntime::cuda;
    const BFloat16 a(round(sign * root));
    gate = static_cast<float>(a > BFloat16(0.0f)
                                 ? BFloat16(1.0f) / (BFloat16(1.0f) + _Exp(-a))
                                 : BFloat16(1.0f) - BFloat16(1.0f) / (BFloat16(1.0f) + _Exp(a)));
  }
  __syncthreads();
  for (int64_t k = threadIdx.x; k < width; k += kThreads) {
    const int64_t value_base = static_cast<int64_t>(blockIdx.x) / value_repeat * width;
    output[base + k] = __float2bfloat16_rn(__bfloat162float(value[value_base + k]) * gate);
  }
}

__global__ void RMSNormRotaryKernel(
    const __nv_bfloat16* input, const float* scale, const int64_t* positions,
    const __nv_bfloat16* cosine, const __nv_bfloat16* sine, __nv_bfloat16* output,
    float* normalized, int heads, int sequence, int input_stride, int rotary_dim,
    int cache_rows, bool scalar_position, float epsilon, bool dense_prefill, bool fp32_rotary) {
  constexpr int threads = 64;
  constexpr int width = 256;
  __shared__ float partial[2];
  __shared__ __nv_bfloat16 values[width];
  const int group = blockIdx.x;
  const int tid = threadIdx.x;
  const int token = group / heads;
  const auto* x = input + static_cast<int64_t>(group) * input_stride;
  float v[4];
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = __bfloat162float(x[tid + i * threads]);
  // Match reduce_square_matrix_columns' four strided values and two-warp shuffle tree.
#pragma unroll
  for (int i = 0; i < 4; ++i) sum += v[i] * v[i];
#pragma unroll
  for (int step = 16; step > 0; step >>= 1) sum += __shfl_down_sync(0xffffffff, sum, step);
  if ((tid & 31) == 0) partial[tid / 32] = sum;
  __syncthreads();
  const float total = partial[0] + partial[1];
  const float rms = sqrtf(__fadd_rn(total / float(width), epsilon));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int column = tid + i * threads;
    const float norm = dense_prefill ? __fmul_rn(v[i], __fdiv_rn(1.f, rms)) : v[i] / rms;
    const auto rounded = __float2bfloat16_rn(__fmul_rn(norm, scale[column]));
    values[column] = rounded;
    if (normalized) normalized[static_cast<int64_t>(group) * width + column] = __bfloat162float(rounded);
  }
  __syncthreads();
  const int64_t base = positions[scalar_position ? 0 : token];
  const bool valid = scalar_position ? base >= 0 && base <= static_cast<int64_t>(cache_rows) - sequence
                                     : base >= 0 && base < cache_rows;
  const int64_t position = valid && scalar_position ? base + token % sequence : base;
  const int half = rotary_dim / 2;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int column = tid + i * threads;
    auto value = values[column];
    if (column < rotary_dim && valid) {
      const int partner = (column + half) % rotary_dim;
      const int64_t cache = position * half + column % half;
      if (!dense_prefill && fp32_rotary) {
        const float a = __bfloat162float(value), b = __bfloat162float(values[partner]);
        value = __float2bfloat16_rn(__fadd_rn(
            __fmul_rn(a, __bfloat162float(cosine[cache])),
            __fmul_rn(column < half ? -b : b, __bfloat162float(sine[cache]))));
      } else {
        const __nv_bfloat16 sign = __float2bfloat16_rn(column < half ? -1.f : 1.f);
        value = value * cosine[cache] + sign * values[partner] * sine[cache];
      }
    }
    output[static_cast<int64_t>(group) * width + column] = value;
  }
}

__device__ __forceinline__ float RoundedBf16Sigmoid(float value, bool sigmoid_in_fp32) {
  if (sigmoid_in_fp32) {
    return __bfloat162float(__float2bfloat16_rn(StableSigmoid(value)));
  }
  using namespace onnxruntime::cuda;
  const BFloat16 a(value);
  return static_cast<float>(a > BFloat16(0.0f)
                               ? BFloat16(1.0f) / (BFloat16(1.0f) + _Exp(-a))
                               : BFloat16(1.0f) - BFloat16(1.0f) / (BFloat16(1.0f) + _Exp(a)));
}

__device__ float ProjectedRowGate(const __nv_bfloat16* activation, const __nv_bfloat16* weight,
                                int64_t row, int64_t input_width, bool sigmoid_in_fp32) {
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage reduction;
  __shared__ float gate;
  float dot = 0.0f;
  for (int64_t k = threadIdx.x; k < input_width; k += kThreads) {
    dot = fmaf(__bfloat162float(activation[row * input_width + k]),
               __bfloat162float(weight[k]), dot);
  }
  const float total = Reduce(reduction).Sum(dot);
  if (threadIdx.x == 0) {
    const float value = __bfloat162float(__float2bfloat16_rn(total));
    gate = RoundedBf16Sigmoid(value, sigmoid_in_fp32);
  }
  __syncthreads();
  return gate;
}

__global__ void PrepareSharedExpertKernel(
    const __nv_bfloat16* up, const __nv_bfloat16* silu_gate, const __nv_bfloat16* activation,
    const __nv_bfloat16* weight, __nv_bfloat16* prepared, float* row_gates,
    int width, int input_width) {
  const int row = blockIdx.x;
  const float gate = ProjectedRowGate(activation, weight, row, input_width, false);
  if (threadIdx.x == 0) row_gates[row] = gate;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * width + column;
    const float value = __bfloat162float(silu_gate[index]);
    const auto silu = __float2bfloat16_rn(value * RoundedBf16Sigmoid(value, false));
    prepared[index] = __float2bfloat16_rn(__bfloat162float(up[index]) * __bfloat162float(silu));
  }
}

__global__ void ProjectedSigmoidGateMulKernel(
    const __nv_bfloat16* attention, const __nv_bfloat16* activation,
    const __nv_bfloat16* weight, const __nv_bfloat16* residual, __nv_bfloat16* output,
    int64_t input_width, int64_t output_width, bool sigmoid_in_fp32) {
  const int64_t row = blockIdx.x;
  const float gate = ProjectedRowGate(activation, weight, row, input_width, sigmoid_in_fp32);
  for (int64_t column = threadIdx.x; column < output_width; column += kThreads) {
    const int64_t index = row * output_width + column;
    __nv_bfloat16 value = __float2bfloat16_rn(__bfloat162float(attention[index]) * gate);
    if (residual != nullptr) {
      value = __float2bfloat16_rn(__bfloat162float(value) + __bfloat162float(residual[index]));
    }
    output[index] = value;
  }
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
                       const int64_t* row_sizes, int64_t rows) {
  const dim3 grid(8, 3);
  RowSelectKernel<<<grid, kThreads, 0, stream>>>(
      selection, inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], outputs[2],
      row_sizes[0], row_sizes[1], row_sizes[2], rows);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchDotProductGate(cudaStream_t stream, const uint16_t* key, const uint16_t* query,
                            const uint16_t* value, uint16_t* output, int64_t groups,
                            int64_t width, int64_t value_repeat, float divisor, float epsilon) {
  ORT_RETURN_IF_NOT(groups >= 0 && groups <= std::numeric_limits<int>::max(),
                    "DotProductGate exceeds CUDA grid limits");
  if (groups == 0) return Status::OK();
  DotProductGateKernel<<<static_cast<int>(groups), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(key), reinterpret_cast<const __nv_bfloat16*>(query),
      reinterpret_cast<const __nv_bfloat16*>(value), reinterpret_cast<__nv_bfloat16*>(output),
      width, value_repeat, divisor, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchPackedMtpMoe(cudaStream_t stream, const uint16_t* input, const uint16_t* router,
                          const uint8_t* gate_up, const uint8_t* down, uint16_t* output,
                          float* scratch, int rows, float gate_up_scale, float down_scale) {
  if (rows == 0) return Status::OK();
  mtp_moe::RouterKernel<<<rows, 32, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(router), scratch);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  mtp_moe::GateUpQuantKernel<2><<<dim3(40, 10, rows), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input), gate_up, scratch, gate_up_scale);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  mtp_moe::DownQuantKernel<2><<<dim3(640, rows), 320, 0, stream>>>(
      down, scratch, reinterpret_cast<__nv_bfloat16*>(output), down_scale);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchMtpProjectionAdd(cudaStream_t stream, const uint16_t* hidden, const uint16_t* embedding,
                              uint16_t* output, int64_t count, int64_t width, int64_t branches) {
  if (count == 0) return Status::OK();
  const int64_t blocks = (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(blocks <= std::numeric_limits<int>::max(), "MtpInputProjection exceeds CUDA grid limits");
  MtpProjectionAddKernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden), reinterpret_cast<const __nv_bfloat16*>(embedding),
      reinterpret_cast<__nv_bfloat16*>(output), count, width, branches);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchRMSNormRotary(cudaStream_t stream, const uint16_t* input, const float* scale,
                          const int64_t* positions, const uint16_t* cosine, const uint16_t* sine,
                          uint16_t* output, float* normalized, int groups, int heads, int sequence,
                          int input_stride, int rotary_dim, int cache_rows, bool scalar_position,
                          float epsilon, bool dense_prefill, bool fp32_rotary) {
  ORT_RETURN_IF_NOT(groups > 1 && heads > 0 && sequence > 0 && input_stride >= 256 &&
                        rotary_dim > 0 && rotary_dim <= 256 && rotary_dim % 2 == 0 && cache_rows > 0 &&
                        (dense_prefill || fp32_rotary),
                    "RMSNormRotary fast path requires valid dimensions and width 256");
  RMSNormRotaryKernel<<<groups, 64, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input), scale, positions,
      reinterpret_cast<const __nv_bfloat16*>(cosine), reinterpret_cast<const __nv_bfloat16*>(sine),
      reinterpret_cast<__nv_bfloat16*>(output), normalized, heads, sequence, input_stride,
      rotary_dim, cache_rows, scalar_position, epsilon, dense_prefill, fp32_rotary);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchPrepareSharedExpert(cudaStream_t stream, const uint16_t* up, const uint16_t* silu_gate,
                                const uint16_t* activation, const uint16_t* weight,
                                uint16_t* prepared, float* row_gates,
                                int rows, int width, int input_width) {
  ORT_RETURN_IF_NOT(rows >= 0 && width > 0 && input_width > 0,
                    "Shared expert preparation requires nonnegative rows and positive widths");
  if (rows == 0) return Status::OK();
  PrepareSharedExpertKernel<<<rows, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(up), reinterpret_cast<const __nv_bfloat16*>(silu_gate),
      reinterpret_cast<const __nv_bfloat16*>(activation), reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<__nv_bfloat16*>(prepared), row_gates, width, input_width);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchProjectedSigmoidGateMul(cudaStream_t stream, const uint16_t* attention,
                                     const uint16_t* activation, const uint16_t* weight,
                                     const uint16_t* residual, uint16_t* output,
                                     int64_t rows, int64_t input_width, int64_t output_width,
                                     bool sigmoid_in_fp32) {
  ORT_RETURN_IF_NOT(rows >= 0 && rows <= std::numeric_limits<int>::max() &&
                        input_width > 0 && output_width > 0,
                    "SigmoidGateMul projected gate dimensions exceed CUDA limits");
  if (rows == 0) return Status::OK();
  ProjectedSigmoidGateMulKernel<<<static_cast<int>(rows), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(attention),
      reinterpret_cast<const __nv_bfloat16*>(activation),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<const __nv_bfloat16*>(residual), reinterpret_cast<__nv_bfloat16*>(output),
      input_width, output_width, sigmoid_in_fp32);
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
#define LAUNCH_GATE(BROADCAST, RESIDUAL, SILU) \
  SigmoidGateMulKernel<BROADCAST, RESIDUAL, SILU><<<static_cast<int>(block_count), kThreads, 0, stream>>>( \
      reinterpret_cast<const __nv_bfloat16*>(attention), \
      reinterpret_cast<const __nv_bfloat16*>(projection), \
      reinterpret_cast<__nv_bfloat16*>(output), count, head_dim, heads, q_stride, \
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
