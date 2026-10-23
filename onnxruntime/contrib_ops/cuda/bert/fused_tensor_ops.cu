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
    output[blockIdx.x] = partial_values == nullptr && indices[0] == INT64_MAX ? 0 : indices[0];
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

__device__ __forceinline__ float StableSigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + expf(-value));
  }
  const float exp_value = expf(value);
  return exp_value / (1.0f + exp_value);
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
  const int64_t row_base = row * output_width;
  if ((output_width & 7) == 0 &&
      (reinterpret_cast<uintptr_t>(attention + row_base) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(output + row_base) & 15) == 0 &&
      (residual == nullptr || (reinterpret_cast<uintptr_t>(residual + row_base) & 15) == 0)) {
    const int64_t vec_width = output_width >> 3;
    const uint4* attn_vec = reinterpret_cast<const uint4*>(attention + row_base);
    const uint4* res_vec = residual != nullptr ? reinterpret_cast<const uint4*>(residual + row_base) : nullptr;
    uint4* out_vec = reinterpret_cast<uint4*>(output + row_base);
    for (int64_t v = threadIdx.x; v < vec_width; v += kThreads) {
      const uint4 a_u4 = attn_vec[v];
      const __nv_bfloat16* a_bf = reinterpret_cast<const __nv_bfloat16*>(&a_u4);
      uint4 o_u4;
      __nv_bfloat16* o_bf = reinterpret_cast<__nv_bfloat16*>(&o_u4);
      if (res_vec != nullptr) {
        const uint4 r_u4 = res_vec[v];
        const __nv_bfloat16* r_bf = reinterpret_cast<const __nv_bfloat16*>(&r_u4);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
          const __nv_bfloat16 scaled = __float2bfloat16_rn(__bfloat162float(a_bf[m]) * gate);
          o_bf[m] = __float2bfloat16_rn(__bfloat162float(scaled) + __bfloat162float(r_bf[m]));
        }
      } else {
#pragma unroll
        for (int m = 0; m < 8; ++m) {
          o_bf[m] = __float2bfloat16_rn(__bfloat162float(a_bf[m]) * gate);
        }
      }
      out_vec[v] = o_u4;
    }
    return;
  }
  for (int64_t column = threadIdx.x; column < output_width; column += kThreads) {
    const int64_t index = row_base + column;
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

}  // namespace onnxruntime::contrib::cuda
