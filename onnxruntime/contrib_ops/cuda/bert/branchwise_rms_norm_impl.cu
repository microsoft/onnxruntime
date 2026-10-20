// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/branchwise_rms_norm_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <limits>
#include <type_traits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/reduction/reduction_functions.h"

namespace onnxruntime::contrib::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kFloat = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
constexpr int kFloat16 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;

__device__ __forceinline__ float LoadScale(const void* data, int64_t index, int element_type) {
  if (element_type == kFloat) {
    return static_cast<const float*>(data)[index];
  }
  if (element_type == kFloat16) {
    return __half2float(static_cast<const half*>(data)[index]);
  }
  return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
}

template <typename T>
__global__ void BranchwiseSquareKernel(const T* x, float* squares, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const float value = to_float<T>(x[index]);
    squares[index] = __fmul_rn(value, value);
  }
}

template <typename T, bool Packed>
__global__ void BranchwiseStandardEpilogueKernel(
    const T* x, const void* scale, int scale_type, const float* sums, T* y, float* widened,
    int64_t count, int branches, int hidden, bool shared_scale, float epsilon, int input_row_stride) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const int64_t group = index / hidden;
    const int64_t scale_index = shared_scale ? index % hidden : index % (static_cast<int64_t>(branches) * hidden);
    const float weight = scale == nullptr ? 1.0f : LoadScale(scale, scale_index, scale_type);
    const float mean = sums[group] / static_cast<float>(hidden);
    const float rms = sqrtf(__fadd_rn(mean, epsilon));
    const int64_t source_index = Packed ? group * input_row_stride + index % hidden : index;
    const float normalized = to_float<T>(x[source_index]) / rms;
    const T rounded = from_float<T>(__fmul_rn(normalized, weight));
    y[index] = rounded;
    if (widened != nullptr) widened[index] = to_float<T>(rounded);
  }
}

template <typename T, int ILP>
__global__ void MixedScaleBranchwiseRMSNormKernel(const T* x, const void* scale, int scale_type,
                                                  T* y, int64_t groups, int branches, int hidden,
                                                  bool shared_scale, float epsilon) {
  const int64_t group = blockIdx.x;
  if (group >= groups) {
    return;
  }

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  using VecT = onnxruntime::cuda::aligned_vector<T, ILP>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float inv_rms;

  const int64_t offset = group * hidden;
  float sum_sq = 0.0f;
  for (int h = threadIdx.x * ILP; h < hidden; h += blockDim.x * ILP) {
    const VecT values = *reinterpret_cast<const VecT*>(x + offset + h);
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
      const float value = to_float<T>(values.val[i]);
      sum_sq += value * value;
    }
  }

  const float total_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);
  if (threadIdx.x == 0) {
    inv_rms = rsqrtf(total_sum_sq / hidden + epsilon);
  }
  __syncthreads();

  const int64_t branch_offset = (group % branches) * hidden;
  for (int h = threadIdx.x * ILP; h < hidden; h += blockDim.x * ILP) {
    const VecT values = *reinterpret_cast<const VecT*>(x + offset + h);
    VecT outputs;
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
      const int scale_index = h + i;
      const int64_t scale_offset = shared_scale ? scale_index : branch_offset + scale_index;
      const float weight = scale == nullptr ? 1.0f : LoadScale(scale, scale_offset, scale_type);
      outputs.val[i] = from_float<T>(to_float<T>(values.val[i]) * inv_rms * weight);
    }
    *reinterpret_cast<VecT*>(y + offset + h) = outputs;
  }
}

}  // namespace

template <typename T>
Status LaunchStandardBranchwiseRMSNorm(cudaStream_t stream, const T* x, const void* scale,
                                       int scale_type, T* y, int groups, int branches,
                                       int hidden, bool shared_scale, float epsilon,
                                       float* widened, float* squares, float* sums, void* reduction_buffer,
                                       size_t reduction_buffer_size, int input_row_stride,
                                       bool deterministic_compute) {
  const int64_t count = static_cast<int64_t>(groups) * hidden;
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  ORT_RETURN_IF_NOT(blocks <= std::numeric_limits<int>::max(), "CUDA launch requires too many blocks");
  // Standard ReduceMean removes singleton dimensions and prefers row reduction for a scalar result.
  // Row reduction uses floating-point atomics; column reduction combines indexed partials in a fixed order.
  if (groups == 1 && !deterministic_compute) {
    BranchwiseSquareKernel<T><<<static_cast<int>(blocks), kThreads, 0, stream>>>(x, squares, count);
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
    ORT_RETURN_IF_ERROR((onnxruntime::cuda::reduce_matrix_rows<float, float>(
        stream, squares, sums, hidden, 1)));
  } else {
    using ReduceT = std::conditional_t<std::is_same_v<T, __nv_bfloat16>, BFloat16, T>;
    ORT_RETURN_IF_ERROR((onnxruntime::cuda::reduce_square_matrix_columns<ReduceT>(
        stream, reinterpret_cast<const ReduceT*>(x), sums, groups, hidden,
        reduction_buffer, reduction_buffer_size, input_row_stride)));
  }
  if (input_row_stride == hidden) {
    BranchwiseStandardEpilogueKernel<T, false><<<static_cast<int>(blocks), kThreads, 0, stream>>>(
        x, scale, scale_type, sums, y, widened, count, branches, hidden, shared_scale, epsilon, input_row_stride);
  } else {
    BranchwiseStandardEpilogueKernel<T, true><<<static_cast<int>(blocks), kThreads, 0, stream>>>(
        x, scale, scale_type, sums, y, widened, count, branches, hidden, shared_scale, epsilon, input_row_stride);
  }
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchMixedScaleBranchwiseRMSNorm(cudaStream_t stream, const T* x, const void* scale,
                                         int scale_type, T* y, int64_t groups, int branches,
                                         int hidden, bool shared_scale, float epsilon) {
  ORT_RETURN_IF_NOT(groups <= std::numeric_limits<int>::max(), "CUDA launch requires too many blocks");
  if constexpr (sizeof(T) == 2) {
    if (hidden % 2 == 0) {
      MixedScaleBranchwiseRMSNormKernel<T, 2><<<static_cast<int>(groups), kThreads, 0, stream>>>(
          x, scale, scale_type, y, groups, branches, hidden, shared_scale, epsilon);
    } else {
      MixedScaleBranchwiseRMSNormKernel<T, 1><<<static_cast<int>(groups), kThreads, 0, stream>>>(
          x, scale, scale_type, y, groups, branches, hidden, shared_scale, epsilon);
    }
  } else {
    MixedScaleBranchwiseRMSNormKernel<T, 1><<<static_cast<int>(groups), kThreads, 0, stream>>>(
        x, scale, scale_type, y, groups, branches, hidden, shared_scale, epsilon);
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                                        \
  template Status LaunchMixedScaleBranchwiseRMSNorm<T>(cudaStream_t, const T*, const void*, int, T*, int64_t, \
                                                       int, int, bool, float);

INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(__nv_bfloat16)

#undef INSTANTIATE

#define INSTANTIATE_STANDARD(T)                                                                    \
  template Status LaunchStandardBranchwiseRMSNorm<T>(cudaStream_t, const T*, const void*, int, T*, \
                                                     int, int, int, bool, float, float*, float*, float*, void*, size_t, int, bool);
INSTANTIATE_STANDARD(float)
INSTANTIATE_STANDARD(half)
INSTANTIATE_STANDARD(__nv_bfloat16)
#undef INSTANTIATE_STANDARD

}  // namespace onnxruntime::contrib::cuda
