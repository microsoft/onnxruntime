// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "einsum_fast_path_kernels.h"

namespace onnxruntime {
namespace cuda {

constexpr int kEinsumFastPathRankLimit = 8;
constexpr int kTraceThreadsPerBlock = 256;

__device__ __forceinline__ int64_t GetEinsumInputOffset(
    size_t output_idx,
    const TArray<int64_t>& input_strides,
    const TArray<int32_t>& input_axis_to_output_axis,
    const TArray<fast_divmod>& output_strides) {
  int64_t output_coordinates[kEinsumFastPathRankLimit] = {};
  int remaining = static_cast<int>(output_idx);
  for (int32_t axis = 0; axis < output_strides.Size(); ++axis) {
    int coordinate = 0;
    output_strides[axis].divmod(remaining, coordinate, remaining);
    output_coordinates[axis] = coordinate;
  }

  int64_t input_offset = 0;
  for (int32_t axis = 0; axis < input_strides.Size(); ++axis) {
    const int32_t output_axis = input_axis_to_output_axis[axis];
    if (output_axis >= 0) {
      input_offset += input_strides[axis] * output_coordinates[output_axis];
    }
  }

  return input_offset;
}

template <typename T>
__global__ void EinsumDiagonalKernel(const T* input_data,
                                     T* output_data,
                                     size_t output_size,
                                     TArray<int64_t> input_strides,
                                     TArray<int32_t> input_axis_to_output_axis,
                                     TArray<fast_divmod> output_strides) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(output_idx, output_size);
  output_data[output_idx] =
      input_data[GetEinsumInputOffset(output_idx, input_strides, input_axis_to_output_axis, output_strides)];
}

template <typename T, typename TAccum>
__global__ void EinsumTraceKernel(const T* input_data,
                                  T* output_data,
                                  size_t output_size,
                                  int64_t trace_dim,
                                  int64_t trace_stride,
                                  TArray<int64_t> input_strides,
                                  TArray<int32_t> input_axis_to_output_axis,
                                  TArray<fast_divmod> output_strides) {
  const size_t output_idx = blockIdx.x;
  if (output_idx >= output_size) {
    return;
  }

  const int64_t input_offset =
      GetEinsumInputOffset(output_idx, input_strides, input_axis_to_output_axis, output_strides);
  TAccum thread_sum = TAccum{};
  for (int64_t diagonal_idx = threadIdx.x; diagonal_idx < trace_dim; diagonal_idx += blockDim.x) {
    thread_sum += static_cast<TAccum>(input_data[input_offset + diagonal_idx * trace_stride]);
  }

  using BlockReduce = cub::BlockReduce<TAccum, kTraceThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const TAccum sum = BlockReduce(temp_storage).Sum(thread_sum);
  if (threadIdx.x == 0) {
    output_data[output_idx] = static_cast<T>(sum);
  }
}

template <>
__global__ void EinsumTraceKernel<half, float>(const half* input_data,
                                               half* output_data,
                                               size_t output_size,
                                               int64_t trace_dim,
                                               int64_t trace_stride,
                                               TArray<int64_t> input_strides,
                                               TArray<int32_t> input_axis_to_output_axis,
                                               TArray<fast_divmod> output_strides) {
  const size_t output_idx = blockIdx.x;
  if (output_idx >= output_size) {
    return;
  }

  const int64_t input_offset =
      GetEinsumInputOffset(output_idx, input_strides, input_axis_to_output_axis, output_strides);
  float thread_sum = 0.0f;
  for (int64_t diagonal_idx = threadIdx.x; diagonal_idx < trace_dim; diagonal_idx += blockDim.x) {
    thread_sum += __half2float(input_data[input_offset + diagonal_idx * trace_stride]);
  }

  using BlockReduce = cub::BlockReduce<float, kTraceThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const float sum = BlockReduce(temp_storage).Sum(thread_sum);
  if (threadIdx.x == 0) {
    output_data[output_idx] = __float2half(sum);
  }
}

template <typename T>
Status LaunchDiagonalTyped(cudaStream_t stream,
                           const void* input_data,
                           void* output_data,
                           size_t output_size,
                           const TArray<int64_t>& input_strides,
                           const TArray<int32_t>& input_axis_to_output_axis,
                           const TArray<fast_divmod>& output_strides) {
  if (output_size == 0) {
    return Status::OK();
  }

  const int blocks = static_cast<int>(
      (output_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
  EinsumDiagonalKernel<T><<<blocks, GridDim::maxThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input_data), reinterpret_cast<T*>(output_data), output_size,
      input_strides, input_axis_to_output_axis, output_strides);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T, typename TAccum>
Status LaunchTraceTyped(cudaStream_t stream,
                        const void* input_data,
                        void* output_data,
                        size_t output_size,
                        int64_t trace_dim,
                        int64_t trace_stride,
                        const TArray<int64_t>& input_strides,
                        const TArray<int32_t>& input_axis_to_output_axis,
                        const TArray<fast_divmod>& output_strides) {
  if (output_size == 0) {
    return Status::OK();
  }

  EinsumTraceKernel<T, TAccum><<<static_cast<int>(output_size), kTraceThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input_data), reinterpret_cast<T*>(output_data), output_size,
      trace_dim, trace_stride, input_strides, input_axis_to_output_axis, output_strides);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchEinsumDiagonal(cudaStream_t stream,
                            const void* input_data,
                            void* output_data,
                            size_t element_size,
                            size_t output_size,
                            const TArray<int64_t>& input_strides,
                            const TArray<int32_t>& input_axis_to_output_axis,
                            const TArray<fast_divmod>& output_strides) {
  switch (element_size) {
    case sizeof(half):
      return LaunchDiagonalTyped<half>(stream, input_data, output_data, output_size, input_strides,
                                       input_axis_to_output_axis, output_strides);
    case sizeof(float):
      return LaunchDiagonalTyped<float>(stream, input_data, output_data, output_size, input_strides,
                                        input_axis_to_output_axis, output_strides);
    case sizeof(double):
      return LaunchDiagonalTyped<double>(stream, input_data, output_data, output_size, input_strides,
                                         input_axis_to_output_axis, output_strides);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "CUDA Einsum diagonal fast path does not support element size ", element_size);
  }
}

Status LaunchEinsumTrace(cudaStream_t stream,
                         const void* input_data,
                         void* output_data,
                         size_t element_size,
                         size_t output_size,
                         int64_t trace_dim,
                         int64_t trace_stride,
                         const TArray<int64_t>& input_strides,
                         const TArray<int32_t>& input_axis_to_output_axis,
                         const TArray<fast_divmod>& output_strides) {
  switch (element_size) {
    case sizeof(half):
      return LaunchTraceTyped<half, float>(stream, input_data, output_data, output_size, trace_dim, trace_stride,
                                           input_strides, input_axis_to_output_axis, output_strides);
    case sizeof(float):
      return LaunchTraceTyped<float, float>(stream, input_data, output_data, output_size, trace_dim, trace_stride,
                                            input_strides, input_axis_to_output_axis, output_strides);
    case sizeof(double):
      return LaunchTraceTyped<double, double>(stream, input_data, output_data, output_size, trace_dim, trace_stride,
                                              input_strides, input_axis_to_output_axis, output_strides);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "CUDA Einsum trace fast path does not support element size ", element_size);
  }
}

}  // namespace cuda
}  // namespace onnxruntime
