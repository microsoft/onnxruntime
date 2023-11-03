// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/concat_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

namespace {
#ifdef USE_ROCM
constexpr int kNumElementsPerThread = 2;
constexpr int kNumThreadsPerBlock = 512;
#else
constexpr int kNumElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kNumThreadsPerBlock = GridDim::maxThreadsPerBlock;
#endif
}  // namespace

// concat dimension are same for all inputs
template <typename T, typename InputDataArray>
__global__ void _ConcatKernelSameConcatDim(const fast_divmod block_size_including_axis_dim_div,
                                           const fast_divmod block_size_inside_axis_dim_div,
                                           const fast_divmod concat_dim_size, T* output_data, InputDataArray input_data,
                                           const CUDA_LONG N) {
  CUDA_LONG start = kNumElementsPerThread * kNumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kNumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      int outer_block_index, block_index, offset, input_index, block_offset;
      block_size_including_axis_dim_div.divmod(id, outer_block_index, offset);
      block_size_inside_axis_dim_div.divmod(offset, block_index, offset);
      concat_dim_size.divmod(block_index, input_index, block_offset);
      CUDA_LONG input_pos =
          (outer_block_index * concat_dim_size.d_ + block_offset) * block_size_inside_axis_dim_div.d_ + offset;
      value[i] = reinterpret_cast<const T*>(input_data[input_index])[input_pos];
      id += kNumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += kNumThreadsPerBlock;
    }
  }
}

template <typename InputDataArray>
Status ConcatSameConcatDimImpl(cudaStream_t stream, const size_t element_bytes, const int block_size_including_axis_dim,
                               const int block_size_inside_axis_dim, const int64_t concat_size, void* output_data,
                               const InputDataArray input_data, const size_t output_size) {
  CUDA_LONG N = static_cast<CUDA_LONG>(output_size);
  int blocksPerGrid = CeilDiv(N, kNumElementsPerThread * kNumThreadsPerBlock);
  fast_divmod block_size_including_axis_dim_div = fast_divmod(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div = fast_divmod(block_size_inside_axis_dim);
  fast_divmod concat_dim_size = fast_divmod(static_cast<int>(concat_size));
  switch (element_bytes) {
#define CASE_ELEMENT_TYPE(type)                                                             \
  case sizeof(type): {                                                                      \
    _ConcatKernelSameConcatDim<<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(          \
        block_size_including_axis_dim_div, block_size_inside_axis_dim_div, concat_dim_size, \
        reinterpret_cast<ToCudaType<type>::MappedType*>(output_data), input_data, N);       \
  } break
    CASE_ELEMENT_TYPE(int8_t);
    CASE_ELEMENT_TYPE(int16_t);
    CASE_ELEMENT_TYPE(int32_t);
    CASE_ELEMENT_TYPE(int64_t);
#undef CASE_ELEMENT_TYPE
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Concat operator");
  }

  return Status::OK();
}

// input tensors addresses in device memory
template Status ConcatSameConcatDimImpl<const void**>(cudaStream_t stream, const size_t element_bytes,
                                                      const int block_size_including_axis_dim,
                                                      const int block_size_inside_axis_dim, const int64_t concat_size,
                                                      void* output_data, const void** input_data,
                                                      const size_t output_size);

// input tensor addresses passed by value
template Status ConcatSameConcatDimImpl<TArray<const void*, 32>>(cudaStream_t stream, const size_t element_bytes,
                                                                 const int block_size_including_axis_dim,
                                                                 const int block_size_inside_axis_dim,
                                                                 const int64_t concat_size, void* output_data,
                                                                 TArray<const void*, 32> input_data,
                                                                 const size_t output_size);

template <typename T>
__global__ void _ConcatKernel(const fast_divmod block_size_including_axis_dim_div,
                              const fast_divmod block_size_inside_axis_dim_div, const int64_t* concat_sizes,
                              const int64_t* concat_sizes_range, const int64_t* axis_dimension_input_output_mapping,
                              T* output_data, const void** input_data, const CUDA_LONG N) {
  CUDA_LONG start = kNumElementsPerThread * kNumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kNumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      int outer_block_index, block_index, offset;
      block_size_including_axis_dim_div.divmod(id, outer_block_index, offset);
      block_size_inside_axis_dim_div.divmod(offset, block_index, offset);
      int input_index = axis_dimension_input_output_mapping[block_index];
      int64_t range_left = (input_index == 0) ? 0 : concat_sizes_range[input_index - 1];
      int block_offset = block_index - static_cast<int>(range_left);
      CUDA_LONG input_pos =
          (outer_block_index * concat_sizes[input_index] + block_offset) * block_size_inside_axis_dim_div.d_ + offset;
      value[i] = reinterpret_cast<const T*>(input_data[input_index])[input_pos];
      id += kNumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += kNumThreadsPerBlock;
    }
  }
}

template <typename T>
Status ConcatImpl(cudaStream_t stream, const size_t element_bytes, const int block_size_including_axis_dim,
                  const int block_size_inside_axis_dim, const int64_t* concat_sizes, const int64_t* concat_sizes_range,
                  const int64_t* axis_dimension_input_output_mapping, T* output_data, const T** input_data,
                  const size_t output_size) {
  CUDA_LONG N = static_cast<CUDA_LONG>(output_size);
  int blocksPerGrid = CeilDiv(N, kNumElementsPerThread * kNumThreadsPerBlock);
  fast_divmod block_size_including_axis_dim_div = fast_divmod(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div = fast_divmod(block_size_inside_axis_dim);

  switch (element_bytes) {
#define CASE_ELEMENT_TYPE(type)                                                                                        \
  case sizeof(type): {                                                                                                 \
    _ConcatKernel<<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(                                                  \
        block_size_including_axis_dim_div, block_size_inside_axis_dim_div, concat_sizes, concat_sizes_range,           \
        axis_dimension_input_output_mapping, reinterpret_cast<ToCudaType<type>::MappedType*>(output_data), input_data, \
        N);                                                                                                            \
  } break;
    CASE_ELEMENT_TYPE(int8_t);
    CASE_ELEMENT_TYPE(int16_t);
    CASE_ELEMENT_TYPE(int32_t);
    CASE_ELEMENT_TYPE(int64_t);
#undef CASE_ELEMENT_TYPE
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Concat operator");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
