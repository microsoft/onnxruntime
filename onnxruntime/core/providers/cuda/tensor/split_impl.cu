// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/split_impl.h"

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

template <typename T, typename OutputDataArray>
__global__ void _SplitKernelSameSplitDim(const fast_divmod block_size_including_axis_dim_div,
                                         const fast_divmod block_size_inside_axis_dim_div,
                                         const fast_divmod split_dim_size, const int num_outputs, const T* input_data,
                                         OutputDataArray output_data, const CUDA_LONG N) {
  CUDA_LONG start = kNumElementsPerThread * kNumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kNumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      value[i] = input_data[id];
      id += kNumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      int outer_block_index, block_index, offset, output_index, block_offset;
      block_size_including_axis_dim_div.divmod(id, outer_block_index, offset);
      block_size_inside_axis_dim_div.divmod(offset, block_index, offset);
      split_dim_size.divmod(block_index, output_index, block_offset);
      CUDA_LONG output_pos =
          (outer_block_index * split_dim_size.d_ + block_offset) * block_size_inside_axis_dim_div.d_ + offset;
      reinterpret_cast<T*>(output_data[output_index])[output_pos] = value[i];
      id += kNumThreadsPerBlock;
    }
  }
}

template <typename OutputDataArray>
Status SplitSameSplitDimImpl(cudaStream_t stream, const size_t element_size, const int block_size_including_axis_dim,
                             const int block_size_inside_axis_dim, const int64_t split_size, const int num_outputs,
                             const void* input_data, OutputDataArray output_data, const size_t input_size) {
  CUDA_LONG N = static_cast<CUDA_LONG>(input_size);
  int blocksPerGrid = CeilDiv(N, kNumElementsPerThread * kNumThreadsPerBlock);
  fast_divmod block_size_including_axis_dim_div = fast_divmod(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div = fast_divmod(block_size_inside_axis_dim);
  fast_divmod split_size_div = fast_divmod(static_cast<int>(split_size));

  switch (element_size) {
#define CASE_ELEMENT_TYPE(type)                                                                         \
  case sizeof(type): {                                                                                  \
    _SplitKernelSameSplitDim<<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(                        \
        block_size_including_axis_dim_div, block_size_inside_axis_dim_div, split_size_div, num_outputs, \
        reinterpret_cast<const ToCudaType<type>::MappedType*>(input_data), output_data, N);             \
  } break
    CASE_ELEMENT_TYPE(int8_t);
    CASE_ELEMENT_TYPE(int16_t);
    CASE_ELEMENT_TYPE(int32_t);
    CASE_ELEMENT_TYPE(int64_t);
#undef CASE_ELEMENT_TYPE
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Slice operator");
  }

  return Status::OK();
}

template Status SplitSameSplitDimImpl<void**>(cudaStream_t stream, const size_t element_size,
                                              const int block_size_including_axis_dim,
                                              const int block_size_inside_axis_dim, const int64_t split_size,
                                              const int num_outputs, const void* input_data, void** output_data,
                                              const size_t input_size);

template Status SplitSameSplitDimImpl<TArray<void*, 32>>(cudaStream_t stream, const size_t element_size,
                                                         const int block_size_including_axis_dim,
                                                         const int block_size_inside_axis_dim, const int64_t split_size,
                                                         const int num_outputs, const void* input_data,
                                                         TArray<void*, 32> output_data, const size_t input_size);

template <typename T>
__global__ void _SplitKernel(const fast_divmod block_size_including_axis_dim_div,
                             const fast_divmod block_size_inside_axis_dim_div, const int64_t* split_sizes,
                             const int64_t* split_sizes_range, const int64_t* axis_dimension_input_output_mapping,
                             const int num_outputs, const T* input_data, void** output_data, const CUDA_LONG N) {
  CUDA_LONG start = kNumElementsPerThread * kNumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kNumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      value[i] = input_data[id];
      id += kNumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      int outer_block_index, block_index, offset;
      block_size_including_axis_dim_div.divmod(id, outer_block_index, offset);
      block_size_inside_axis_dim_div.divmod(offset, block_index, offset);
      int output_index = axis_dimension_input_output_mapping[block_index];
      int64_t range_left = (output_index == 0) ? 0 : split_sizes_range[output_index - 1];
      int block_offset = block_index - static_cast<int>(range_left);
      CUDA_LONG output_pos =
          (outer_block_index * split_sizes[output_index] + block_offset) * block_size_inside_axis_dim_div.d_ + offset;
      reinterpret_cast<T*>(output_data[output_index])[output_pos] = value[i];
      id += kNumThreadsPerBlock;
    }
  }
}

Status SplitImpl(cudaStream_t stream, const size_t element_size, const int block_size_including_axis_dim,
                 const int block_size_inside_axis_dim, const int64_t* split_sizes, const int64_t* split_sizes_range,
                 const int64_t* axis_dimension_input_output_mapping, const int num_outputs, const void* input_data,
                 void** output_data, const size_t input_size) {
  CUDA_LONG N = static_cast<CUDA_LONG>(input_size);
  int blocksPerGrid = CeilDiv(N, kNumElementsPerThread * kNumThreadsPerBlock);
  fast_divmod block_size_including_axis_dim_div = fast_divmod(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div = fast_divmod(block_size_inside_axis_dim);

  switch (element_size) {
#define CASE_ELEMENT_TYPE(type)                                                                            \
  case sizeof(type): {                                                                                     \
    _SplitKernel<<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(                                       \
        block_size_including_axis_dim_div, block_size_inside_axis_dim_div, split_sizes, split_sizes_range, \
        axis_dimension_input_output_mapping, num_outputs,                                                  \
        reinterpret_cast<const ToCudaType<type>::MappedType*>(input_data), output_data, N);                \
  } break
    CASE_ELEMENT_TYPE(int8_t);
    CASE_ELEMENT_TYPE(int16_t);
    CASE_ELEMENT_TYPE(int32_t);
    CASE_ELEMENT_TYPE(int64_t);
#undef CASE_ELEMENT_TYPE
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Slice operator");
  }

  return Status::OK();
}

template <typename T>
__global__ void _Split3InnerKernel(const int64_t size0,
                                   const int64_t size1,
                                   const int64_t size2,
                                   const T* input_data,
                                   T* output_data0,
                                   T* output_data1,
                                   T* output_data2,
                                   const int64_t outer_size,
                                   const int64_t inner_size) {
  int64_t data_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t row_id = data_id / inner_size;
  int64_t col_id = data_id % inner_size;

  if (row_id >= outer_size || col_id >= inner_size) {
    return;
  }
  if (col_id < size0) {
    output_data0[row_id * size0 + col_id] = input_data[data_id];
  } else if (col_id < size0 + size1) {
    output_data1[row_id * size1 + col_id - size0] = input_data[data_id];
  } else {
    output_data2[row_id * size2 + col_id - size0 - size1] = input_data[data_id];
  }
}

Status Split3Inner(cudaStream_t stream, const size_t element_size, const int64_t size0, const int64_t size1,
                   const int64_t size2, const void* input_data, void* output_data0, void* output_data1,
                   void* output_data2, const gsl::span<const int64_t>& input_shape) {
  int64_t outer_size = 1;
  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      outer_size *= input_shape[i];
  }
  int64_t inner_size = input_shape[input_shape.size() - 1];
  assert (inner_size == (size0 + size1 + size2));

  int64_t N = outer_size * inner_size;
  int blocksPerGrid = CeilDiv(N, kNumThreadsPerBlock);
  dim3 block(kNumThreadsPerBlock);
  dim3 grid(blocksPerGrid);

  switch (element_size) {
#define CASE_ELEMENT_TYPE(type)                                                                 \
  case sizeof(type): {                                                                          \
    _Split3InnerKernel<<<grid, block, 0, stream>>>(size0, size1, size2,                           \
        reinterpret_cast<const ToCudaType<type>::MappedType*>(input_data),                       \
        reinterpret_cast<ToCudaType<type>::MappedType*>(output_data0),                           \
        reinterpret_cast<ToCudaType<type>::MappedType*>(output_data1),                           \
        reinterpret_cast<ToCudaType<type>::MappedType*>(output_data2), outer_size, inner_size);  \
  } break
    CASE_ELEMENT_TYPE(int8_t);
    CASE_ELEMENT_TYPE(int16_t);
    CASE_ELEMENT_TYPE(int32_t);
    CASE_ELEMENT_TYPE(int64_t);
#undef CASE_ELEMENT_TYPE
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Split3Inner operator");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
