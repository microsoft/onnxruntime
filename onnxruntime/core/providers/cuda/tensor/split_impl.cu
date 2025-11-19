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

#ifndef USE_ROCM
template <typename T>
__global__ void _Split3InnerKernel(const int64_t size0_in_byte,
                                   const int64_t size1_in_byte,
                                   const int64_t size2_in_byte,
                                   const void* input_data,
                                   void* output_data0,
                                   void* output_data1,
                                   void* output_data2,
                                   const int64_t inner_size_in_byte) {
  // each block copy one row of input data
  auto size0 = size0_in_byte / sizeof(T);
  auto size1 = size1_in_byte / sizeof(T);
  auto size2 = size2_in_byte / sizeof(T);
  auto inner_size = inner_size_in_byte / sizeof(T);
  auto output0_vec = reinterpret_cast<T*>(output_data0) + blockIdx.x * size0;
  auto output1_vec = reinterpret_cast<T*>(output_data1) + blockIdx.x * size1;
  auto output2_vec = reinterpret_cast<T*>(output_data2) + blockIdx.x * size2;
  auto input_vec = reinterpret_cast<const T*>(input_data) + blockIdx.x * inner_size;
  // all size and pointer are aligned to sizeof(T)
  // so here use all threads in the block to do vectorized copy

  for (auto tid = threadIdx.x; tid < inner_size; tid += blockDim.x) {
    auto data = input_vec[tid];
    if (tid < size0) {
      output0_vec[tid] = data;
    } else if (tid < (size0 + size1)) {
      output1_vec[tid - size0] = data;
    } else {
      output2_vec[tid - size0 - size1] = data;
    }
  }
}

Status Split3Inner(cudaStream_t stream, const size_t element_size, const int64_t size0, const int64_t size1,
                   const int64_t size2, const void* input_data, void* output_data0, void* output_data1,
                   void* output_data2, const gsl::span<const int64_t>& input_shape) {
  CUDA_LONG outer_size = 1;
  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
    outer_size *= static_cast<CUDA_LONG>(input_shape[i]);
  }
  CUDA_LONG inner_size_in_byte = static_cast<CUDA_LONG>(input_shape[input_shape.size() - 1] * element_size);

  auto select = [](size_t value) {
    if (value % 16 == 0) {
      return 16;
    } else if (value % 8 == 0) {
      return 8;
    } else if (value % 4 == 0) {
      return 4;
    } else if (value % 2 == 0) {
      return 2;
    } else {
      return 1;
    }
  };

  auto input_v = reinterpret_cast<size_t>(input_data);
  auto output_v0 = reinterpret_cast<size_t>(output_data0);
  auto output_v1 = reinterpret_cast<size_t>(output_data1);
  auto output_v2 = reinterpret_cast<size_t>(output_data2);
  auto size0_in_byte = size0 * element_size;
  auto size1_in_byte = size1 * element_size;
  auto size2_in_byte = size2 * element_size;

  auto VEC_SIZE = std::min(select(size0_in_byte), std::min(select(size1_in_byte), select(size2_in_byte)));
  auto min_output_vec_size = std::min(select(output_v0), std::min(select(output_v1), select(output_v2)));
  VEC_SIZE = std::min(VEC_SIZE, std::min(select(input_v), min_output_vec_size));

  // determine threads based on the size of the output
  auto threadsPerBlock = kNumThreadsPerBlock;
  if ((inner_size_in_byte / VEC_SIZE) <= 128) {
    // use less threads when the size is small
    threadsPerBlock = 128;
  }

  switch (VEC_SIZE) {
#define CASE_ELEMENT_TYPE(type)                                         \
  _Split3InnerKernel<type><<<outer_size, threadsPerBlock, 0, stream>>>( \
      size0_in_byte,                                                    \
      size1_in_byte,                                                    \
      size2_in_byte,                                                    \
      input_data,                                                       \
      output_data0,                                                     \
      output_data1,                                                     \
      output_data2,                                                     \
      inner_size_in_byte)
    case 16:
      CASE_ELEMENT_TYPE(int4);
      break;
    case 8:
      CASE_ELEMENT_TYPE(int64_t);
      break;
    case 4:
      CASE_ELEMENT_TYPE(int32_t);
      break;
    case 2:
      CASE_ELEMENT_TYPE(int16_t);
      break;
    default:
      CASE_ELEMENT_TYPE(int8_t);
      break;
#undef CASE_ELEMENT_TYPE
  }

  return Status::OK();
}
#endif

}  // namespace cuda
}  // namespace onnxruntime
