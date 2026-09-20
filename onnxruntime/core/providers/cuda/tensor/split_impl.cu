// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/split_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

namespace {
constexpr int kNumElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kNumThreadsPerBlock = GridDim::maxThreadsPerBlock;
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
__global__ void _SplitSmallInnerKernel(TArray<int64_t, kMaxSmallInnerSplitOutputs> split_sizes_in_byte,
                                       const void* input_data,
                                       TArray<void*, kMaxSmallInnerSplitOutputs> output_data,
                                       const int64_t inner_size_in_byte,
                                       const CUDA_LONG N) {
  auto inner_size = inner_size_in_byte / sizeof(T);
  auto input_vec = reinterpret_cast<const T*>(input_data);

  CUDA_LONG id = kNumElementsPerThread * kNumThreadsPerBlock * blockIdx.x + threadIdx.x;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id >= N) {
      return;
    }

    const auto outer_index = id / inner_size;
    const auto inner_index = id % inner_size;
    int64_t output_start = 0;
    for (int output_index = 0; output_index < split_sizes_in_byte.Size(); ++output_index) {
      const int64_t output_size = split_sizes_in_byte[output_index] / sizeof(T);
      if (inner_index < output_start + output_size) {
        auto output_vec = reinterpret_cast<T*>(output_data[output_index]);
        output_vec[outer_index * output_size + inner_index - output_start] = input_vec[id];
        break;
      }
      output_start += output_size;
    }
    id += kNumThreadsPerBlock;
  }
}

Status SplitSmallInner(cudaStream_t stream, const size_t element_size,
                       const TArray<int64_t, kMaxSmallInnerSplitOutputs>& split_sizes,
                       const void* input_data,
                       const TArray<void*, kMaxSmallInnerSplitOutputs>& output_data,
                       const gsl::span<const int64_t>& input_shape) {
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
  auto VEC_SIZE = select(input_v);
  TArray<int64_t, kMaxSmallInnerSplitOutputs> split_sizes_in_byte(split_sizes.Size());
  for (int i = 0; i < split_sizes.Size(); ++i) {
    split_sizes_in_byte[i] = split_sizes[i] * element_size;
    VEC_SIZE = std::min(VEC_SIZE, select(split_sizes_in_byte[i]));
    VEC_SIZE = std::min(VEC_SIZE, select(reinterpret_cast<size_t>(output_data[i])));
  }

  const CUDA_LONG N = outer_size * inner_size_in_byte / VEC_SIZE;
  const int blocks_per_grid = CeilDiv(N, kNumElementsPerThread * kNumThreadsPerBlock);

  switch (VEC_SIZE) {
#define CASE_ELEMENT_TYPE(type)                                                      \
  _SplitSmallInnerKernel<type><<<blocks_per_grid, kNumThreadsPerBlock, 0, stream>>>( \
      split_sizes_in_byte, input_data, output_data, inner_size_in_byte, N)
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

}  // namespace cuda
}  // namespace onnxruntime
