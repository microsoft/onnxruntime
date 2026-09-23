// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/concat_impl.h"

#include <climits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

namespace {
constexpr int kNumElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kNumThreadsPerBlock = GridDim::maxThreadsPerBlock;
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
                                                      void* output_data, const void** const input_data,
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

Status ConcatImpl(cudaStream_t stream, const size_t element_bytes, const int block_size_including_axis_dim,
                  const int block_size_inside_axis_dim, const int64_t* concat_sizes, const int64_t* concat_sizes_range,
                  const int64_t* axis_dimension_input_output_mapping, void* output_data, const void** input_data,
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

// Each block copies a contiguous chunk of one input: the input is chosen once per block and reads are coalesced.
template <typename T>
__global__ void _ConcatKernelInputMajor(const TArray<const void*, 32> input_data, const TArray<int32_t, 32> block_end,
                                        const TArray<int32_t, 32> output_offset,
                                        const TArray<fast_divmod, 32> input_pitch_div, const int outer_count,
                                        const int output_pitch, T* output_data) {
  const int block_id = static_cast<int>(blockIdx.x);
  int input_index = 0;
  while (block_id >= block_end[input_index]) ++input_index;
  const int block_begin = input_index == 0 ? 0 : block_end[input_index - 1];
  const fast_divmod pitch_div = input_pitch_div[input_index];
  const CUDA_LONG input_size = static_cast<CUDA_LONG>(outer_count) * static_cast<CUDA_LONG>(pitch_div.d_);
  const T* input = reinterpret_cast<const T*>(input_data[input_index]);
  const int base = output_offset[input_index];

  CUDA_LONG start = kNumElementsPerThread * kNumThreadsPerBlock * (block_id - block_begin) + threadIdx.x;
  T value[kNumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < input_size) {
      value[i] = input[id];
      id += kNumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < input_size) {
      int outer_index, inner_offset;
      pitch_div.divmod(id, outer_index, inner_offset);
      output_data[outer_index * output_pitch + base + inner_offset] = value[i];
      id += kNumThreadsPerBlock;
    }
  }
}

// Each thread finds the input of its output element by searching the prefix sums, staged in shared memory
// so that lanes looking up different inputs do not serialize on the constant cache.
template <typename T>
__global__ void _ConcatKernelOutputMajor(const fast_divmod block_size_including_axis_dim_div,
                                         const fast_divmod block_size_inside_axis_dim_div,
                                         const TArray<int64_t, 32> concat_sizes_range, T* output_data,
                                         const TArray<const void*, 32> input_data, const CUDA_LONG N) {
  __shared__ int range[32];
  const int input_count = concat_sizes_range.Size();
  const int thread_id = static_cast<int>(threadIdx.x);
  if (thread_id < 32) {
    range[thread_id] = thread_id < input_count ? static_cast<int>(concat_sizes_range[thread_id]) : INT_MAX;
  }
  __syncthreads();

  // Largest power of two below input_count; the branchless search then probes at most 2 * first_step - 1 slots.
  int first_step = 1;
  while (first_step * 2 < input_count) first_step *= 2;

  CUDA_LONG start = kNumElementsPerThread * kNumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kNumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      int outer_block_index, block_index, offset;
      block_size_including_axis_dim_div.divmod(id, outer_block_index, offset);
      block_size_inside_axis_dim_div.divmod(offset, block_index, offset);
      int input_index = 0;
      if (input_count <= 4) {
        while (block_index >= range[input_index]) ++input_index;
      } else {
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
          if (step <= first_step && range[input_index + step - 1] <= block_index) input_index += step;
        }
      }
      const int range_left = (input_index == 0) ? 0 : range[input_index - 1];
      const int concat_size = range[input_index] - range_left;
      const int block_offset = block_index - range_left;
      CUDA_LONG input_pos =
          (outer_block_index * concat_size + block_offset) * block_size_inside_axis_dim_div.d_ + offset;
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

Status ConcatImpl(cudaStream_t stream, const size_t element_bytes, const int block_size_including_axis_dim,
                  const int block_size_inside_axis_dim, const TArray<int64_t, 32>& concat_sizes_range,
                  void* output_data, const TArray<const void*, 32>& input_data, const size_t output_size) {
  // Shorter per-input rows make the input-major writes too scattered to coalesce. The output-major kernel is
  // memory bound for 4- and 8-byte types but instruction bound for narrower ones (thresholds tuned on A100).
  const int64_t min_row_elements_for_input_major = element_bytes <= 2 ? 32 : 256;
  const int input_count = concat_sizes_range.Size();
  bool input_major = true;
  for (int i = 0; i < input_count && input_major; ++i) {
    const int64_t concat_size = concat_sizes_range[i] - (i == 0 ? 0 : concat_sizes_range[i - 1]);
    input_major = concat_size * block_size_inside_axis_dim >= min_row_elements_for_input_major;
  }

  CUDA_LONG N = static_cast<CUDA_LONG>(output_size);
  if (input_major) {
    const int outer_count = static_cast<int>(output_size / block_size_including_axis_dim);
    TArray<int32_t, 32> block_end(input_count);
    TArray<int32_t, 32> output_offset(input_count);
    TArray<fast_divmod, 32> input_pitch_div(input_count);
    int blocks_per_grid = 0;
    for (int i = 0; i < input_count; ++i) {
      const int64_t range_left = i == 0 ? 0 : concat_sizes_range[i - 1];
      const int input_pitch = static_cast<int>((concat_sizes_range[i] - range_left) * block_size_inside_axis_dim);
      blocks_per_grid += CeilDiv(outer_count * input_pitch, kNumElementsPerThread * kNumThreadsPerBlock);
      block_end[i] = blocks_per_grid;
      output_offset[i] = static_cast<int>(range_left * block_size_inside_axis_dim);
      input_pitch_div[i] = fast_divmod(input_pitch);
    }

    switch (element_bytes) {
#define CASE_ELEMENT_TYPE(type)                                                                            \
  case sizeof(type): {                                                                                     \
    _ConcatKernelInputMajor<<<blocks_per_grid, kNumThreadsPerBlock, 0, stream>>>(                          \
        input_data, block_end, output_offset, input_pitch_div, outer_count, block_size_including_axis_dim, \
        reinterpret_cast<ToCudaType<type>::MappedType*>(output_data));                                     \
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

  int blocksPerGrid = CeilDiv(N, kNumElementsPerThread * kNumThreadsPerBlock);
  fast_divmod block_size_including_axis_dim_div = fast_divmod(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div = fast_divmod(block_size_inside_axis_dim);

  switch (element_bytes) {
#define CASE_ELEMENT_TYPE(type)                                                                \
  case sizeof(type): {                                                                         \
    _ConcatKernelOutputMajor<<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(               \
        block_size_including_axis_dim_div, block_size_inside_axis_dim_div, concat_sizes_range, \
        reinterpret_cast<ToCudaType<type>::MappedType*>(output_data), input_data, N);          \
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
