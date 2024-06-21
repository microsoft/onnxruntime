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
__device__ __forceinline__ void _vec_copy_data(
    void* output_ptr,
    const void* input_ptr,
    const int64_t size_in_byte) {
  auto output_ptr_vec = reinterpret_cast<T*>(output_ptr);
  auto input_ptr_vec = reinterpret_cast<const T*>(input_ptr);
  auto new_size = size_in_byte / sizeof(T);
  for (int i = threadIdx.x; i < new_size; i += blockDim.x) {
    output_ptr_vec[i] = input_ptr_vec[i];
  }
  auto remain = size_in_byte % sizeof(T);
  auto output_ptr_r = reinterpret_cast<char*>(output_ptr_vec + new_size);
  auto input_ptr_r = reinterpret_cast<const char*>(input_ptr_vec + new_size);
  for (int i = threadIdx.x; i < remain; i += blockDim.x) {
    output_ptr_r[i] = input_ptr_r[i];
  }
}

template <typename T>
__device__ __forceinline__ void _block_copy_data(T* out, const T* input, int size) {
  auto copy_size_in_byte = size * sizeof(T);

  auto select = [](int value) {
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
  // cast the input and output ptr into long value, and check the alignment size of there pointers.
  auto input_v = reinterpret_cast<size_t>(input);
  auto out_v = reinterpret_cast<size_t>(out);
  // select the min alignment of input and output pointer
  auto vec_size = std::min(select(input_v), select(out_v));

  if (vec_size == 16) {
    _vec_copy_data<int4>(out, input, copy_size_in_byte);
  } else if (vec_size == 8) {
    _vec_copy_data<int2>(out, input, copy_size_in_byte);
  } else if (vec_size == 4) {
    _vec_copy_data<int32_t>(out, input, copy_size_in_byte);
  } else if (vec_size == 2) {
    _vec_copy_data<int16_t>(out, input, copy_size_in_byte);
  } else {
    _vec_copy_data<char>(out, input, copy_size_in_byte);
  }
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
  // each block copy a slice row of input data
  int64_t row_id = blockIdx.x / 3;
  int64_t slice_id = blockIdx.x % 3;
#define _SELECT_1_in_3(slice_id, s1, s2, s3) \
  ((slice_id == 0) ? (s1) : ((slice_id == 1) ? (s2) : (s3)))

  // move input pointer to the corresponding row, and based on the slice_id, move to the begin of the slice
  auto input_ptr = input_data + row_id * inner_size + _SELECT_1_in_3(slice_id, 0, size0, size0+size1);

  auto output_ptr = _SELECT_1_in_3(slice_id,
                                  output_data0 + row_id * size0,
                                  output_data1 + row_id * size1,
                                  output_data2 + row_id * size2);
  auto copy_size = _SELECT_1_in_3(slice_id, size0, size1, size2);

  _block_copy_data(output_ptr, input_ptr, copy_size);
#undef _SELECT_1_in_3
}

Status Split3Inner(cudaStream_t stream, const size_t element_size, const int64_t size0, const int64_t size1,
                   const int64_t size2, const void* input_data, void* output_data0, void* output_data1,
                   void* output_data2, const gsl::span<const int64_t>& input_shape) {
  CUDA_LONG outer_size = 1;
  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      outer_size *= static_cast<CUDA_LONG>(input_shape[i]);
  }
  CUDA_LONG inner_size = static_cast<CUDA_LONG>(input_shape[input_shape.size() - 1]);

  int blocksPerGrid = outer_size * 3;

  switch (element_size) {
#define CASE_ELEMENT_TYPE(type)                                                                 \
  case sizeof(type): {                                                                          \
    _Split3InnerKernel<<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(size0, size1, size2,   \
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
