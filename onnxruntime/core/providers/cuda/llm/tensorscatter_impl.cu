// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/llm/tensorscatter_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T, bool circular>
__global__ void _TensorScatterKernel(
    T* output_data,
    const T* update_data,
    const int64_t* write_indices,
    int64_t prefix_count,
    int64_t prefix_stride_for_batch,
    int64_t max_seq_len,
    int64_t seq_len,
    int64_t suffix_count,
    size_t total_elements) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, total_elements);

  int64_t seq_suffix = seq_len * suffix_count;
  int64_t prefix_idx = id / seq_suffix;
  int64_t remainder = id - prefix_idx * seq_suffix;
  int64_t seq_idx = remainder / suffix_count;
  int64_t suffix_idx = remainder - seq_idx * suffix_count;

  int64_t batch_idx = prefix_idx / prefix_stride_for_batch;
  int64_t wi = (write_indices != nullptr) ? write_indices[batch_idx] : 0;
  // write_indices are validated on the host before kernel launch.
  int64_t cache_pos;
  if (circular) {
    cache_pos = (wi + seq_idx) % max_seq_len;
  } else {
    cache_pos = wi + seq_idx;
  }

  int64_t out_offset = prefix_idx * (max_seq_len * suffix_count) + cache_pos * suffix_count + suffix_idx;
  output_data[out_offset] = update_data[id];
}

template <typename T>
Status _TensorScatterDispatchCircular(
    cudaStream_t stream,
    T* output_data,
    const T* update_data,
    const int64_t* write_indices,
    int64_t prefix_count,
    int64_t prefix_stride_for_batch,
    int64_t max_seq_len,
    int64_t seq_len,
    int64_t suffix_count,
    bool circular) {
  size_t total_elements = static_cast<size_t>(prefix_count) * static_cast<size_t>(seq_len) * static_cast<size_t>(suffix_count);
  if (total_elements == 0) return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(total_elements, static_cast<size_t>(GridDim::maxThreadsPerBlock)));

  if (circular) {
    _TensorScatterKernel<T, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        output_data, update_data, write_indices,
        prefix_count, prefix_stride_for_batch, max_seq_len, seq_len, suffix_count,
        total_elements);
  } else {
    _TensorScatterKernel<T, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        output_data, update_data, write_indices,
        prefix_count, prefix_stride_for_batch, max_seq_len, seq_len, suffix_count,
        total_elements);
  }

  return CUDA_CALL(cudaGetLastError());
}

Status TensorScatterImpl(
    cudaStream_t stream,
    void* output_data,
    const void* update_data,
    const int64_t* write_indices,
    size_t element_size,
    int64_t prefix_count,
    int64_t prefix_stride_for_batch,
    int64_t max_seq_len,
    int64_t seq_len,
    int64_t suffix_count,
    bool circular) {
  switch (element_size) {
    case sizeof(int8_t):
      return _TensorScatterDispatchCircular<int8_t>(
          stream, reinterpret_cast<int8_t*>(output_data),
          reinterpret_cast<const int8_t*>(update_data), write_indices,
          prefix_count, prefix_stride_for_batch, max_seq_len, seq_len, suffix_count, circular);

    case sizeof(int16_t):
      return _TensorScatterDispatchCircular<int16_t>(
          stream, reinterpret_cast<int16_t*>(output_data),
          reinterpret_cast<const int16_t*>(update_data), write_indices,
          prefix_count, prefix_stride_for_batch, max_seq_len, seq_len, suffix_count, circular);

    case sizeof(int32_t):
      return _TensorScatterDispatchCircular<int32_t>(
          stream, reinterpret_cast<int32_t*>(output_data),
          reinterpret_cast<const int32_t*>(update_data), write_indices,
          prefix_count, prefix_stride_for_batch, max_seq_len, seq_len, suffix_count, circular);

    case sizeof(int64_t):
      return _TensorScatterDispatchCircular<int64_t>(
          stream, reinterpret_cast<int64_t*>(output_data),
          reinterpret_cast<const int64_t*>(update_data), write_indices,
          prefix_count, prefix_stride_for_batch, max_seq_len, seq_len, suffix_count, circular);

    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported element size for TensorScatter: ", element_size);
  }
}

}  // namespace cuda
}  // namespace onnxruntime
