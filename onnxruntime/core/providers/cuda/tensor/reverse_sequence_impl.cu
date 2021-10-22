// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

static const int kReverseSequenceElementsPerThread = 4;

template <typename T, bool time_major>
__global__ void ReverseSequenceImplKernel(
    const T* x_data,
    const int64_t* seq_len_data,
    T* y_data,
    const int batch_size,
    const int max_seq_len,
    const int element_size,
    const int group_count,
    const fast_divmod fdm_grouped_stride_0,
    const fast_divmod fdm_grouped_stride_1) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(grouped_index, group_count);
  int batch_id, seq_id, gid = grouped_index;
  if (time_major) {
    fdm_grouped_stride_0.divmod(gid, seq_id, gid);
    fdm_grouped_stride_1.divmod(gid, batch_id, gid);
  } else {
    fdm_grouped_stride_0.divmod(gid, batch_id, gid);
    fdm_grouped_stride_1.divmod(gid, seq_id, gid);
  }
  int eid = gid * kReverseSequenceElementsPerThread;
  int target_seq_id = (seq_id < (int)seq_len_data[batch_id]) ? ((int)seq_len_data[batch_id] - 1 - seq_id) : seq_id;
  int flat_src_idx, flat_target_idx;
  if (time_major) {
    flat_src_idx = seq_id * batch_size * element_size + batch_id * element_size + eid;
    flat_target_idx = target_seq_id * batch_size * element_size + batch_id * element_size + eid;
  } else {
    flat_src_idx = batch_id * max_seq_len * element_size + seq_id * element_size + eid;
    flat_target_idx = batch_id * max_seq_len * element_size + target_seq_id * element_size + eid;
  }

  y_data[flat_target_idx] = x_data[flat_src_idx];
#pragma unroll
  for (int i = 1; i < kReverseSequenceElementsPerThread; ++i) {
    if (eid + i < element_size) {
      y_data[flat_target_idx + i] = x_data[flat_src_idx + i];
    }
  }
}

template <typename T>
cudaError_t ReverseSequenceCudaImpl(
    cudaStream_t stream,
    const T* x_data,
    const int64_t* seq_len_data,
    T* y_data,
    const int batch_size,
    const int max_seq_len,
    const int element_size,
    const bool time_major) {
  int element_group_size = CeilDiv(element_size, kReverseSequenceElementsPerThread);
  fast_divmod fdm_grouped_stride_1(element_group_size);
  fast_divmod fdm_grouped_stride_0(element_group_size * ((time_major) ? batch_size : max_seq_len));
  int group_count = batch_size * max_seq_len * element_group_size;
  int blocksPerGrid = CeilDiv(group_count, GridDim::maxThreadsPerBlock);

  if (time_major) {
    ReverseSequenceImplKernel<T, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        x_data, seq_len_data, y_data, batch_size, max_seq_len, element_size,
        group_count, fdm_grouped_stride_0, fdm_grouped_stride_1);
  } else {
    ReverseSequenceImplKernel<T, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        x_data, seq_len_data, y_data, batch_size, max_seq_len, element_size,
        group_count, fdm_grouped_stride_0, fdm_grouped_stride_1);
  }
  return cudaSuccess;
}

#define InstantiateReverseSequenceImpl(T)       \
  template cudaError_t ReverseSequenceCudaImpl( \
      cudaStream_t stream,                \
      const T* x_data,                          \
      const int64_t* seq_len_data,              \
      T* y_data,                                \
      const int batch_size,                     \
      const int max_seq_len,                    \
      const int element_size,                   \
      const bool time_major)

InstantiateReverseSequenceImpl(int64_t);
InstantiateReverseSequenceImpl(int32_t);
InstantiateReverseSequenceImpl(int16_t);
InstantiateReverseSequenceImpl(int8_t);

}  // namespace cuda
}  // namespace onnxruntime
