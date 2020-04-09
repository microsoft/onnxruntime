// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "where_impl.h"

namespace onnxruntime {
namespace cuda {

// broadcast by computing output coordinate from offset, using fast_divmod
template <typename T, bool cond_need_compute, bool x_need_compute, bool y_need_compute>
__global__ void _TenaryElementWise(
    size_t output_rank,
    const TArray<int64_t> cond_padded_strides,
    const bool* cond_data,
    const TArray<int64_t> x_padded_strides,
    const T* x_data,
    const TArray<int64_t> y_padded_strides,
    const T* y_data,
    const TArray<fast_divmod> fdm_output_strides,
    T* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG cond_index = (cond_need_compute ? 0 : id);
  CUDA_LONG x_index = (x_need_compute ? 0 : id);
  CUDA_LONG y_index = (y_need_compute ? 0 : id);

  // compute indexes with broadcasting rules: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
  CUDA_LONG offset = id;
  for (auto dim = 0; dim < fdm_output_strides.GetCapacity(); dim++) {
    if (dim >= output_rank) {
      break;
    }

    int q, r;
    fdm_output_strides[dim].divmod(offset, q, r);

    if (cond_need_compute) {
      cond_index += static_cast<int>(cond_padded_strides[dim]) * q;
    }

    if (x_need_compute) {
      x_index += static_cast<int>(x_padded_strides[dim]) * q;
    }

    if (y_need_compute) {
      y_index += static_cast<int>(y_padded_strides[dim]) * q;
    }

    offset = r;
  }

  output_data[id] = cond_data[cond_index] ? x_data[x_index] : y_data[y_index];
}

// for scalar broadcast or non-broadcast case
template <typename T>
__global__ void _TenaryElementWiseSimple(
    const bool* cond_data,
    const T* x_data,
    const T* y_data,
    T* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = cond_data[id] ? x_data[id] : y_data[id];
}

template <typename T>
void WhereImpl(
    size_t output_rank_or_simple_broadcast,
    const TArray<int64_t>& cond_padded_strides,
    const bool* cond_data,
    const TArray<int64_t>& x_padded_strides,
    const T* x_data,
    const TArray<int64_t>& y_padded_strides,
    const T* y_data,
    const TArray<fast_divmod>& fdm_output_strides,
    T* output_data,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  if (output_rank_or_simple_broadcast == static_cast<size_t>(SimpleBroadcast::NoBroadcast)) {
    _TenaryElementWiseSimple<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        cond_data,
        x_data,
        y_data,
        output_data,
        N);
  } else {
    if (cond_padded_strides.size_ && x_padded_strides.size_ && y_padded_strides.size_) {
      _TenaryElementWise<T, true, true, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank_or_simple_broadcast,
          cond_padded_strides,
          cond_data,
          x_padded_strides,
          x_data,
          y_padded_strides,
          y_data,
          fdm_output_strides,
          output_data,
          N);
    } else if (cond_padded_strides.size_ && x_padded_strides.size_ && !y_padded_strides.size_) {
      _TenaryElementWise<T, true, true, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank_or_simple_broadcast,
          cond_padded_strides,
          cond_data,
          x_padded_strides,
          x_data,
          y_padded_strides,
          y_data,
          fdm_output_strides,
          output_data,
          N);
    } else if (cond_padded_strides.size_ && !x_padded_strides.size_ && y_padded_strides.size_) {
      _TenaryElementWise<T, true, false, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank_or_simple_broadcast,
          cond_padded_strides,
          cond_data,
          x_padded_strides,
          x_data,
          y_padded_strides,
          y_data,
          fdm_output_strides,
          output_data,
          N);
    } else if (!cond_padded_strides.size_ && x_padded_strides.size_ && y_padded_strides.size_) {
      _TenaryElementWise<T, false, true, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank_or_simple_broadcast,
          cond_padded_strides,
          cond_data,
          x_padded_strides,
          x_data,
          y_padded_strides,
          y_data,
          fdm_output_strides,
          output_data,
          N);
    } else if (cond_padded_strides.size_ && !x_padded_strides.size_ && !y_padded_strides.size_) {
      _TenaryElementWise<T, true, false, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_rank_or_simple_broadcast,
          cond_padded_strides,
          cond_data,
          x_padded_strides,
          x_data,
          y_padded_strides,
          y_data,
          fdm_output_strides,
          output_data,
          N);
      } else if (!cond_padded_strides.size_ && x_padded_strides.size_ && !y_padded_strides.size_) {
        _TenaryElementWise<T, false, true, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            output_rank_or_simple_broadcast,
            cond_padded_strides,
            cond_data,
            x_padded_strides,
            x_data,
            y_padded_strides,
            y_data,
            fdm_output_strides,
            output_data,
            N);
      } else if (!cond_padded_strides.size_ && !x_padded_strides.size_ && y_padded_strides.size_) {
        _TenaryElementWise<T, false, false, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            output_rank_or_simple_broadcast,
            cond_padded_strides,
            cond_data,
            x_padded_strides,
            x_data,
            y_padded_strides,
            y_data,
            fdm_output_strides,
            output_data,
            N);
      } else {
        _TenaryElementWise<T, false, false, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            output_rank_or_simple_broadcast,
            cond_padded_strides,
            cond_data,
            x_padded_strides,
            x_data,
            y_padded_strides,
            y_data,
            fdm_output_strides,
            output_data,
            N);
      }
  }
}

#define SPECIALIZED_IMPL(T)                                          \
  template void WhereImpl<T>(size_t output_rank_or_simple_broadcast, \
                             const TArray<int64_t>& cond_padded_strides,     \
                             const bool* cond_data,                  \
                             const TArray<int64_t>& x_padded_strides,        \
                             const T* x_data,                        \
                             const TArray<int64_t>& y_padded_strides,        \
                             const T* y_data,                        \
                             const TArray<fast_divmod>& fdm_output_strides,  \
                             T* output_data,                         \
                             size_t count);

SPECIALIZED_IMPL(uint8_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime