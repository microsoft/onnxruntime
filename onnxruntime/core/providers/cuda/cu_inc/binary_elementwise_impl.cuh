// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

// broadcast by computing output coordinate from offset, using fast_divmod
template <typename T, typename T1, typename T2, typename FuncT,
  bool lhs_need_compute, bool rhs_need_compute, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWise(
    int32_t output_rank,
    const TArray<int64_t> lhs_padded_strides,
    const T1* lhs_data,
    const TArray<int64_t> rhs_padded_strides,
    const T2* rhs_data,
    const TArray<fast_divmod> fdm_output_strides,
    T* output_data,
    const FuncT& functor,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T1 lvalue[NumElementsPerThread];
  T2 rvalue[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG lhs_index = (lhs_need_compute ? 0 : id);
      CUDA_LONG rhs_index = (rhs_need_compute ? 0 : id);
      // compute indexes with broadcasting rules: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
      CUDA_LONG offset = id;
#pragma unroll
      for (auto dim = 0; dim < fdm_output_strides.Capacity(); dim++) {
        if (dim >= output_rank) {
          break;
        }
        int q, r;
        fdm_output_strides[dim].divmod(offset, q, r);
        if (lhs_need_compute) {
          lhs_index += static_cast<int>(lhs_padded_strides[dim]) * q;
        }

        if (rhs_need_compute) {
          rhs_index += static_cast<int>(rhs_padded_strides[dim]) * q;
        }
        offset = r;
      }
      lvalue[i] = lhs_data[lhs_index];
      rvalue[i] = rhs_data[rhs_index];

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = functor(lvalue[i], rvalue[i]);

      id += NumThreadsPerBlock;
    }
  }
}

// for scalar broadcast or non-broadcast case
template <bool IncL, bool IncR, typename T, typename T1, typename T2, typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseSimple(
    const T1* lhs_data,
    const T2* rhs_data,
    T* output_data,
    const FuncT& func,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T1 lvalue[NumElementsPerThread];
  T2 rvalue[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      lvalue[i] = lhs_data[IncL ? id : 0];
      rvalue[i] = rhs_data[IncR ? id : 0];

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = func(lvalue[i], rvalue[i]);

      id += NumThreadsPerBlock;
    }
  }
}

// for rhs per-channel broadcast case
template <typename T, typename T1, typename T2, typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseRhsPerChannelBatch1(
    const T1* lhs_data,
    const T2* rhs_data,
    const fast_divmod fdm_H,
    T* output_data,
    FuncT func,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T1 lvalue[NumElementsPerThread];
  T2 rvalue[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG rhs_id = fdm_H.div(id);
      lvalue[i] = lhs_data[id];
      rvalue[i] = rhs_data[rhs_id];

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = func(lvalue[i], rvalue[i]);

      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, typename T1, typename T2, typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _BinaryElementWiseRhsPerChannelBatchN(
    const T1* lhs_data,
    const T2* rhs_data,
    const fast_divmod fdm_H,
    const fast_divmod fdm_C,
    T* output_data,
    FuncT func,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T1 lvalue[NumElementsPerThread];
  T2 rvalue[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG rhs_id = fdm_H.div(id);
      int q, r;
      fdm_C.divmod(rhs_id, q, r);
      rhs_id = r;

      lvalue[i] = lhs_data[id];
      rvalue[i] = rhs_data[rhs_id];

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = func(lvalue[i], rvalue[i]);

      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, typename T1, typename T2, typename FuncT>
void BinaryElementWiseNoBroadcastImpl(
    cudaStream_t stream,
    const T1* lhs_data,
    const T2* rhs_data,
    T* output_data,
    const FuncT& func,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the output shape
    return;

  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _BinaryElementWiseSimple<true, true, T, T1, T2, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      lhs_data,
      rhs_data,
      output_data,
      func,
      N);
}

template <typename T, typename T1, typename T2, typename FuncT>
void BinaryElementWiseImpl(
    cudaStream_t stream,
    int32_t output_rank_or_simple_broadcast,
    const TArray<int64_t>* lhs_padded_strides,
    const T1* lhs_data,
    const TArray<int64_t>* rhs_padded_strides,
    const T2* rhs_data,
    const TArray<fast_divmod>* fdm_output_strides,
    const fast_divmod& fdm_H,
    const fast_divmod& fdm_C,
    T* output_data,
    const FuncT& func,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the output shape
    return;

  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (output_rank_or_simple_broadcast == static_cast<int32_t>(SimpleBroadcast::NoBroadcast)) {
    _BinaryElementWiseSimple<true, true, T, T1, T2, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        lhs_data,
        rhs_data,
        output_data,
        func,
        N);
  } else if (output_rank_or_simple_broadcast == static_cast<int32_t>(SimpleBroadcast::LeftScalar)) {
    _BinaryElementWiseSimple<false, true, T, T1, T2, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        lhs_data,
        rhs_data,
        output_data,
        func,
        N);
  } else if (output_rank_or_simple_broadcast == static_cast<int32_t>(SimpleBroadcast::RightScalar)) {
    _BinaryElementWiseSimple<true, false, T, T1, T2, FuncT, GridDim::maxThreadsPerBlock,
                             GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        lhs_data,
        rhs_data,
        output_data,
        func,
        N);
  } else if (output_rank_or_simple_broadcast == static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatch1)) {
    _BinaryElementWiseRhsPerChannelBatch1<T, T1, T2, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        lhs_data,
        rhs_data,
        fdm_H,
        output_data,
        func,
        N);
  } else if (output_rank_or_simple_broadcast == static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatchN)) {
    _BinaryElementWiseRhsPerChannelBatchN<T, T1, T2, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        lhs_data,
        rhs_data,
        fdm_H,
        fdm_C,
        output_data,
        func,
        N);
  } else {
    if (lhs_padded_strides && rhs_padded_strides && lhs_padded_strides->Size() && rhs_padded_strides->Size())
      _BinaryElementWise<T, T1, T2, FuncT, true, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank_or_simple_broadcast,
          *lhs_padded_strides,
          lhs_data,
          *rhs_padded_strides,
          rhs_data,
          *fdm_output_strides,
          output_data,
          func,
          N);
    else if (lhs_padded_strides && lhs_padded_strides->Size())
      _BinaryElementWise<T, T1, T2, FuncT, true, false, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank_or_simple_broadcast,
          *lhs_padded_strides,
          lhs_data,
          TArray<int64_t>(), // rhs is not computed, so no need to deference rhs_padded_strides
          rhs_data,
          *fdm_output_strides,
          output_data,
          func,
          N);
    else if (rhs_padded_strides && rhs_padded_strides->Size())
      _BinaryElementWise<T, T1, T2, FuncT, false, true, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank_or_simple_broadcast,
          TArray<int64_t>(), // lhs is not computed, so no need to deference lhs_padded_strides
          lhs_data,
          *rhs_padded_strides,
          rhs_data,
          *fdm_output_strides,
          output_data,
          func,
          N);
  }
}

}  // namespace cuda
}  // namespace onnxruntime
