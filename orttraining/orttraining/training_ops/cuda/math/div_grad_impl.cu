// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace cuda {

// for now this operator classes are no different than a funciton.
// Eventually once multiple binary gradient ops are needed, we will pass
// its instance from API instead of direct function call.
template <class T>
struct OP_A_DivGrad {
  __device__ __inline__ T operator()(T dy, T b) const {
    return dy / b;
  }
};
template <class T>
struct OP_B_DivGrad {
  __device__ __inline__ T operator()(T dy, T a, T b) const {
    return -dy * a / (b * b);
  }
};

template <typename T, bool a_is_scalar, bool b_is_scalar>
__global__ void _DivGradSimple(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    T* output_da_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG a_index = (a_is_scalar ? 0 : id);
  CUDA_LONG b_index = (b_is_scalar ? 0 : id);
  output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
  output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
}

template <typename T, bool a_is_scalar, bool b_is_scalar>
__global__ void _DivGradSimple_A(
    const T* b_data,
    const T* dy_data,
    T* output_da_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG b_index = (b_is_scalar ? 0 : id);
  output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
}

template <typename T, bool a_is_scalar, bool b_is_scalar>
__global__ void _DivGradSimple_B(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG a_index = (a_is_scalar ? 0 : id);
  CUDA_LONG b_index = (b_is_scalar ? 0 : id);
  output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
}

template <typename T>
__global__ void _DivGradRhsPerChannelBatch1(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    T* output_da_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG a_index = id;
  CUDA_LONG b_index = fdm_H.div(id);
  output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
  output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
}

template <typename T>
__global__ void _DivGradRhsPerChannelBatch1_A(
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    T* output_da_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG b_index = fdm_H.div(id);
  output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
}

template <typename T>
__global__ void _DivGradRhsPerChannelBatch1_B(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG a_index = id;
  CUDA_LONG b_index = fdm_H.div(id);
  output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
}

template <typename T>
__global__ void _DivGradRhsPerChannelBatchN(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    const fast_divmod fdm_C,
    T* output_da_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG a_index = id;
  CUDA_LONG b_index = fdm_H.div(id);
  int q, r;
  fdm_C.divmod(b_index, q, r);
  b_index = r;
  output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
  output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
}

template <typename T>
__global__ void _DivGradRhsPerChannelBatchN_A(
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    const fast_divmod fdm_C,
    T* output_da_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG b_index = fdm_H.div(id);
  int q, r;
  fdm_C.divmod(b_index, q, r);
  b_index = r;
  output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
}

template <typename T>
__global__ void _DivGradRhsPerChannelBatchN_B(
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    const fast_divmod fdm_H,
    const fast_divmod fdm_C,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG a_index = id;
  CUDA_LONG b_index = fdm_H.div(id);
  int q, r;
  fdm_C.divmod(b_index, q, r);
  b_index = r;
  output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
}

template <typename T, bool a_need_compute, bool b_need_compute>
__global__ void _DivGrad(
    int32_t output_rank,
    const TArray<int64_t> a_padded_strides,
    const T* a_data,
    const TArray<int64_t> b_padded_strides,
    const T* b_data,
    const T* dy_data,
    const TArray<fast_divmod> fdm_output_strides,
    T* output_da_data,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG a_index = (a_need_compute ? 0 : id);
  CUDA_LONG b_index = (b_need_compute ? 0 : id);
  CUDA_LONG offset = id;
#pragma unroll
  for (auto dim = 0; dim < fdm_output_strides.Capacity(); dim++) {
    if (dim >= output_rank) {
      break;
    }
    int q, r;
    fdm_output_strides[dim].divmod(offset, q, r);
    if (a_need_compute) {
      a_index += static_cast<int>(a_padded_strides[dim]) * q;
    }

    if (b_need_compute) {
      b_index += static_cast<int>(b_padded_strides[dim]) * q;
    }
    offset = r;
  }
  output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
  output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
}

template <typename T, bool b_need_compute>
__global__ void _DivGrad_A(
    int32_t output_rank,
    const TArray<int64_t> b_padded_strides,
    const T* b_data,
    const T* dy_data,
    const TArray<fast_divmod> fdm_output_strides,
    T* output_da_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG b_index = (b_need_compute ? 0 : id);
  CUDA_LONG offset = id;
#pragma unroll
  for (auto dim = 0; dim < fdm_output_strides.Capacity(); dim++) {
    if (dim >= output_rank) {
      break;
    }
    int q, r;
    fdm_output_strides[dim].divmod(offset, q, r);
    if (b_need_compute) {
      b_index += static_cast<int>(b_padded_strides[dim]) * q;
    }
    offset = r;
  }
  output_da_data[id] = OP_A_DivGrad<T>()(dy_data[id], b_data[b_index]);
}

template <typename T, bool a_need_compute, bool b_need_compute>
__global__ void _DivGrad_B(
    int32_t output_rank,
    const TArray<int64_t> a_padded_strides,
    const T* a_data,
    const TArray<int64_t> b_padded_strides,
    const T* b_data,
    const T* dy_data,
    const TArray<fast_divmod> fdm_output_strides,
    T* output_db_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG a_index = (a_need_compute ? 0 : id);
  CUDA_LONG b_index = (b_need_compute ? 0 : id);
  CUDA_LONG offset = id;
#pragma unroll
  for (auto dim = 0; dim < fdm_output_strides.Capacity(); dim++) {
    if (dim >= output_rank) {
      break;
    }
    int q, r;
    fdm_output_strides[dim].divmod(offset, q, r);
    if (a_need_compute) {
      a_index += static_cast<int>(a_padded_strides[dim]) * q;
    }

    if (b_need_compute) {
      b_index += static_cast<int>(b_padded_strides[dim]) * q;
    }
    offset = r;
  }
  output_db_data[id] = OP_B_DivGrad<T>()(dy_data[id], a_data[a_index], b_data[b_index]);
}

template <typename T>
void ImplDivGradSimple(
    cudaStream_t stream,
    SimpleBroadcast simpleBroadcast,
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    size_t count,
    T* da_output_data,
    T* db_output_data) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  switch (simpleBroadcast) {
    case SimpleBroadcast::NoBroadcast:
      // a, b and dy has the same shape: a_is_scalar = false, b_is_scalar = false
      if (da_output_data && db_output_data)
        _DivGradSimple<T, false, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            a_data,
            b_data,
            dy_data,
            da_output_data,
            db_output_data,
            N);
      else if (da_output_data)
        _DivGradSimple_A<T, false, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            b_data,
            dy_data,
            da_output_data,
            N);
      else
        _DivGradSimple_B<T, false, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            a_data,
            b_data,
            dy_data,
            db_output_data,
            N);
      return;
    case SimpleBroadcast::LeftScalar:
      // a is a scalar, b and dy has the same shape
      if (da_output_data && db_output_data)
        _DivGradSimple<T, true, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            a_data,
            b_data,
            dy_data,
            da_output_data,
            db_output_data,
            N);
      else if (da_output_data)
        _DivGradSimple_A<T, true, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            b_data,
            dy_data,
            da_output_data,
            N);
      else
        _DivGradSimple_B<T, true, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            a_data,
            b_data,
            dy_data,
            db_output_data,
            N);
      return;
    case SimpleBroadcast::RightScalar:
      // b is a scalar, a and dy has the same shape
      if (da_output_data && db_output_data)
        _DivGradSimple<T, false, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            a_data,
            b_data,
            dy_data,
            da_output_data,
            db_output_data,
            N);
      else if (da_output_data)
        _DivGradSimple_A<T, false, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            b_data,
            dy_data,
            da_output_data,
            N);
      else
        _DivGradSimple_B<T, false, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            a_data,
            b_data,
            dy_data,
            db_output_data,
            N);
      return;
    default:
      assert(false);
  }
}

template <typename T>
void ImplDivGradRhsPerChannelBatch1(
    cudaStream_t stream,
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    size_t count,
    const fast_divmod& fdm_H,
    T* da_output_data,
    T* db_output_data) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (da_output_data && db_output_data)
    _DivGradRhsPerChannelBatch1<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        a_data,
        b_data,
        dy_data,
        fdm_H,
        da_output_data,
        db_output_data,
        N);
  else if (da_output_data)
    _DivGradRhsPerChannelBatch1_A<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        b_data,
        dy_data,
        fdm_H,
        da_output_data,
        N);
  else
    _DivGradRhsPerChannelBatch1_B<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        a_data,
        b_data,
        dy_data,
        fdm_H,
        db_output_data,
        N);
}

template <typename T>
void ImplDivGradRhsPerChannelBatchN(
    cudaStream_t stream,
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    size_t count,
    const fast_divmod& fdm_H,
    const fast_divmod& fdm_C,
    T* da_output_data,
    T* db_output_data) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  if (da_output_data && db_output_data)
    _DivGradRhsPerChannelBatchN<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        a_data,
        b_data,
        dy_data,
        fdm_H,
        fdm_C,
        da_output_data,
        db_output_data,
        N);
  else if (da_output_data)
    _DivGradRhsPerChannelBatchN_A<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        b_data,
        dy_data,
        fdm_H,
        fdm_C,
        da_output_data,
        N);
  else
    _DivGradRhsPerChannelBatchN_B<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        a_data,
        b_data,
        dy_data,
        fdm_H,
        fdm_C,
        db_output_data,
        N);
}

template <typename T>
void ImplDivGrad(
    cudaStream_t stream,
    int32_t output_rank,
    const TArray<int64_t>* a_padded_strides,
    const T* a_data,
    const TArray<int64_t>* b_padded_strides,
    const T* b_data,
    const T* dy_data,
    size_t count,
    const TArray<fast_divmod>* fdm_output_strides,
    T* da_output_data,
    T* db_output_data) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (a_padded_strides && a_padded_strides->Size() && b_padded_strides && b_padded_strides->Size()) {
    if (da_output_data && db_output_data)
      _DivGrad<T, true, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank,
          *a_padded_strides,
          a_data,
          *b_padded_strides,
          b_data,
          dy_data,
          *fdm_output_strides,
          da_output_data,
          db_output_data,
          N);
    else if (da_output_data)
      _DivGrad_A<T, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank,
          *b_padded_strides,
          b_data,
          dy_data,
          *fdm_output_strides,
          da_output_data,
          N);
    else
      _DivGrad_B<T, true, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank,
          *a_padded_strides,
          a_data,
          *b_padded_strides,
          b_data,
          dy_data,
          *fdm_output_strides,
          db_output_data,
          N);
  } else if (a_padded_strides && a_padded_strides->Size()) {
    if (da_output_data && db_output_data)
      _DivGrad<T, true, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank,
          *a_padded_strides,
          a_data,
          *b_padded_strides,
          b_data,
          dy_data,
          *fdm_output_strides,
          da_output_data,
          db_output_data,
          N);
    else if (da_output_data)
      _DivGrad_A<T, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank,
          *b_padded_strides,
          b_data,
          dy_data,
          *fdm_output_strides,
          da_output_data,
          N);
    else
      _DivGrad_B<T, true, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank,
          *a_padded_strides,
          a_data,
          *b_padded_strides,
          b_data,
          dy_data,
          *fdm_output_strides,
          db_output_data,
          N);
  } else {
    if (da_output_data && db_output_data)
      _DivGrad<T, false, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank,
          *a_padded_strides,
          a_data,
          *b_padded_strides,
          b_data,
          dy_data,
          *fdm_output_strides,
          da_output_data,
          db_output_data,
          N);
    else if (da_output_data)
      _DivGrad_A<T, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank,
          *b_padded_strides,
          b_data,
          dy_data,
          *fdm_output_strides,
          da_output_data,
          N);
    else
      _DivGrad_B<T, false, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_rank,
          *a_padded_strides,
          a_data,
          *b_padded_strides,
          b_data,
          dy_data,
          *fdm_output_strides,
          db_output_data,
          N);
  }
}  // namespace cuda

#define SPECIALIZED_DIV_GRAD_IMPL(T)                 \
  template void ImplDivGrad<T>(                      \
      cudaStream_t stream,                           \
      int32_t output_rank,                           \
      const TArray<int64_t>* a_padded_strides,       \
      const T* a_data,                               \
      const TArray<int64_t>* b_padded_strides,       \
      const T* b_data,                               \
      const T* dy_data,                              \
      size_t count,                                  \
      const TArray<fast_divmod>* fdm_output_strides, \
      T* da_output_data,                             \
      T* db_output_data);                            \
  template void ImplDivGradRhsPerChannelBatch1<T>(   \
      cudaStream_t stream,                           \
      const T* a_data,                               \
      const T* b_data,                               \
      const T* dy_data,                              \
      size_t count,                                  \
      const fast_divmod& fdm_H,                      \
      T* da_output_data,                             \
      T* db_output_data);                            \
  template void ImplDivGradRhsPerChannelBatchN<T>(   \
      cudaStream_t stream,                           \
      const T* a_data,                               \
      const T* b_data,                               \
      const T* dy_data,                              \
      size_t count,                                  \
      const fast_divmod& fdm_H,                      \
      const fast_divmod& fdm_C,                      \
      T* da_output_data,                             \
      T* db_output_data);                            \
  template void ImplDivGradSimple<T>(                \
      cudaStream_t stream,                           \
      SimpleBroadcast simpleBroadcast,               \
      const T* a_data,                               \
      const T* b_data,                               \
      const T* dy_data,                              \
      size_t count,                                  \
      T* da_output_data,                             \
      T* db_output_data);

SPECIALIZED_DIV_GRAD_IMPL(half)
SPECIALIZED_DIV_GRAD_IMPL(float)
SPECIALIZED_DIV_GRAD_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime
