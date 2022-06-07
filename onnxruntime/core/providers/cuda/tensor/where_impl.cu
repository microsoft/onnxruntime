// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __GNUC__
#include "onnxruntime_config.h"
#pragma GCC diagnostic ignored "-Wswitch"
#endif
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "where_impl.h"

namespace onnxruntime {
namespace cuda {

// broadcast by computing output coordinate from offset, using fast_divmod
template <typename T, BroadcastIndexType CondIndexType, BroadcastIndexType XIndexType, BroadcastIndexType YIndexType, int NumThreadsPerBlock, int NumElementsPerThread>
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
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  bool cond_value[NumElementsPerThread];
  T x_value[NumElementsPerThread];
  T y_value[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      // compute indexes with broadcasting rules: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
      CUDA_LONG cond_index = (CondIndexType == BroadcastIndexType::NoBroadcast ? id : 0);
      CUDA_LONG x_index = (XIndexType == BroadcastIndexType::NoBroadcast ? id : 0);
      CUDA_LONG y_index = (YIndexType == BroadcastIndexType::NoBroadcast ? id : 0);
      CUDA_LONG offset = id;
#pragma unroll
      for (auto dim = 0; dim < fdm_output_strides.Capacity(); dim++) {
        if (dim >= output_rank) {
          break;
        }

        int q, r;
        fdm_output_strides[dim].divmod(offset, q, r);

        if (CondIndexType == BroadcastIndexType::NeedCompute) {
          cond_index += static_cast<int>(cond_padded_strides[dim]) * q;
        }

        if (XIndexType == BroadcastIndexType::NeedCompute) {
          x_index += static_cast<int>(x_padded_strides[dim]) * q;
        }

        if (YIndexType == BroadcastIndexType::NeedCompute) {
          y_index += static_cast<int>(y_padded_strides[dim]) * q;
        }

        offset = r;
      }

      cond_value[i] = cond_data[cond_index];
      x_value[i] = x_data[x_index];
      y_value[i] = y_data[y_index];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = cond_value[i] ? x_value[i] : y_value[i];
      id += NumThreadsPerBlock;
    }
  }
}

// for scalar broadcast or non-broadcast case
template <typename T, BroadcastIndexType CondIndexType, BroadcastIndexType XIndexType, BroadcastIndexType YIndexType, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _TenaryElementWiseSimple(
    const bool* cond_data,
    const T* x_data,
    const T* y_data,
    T* output_data,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  bool cond_value[NumElementsPerThread];
  T x_value[NumElementsPerThread];
  T y_value[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      cond_value[i] = cond_data[CondIndexType == BroadcastIndexType::NoBroadcast ? id : 0];
      x_value[i] = x_data[XIndexType == BroadcastIndexType::NoBroadcast ? id : 0];
      y_value[i] = y_data[YIndexType == BroadcastIndexType::NoBroadcast ? id : 0];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = cond_value[i] ? x_value[i] : y_value[i];
      id += NumThreadsPerBlock;
    }
  }
}

#define HANDLE_Y_INDEX_TYPE_SIMPLE(COND_INDEX_TYPE, X_INDEX_TYPE, Y_INDEX_TYPE) \
  case Y_INDEX_TYPE: {                                                          \
    _TenaryElementWiseSimple<T,                                                 \
                             COND_INDEX_TYPE,                                   \
                             X_INDEX_TYPE,                                      \
                             Y_INDEX_TYPE,                                      \
                             GridDim::maxThreadsPerBlock,                       \
                             GridDim::maxElementsPerThread>                     \
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(cond_data,  \
                                                            x_data,             \
                                                            y_data,             \
                                                            output_data,        \
                                                            N);                 \
  } break

#define HANDLE_X_INDEX_TYPE_SIMPLE(COND_INDEX_TYPE, X_INDEX_TYPE, Y_INDEX_TYPE_VAL)               \
  case X_INDEX_TYPE: {                                                                            \
    switch (Y_INDEX_TYPE_VAL) {                                                                   \
      HANDLE_Y_INDEX_TYPE_SIMPLE(COND_INDEX_TYPE, X_INDEX_TYPE, BroadcastIndexType::NoBroadcast); \
      HANDLE_Y_INDEX_TYPE_SIMPLE(COND_INDEX_TYPE, X_INDEX_TYPE, BroadcastIndexType::Scalar);      \
    }                                                                                             \
  } break

#define HANDLE_COND_INDEX_TYPE_SIMPLE(COND_INDEX_TYPE, X_INDEX_TYPE_VAL, Y_INDEX_TYPE_VAL)            \
  case COND_INDEX_TYPE: {                                                                             \
    switch (X_INDEX_TYPE_VAL) {                                                                       \
      HANDLE_X_INDEX_TYPE_SIMPLE(COND_INDEX_TYPE, BroadcastIndexType::NoBroadcast, Y_INDEX_TYPE_VAL); \
      HANDLE_X_INDEX_TYPE_SIMPLE(COND_INDEX_TYPE, BroadcastIndexType::Scalar, Y_INDEX_TYPE_VAL);      \
    }                                                                                                 \
  } break

#define HANDLE_Y_INDEX_TYPE(COND_INDEX_TYPE, X_INDEX_TYPE, Y_INDEX_TYPE)                     \
  case Y_INDEX_TYPE: {                                                                       \
    _TenaryElementWise<T,                                                                    \
                       COND_INDEX_TYPE,                                                      \
                       X_INDEX_TYPE,                                                         \
                       Y_INDEX_TYPE,                                                         \
                       GridDim::maxThreadsPerBlock,                                          \
                       GridDim::maxElementsPerThread>                                        \
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(output_rank_or_simple_broadcast, \
                                                            cond_padded_strides,             \
                                                            cond_data,                       \
                                                            x_padded_strides,                \
                                                            x_data,                          \
                                                            y_padded_strides,                \
                                                            y_data,                          \
                                                            fdm_output_strides,              \
                                                            output_data,                     \
                                                            N);                              \
  } break

#define HANDLE_X_INDEX_TYPE(COND_INDEX_TYPE, X_INDEX_TYPE, Y_INDEX_TYPE_VAL)               \
  case X_INDEX_TYPE: {                                                                     \
    switch (Y_INDEX_TYPE_VAL) {                                                            \
      HANDLE_Y_INDEX_TYPE(COND_INDEX_TYPE, X_INDEX_TYPE, BroadcastIndexType::NoBroadcast); \
      HANDLE_Y_INDEX_TYPE(COND_INDEX_TYPE, X_INDEX_TYPE, BroadcastIndexType::Scalar);      \
      HANDLE_Y_INDEX_TYPE(COND_INDEX_TYPE, X_INDEX_TYPE, BroadcastIndexType::NeedCompute); \
    }                                                                                      \
  } break

#define HANDLE_COND_INDEX_TYPE(COND_INDEX_TYPE, X_INDEX_TYPE_VAL, Y_INDEX_TYPE_VAL)            \
  case COND_INDEX_TYPE: {                                                                      \
    switch (X_INDEX_TYPE_VAL) {                                                                \
      HANDLE_X_INDEX_TYPE(COND_INDEX_TYPE, BroadcastIndexType::NoBroadcast, Y_INDEX_TYPE_VAL); \
      HANDLE_X_INDEX_TYPE(COND_INDEX_TYPE, BroadcastIndexType::Scalar, Y_INDEX_TYPE_VAL);      \
      HANDLE_X_INDEX_TYPE(COND_INDEX_TYPE, BroadcastIndexType::NeedCompute, Y_INDEX_TYPE_VAL); \
    }                                                                                          \
  } break

template <typename T>
void WhereImpl(
    cudaStream_t stream,
    size_t output_rank_or_simple_broadcast,
    BroadcastIndexType cond_index_type,
    const TArray<int64_t>& cond_padded_strides,
    const bool* cond_data,
    BroadcastIndexType x_index_type,
    const TArray<int64_t>& x_padded_strides,
    const T* x_data,
    BroadcastIndexType y_index_type,
    const TArray<int64_t>& y_padded_strides,
    const T* y_data,
    const TArray<fast_divmod>& fdm_output_strides,
    T* output_data,
    size_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (output_rank_or_simple_broadcast == static_cast<size_t>(SimpleBroadcast::NoBroadcast)) {
    switch (cond_index_type) {
      HANDLE_COND_INDEX_TYPE_SIMPLE(BroadcastIndexType::NoBroadcast, x_index_type, y_index_type);
      HANDLE_COND_INDEX_TYPE_SIMPLE(BroadcastIndexType::Scalar, x_index_type, y_index_type);
    }
  } else {
    switch (cond_index_type) {
      HANDLE_COND_INDEX_TYPE(BroadcastIndexType::NoBroadcast, x_index_type, y_index_type);
      HANDLE_COND_INDEX_TYPE(BroadcastIndexType::Scalar, x_index_type, y_index_type);
      HANDLE_COND_INDEX_TYPE(BroadcastIndexType::NeedCompute, x_index_type, y_index_type);
    }
  }
}

#define SPECIALIZED_IMPL(T)                                                 \
  template void WhereImpl<T>(cudaStream_t stream,                           \
                             size_t output_rank_or_simple_broadcast,        \
                             BroadcastIndexType cond_index_type,            \
                             const TArray<int64_t>& cond_padded_strides,    \
                             const bool* cond_data,                         \
                             BroadcastIndexType x_index_type,               \
                             const TArray<int64_t>& x_padded_strides,       \
                             const T* x_data,                               \
                             BroadcastIndexType y_index_type,               \
                             const TArray<int64_t>& y_padded_strides,       \
                             const T* y_data,                               \
                             const TArray<fast_divmod>& fdm_output_strides, \
                             T* output_data,                                \
                             size_t count);

SPECIALIZED_IMPL(uint8_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double_t)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
