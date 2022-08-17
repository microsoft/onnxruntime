// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

__device__ inline float ToFloat(const __half h) { return __half2float(h); }

__device__ inline float ToFloat(const float f) { return f; }

template <typename T>
__inline__ __device__ T
WarpReduceSum(T val) {
  val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
  return val;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
