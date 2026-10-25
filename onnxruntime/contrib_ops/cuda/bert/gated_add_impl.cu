// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>

#include "contrib_ops/cuda/bert/gated_add_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
__device__ __forceinline__ T RoundedMulAdd(T x, T y, T gate);

template <>
__device__ __forceinline__ float RoundedMulAdd(float x, float y, float gate) {
  return __fadd_rn(x, __fmul_rn(y, gate));
}

template <>
__device__ __forceinline__ half RoundedMulAdd(half x, half y, half gate) {
  unsigned short product;
  unsigned short output;
  asm volatile("mul.rn.f16 %0, %1, %2;"
               : "=h"(product)
               : "h"(__half_as_ushort(y)), "h"(__half_as_ushort(gate)));
  asm volatile("add.rn.f16 %0, %1, %2;"
               : "=h"(output)
               : "h"(__half_as_ushort(x)), "h"(product));
  return __ushort_as_half(output);
}

template <>
__device__ __forceinline__ __nv_bfloat16 RoundedMulAdd(
    __nv_bfloat16 x, __nv_bfloat16 y, __nv_bfloat16 gate) {
  // FP32 represents BF16 products exactly; round between operations to match separate Mul and Add.
  const __nv_bfloat16 product = __float2bfloat16_rn(
      __fmul_rn(__bfloat162float(y), __bfloat162float(gate)));
  return __float2bfloat16_rn(
      __fadd_rn(__bfloat162float(x), __bfloat162float(product)));
}

template <typename T, typename IndexT, typename DivModT>
__global__ void GatedAddKernel(T* output,
                               const T* x,
                               const T* y,
                               const T* gate,
                               IndexT count,
                               DivModT hidden_size) {
  const IndexT index = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = RoundedMulAdd(x[index], y[index], gate[hidden_size.div(index)]);
  }
}

template <typename T>
__global__ void GatedAddVec8Kernel(uint4* output,
                                   const uint4* x,
                                   const uint4* y,
                                   const T* gate,
                                   int vec_count,
                                   onnxruntime::cuda::fast_divmod vec_hidden_size) {
  const int v = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v < vec_count) {
    const T g = gate[vec_hidden_size.div(v)];
    const uint4 x_u4 = x[v];
    const uint4 y_u4 = y[v];
    uint4 out_u4;
    const T* x_t = reinterpret_cast<const T*>(&x_u4);
    const T* y_t = reinterpret_cast<const T*>(&y_u4);
    T* out_t = reinterpret_cast<T*>(&out_u4);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      out_t[m] = RoundedMulAdd(x_t[m], y_t[m], g);
    }
    output[v] = out_u4;
  }
}

}  // namespace

template <typename T>
Status LaunchGatedAddKernel(cudaStream_t stream,
                            T* output,
                            const T* x,
                            const T* y,
                            const T* gate,
                            int64_t count,
                            int64_t hidden_size) {
  if (count == 0) {
    return Status::OK();
  }

  constexpr int kThreads = 256;
  if constexpr (sizeof(T) == 2) {
    if ((hidden_size & 7) == 0 && (count >> 3) <= std::numeric_limits<int>::max() &&
        (reinterpret_cast<uintptr_t>(output) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(x) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(y) & 15) == 0) {
      const int vec_count = static_cast<int>(count >> 3);
      const int vec_blocks = (vec_count - 1) / kThreads + 1;
      GatedAddVec8Kernel<T><<<vec_blocks, kThreads, 0, stream>>>(
          reinterpret_cast<uint4*>(output),
          reinterpret_cast<const uint4*>(x),
          reinterpret_cast<const uint4*>(y),
          gate,
          vec_count,
          onnxruntime::cuda::fast_divmod(static_cast<int>(hidden_size >> 3)));
      return CUDA_CALL(cudaGetLastError());
    }
  }
  const int64_t blocks = (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(blocks <= std::numeric_limits<int>::max(),
                    "GatedAdd launch requires too many blocks");
  if (count <= std::numeric_limits<int>::max()) {
    GatedAddKernel<T, int, onnxruntime::cuda::fast_divmod><<<static_cast<int>(blocks), kThreads, 0, stream>>>(
        output, x, y, gate, static_cast<int>(count),
        onnxruntime::cuda::fast_divmod(static_cast<int>(hidden_size)));
  } else {
    GatedAddKernel<T, int64_t, onnxruntime::cuda::DivMod<int64_t>><<<static_cast<int>(blocks), kThreads, 0, stream>>>(
        output, x, y, gate, count, onnxruntime::cuda::DivMod<int64_t>(hidden_size));
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_GATED_ADD(T)                                      \
  template Status LaunchGatedAddKernel<T>(cudaStream_t, T*, const T*, \
                                          const T*, const T*, int64_t, int64_t);

INSTANTIATE_GATED_ADD(float)
INSTANTIATE_GATED_ADD(half)
INSTANTIATE_GATED_ADD(__nv_bfloat16)

#undef INSTANTIATE_GATED_ADD

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime