// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dsv4_swiglu_impl.h"

#include "contrib_ops/cuda/math/dsv4_common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kThreads = 256;

// ORT's Sigmoid keeps the exponent non-positive on both branches.
__device__ __forceinline__ float Sigmoid(float a) {
  return a > 0.0f ? 1.0f / (1.0f + expf(-a)) : 1.0f - 1.0f / (1.0f + expf(a));
}

// The shared expert's gated activation. The graph spells this as two casts, two clips, a
// sigmoid, two multiplies and a cast back -- eight passes over the same buffer.
// With `Fused`, gate and up are the two halves of one `[.., 2 * cols]` row, so the source
// index skips the half that the other operand owns.
template <typename CudaT, bool Fused>
__global__ void DSV4SwiGLUKernel(int64_t count, int64_t cols, float limit,
                                 const CudaT* __restrict__ gate, const CudaT* __restrict__ up,
                                 CudaT* __restrict__ output) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    const int64_t src = Fused ? i + (i / cols) * cols : i;
    float g = DSV4Conv<CudaT>::ToFloat(gate[src]);
    float l = DSV4Conv<CudaT>::ToFloat(up[src]);
    if (limit > 0.0f) {
      g = fminf(g, limit);
      l = fminf(fmaxf(l, -limit), limit);
    }
    output[i] = DSV4Conv<CudaT>::FromFloat(g * Sigmoid(g) * l);
  }
}

}  // namespace

template <typename T>
Status LaunchDSV4SwiGLU(cudaStream_t stream, int64_t count, int64_t cols, float limit,
                        const T* gate, const T* up, T* output) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const int64_t want = (count + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(want < 65535 ? want : 65535);
  auto* g = reinterpret_cast<const CudaT*>(gate);
  auto* u = reinterpret_cast<const CudaT*>(up);
  auto* o = reinterpret_cast<CudaT*>(output);
  if (cols > 0) {
    DSV4SwiGLUKernel<CudaT, true><<<blocks, kThreads, 0, stream>>>(count, cols, limit, g, u, o);
  } else {
    DSV4SwiGLUKernel<CudaT, false><<<blocks, kThreads, 0, stream>>>(count, 0, limit, g, u, o);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

#define INSTANTIATE(T)                                                                 \
  template Status LaunchDSV4SwiGLU<T>(cudaStream_t, int64_t, int64_t, float, const T*, \
                                      const T*, T*)

INSTANTIATE(float);
INSTANTIATE(MLFloat16);
INSTANTIATE(BFloat16);

#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
