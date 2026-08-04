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
template <typename CudaT>
__global__ void DSV4SwiGLUKernel(int64_t count, float limit, const CudaT* __restrict__ gate,
                                 const CudaT* __restrict__ up, CudaT* __restrict__ output) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    float g = DSV4Conv<CudaT>::ToFloat(gate[i]);
    float l = DSV4Conv<CudaT>::ToFloat(up[i]);
    if (limit > 0.0f) {
      g = fminf(g, limit);
      l = fminf(fmaxf(l, -limit), limit);
    }
    output[i] = DSV4Conv<CudaT>::FromFloat(g * Sigmoid(g) * l);
  }
}

}  // namespace

template <typename T>
Status LaunchDSV4SwiGLU(cudaStream_t stream, int64_t count, float limit, const T* gate,
                        const T* up, T* output) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const int64_t want = (count + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(want < 65535 ? want : 65535);
  DSV4SwiGLUKernel<CudaT><<<blocks, kThreads, 0, stream>>>(
      count, limit, reinterpret_cast<const CudaT*>(gate), reinterpret_cast<const CudaT*>(up),
      reinterpret_cast<CudaT*>(output));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

#define INSTANTIATE(T) \
  template Status LaunchDSV4SwiGLU<T>(cudaStream_t, int64_t, float, const T*, const T*, T*)

INSTANTIATE(float);
INSTANTIATE(MLFloat16);
INSTANTIATE(BFloat16);

#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
