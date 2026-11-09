// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/swiglu_impl.h"

#include "contrib_ops/cuda/math/swiglu_activation.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kThreads = 256;

// The gated activation of a feed-forward or MoE expert block. The graph spells this as two
// casts, two clips, a sigmoid, two multiplies and a cast back -- eight passes over the same
// buffer. Matches MoE/QMoE's alpha/beta formula and positive-limit clamping rule.
// With `Fused`, gate and up are the two halves of one `[.., 2 * cols]` row, so the source
// index skips the half that the other operand owns.
// With `round_activation`, G * Sigmoid(alpha * G) is narrowed to T before the multiply by (L + beta),
// exactly as a QuickGelu node followed by a Mul in T would, so SwiGluFusion does not change results.
template <typename CudaT, bool Fused>
__global__ void SwiGLUKernel(int64_t count, int64_t cols, float limit, float alpha, float beta,
                             bool round_activation, const CudaT* __restrict__ gate,
                             const CudaT* __restrict__ up, CudaT* __restrict__ output) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    const int64_t src = Fused ? i + (i / cols) * cols : i;
    const float g = static_cast<float>(gate[src]);
    const float l = static_cast<float>(up[src]);
    output[i] = static_cast<CudaT>(SwiGluActivation<CudaT>(g, l, limit, alpha, beta, round_activation));
  }
}

}  // namespace

template <typename T>
Status LaunchSwiGLU(cudaStream_t stream, int64_t count, int64_t cols, float limit,
                    float alpha, float beta, bool round_activation,
                    const T* gate, const T* up, T* output) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const int64_t want = (count + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(want < 65535 ? want : 65535);
  auto* g = reinterpret_cast<const CudaT*>(gate);
  auto* u = reinterpret_cast<const CudaT*>(up);
  auto* o = reinterpret_cast<CudaT*>(output);
  if (cols > 0) {
    SwiGLUKernel<CudaT, true><<<blocks, kThreads, 0, stream>>>(count, cols, limit, alpha, beta,
                                                               round_activation, g, u, o);
  } else {
    SwiGLUKernel<CudaT, false><<<blocks, kThreads, 0, stream>>>(count, 0, limit, alpha, beta,
                                                                round_activation, g, u, o);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

#define INSTANTIATE(T)                                                                       \
  template Status LaunchSwiGLU<T>(cudaStream_t, int64_t, int64_t, float, float, float, bool, \
                                  const T*, const T*, T*)

INSTANTIATE(float);
INSTANTIATE(MLFloat16);
INSTANTIATE(BFloat16);

#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
