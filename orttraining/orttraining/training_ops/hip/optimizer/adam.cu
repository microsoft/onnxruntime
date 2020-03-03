// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/cu_inc/common.cuh"
#include "adam.h"

namespace onnxruntime {
namespace hip {

template <typename T1, typename T3, typename T4, typename T_GRAD>
__global__ void _AdamOptimizer(
    const T1* eta,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
    const T4 alpha,
    const T4 beta,
    const T4 lambda,
    const T4 epsilon,
    T4* moment_1_out,
    T4* moment_2_out,
    T3* weights_out,
    T_GRAD* grads_out,
    half* fp16_weights_out,
    HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  T4 g_scale = loss_scale ? T4(*loss_scale) : T4(1.f);

  // Gradient scaling/clipping.
  const T4 g = T4(grads[id]) / g_scale;

  // A shared constant.
  const T4 one = T4(1.0f);

  // Compute exponentially-averaged historical gradient.
  const T4 m1o = alpha * moment_1[id] + (one - alpha) * g;

  // Compute exponentially-averaged historical squared gradient.
  const T4 m2o = beta * moment_2[id] + (one - beta) * g * g;

  // Compute weight update.
  const T4 denom = _Sqrt(m2o) + epsilon;
  const T4 update = (m1o / denom) + (lambda * T4(weights[id]));
  const T4 delta = -T4(*eta) * update;

  // Compute the new gradient.
  if (grads_out) {
    grads_out[id] = T_GRAD(delta);
  }

  // Compute the new weight.
  if (weights_out) {
    weights_out[id] = weights[id] + T3(delta);

    if (fp16_weights_out) {
      fp16_weights_out[id] = static_cast<half>(weights_out[id]);
    }
  }

  moment_1_out[id] = m1o;
  moment_2_out[id] = m2o;
}

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
void AdamOptimizerImpl(
    const T1* eta,
    const T2 /*update_count*/,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
    T4 alpha,
    T4 beta,
    T4 lambda,
    T4 epsilon,
    T4* moment_1_out,
    T4* moment_2_out,
    T3* weights_out,
    T_GRAD* grads_out,
    half* fp16_weights_out,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  HIP_LONG N = static_cast<HIP_LONG>(count);

  hipLaunchKernelGGL(_AdamOptimizer<T1, T3, T4, T_GRAD>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
      eta,
      weights,
      grads,
      moment_1,
      moment_2,
      loss_scale,
      alpha,
      beta,
      lambda,
      epsilon,
      moment_1_out,
      moment_2_out,
      weights_out,
      grads_out,
      fp16_weights_out,
      N);
}

#define SPECIALIZED_AdamOptimizerImpl(T1, T2, T3, T4, T_GRAD)              \
  template void AdamOptimizerImpl(                                         \
      const T1* eta,                                                       \
      const T2 update_count,                                               \
      const T3* weights,                                                   \
      const T_GRAD* grads,                                                 \
      const T4* moment_1,                                                  \
      const T4* moment_2,                                                  \
      const T3* loss_scale,                                                \
      T4 alpha,                                                            \
      T4 beta,                                                             \
      T4 lambda,                                                           \
      T4 epsilon,                                                          \
      T4* moment_1_out,                                                    \
      T4* moment_2_out,                                                    \
      T3* weights_out,                                                     \
      T_GRAD* grads_out,                                                   \
      half* fp16_weights_out,                                              \
      size_t count);

SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, float, float)
SPECIALIZED_AdamOptimizerImpl(half, int64_t, float, half, float)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, half, float)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, float, half)
SPECIALIZED_AdamOptimizerImpl(half, int64_t, float, half, half)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, half, half)

}  // namespace hip
}  // namespace onnxruntime
