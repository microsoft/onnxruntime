// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"
#include "adam.h"

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD, typename T_GRAD_NORM, bool update_fp16_weight, bool has_loss_scale>
__global__ void _AdamOptimizer(
    const T1* eta,
    const T2 update_count,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
    const T_GRAD_NORM* grad_norm,
    T4 alpha,
    T4 beta,
    T4 lambda,
    T4 epsilon,
    T3* weights_out,
    T4* moment_1_out,
    T4* moment_2_out,
    half* fp16_weights_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  T_GRAD_NORM g_scale = has_loss_scale ? T_GRAD_NORM(*loss_scale) : T_GRAD_NORM(1.f);
  if (grad_norm != nullptr && *grad_norm > g_scale) {
    const T_GRAD_NORM actual_g_norm = *grad_norm / g_scale;
    g_scale *= actual_g_norm;
  }

  // Gradient scaling/clipping.
  const T_GRAD new_grad = T_GRAD(T_GRAD_NORM(grads[id]) / g_scale);

  // Regularize gradient.
  const T4 g_regularized = lambda * T4(weights[id]) + T4(new_grad);

  // A shared constant.
  const T4 one = T4(1.0f);

  // Compute exponentially-averaged historical gradient.
  T4 m1o = alpha * moment_1[id] + (one - alpha) * g_regularized;

  // Compute exponentially-averaged historical squared gradient.
  T4 m2o = beta * moment_2[id] + (one - beta) * g_regularized * g_regularized;

  // Update learning rate - Use the updated eta for the final weight update.
  const T4 count = T4(static_cast<long long>(update_count));
  const T4 numerator = _Sqrt(one - _Pow(beta, count));
  const T4 denom = one - _Pow(alpha, count);
  const T4 eta_new = T4(*eta) * numerator / denom;

  // Compute the new weight.
  weights_out[id] = weights[id] -
                    T3(eta_new * m1o / (_Sqrt(m2o) + epsilon));

  if (update_fp16_weight) {
    fp16_weights_out[id] = static_cast<half>(weights_out[id]);
  }

  moment_1_out[id] = m1o;
  moment_2_out[id] = m2o;
}

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD, typename T_GRAD_NORM>
void AdamOptimizerImpl(
    const T1* eta,
    const T2 update_count,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
    const T_GRAD_NORM* grad_norm,
    T4 alpha,
    T4 beta,
    T4 lambda,
    T4 epsilon,
    T3* weights_out,
    T4* moment_1_out,
    T4* moment_2_out,
    half* fp16_weights_out,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  if (fp16_weights_out != nullptr && loss_scale != nullptr) {
    _AdamOptimizer<T1, T2, T3, T4, T_GRAD, T_GRAD_NORM, true, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        update_count,
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
        grad_norm,
        alpha,
        beta,
        lambda,
        epsilon,
        weights_out,
        moment_1_out,
        moment_2_out,
        fp16_weights_out,
        N);
  } else if (fp16_weights_out != nullptr && loss_scale == nullptr) {
    _AdamOptimizer<T1, T2, T3, T4, T_GRAD, T_GRAD_NORM, true, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        update_count,
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
        grad_norm,
        alpha,
        beta,
        lambda,
        epsilon,
        weights_out,
        moment_1_out,
        moment_2_out,
        fp16_weights_out,
        N);
  } else if (fp16_weights_out == nullptr && loss_scale != nullptr) {
    _AdamOptimizer<T1, T2, T3, T4, T_GRAD, T_GRAD_NORM, false, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        update_count,
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
        grad_norm,
        alpha,
        beta,
        lambda,
        epsilon,
        weights_out,
        moment_1_out,
        moment_2_out,
        fp16_weights_out,
        N);
  } else {
    _AdamOptimizer<T1, T2, T3, T4, T_GRAD, T_GRAD_NORM, false, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        update_count,
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
        grad_norm,
        alpha,
        beta,
        lambda,
        epsilon,
        weights_out,
        moment_1_out,
        moment_2_out,
        nullptr,
        N);
  }
}

#define SPECIALIZED_AdamOptimizerImpl(T1, T2, T3, T4, T_GRAD, T_GRAD_NORM) \
  template void AdamOptimizerImpl(                                         \
      const T1* eta,                                                       \
      const T2 update_count,                                               \
      const T3* weights,                                                   \
      const T_GRAD* grads,                                                 \
      const T4* moment_1,                                                  \
      const T4* moment_2,                                                  \
      const T3* loss_scale,                                                \
      const T_GRAD_NORM* grad_norm,                                        \
      T4 alpha,                                                            \
      T4 beta,                                                             \
      T4 lambda,                                                           \
      T4 epsilon,                                                          \
      T3* weights_out,                                                     \
      T4* moment_1_out,                                                    \
      T4* moment_2_out,                                                    \
      half* fp16_weights_out,                                              \
      size_t count);

SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, float, float, float)
SPECIALIZED_AdamOptimizerImpl(half, int64_t, float, half, float, float)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, half, float, float)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, float, half, half)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, float, half, float)
SPECIALIZED_AdamOptimizerImpl(half, int64_t, float, half, half, half)
SPECIALIZED_AdamOptimizerImpl(half, int64_t, float, half, half, float)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, half, half, half)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, half, half, float)

}  // namespace cuda
}  // namespace onnxruntime
