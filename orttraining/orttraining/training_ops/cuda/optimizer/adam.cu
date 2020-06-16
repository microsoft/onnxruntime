// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/optimizer/common.cuh"
#include "orttraining/training_ops/cuda/optimizer/adam.h"
#include "orttraining/training_ops/cuda/optimizer/common.h"

namespace onnxruntime {
namespace cuda {
template <typename T1, typename T3, typename T4, typename T_GRAD, typename T_GRAD_NORM>
__global__ void _AdamOptimizer_mode0(
    const T1* eta,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
    const T_GRAD_NORM* grad_norm,
    const T4 alpha,
    const T4 beta,
    const T4 lambda,
    const T4 epsilon,
    const T4 alpha_correction,
    const T4 beta_correction,
    T4* moment_1_out,
    T4* moment_2_out,
    T3* weights_out,
    T_GRAD* grads_out,
    half* fp16_weights_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  const T4 actual_scale = _ComputeGradScale<T3, T_GRAD_NORM, T4>(loss_scale, grad_norm);

  // Gradient scaling/clipping.
  const T4 g = T4(grads[id]) / actual_scale;
  // A shared constant.
  const T4 one = T4(1.0f);

  // Compute exponentially-averaged historical gradient.
  const T4 m1o = alpha * moment_1[id] + (one - alpha) * g;
  const T4 m1o_corrected = m1o / alpha_correction;

  // Compute exponentially-averaged historical squared gradient.
  const T4 m2o = beta * moment_2[id] + (one - beta) * g * g;
  const T4 m2o_corrected = m2o / beta_correction;

  // Compute weight update.
  const T4 denom = _Sqrt(m2o_corrected) + epsilon;
  const T4 update = (m1o_corrected / denom) + (lambda * T4(weights[id]));

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

template <typename T1, typename T3, typename T4, typename T_GRAD, typename T_GRAD_NORM>
__global__ void _AdamOptimizer_mode1(
    const T1* eta,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
    const T_GRAD_NORM* grad_norm,
    const T4 alpha,
    const T4 beta,
    const T4 lambda,
    const T4 epsilon,
    const T4 alpha_correction,
    const T4 beta_correction,
    T4* moment_1_out,
    T4* moment_2_out,
    T3* weights_out,
    T_GRAD* grads_out,
    half* fp16_weights_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  const T4 actual_scale = _ComputeGradScale<T3, T_GRAD_NORM, T4>(loss_scale, grad_norm);

  // Gradient scaling/clipping.
  const T4 g = T4(grads[id]) / actual_scale;
  // A shared constant.
  const T4 one = T4(1.0f);

  // Compute exponentially-averaged historical gradient.
  const T4 m1o = alpha * moment_1[id] + (one - alpha) * g;

  // Compute exponentially-averaged historical squared gradient.
  const T4 m2o = beta * moment_2[id] + (one - beta) * g * g;

  const T4 denom = _Sqrt(m2o) + epsilon;

  // Apply bias correction terms on learning rate
  const T4 step_size = T4(*eta) * _Sqrt(beta_correction) / alpha_correction;

  // Huggingface updates weights in the following logic:
  // param' = param - step_size * m1o / denom
  // param_out = param' - original_lr * lambda * param'
  // then param_out = param - step_size * m1o / denom - original_lr * lambda * (param - step_size * m1o / denom)
  // so delta = -step_size * m1o / denom - original_lr * lambda * (param - step_size * m1o / denom)
  const T4 delta = -step_size * m1o / denom - T4(*eta) * lambda * (T4(weights[id]) - step_size * m1o / denom);

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
    const T4 alpha,
    const T4 beta,
    const T4 lambda,
    const T4 epsilon,
    const bool do_bias_correction,
    const int64_t weight_decay_mode,
    T4* moment_1_out,
    T4* moment_2_out,
    T3* weights_out,
    T_GRAD* grads_out,
    half* fp16_weights_out,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  // If bias correction coefficients are set to 1s, it's equivalent to disabling bias correction. 
  const T4 alpha_correction = do_bias_correction ? 
    onnxruntime::contrib::compute_bias_correction_coefficient(alpha, update_count) : T4(1.f);
  const T4 beta_correction = do_bias_correction ?
    onnxruntime::contrib::compute_bias_correction_coefficient(beta, update_count) : T4(1.f);
  
  // Currently two modes of Adamw are supported:
  // Mode 0: Pytorch https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW,
  //         bias correction is applied on m and v individually,
  //         weight decay is applied before weight is updated.
  // Mode 1: Huggingface https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW.,
  //         bias correction is applied on learning rate,
  //         weight decay is applied after weight is updated.
  if (weight_decay_mode == 0) {
    _AdamOptimizer_mode0<T1, T3, T4, T_GRAD, T_GRAD_NORM><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      eta,
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
      alpha_correction,
      beta_correction,
      moment_1_out,
      moment_2_out,
      weights_out,
      grads_out,
      fp16_weights_out,
      N);
  }
  else if (weight_decay_mode == 1) {
    _AdamOptimizer_mode1<T1, T3, T4, T_GRAD, T_GRAD_NORM><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      eta,
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
      alpha_correction,
      beta_correction,
      moment_1_out,
      moment_2_out,
      weights_out,
      grads_out,
      fp16_weights_out,
      N);
  }
  else {
    // Shouldn't reach here
    ORT_THROW("Unsupported Adamw optimizer mode.");
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
      const T4 alpha,                                                      \
      const T4 beta,                                                       \
      const T4 lambda,                                                     \
      const T4 epsilon,                                                    \
      const bool do_bias_correction,                                       \
      const int64_t weight_decay_mode,                                     \
      T4* moment_1_out,                                                    \
      T4* moment_2_out,                                                    \
      T3* weights_out,                                                     \
      T_GRAD* grads_out,                                                   \
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
