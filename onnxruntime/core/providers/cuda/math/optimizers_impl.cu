// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "optimizers.h"
#include "core/providers/cuda/cuda_common.h"


namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _SGDOptimizer(
    const T* eta,
    const T* weights,
    const T* gradients,
    T* weights_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  weights_out[id] = weights[id] - ((*eta) * gradients[id]);
}

template <typename T>
void SGDOptimizerImpl(
    const T* eta,
    const T* weights,
    const T* gradients,
    T* weights_out,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _SGDOptimizer<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      eta,
      weights,
      gradients,
      weights_out,
      N);
}

#define SPECIALIZED_IMPL__SGDOptimizerImpl(T)      \
template void SGDOptimizerImpl(                    \
    const T* eta,                                  \
    const T* weights,                              \
    const T* gradients,                            \
    T* weights_out,                                \
    size_t count);

SPECIALIZED_IMPL__SGDOptimizerImpl(float)

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
__global__ void _AdamOptimizer(
    const T1* eta,
    const T2* update_count,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    T4 alpha,
    T4 beta,
    T4 lambda,
    T4 epsilon,
    T3* weights_out,
    T4* moment_1_out,
    T4* moment_2_out,
    int64_t* update_count_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  // Regularize gradient.
  const T4 g_regularized = lambda * T4(weights[id]) + T4(grads[id]);

  // A shared constant. 
  const T4 one = T4(1.0f);

  // Update exponentially-averaged historical gradient.
  moment_1_out[id] = \
    alpha * moment_1[id] + (one - alpha) * g_regularized;

  // Update exponentially-averaged historical squared gradient.
  moment_2_out[id] = \
    beta * moment_2[id] + (one  - beta) * g_regularized * g_regularized;

  // Update learning rate - Use the updated eta for the final weight update.
  const T4 count = T4(static_cast<long long>(*update_count));
  const T4 numerator = _Sqrt(one - _Pow(beta, count));
  const T4 denom = one - _Pow(alpha, count);
  const T4 eta_new = T4(*eta) * numerator / denom;

  // Compute the new weight.
  weights_out[id] = weights[id] - \
    T3(eta_new * moment_1_out[id] / (_Sqrt(moment_2_out[id]) + epsilon));
  *update_count_out = (*update_count) + 1;
}

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
void AdamOptimizerImpl(
    const T1* eta,
    const T2* update_count,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    T4 alpha,
    T4 beta,
    T4 lambda,
    T4 epsilon,
    T3* weights_out,
    T4* moment_1_out,
    T4* moment_2_out,
    T2* update_count_out,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _AdamOptimizer<T1, T2, T3, T4, T_GRAD><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      eta,
      update_count,
      weights,
      grads,
      moment_1,
      moment_2,
      alpha,
      beta,
      lambda,
      epsilon,
      weights_out,
      moment_1_out,
      moment_2_out,
      update_count_out,
      N);
}

#define SPECIALIZED_AdamOptimizerImpl(T1, T2, T3, T4, T_GRAD) \
template void AdamOptimizerImpl(                              \
    const T1* eta,                                            \
    const T2* update_count,                                   \
    const T3* weights,                                        \
    const T_GRAD* grads,                                      \
    const T4* moment_1,                                       \
    const T4* moment_2,                                       \
    T4 alpha,                                                 \
    T4 beta,                                                  \
    T4 lambda,                                                \
    T4 epsilon,                                               \
    T3* weights_out,                                          \
    T4* moment_1_out,                                         \
    T4* moment_2_out,                                         \
    T2* update_count_out,                                     \
    size_t count);

SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, float, float)
SPECIALIZED_AdamOptimizerImpl(half, int64_t, float, half, float)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, half, float)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, float, half)
SPECIALIZED_AdamOptimizerImpl(half, int64_t, float, half, half)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, half, half)

template <typename T1, typename T2, typename T3>
__global__ void _LambComputeDirection(
    const T1* weights,
    const T2* grads,
    const T3* moment_1,
    const T3* moment_2,
    T3 alpha,
    T3 beta,
    T1 lambda,
    T3 epsilon,
    T2* update_direction,
    T3* moment_1_out,
    T3* moment_2_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  const T3 one = T3(1.0);
  const T3 g = T3(grads[id]);

  // Update exponentially-averaged historical gradient
  moment_1_out[id] = alpha * moment_1[id] + \
    (one - alpha) * g;

  // Update exponentially-averaged historical squared gradient
  moment_2_out[id] = beta * moment_2[id] + \
    (one - beta) * g * g;

  // Save regularized update direction to output.
  update_direction[id] = lambda * weights[id] + \
    T1(moment_1_out[id] / (_Sqrt(moment_2_out[id]) + epsilon));
}

template <typename T1, typename T2, typename T3>
void LambComputeDirectionImpl(
    const T1* weights,
    const T2* grads,
    const T3* moment_1,
    const T3* moment_2,
    T3 alpha,
    T3 beta,
    T1 lambda,
    T3 epsilon,
    T2* update_direction,
    T3* moment_1_out,
    T3* moment_2_out,
    size_t count) {
  int blocksPerGrid = \
    (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _LambComputeDirection<T1, T2, T3>\
    <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      weights,
      grads,
      moment_1,
      moment_2,
      alpha,
      beta,
      lambda,
      epsilon,
      update_direction,
      moment_1_out,
      moment_2_out,
      N);
}

#define SPECIALIZED_IMPL_LambComputeDirectionImpl(T1, T2, T3) \
template void LambComputeDirectionImpl(                  \
    const T1* weights,                                   \
    const T2* grads,                                     \
    const T3* moment_1,                                  \
    const T3* moment_2,                                  \
    T3 alpha,                                            \
    T3 beta,                                             \
    T1 lambda,                                           \
    T3 epsilon,                                          \
    T2* weights_out,                                     \
    T3* moment_1_out,                                    \
    T3* moment_2_out,                                    \
    size_t count);

SPECIALIZED_IMPL_LambComputeDirectionImpl(float, float, float)
SPECIALIZED_IMPL_LambComputeDirectionImpl(double, double, double)
SPECIALIZED_IMPL_LambComputeDirectionImpl(float, half, half)
SPECIALIZED_IMPL_LambComputeDirectionImpl(float, half, float)

template <typename T1, typename T2>
__global__ void _LambUpdate(
    const T1* eta,
    const T2* r_norm,
    const T2* w_norm,
    const T2* weights,
    const T1* update_direction,
    T2* weights_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  // Compute new weight using the saved update direction.
  weights_out[id] = weights[id] - \
    (*w_norm) / (*r_norm) * T2((*eta) * update_direction[id]);
}

template <typename T1, typename T2>
void LambUpdateImpl(
    const T1* eta,
    const T2* r_norm,
    const T2* w_norm,
    const T2* weights,
    const T1* update_direction,
    T2* weights_out,
    size_t count) {
  int blocksPerGrid = \
    (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _LambUpdate<T1, T2><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      eta,
      r_norm,
      w_norm,
      weights,
      update_direction,
      weights_out,
      N);
}

#define SPECIALIZED_IMPL_LambUpdate(T1, T2) \
template void LambUpdateImpl(               \
    const T1* eta,                          \
    const T2* r_norm,                       \
    const T2* w_norm,                       \
    const T2* weights,                      \
    const T1* update_direction,             \
    T2* weights_out,                        \
    size_t count);

SPECIALIZED_IMPL_LambUpdate(float, float)
SPECIALIZED_IMPL_LambUpdate(double, double)
SPECIALIZED_IMPL_LambUpdate(half, float)

template <typename T1, typename T2>
__global__ void _LambScalarL2NormReduction(
    const T1* value,
    T2* value_out) {
  *value_out = _Abs(*value);
}

template <typename T1, typename T2>
void LambScalarL2NormReductionImpl(
    const T1* value,
    T2* value_out) {
  _LambScalarL2NormReduction<T1, T2><<<1, 1, 0>>>(
      value,
      value_out);
}

#define SPECIALIZED_IMPL_LambScalarL2NormReduction(T1, T2) \
template void LambScalarL2NormReductionImpl(               \
    const T1* value,                                       \
    T2* value_out);

SPECIALIZED_IMPL_LambScalarL2NormReduction(float, float)
SPECIALIZED_IMPL_LambScalarL2NormReduction(double, double)
SPECIALIZED_IMPL_LambScalarL2NormReduction(half, float)

}  // namespace cuda
}  // namespace onnxruntime
