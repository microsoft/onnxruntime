// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "optimizers.h"
#include "core/providers/cuda/cuda_common.h"


namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _AdamOptimizer(
    const T* eta,
    int64_t* update_count,
    const T* weights,
    const T* grads,
    const T* moment_1,
    const T* moment_2,
    float alpha,
    float beta,
    float lambda,
    float epsilon,
    T* weights_out,
    T* moment_1_out,
    T* moment_2_out,
    CUDA_LONG N) {

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // Regularize gradient
  T g_regularized = lambda * weights[id] + grads[id];

  // Update exponentially-averaged historical gradient
  moment_1_out[id] = alpha * moment_1[id] + ((1 - alpha) * g_regularized);

  // Update exponentially-averaged historical squared gradient
  moment_2_out[id] = beta * moment_2[id] + ((1 - beta) * g_regularized * g_regularized);

  // Update learning rate - Use the updated eta for the final weight update
  float numerator = _Sqrt(1 - _Pow(beta, static_cast<float>(*update_count)));
  float denom = (1 - _Pow(alpha, static_cast<float>(*update_count)));
  float eta_new = (*eta) * numerator / denom;

  weights_out[id] = weights[id] - ((eta_new * moment_1_out[id]) / (_Sqrt(moment_2_out[id]) + epsilon));
  (*update_count)++;
}

template <typename T>
void AdamOptimizerImpl(
    const T* eta,
    int64_t* update_count,
    const T* weights,
    const T* grads,
    const T* moment_1,
    const T* moment_2,
    float alpha,
    float beta,
    float lambda,
    float epsilon,
    T* weights_out,
    T* moment_1_out,
    T* moment_2_out,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _AdamOptimizer<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
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
      N);
}

#define SPECIALIZED_IMPL__AdamOptimizerImpl(T)      \
template void AdamOptimizerImpl(                    \
    const T* eta,                                   \
    int64_t* update_count,                          \
    const T* weights,                               \
    const T* grads,                                 \
    const T* moment_1,                              \
    const T* moment_2,                              \
    float alpha,                                    \
    float beta,                                     \
    float lambda,                                   \
    float epsilon,                                  \
    T* weights_out,                                 \
    T* moment_1_out,                                \
    T* moment_2_out,                                \
    size_t count);

SPECIALIZED_IMPL__AdamOptimizerImpl(float)

}  // namespace cuda
}  // namespace onnxruntime
