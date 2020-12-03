// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "orttraining/training_ops/cuda/optimizer/common.h"
#include "orttraining/training_ops/cuda/optimizer/lamb.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/atomic/common.cuh"
#include "orttraining/training_ops/cuda/math/isfinite.cuh"
#include "orttraining/training_ops/cuda/optimizer/common.cuh"
#include "orttraining/training_ops/cuda/optimizer/lamb.h"
namespace onnxruntime {
namespace cuda {

__forceinline__ __host__ __device__ int least_pow2_bound(int value) {
  unsigned int value_ = static_cast<unsigned int>(value);
  --value_;
  value_ |= value_ >> 1;
  value_ |= value_ >> 2;
  value_ |= value_ >> 4;
  value_ |= value_ >> 8;
  value_ |= value_ >> 16;
  return static_cast<int>(++value_);
}

template <typename T1, typename T2, typename T3>
__device__ __forceinline__ void _LambComputeDirectionRule(
    const T1& g_scale,
    const T1& w,
    const T2& g,
    const T3& m1,
    const T3& m2,
    const T3& alpha,
    const T3& beta,
    const T1& lambda,
    const T3& epsilon,
    const T3& alpha_correction,
    const T3& beta_correction,
    T2& d,
    T3& m1_new,
    T3& m2_new) {
  // Actual gradient. The scale is a product of loss' scale and
  // global gradient norm (if the norm > 1).
  const T3 g_scaled = T3(T1(g) / g_scale);

  // A constant in Lamb's equation.
  const T3 one = T3(1.0f);

  // Update exponentially-averaged historical gradient
  const T3 m1_new_tmp = alpha * m1 + (one - alpha) * g_scaled;

  // Update exponentially-averaged historical squared gradient
  const T3 m2_new_tmp = beta * m2 + (one - beta) * g_scaled * g_scaled;

  // Compute unbiased 1st-order momentom.
  // The value alpha_correction is usually (1-alpha^t),
  // where t is the number of executed training iterations.
  const T3 m1_new_tmp_corrected = m1_new_tmp / alpha_correction;

  // Compute unbiased 2nd-order momentom.
  // The value beta_correction is usually (1-beta^t),
  // where t is the number of executed training iterations.
  const T3 m2_new_tmp_corrected = m2_new_tmp / beta_correction;

  // Save regularized update direction to output.
  const T2 d_tmp = lambda * w + 
    T1(m1_new_tmp_corrected / (_Sqrt(m2_new_tmp_corrected) + epsilon));

  // Things are updated only if the direction is finite.
  if (_IsFiniteScalar(d_tmp)) {
    d = d_tmp;
    m1_new = m1_new_tmp;
    m2_new = m2_new_tmp;
  } else {
    d = T2(0);
    m1_new = m1;
    m2_new = m2;
  }
}

template <typename T1, typename T2, typename T3, typename T_GRAD_NORM>
__global__ void _LambComputeDirectionImpl(
    const T1* weights,
    const T2* grads,
    const T3* moment_1,
    const T3* moment_2,
    const T1* loss_scale,
    const T_GRAD_NORM* g_norm,
    T3 alpha,
    T3 beta,
    T1 lambda,
    T3 epsilon,
    T3 alpha_correction,
    T3 beta_correction,
    T2* update_direction,
    T3* moment_1_out,
    T3* moment_2_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  const T1 scale = _ComputeGradScale<T1, T_GRAD_NORM, T1>(loss_scale, g_norm);

  _LambComputeDirectionRule(
      scale,
      weights[id],
      grads[id],
      moment_1[id],
      moment_2[id],
      alpha,
      beta,
      lambda,
      epsilon,
      alpha_correction,
      beta_correction,
      update_direction[id],
      moment_1_out[id],
      moment_2_out[id]);
}

template <typename T1, typename T2, typename T3, typename T_GRAD_NORM>
void LambComputeDirection(
    const T1* weights,
    const T2* grads,
    const T3* moment_1,
    const T3* moment_2,
    const T1* loss_scale,
    const T_GRAD_NORM* grad_norm,
    T3 alpha,
    T3 beta,
    T1 lambda,
    T3 epsilon,
    T3 alpha_correction,
    T3 beta_correction,
    T2* update_direction,
    T3* moment_1_out,
    T3* moment_2_out,
    size_t count) {
  int blocksPerGrid =
      (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _LambComputeDirectionImpl<T1, T2, T3, T_GRAD_NORM><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
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
      update_direction,
      moment_1_out,
      moment_2_out,
      N);
}

#define SPECIALIZED_LAMB_COMPUTE_DIRECTION(T1, T2, T3, T_GRAD_NORM) \
  template void LambComputeDirection(                     \
      const T1* weights,                                  \
      const T2* grads,                                    \
      const T3* moment_1,                                 \
      const T3* moment_2,                                 \
      const T1* loss_scale,                               \
      const T_GRAD_NORM* grad_norm,                       \
      T3 alpha,                                           \
      T3 beta,                                            \
      T1 lambda,                                          \
      T3 epsilon,                                         \
      T3 alpha_correction,                                \
      T3 beta_correction,                                 \
      T2* weights_out,                                    \
      T3* moment_1_out,                                   \
      T3* moment_2_out,                                   \
      size_t count);

SPECIALIZED_LAMB_COMPUTE_DIRECTION(float, float, float, float)
SPECIALIZED_LAMB_COMPUTE_DIRECTION(double, double, double, double)
SPECIALIZED_LAMB_COMPUTE_DIRECTION(float, half, half, half)
SPECIALIZED_LAMB_COMPUTE_DIRECTION(float, half, half, float)
SPECIALIZED_LAMB_COMPUTE_DIRECTION(float, half, float, half)
SPECIALIZED_LAMB_COMPUTE_DIRECTION(float, half, float, float)

template <typename T1, typename T2, typename T3, typename T_MIXED_PRECISION_FP>
__device__ __forceinline__ void _LambUpdateRule(
    const T1 eta,
    const float ratio_min,
    const float ratio_max,
    const T2 r_norm,
    const T2 w_norm,
    const T2 w,
    const T3 d,
    T2* w_new,
    T3* g_new,
    T_MIXED_PRECISION_FP* w_mixed_precision_new) {
  // Confidence coefficeint of this update. 
  const T2 ratio = (w_norm != T2(0.0f) && r_norm != T2(0.0f)) ?
    T2(eta) * _Max(T2(ratio_min), _Min(T2(ratio_max), _Sqrt(w_norm / r_norm))) : T2(eta);

  // Compute delta using the saved update direction.
  const T2 delta = -ratio * T2(d);
  const T2 w_new_tmp = w + delta;

  if (_IsFiniteScalar(w_new_tmp)) {
    if (g_new) {
      *g_new = T3(delta);
    }
    if (w_new) {
      *w_new = w_new_tmp;
      if (w_mixed_precision_new) {
        *w_mixed_precision_new = T_MIXED_PRECISION_FP(w_new_tmp);
      }
    }
  } else {
    if (g_new) {
      *g_new = T3(0);
    }
    if (w_new) {
      *w_new = w;
      if (w_mixed_precision_new) {
        *w_mixed_precision_new = T_MIXED_PRECISION_FP(w);
      }
    }
  }
}

template <typename T1, typename T2, typename T3, typename T_MIXED_PRECISION_FP>
__global__ void _LambUpdateImpl(
    const T1* eta,
    const float ratio_min,
    const float ratio_max,
    const T2* r_norm,
    const T2* w_norm,
    const T2* weights,
    const T3* update_direction,
    T2* weights_out,
    T3* gradients_out,
    T_MIXED_PRECISION_FP* mixed_precision_weights_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  _LambUpdateRule(
      *eta,
      ratio_min,
      ratio_max,
      *r_norm,
      *w_norm,
      weights[id],
      update_direction[id],
      weights_out != nullptr ? weights_out + id : nullptr,
      gradients_out != nullptr ? gradients_out + id : nullptr,
      mixed_precision_weights_out != nullptr ? mixed_precision_weights_out + id : nullptr);
}

template <typename T1, typename T2, typename T3, typename T_MIXED_PRECISION_FP>
void LambUpdate(
    const T1* eta,
    const float ratio_min,
    const float ratio_max,
    const T2* r_norm,
    const T2* w_norm,
    const T2* weights,
    const T3* update_direction,
    T2* weights_out,
    T3* gradients_out,
    T_MIXED_PRECISION_FP* mixed_precision_weights_out,
    size_t count) {
  int blocksPerGrid =
      (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _LambUpdateImpl<T1, T2, T3, T_MIXED_PRECISION_FP><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      eta,
      ratio_min,
      ratio_max,
      r_norm,
      w_norm,
      weights,
      update_direction,
      weights_out,
      gradients_out,
      mixed_precision_weights_out,
      N);
}

#define INSTANTIATE_LAMB_UPDATE(T1, T2, T3, T_MIXED_PRECISION_FP) \
  template void LambUpdate(                                       \
      const T1* eta,                                              \
      const float ratio_min,                                      \
      const float ratio_max,                                      \
      const T2* r_norm,                                           \
      const T2* w_norm,                                           \
      const T2* weights,                                          \
      const T3* update_direction,                                 \
      T2* weights_out,                                            \
      T3* gradients_out,                                          \
      T_MIXED_PRECISION_FP* mixed_precision_weights_out,          \
      size_t count);

INSTANTIATE_LAMB_UPDATE(float, float, float, half)
INSTANTIATE_LAMB_UPDATE(double, double, double, half)
INSTANTIATE_LAMB_UPDATE(half, float, half, half)
INSTANTIATE_LAMB_UPDATE(float, float, half, half)

template <typename T1, typename T2, typename T3, typename T_GRAD_NORM>
__global__ void LambMultiTensorComputeDirectionImpl(
    ChunkGroup<6> chunk_group,
    const T1* loss_scale,
    const T_GRAD_NORM* g_norm,
    const T1 lambda,
    const T3 alpha,
    const T3 beta,
    const T3 epsilon,
    const T3 alpha_correction,
    const T3 beta_correction) {
  const int group_index = chunk_group.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunk_group.tensor_sizes[group_index];
  const int chunk_size = chunk_group.chunk_size;
  const int chunk_start = chunk_group.block_index_to_chunk_start_index[blockIdx.x];
  const T1* w = reinterpret_cast<const T1*>(chunk_group.tensor_ptrs[0][group_index]) + chunk_start;
  T2* g = reinterpret_cast<T2*>(chunk_group.tensor_ptrs[1][group_index]) + chunk_start;
  const T3* m1 = reinterpret_cast<const T3*>(chunk_group.tensor_ptrs[2][group_index]) + chunk_start;
  const T3* m2 = reinterpret_cast<const T3*>(chunk_group.tensor_ptrs[3][group_index]) + chunk_start;
  T3* m1_new = reinterpret_cast<T3*>(chunk_group.tensor_ptrs[4][group_index]) + chunk_start;
  T3* m2_new = reinterpret_cast<T3*>(chunk_group.tensor_ptrs[5][group_index]) + chunk_start;
  const T1 scale = _ComputeGradScale<T1, T_GRAD_NORM, T1>(loss_scale, g_norm);

  #pragma unroll
  for (int i = threadIdx.x; i < chunk_size && i + chunk_start < tensor_size; i += blockDim.x) {
    _LambComputeDirectionRule(
        scale,
        w[i],
        g[i],
        m1[i],
        m2[i],
        alpha,
        beta,
        lambda,
        epsilon,
        alpha_correction,
        beta_correction,
        g[i],
        m1_new[i],
        m2_new[i]);
  }
}

template <typename T1, typename T2, typename T3, typename T_GRAD_NORM>
void LambMultiTensorComputeDirectionFunctor<T1, T2, T3, T_GRAD_NORM>::operator()(
    ChunkGroup<6> chunk_group,
    const T1* loss_scale,
    const T_GRAD_NORM* g_norm,
    const T1 lambda,
    const T3 alpha,
    const T3 beta,
    const T3 epsilon,
    const T3 alpha_correction,
    const T3 beta_correction) {
  const int thread_count = ChunkGroup<6>::thread_count_per_block;
  const int block_count = chunk_group.chunk_count;

  LambMultiTensorComputeDirectionImpl<T1, T2, T3><<<block_count, thread_count, 0>>>(
      chunk_group,
      loss_scale,
      g_norm,
      lambda,
      alpha,
      beta,
      epsilon,
      alpha_correction,
      beta_correction);
}

#define INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(T1, T2, T3, T_GRAD_NORM)   \
  template void LambMultiTensorComputeDirectionFunctor<T1, T2, T3, T_GRAD_NORM>::operator()( \
      ChunkGroup<6> chunk_group,                                                \
      const T1* loss_scale,                                                     \
      const T_GRAD_NORM* g_norm,                                                \
      const T1 lambda,                                                          \
      const T3 alpha,                                                           \
      const T3 beta,                                                            \
      const T3 epsilon,                                                         \
      const T3 alpha_correction,                                                \
      const T3 beta_correction);

INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(float, float, float, float)
INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(double, double, double, double)
INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(float, half, half, half)
INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(float, half, half, float)
INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(float, half, float, half)
INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(float, half, float, float)

template <typename T1, typename T2, typename T3, typename T_MIXED_PRECISION_FP>
__global__ void LambMultiTensorUpdateImpl(
    ChunkGroup<7> chunk_group,
    const T1* eta,
    const float ratio_min,
    const float ratio_max) {
  const int group_index = chunk_group.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunk_group.tensor_sizes[group_index];
  const int chunk_size = chunk_group.chunk_size;
  const int chunk_start = chunk_group.block_index_to_chunk_start_index[blockIdx.x];

  const T2* w_norm = reinterpret_cast<const T2*>(chunk_group.tensor_ptrs[0][group_index]);
  const T2* r_norm = reinterpret_cast<const T2*>(chunk_group.tensor_ptrs[1][group_index]);
  const T2* w = reinterpret_cast<const T2*>(chunk_group.tensor_ptrs[2][group_index]) + chunk_start;
  const T3* d = reinterpret_cast<const T3*>(chunk_group.tensor_ptrs[3][group_index]) + chunk_start;
  T2* w_new = chunk_group.tensor_ptrs[4][group_index] != nullptr ? reinterpret_cast<T2*>(chunk_group.tensor_ptrs[4][group_index]) + chunk_start : nullptr;
  T3* g_new = chunk_group.tensor_ptrs[5][group_index] != nullptr ? reinterpret_cast<T3*>(chunk_group.tensor_ptrs[5][group_index]) + chunk_start : nullptr;
  T_MIXED_PRECISION_FP* w_mixed_precision_new = chunk_group.tensor_ptrs[6][group_index] != nullptr ? reinterpret_cast<T_MIXED_PRECISION_FP*>(chunk_group.tensor_ptrs[6][group_index]) + chunk_start : nullptr;

  for (int i = threadIdx.x; i < chunk_size && i + chunk_start < tensor_size; i += blockDim.x) {
    _LambUpdateRule(
        *eta,
        ratio_min,
        ratio_max,
        *r_norm,
        *w_norm,
        w[i],
        d[i],
        w_new != nullptr ? w_new + i : nullptr,
        g_new != nullptr ? g_new + i : nullptr,
        w_mixed_precision_new != nullptr ? w_mixed_precision_new + i : nullptr);
  }
}

template <typename T1, typename T2, typename T3, typename T_MIXED_PRECISION_FP>
void LambMultiTensorUpdateFunctor<T1, T2, T3, T_MIXED_PRECISION_FP>::operator()(
    ChunkGroup<7> chunk_group,
    const T1* eta,
    const float ratio_min,
    const float ratio_max) {
  const int thread_count = ChunkGroup<7>::thread_count_per_block;
  const int block_count = chunk_group.chunk_count;

  LambMultiTensorUpdateImpl<T1, T2, T3, T_MIXED_PRECISION_FP><<<block_count, thread_count, 0>>>(
      chunk_group,
      eta,
      ratio_min,
      ratio_max);
}

#define INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(T1, T2, T3, T_MIXED_PRECISION_FP)      \
  template void LambMultiTensorUpdateFunctor<T1, T2, T3, T_MIXED_PRECISION_FP>::operator()( \
      ChunkGroup<7> chunk_group,                                                            \
      const T1* eta,                                                                        \
      const float ratio_min,                                                                \
      const float ratio_max);

INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(float, float, float, half)
INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(double, double, double, half)
INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(half, float, half, half)
INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(float, float, half, half)

}  // namespace cuda
}  // namespace onnxruntime
