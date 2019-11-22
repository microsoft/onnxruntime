// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "optimizers.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/atomic/common.cuh"

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

#define SPECIALIZED_IMPL__SGDOptimizerImpl(T) \
  template void SGDOptimizerImpl(             \
      const T* eta,                           \
      const T* weights,                       \
      const T* gradients,                     \
      T* weights_out,                         \
      size_t count);

SPECIALIZED_IMPL__SGDOptimizerImpl(float)

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD, bool update_fp16_weight, bool has_loss_scale>
    __global__ void _AdamOptimizer(
    const T1* eta,
    const T2 update_count,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
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
  // Regularize gradient.
  T4 new_grad = T4(grads[id]);
  if (has_loss_scale) {
    new_grad /= T4(*loss_scale);
  }
  const T4 g_regularized = lambda * T4(weights[id]) + new_grad;

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

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
void AdamOptimizerImpl(
    const T1* eta,
    const T2 update_count,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
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
    _AdamOptimizer<T1, T2, T3, T4, T_GRAD, true, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        update_count,
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
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
    _AdamOptimizer<T1, T2, T3, T4, T_GRAD, true, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        update_count,
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
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
    _AdamOptimizer<T1, T2, T3, T4, T_GRAD, false, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        update_count,
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
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
    _AdamOptimizer<T1, T2, T3, T4, T_GRAD, false, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        update_count,
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
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

#define SPECIALIZED_AdamOptimizerImpl(T1, T2, T3, T4, T_GRAD) \
  template void AdamOptimizerImpl(                            \
      const T1* eta,                                          \
      const T2 update_count,                                  \
      const T3* weights,                                      \
      const T_GRAD* grads,                                    \
      const T4* moment_1,                                     \
      const T4* moment_2,                                     \
      const T3* loss_scale,                            \
      T4 alpha,                                               \
      T4 beta,                                                \
      T4 lambda,                                              \
      T4 epsilon,                                             \
      T3* weights_out,                                        \
      T4* moment_1_out,                                       \
      T4* moment_2_out,                                       \
      half* fp16_weights_out,                                 \
      size_t count);

SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, float, float)
SPECIALIZED_AdamOptimizerImpl(half, int64_t, float, half, float)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, half, float)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, float, half)
SPECIALIZED_AdamOptimizerImpl(half, int64_t, float, half, half)
SPECIALIZED_AdamOptimizerImpl(float, int64_t, float, half, half)
template <typename T1, typename T2, typename T3, bool has_loss_scale>
__global__ void _LambComputeDirection(
    const T1* weights,
    const T2* grads,
    const T3* moment_1,
    const T3* moment_2,
    const T1* loss_scale,
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
  T3 g = T3(grads[id]);

  if (has_loss_scale) {
    g /= T3(*loss_scale);
  }
  // Update exponentially-averaged historical gradient
  moment_1_out[id] = alpha * moment_1[id] +
                     (one - alpha) * g;

  // Update exponentially-averaged historical squared gradient
  moment_2_out[id] = beta * moment_2[id] +
                     (one - beta) * g * g;

  // Save regularized update direction to output.
  update_direction[id] = lambda * weights[id] +
                         T1(moment_1_out[id] / (_Sqrt(moment_2_out[id]) + epsilon));
}

template <typename T1, typename T2, typename T3>
void LambComputeDirectionImpl(
    const T1* weights,
    const T2* grads,
    const T3* moment_1,
    const T3* moment_2,
    const T1* loss_scale,
    T3 alpha,
    T3 beta,
    T1 lambda,
    T3 epsilon,
    T2* update_direction,
    T3* moment_1_out,
    T3* moment_2_out,
    size_t count) {
  int blocksPerGrid =
      (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (loss_scale == nullptr) {
    _LambComputeDirection<T1, T2, T3, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
        alpha,
        beta,
        lambda,
        epsilon,
        update_direction,
        moment_1_out,
        moment_2_out,
        N);
  } else {
    _LambComputeDirection<T1, T2, T3, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        weights,
        grads,
        moment_1,
        moment_2,
        loss_scale,
        alpha,
        beta,
        lambda,
        epsilon,
        update_direction,
        moment_1_out,
        moment_2_out,
        N);
  }
}

#define SPECIALIZED_IMPL_LambComputeDirectionImpl(T1, T2, T3) \
  template void LambComputeDirectionImpl(                     \
      const T1* weights,                                      \
      const T2* grads,                                        \
      const T3* moment_1,                                     \
      const T3* moment_2,                                     \
      const T1* loss_scale,                            \
      T3 alpha,                                               \
      T3 beta,                                                \
      T1 lambda,                                              \
      T3 epsilon,                                             \
      T2* weights_out,                                        \
      T3* moment_1_out,                                       \
      T3* moment_2_out,                                       \
      size_t count);

SPECIALIZED_IMPL_LambComputeDirectionImpl(float, float, float)
SPECIALIZED_IMPL_LambComputeDirectionImpl(double, double, double)
SPECIALIZED_IMPL_LambComputeDirectionImpl(float, half, half)
SPECIALIZED_IMPL_LambComputeDirectionImpl(float, half, float)

template <typename T1, typename T2, typename T3, bool update_fp16_weight>
__global__ void _LambUpdate(
    const T1* eta,
    const T2* r_norm,
    const T2* w_norm,
    const T2* weights,
    const T2 threshold,
    const T3* update_direction,
    T2* weights_out,
    half* fp16_weights_out,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  // The reason to have _Min(...):
  //   The confidence level should not exceed 1 for numerical stability.
  //   The threshold will be used even if r_norm and w_norm are 0 because
  //   NaN > threshold ? NaN : threshold returns threshold.
  // The reason to have *w_norm != 0?:
  //   If a tensor is zero-initialized, its w_norm will be 0 and therefore its
  //   ratio is always 0 without the _Max(...). If a tensor's ratio is always
  //   0, that tensor will never be updated.
  const auto ratio = *w_norm != T2(0.0f)? _Min(_Sqrt(*w_norm / *r_norm), threshold) : T2(1.0f);
  // Compute new weight using the saved update direction.
  weights_out[id] = weights[id] - ratio * T2((*eta) * T1(update_direction[id]));

  if (update_fp16_weight) {
    fp16_weights_out[id] = static_cast<half>(weights_out[id]);
  }
}

template <typename T1, typename T2, typename T3>
void LambUpdateImpl(
    const T1* eta,
    const T2* r_norm,
    const T2* w_norm,
    const T2* weights,
    const T2 threshold,
    const T3* update_direction,
    T2* weights_out,
    half* fp16_weights_out,
    size_t count) {
  int blocksPerGrid =
      (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (fp16_weights_out != nullptr) {
    _LambUpdate<T1, T2, T3, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        r_norm,
        w_norm,
        weights,
        threshold,
        update_direction,
        weights_out,
        fp16_weights_out,
        N);
  } else {
    _LambUpdate<T1, T2, T3, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        eta,
        r_norm,
        w_norm,
        weights,
        threshold,
        update_direction,
        weights_out,
        nullptr,
        N);
  }
}

#define SPECIALIZED_IMPL_LambUpdate(T1, T2, T3) \
  template void LambUpdateImpl(                 \
      const T1* eta,                            \
      const T2* r_norm,                         \
      const T2* w_norm,                         \
      const T2* weights,                        \
      const T2 threshold,                       \
      const T3* update_direction,               \
      T2* weights_out,                          \
      half* fp16_weights_out,                   \
      size_t count);

SPECIALIZED_IMPL_LambUpdate(float, float, float)
SPECIALIZED_IMPL_LambUpdate(double, double, double)
SPECIALIZED_IMPL_LambUpdate(half, float, half)
SPECIALIZED_IMPL_LambUpdate(float, float, half)

template <typename T, typename T_GRAD>
__global__ void _AccumulateGradient(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  accumulated_gradient[id] = gradient_buffer[id] + T(gradient[id]);
}

template <typename T, typename T_GRAD>
void AccumulateGradientImpl(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _AccumulateGradient<T, T_GRAD><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      gradient_buffer,
      gradient,
      accumulated_gradient,
      N);
}

#define SPECIALIZED_IMPL_AccumulateGradient(T, T_GRAD) \
  template void AccumulateGradientImpl(                \
      const T* gradient_buffer,                        \
      const T_GRAD* gradient,                          \
      T* accumulated_gradient,                         \
      size_t count);

SPECIALIZED_IMPL_AccumulateGradient(float, float)
SPECIALIZED_IMPL_AccumulateGradient(float, half)

// A mapping from [w, g, m1, m2] to [d, m1_new, m2_new].
// We reuse "g" to store "d" to save some memory.
template<typename T1, typename T2, typename T3>
__global__ void _LambStage1(ChunkGroup<6> chunk_group, const T1 *loss_scale,
  const T1 lambda, const T3 alpha, const T3 beta, const T3 epsilon) {
  const int group_index = chunk_group.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunk_group.tensor_sizes[group_index];
  const int chunk_size = chunk_group.chunk_size;
  const int chunk_start = chunk_group.block_index_to_chunk_start_index[blockIdx.x];
  const T1 *w = reinterpret_cast<const T1*>(chunk_group.tensor_ptrs[0][group_index]) + chunk_start;
  T2 *g = reinterpret_cast<T2*>(chunk_group.tensor_ptrs[1][group_index]) + chunk_start;
  const T3 *m1 = reinterpret_cast<const T3*>(chunk_group.tensor_ptrs[2][group_index]) + chunk_start;
  const T3 *m2 = reinterpret_cast<const T3*>(chunk_group.tensor_ptrs[3][group_index]) + chunk_start;
  T3 *m1_new = reinterpret_cast<T3*>(chunk_group.tensor_ptrs[4][group_index]) + chunk_start;
  T3 *m2_new = reinterpret_cast<T3*>(chunk_group.tensor_ptrs[5][group_index]) + chunk_start;

  const T3 one = T3(1.0);
  const T3 inverse_scale = loss_scale? one / T3(*loss_scale) : T3(1.f);

#pragma unroll(4)
  for (int i = threadIdx.x; i < chunk_size && i + chunk_start < tensor_size; i += blockDim.x) {
    T3 g_scaled = T3(g[i]) * inverse_scale;

    // Update exponentially-averaged historical gradient, the 1st-order momentum.
    m1_new[i] = alpha * m1[i] + (one - alpha) * g_scaled;

    // Update exponentially-averaged historical squared gradient, the 2st-order momentum.
    m2_new[i] = beta * m2[i] + (one - beta) * g_scaled * g_scaled;

    // Save regularized update direction to output.
    g[i] = lambda * w[i] + T1(m1_new[i] / (_Sqrt(m2_new[i]) + epsilon));
  }
}

template<typename T1, typename T2, typename T3>
  __global__ void LambStage1MultiTensorImpl(
    ChunkGroup<6> chunk_group,
    const T1 *loss_scale,
    const T1 lambda,
    const T3 alpha,
    const T3 beta,
    const T3 epsilon) {
  const int group_index = chunk_group.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunk_group.tensor_sizes[group_index];
  const int chunk_size = chunk_group.chunk_size;
  const int chunk_start = chunk_group.block_index_to_chunk_start_index[blockIdx.x];
  const T1 *w = reinterpret_cast<const T1*>(chunk_group.tensor_ptrs[0][group_index]) + chunk_start;
  T2 *g = reinterpret_cast<T2*>(chunk_group.tensor_ptrs[1][group_index]) + chunk_start;
  const T3 *m1 = reinterpret_cast<const T3*>(chunk_group.tensor_ptrs[2][group_index]) + chunk_start;
  const T3 *m2 = reinterpret_cast<const T3*>(chunk_group.tensor_ptrs[3][group_index]) + chunk_start;
  T3 *m1_new = reinterpret_cast<T3*>(chunk_group.tensor_ptrs[4][group_index]) + chunk_start;
  T3 *m2_new = reinterpret_cast<T3*>(chunk_group.tensor_ptrs[5][group_index]) + chunk_start;

  const T3 one = T3(1.f);
  const T3 scale = loss_scale? T3(*loss_scale) : T3(1.f);

#pragma unroll
  for (int i = threadIdx.x; i < chunk_size && i + chunk_start < tensor_size; i += blockDim.x) {
    T3 g_scaled = T3(g[i]) / scale;

    // Update exponentially-averaged historical gradient, the 1st-order momentum.
    m1_new[i] = alpha * m1[i] + (one - alpha) * g_scaled;

    // Update exponentially-averaged historical squared gradient, the 2st-order momentum.
    m2_new[i] = beta * m2[i] + (one - beta) * g_scaled * g_scaled;

    // Save regularized update direction to output.
    g[i] = lambda * w[i] + T1(m1_new[i] / (_Sqrt(m2_new[i]) + epsilon));
  }
}

template <typename T1, typename T2, typename T3>
void LambStage1MultiTensorFunctor<T1, T2, T3>::operator()(
  ChunkGroup<6> chunk_group,
  const T1 *loss_scale,
  const T1 lambda,
  const T3 alpha,
  const T3 beta,
  const T3 epsilon) {
  const int thread_count = ChunkGroup<6>::thread_count_per_block;
  const int block_count = chunk_group.chunk_count;

  LambStage1MultiTensorImpl<T1, T2, T3><<<block_count, thread_count, 0>>>(
    chunk_group,
    loss_scale,
    lambda,
    alpha,
    beta,
    epsilon);
}

#define INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(T1, T2, T3) \
  template void LambStage1MultiTensorFunctor<T1, T2, T3>::operator()( \
    ChunkGroup<6> chunk_group, \
    const T1 *loss_scale, \
    const T1 lambda, \
    const T3 alpha, \
    const T3 beta, \
    const T3 epsilon);

INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(float, float, float)
INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(double, double, double)
INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(float, half, half)
INSTANTIATE_LAMB_STAGE1_MULTI_TENSOR_FUNCTOR(float, half, float)

template<typename T1, typename T2, typename T3>
__global__ void LambStage2MultiTensorImpl(
  ChunkGroup<6> chunk_group,
  const T1* eta,
  const T2 threshold) {
  const int group_index = chunk_group.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunk_group.tensor_sizes[group_index];
  const int chunk_size = chunk_group.chunk_size;
  const int chunk_start = chunk_group.block_index_to_chunk_start_index[blockIdx.x];

  const T2* w_norm = reinterpret_cast<const T2*>(chunk_group.tensor_ptrs[0][group_index]);
  const T2* r_norm = reinterpret_cast<const T2*>(chunk_group.tensor_ptrs[1][group_index]);
  const T2* w = reinterpret_cast<const T2*>(chunk_group.tensor_ptrs[2][group_index]) + chunk_start;
  const T3* d = reinterpret_cast<const T3*>(chunk_group.tensor_ptrs[3][group_index]) + chunk_start;
  T2* w_new = reinterpret_cast<T2*>(chunk_group.tensor_ptrs[4][group_index]) + chunk_start;
  half* w_fp16_new = chunk_group.tensor_ptrs[5][group_index] != nullptr? reinterpret_cast<half*>(chunk_group.tensor_ptrs[5][group_index]) + chunk_start : nullptr;

  for (int i = threadIdx.x; i < chunk_size && i + chunk_start < tensor_size; i += blockDim.x) {
    // Some reasons to have _Min(...):
    //   1. The confidence level should not exceed 1 for numerical stability.
    //   2. The threshold will be used even if r_norm and w_norm are 0 because
    //      NaN > threshold ? NaN : threshold returns threshold.
    // Some reasons to have *w_norm != 0?:
    //   If a tensor is zero-initialized, its w_norm will be 0 and therefore its
    //   ratio is always 0 without the _Max(...). If a tensor's ratio is always
    //   0, that tensor will never be updated.
    const auto ratio = *w_norm != T2(0.0f)? _Min(_Sqrt(*w_norm / *r_norm), threshold) : T2(1.0f);
    // Compute new weight using the saved update direction.
    w_new[i] = w[i] - ratio * T2((*eta) * T1(d[i]));

    if (w_fp16_new != nullptr) {
      w_fp16_new[i] = static_cast<half>(w_new[i]);
    }
  }
}

template<typename T1, typename T2, typename T3>
void LambStage2MultiTensorFunctor<T1, T2, T3>::operator()(
  ChunkGroup<6> chunk_group,
  const T1* eta,
  const T2 threshold) {
  const int thread_count = ChunkGroup<6>::thread_count_per_block;
  const int block_count = chunk_group.chunk_count;

  LambStage2MultiTensorImpl<T1, T2, T3><<<block_count, thread_count, 0>>>(
    chunk_group,
    eta,
    threshold);
}

#define INSTANTIATE_LAMB_STAGE2_MULTI_TENSOR_FUNCTOR(T1, T2, T3) \
  template void LambStage2MultiTensorFunctor<T1, T2, T3>::operator()( \
    ChunkGroup<6> chunk_group,                                        \
    const T1* eta,                                                    \
    const T2 threshold);

INSTANTIATE_LAMB_STAGE2_MULTI_TENSOR_FUNCTOR(float, float, float)
INSTANTIATE_LAMB_STAGE2_MULTI_TENSOR_FUNCTOR(double, double, double)
INSTANTIATE_LAMB_STAGE2_MULTI_TENSOR_FUNCTOR(half, float, half)
INSTANTIATE_LAMB_STAGE2_MULTI_TENSOR_FUNCTOR(float, float, half)

template<typename TIn1, typename TIn2, typename TOut1, typename TOut2, typename TBuf>
__global__ void LambReductionMultiTensorImpl(ChunkGroup<4> chunk_group) {
  const int group_index = chunk_group.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunk_group.tensor_sizes[group_index];
  const int chunk_size = chunk_group.chunk_size;
  const int chunk_start = chunk_group.block_index_to_chunk_start_index[blockIdx.x];
  const TIn1 *w = reinterpret_cast<const TIn1*>(chunk_group.tensor_ptrs[0][group_index]) + chunk_start;
  const TIn2 *d = reinterpret_cast<const TIn2*>(chunk_group.tensor_ptrs[1][group_index]) + chunk_start;
  TOut1 *w_norm = reinterpret_cast<TOut1*>(chunk_group.tensor_ptrs[2][group_index]);
  TOut2 *d_norm = reinterpret_cast<TOut2*>(chunk_group.tensor_ptrs[3][group_index]);

  TBuf d_sum = TBuf(0.f);
  TBuf w_sum = TBuf(0.f);
  constexpr int load_count_per_thread = 4;
  for (int i = threadIdx.x; i < chunk_size && i + chunk_start < tensor_size; i += blockDim.x * load_count_per_thread) {
#pragma unroll
    for (int j = 0; j < load_count_per_thread; ++j) {
      const int index_in_chunk = i + j * blockDim.x;
      const int index_in_tensor = chunk_start + index_in_chunk;
      if (index_in_chunk < chunk_size && index_in_tensor < tensor_size) {
        const TBuf w_element = TBuf(w[index_in_chunk]);
        const TBuf d_element = TBuf(d[index_in_chunk]);
        w_sum += w_element * w_element;
        d_sum += d_element * d_element;
      }
    }
  }

  // Thread count in a block must be a multiple of 32.
  constexpr int warp_size = 32;
#pragma unroll
  for (int stride = warp_size / 2; stride > 0; stride /= 2) {
    w_sum += __shfl_down_sync(0xFFFFFFFF, w_sum, stride);
    d_sum += __shfl_down_sync(0xFFFFFFFF, d_sum, stride);
  }

  const int warp_count_in_block = blockDim.x / warp_size;
  const int lid = threadIdx.x % warp_size;
  const int wid = threadIdx.x / warp_size;

  // Shape is 2 x warp_count_in_block.
  extern __shared__ unsigned char shared_memory_[];
  TBuf *shared_memory = reinterpret_cast<TBuf*>(shared_memory_);
  TBuf* w_shared_memory_ = shared_memory;
  TBuf* d_shared_memory_ = shared_memory + warp_count_in_block;

  if (lid == 0) {
    w_shared_memory_[wid] = w_sum;
    d_shared_memory_[wid] = d_sum;
  }

  __syncthreads();

#pragma unroll
  for (int stride = warp_count_in_block / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      w_shared_memory_[threadIdx.x] += w_shared_memory_[threadIdx.x + stride];
      d_shared_memory_[threadIdx.x] += d_shared_memory_[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomic_add(w_norm, TOut1(w_shared_memory_[0]));
    atomic_add(d_norm, TOut2(d_shared_memory_[0]));
  }
};


template<typename TIn1, typename TIn2, typename TOut1, typename TOut2, typename TBuf>
void LambReductionMultiTensorFunctor<TIn1, TIn2, TOut1, TOut2, TBuf>::operator()(ChunkGroup<4> chunk_group) {
  // thread count per block.
  constexpr int thread_count = ChunkGroup<4>::thread_count_per_block;
  // warp size of GPU.
  constexpr int warp_size = 32;
  // shared memory's size per block.
  const int shared_memory_size = thread_count / warp_size * 2 * sizeof(TBuf);

  // Enforce assumptions used inside this reduction CUDA kernel.
  ORT_ENFORCE(thread_count % warp_size == 0);
  ORT_ENFORCE((thread_count & (thread_count - 1)) == 0);

  LambReductionMultiTensorImpl<TIn1, TIn2, TOut1, TOut2, TBuf><<<chunk_group.chunk_count, thread_count, shared_memory_size>>>(chunk_group);
}

#define INSTANTIATE_LAMB_REDUCTION_MULTI_TENSOR_FUNCTOR(TIn1, TIn2, TOut1, TOut2, TBuf) \
  template void LambReductionMultiTensorFunctor<TIn1, TIn2, TOut1, TOut2, TBuf>::operator()(ChunkGroup<4> chunk_group);

INSTANTIATE_LAMB_REDUCTION_MULTI_TENSOR_FUNCTOR(float, float, float, float, float)
INSTANTIATE_LAMB_REDUCTION_MULTI_TENSOR_FUNCTOR(double, double, double, double, double)
INSTANTIATE_LAMB_REDUCTION_MULTI_TENSOR_FUNCTOR(float, half, float, half, float)
INSTANTIATE_LAMB_REDUCTION_MULTI_TENSOR_FUNCTOR(float, half, float, float, float)
INSTANTIATE_LAMB_REDUCTION_MULTI_TENSOR_FUNCTOR(half, half, half, half, float)

}  // namespace cuda
}  // namespace onnxruntime
