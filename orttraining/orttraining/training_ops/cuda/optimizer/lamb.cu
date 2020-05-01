// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/atomic/common.cuh"
#include "orttraining/training_ops/cuda/math/isfinite.cuh"
#include "orttraining/training_ops/cuda/optimizer/common.cuh"
#include "orttraining/training_ops/cuda/optimizer/lamb.h"
namespace onnxruntime {
namespace cuda {
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

template <typename T1, typename T2, typename T3>
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
    half* w_fp16_new) {
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
      if (w_fp16_new) {
        *w_fp16_new = half(w_new_tmp);
      }
    }
  } else {
    if (g_new) {
      *g_new = T3(0);
    }
    if (w_new) {
      *w_new = w;
      if (w_fp16_new) {
        *w_fp16_new = half(w);
      }
    }
  }
}

template <typename T1, typename T2, typename T3>
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
    half* fp16_weights_out,
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
      fp16_weights_out != nullptr ? fp16_weights_out + id : nullptr);
}

template <typename T1, typename T2, typename T3>
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
    half* fp16_weights_out,
    size_t count) {
  int blocksPerGrid =
      (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _LambUpdateImpl<T1, T2, T3><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      eta,
      ratio_min,
      ratio_max,
      r_norm,
      w_norm,
      weights,
      update_direction,
      weights_out,
      gradients_out,
      fp16_weights_out,
      N);
}

#define INSTANTIATE_LAMB_UPDATE(T1, T2, T3) \
  template void LambUpdate(                     \
      const T1* eta,                            \
      const float ratio_min,                    \
      const float ratio_max,                    \
      const T2* r_norm,                         \
      const T2* w_norm,                         \
      const T2* weights,                        \
      const T3* update_direction,               \
      T2* weights_out,                          \
      T3* gradients_out,                        \
      half* fp16_weights_out,                   \
      size_t count);

INSTANTIATE_LAMB_UPDATE(float, float, float)
INSTANTIATE_LAMB_UPDATE(double, double, double)
INSTANTIATE_LAMB_UPDATE(half, float, half)
INSTANTIATE_LAMB_UPDATE(float, float, half)

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

template <typename T1, typename T2, typename T3>
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
  half* w_fp16_new = chunk_group.tensor_ptrs[6][group_index] != nullptr ? reinterpret_cast<half*>(chunk_group.tensor_ptrs[6][group_index]) + chunk_start : nullptr;

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
        w_fp16_new != nullptr ? w_fp16_new + i : nullptr);
  }
}

template <typename T1, typename T2, typename T3>
void LambMultiTensorUpdateFunctor<T1, T2, T3>::operator()(
    ChunkGroup<7> chunk_group,
    const T1* eta,
    const float ratio_min,
    const float ratio_max) {
  const int thread_count = ChunkGroup<7>::thread_count_per_block;
  const int block_count = chunk_group.chunk_count;

  LambMultiTensorUpdateImpl<T1, T2, T3><<<block_count, thread_count, 0>>>(
      chunk_group,
      eta,
      ratio_min,
      ratio_max);
}

#define INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(T1, T2, T3)      \
  template void LambMultiTensorUpdateFunctor<T1, T2, T3>::operator()( \
      ChunkGroup<7> chunk_group,                                      \
      const T1* eta,                                                  \
      const float ratio_min,                                          \
      const float ratio_max);

INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(float, float, float)
INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(double, double, double)
INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(half, float, half)
INSTANTIATE_LAMB_MULTI_TENSOR_UPDATE_FUNCTOR(float, float, half)

template <typename TIn1, typename TIn2, typename TOut1, typename TOut2, typename TBuf>
__global__ void LambMultiTensorReductionImpl(ChunkGroup<4> chunk_group) {
  const int group_index = chunk_group.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunk_group.tensor_sizes[group_index];
  const int chunk_size = chunk_group.chunk_size;
  const int chunk_start = chunk_group.block_index_to_chunk_start_index[blockIdx.x];
  const TIn1* w = reinterpret_cast<const TIn1*>(chunk_group.tensor_ptrs[0][group_index]) + chunk_start;
  const TIn2* d = reinterpret_cast<const TIn2*>(chunk_group.tensor_ptrs[1][group_index]) + chunk_start;
  TOut1* w_norm = reinterpret_cast<TOut1*>(chunk_group.tensor_ptrs[2][group_index]);
  TOut2* d_norm = reinterpret_cast<TOut2*>(chunk_group.tensor_ptrs[3][group_index]);

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

  // Thread count in a block must be a multiple of GPU_WARP_SIZE.
#pragma unroll
  for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
    w_sum += WARP_SHFL_DOWN(w_sum, stride);
    d_sum += WARP_SHFL_DOWN(d_sum, stride);
  }

  const int warp_count_in_block = blockDim.x / GPU_WARP_SIZE;
  const int lid = threadIdx.x % GPU_WARP_SIZE;
  const int wid = threadIdx.x / GPU_WARP_SIZE;

  // Shape is 2 x warp_count_in_block.
  extern __shared__ unsigned char shared_memory_[];
  TBuf* shared_memory = reinterpret_cast<TBuf*>(shared_memory_);
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

template <typename TIn1, typename TIn2, typename TOut1, typename TOut2, typename TBuf>
void LambMultiTensorReductionFunctor<TIn1, TIn2, TOut1, TOut2, TBuf>::operator()(ChunkGroup<4> chunk_group) {
  // thread count per block.
  constexpr int thread_count = ChunkGroup<4>::thread_count_per_block;
  // shared memory's size per block.
  const int shared_memory_size = thread_count / GPU_WARP_SIZE * 2 * sizeof(TBuf);

  // Enforce assumptions used inside this reduction CUDA kernel.
  ORT_ENFORCE(thread_count % GPU_WARP_SIZE == 0);
  ORT_ENFORCE((thread_count & (thread_count - 1)) == 0);

  LambMultiTensorReductionImpl<TIn1, TIn2, TOut1, TOut2, TBuf><<<chunk_group.chunk_count, thread_count, shared_memory_size>>>(chunk_group);
}

#define INSTANTIATE_LAMB_MULTI_TENSOR_REDUCTION_FUNCTOR(TIn1, TIn2, TOut1, TOut2, TBuf) \
  template void LambMultiTensorReductionFunctor<TIn1, TIn2, TOut1, TOut2, TBuf>::operator()(ChunkGroup<4> chunk_group);

INSTANTIATE_LAMB_MULTI_TENSOR_REDUCTION_FUNCTOR(float, float, float, float, float)
INSTANTIATE_LAMB_MULTI_TENSOR_REDUCTION_FUNCTOR(double, double, double, double, double)
INSTANTIATE_LAMB_MULTI_TENSOR_REDUCTION_FUNCTOR(float, half, float, half, float)
INSTANTIATE_LAMB_MULTI_TENSOR_REDUCTION_FUNCTOR(float, half, float, float, float)
INSTANTIATE_LAMB_MULTI_TENSOR_REDUCTION_FUNCTOR(half, half, half, half, float)

}  // namespace cuda
}  // namespace onnxruntime