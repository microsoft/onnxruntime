// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"

#include "orttraining/training_ops/cuda/optimizer/adamw/adamw_impl.h"
#include "orttraining/training_ops/cuda/optimizer/common.cuh"
#include "orttraining/training_ops/cuda/optimizer/common.h"

namespace onnxruntime {
namespace cuda {

template <typename T_WEIGHT, typename T_GRAD, typename T_MOMENTUM>
__device__ void PrepareMTAData(
    const ChunkGroup<MTA_ADAMW_GROUP_SIZE>& chunks,
    const int& block_idx,
    T_WEIGHT*& weight_chunk_ptr,
    T_GRAD*& grad_chunk_ptr,
    T_MOMENTUM*& momentum_1_chunk_ptr,
    T_MOMENTUM*& momentum_2_chunk_ptr,
    int& chunk_size) {
  const int tensor_idx = chunks.block_index_to_tensor_group_index[block_idx];
  const int tensor_size = chunks.tensor_sizes[tensor_idx];
  T_WEIGHT* weight_tensor_ptr = static_cast<T_WEIGHT*>(chunks.tensor_ptrs[0][tensor_idx]);
  T_GRAD* grad_tensor_ptr = static_cast<T_GRAD*>(chunks.tensor_ptrs[1][tensor_idx]);
  T_MOMENTUM* momentum_1_tensor_ptr = static_cast<T_MOMENTUM*>(chunks.tensor_ptrs[2][tensor_idx]);
  T_MOMENTUM* momentum_2_tensor_ptr = static_cast<T_MOMENTUM*>(chunks.tensor_ptrs[3][tensor_idx]);
  const int chunk_start_idx = chunks.block_index_to_chunk_start_index[block_idx];
  // chunk_size is chunks.chunk_size if the loaded chunk is full. Otherwise (this
  // chunk is the last one in the source tensor), the actual size is determined
  // by the bound of the source tensor.
  chunk_size = min(tensor_size, chunk_start_idx + chunks.chunk_size) - chunk_start_idx;

  weight_chunk_ptr = weight_tensor_ptr + chunk_start_idx;
  grad_chunk_ptr = grad_tensor_ptr + chunk_start_idx;
  momentum_1_chunk_ptr = momentum_1_tensor_ptr + chunk_start_idx;
  momentum_2_chunk_ptr = momentum_2_tensor_ptr + chunk_start_idx;
}

// Torch Adam equivalence.
template <typename T_WEIGHT, typename T_GRAD, typename T_MOMENTUM>
__global__ void AdamWComputeMode0(
    ChunkGroup<MTA_ADAMW_GROUP_SIZE> chunks,
    const float alpha,
    const float beta,
    const float epsilon,
    const float lr,
    const float alpha_correction,
    const float beta_correction,
    const float decay) {
  const int block_idx = blockIdx.x;

  T_WEIGHT* weight_chunk_ptr;
  T_GRAD* grad_chunk_ptr;
  T_MOMENTUM* momentum_1_chunk_ptr;
  T_MOMENTUM* momentum_2_chunk_ptr;

  // TODO(pengwa): unroll this one for better perf.
  int chunk_size;

  PrepareMTAData(chunks, block_idx, weight_chunk_ptr, grad_chunk_ptr,
                 momentum_1_chunk_ptr, momentum_2_chunk_ptr, chunk_size);

#pragma unroll 4
  for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
    float w = static_cast<float>(weight_chunk_ptr[i]);
    float g = static_cast<float>(grad_chunk_ptr[i]);
    float m1 = static_cast<float>(momentum_1_chunk_ptr[i]);
    float m2 = static_cast<float>(momentum_2_chunk_ptr[i]);

    // Perform weight decay.
    w = w - (w * lr * decay);

    // Compute exponentially-averaged historical gradient.
    m1 = alpha * m1 + (1.f - alpha) * g;

    // Compute exponentially-averaged historical squared gradient.
    m2 = beta * m2 + (1.f - beta) * g * g;

    // Compute the new weight.
    const float denom = (_Sqrt(m2) / _Sqrt(beta_correction)) + epsilon;
    w = w - (lr * m1) / (alpha_correction * denom);

    // Update the new weight and momentums.
    weight_chunk_ptr[i] = static_cast<T_WEIGHT>(w);
    momentum_1_chunk_ptr[i] = static_cast<T_MOMENTUM>(m1);
    momentum_2_chunk_ptr[i] = static_cast<T_MOMENTUM>(m2);
  }
}

// Huggingface AdamW equivalence.
template <typename T_WEIGHT, typename T_GRAD, typename T_MOMENTUM>
__global__ void AdamWComputeMode1(
    ChunkGroup<MTA_ADAMW_GROUP_SIZE> chunks,
    const float alpha,
    const float beta,
    const float epsilon,
    const float lr,
    const float lr_corrected,
    const float decay) {
  const int block_idx = blockIdx.x;

  T_WEIGHT* weight_chunk_ptr;
  T_GRAD* grad_chunk_ptr;
  T_MOMENTUM* momentum_1_chunk_ptr;
  T_MOMENTUM* momentum_2_chunk_ptr;
  int chunk_size;

  PrepareMTAData(chunks, block_idx, weight_chunk_ptr, grad_chunk_ptr,
                 momentum_1_chunk_ptr, momentum_2_chunk_ptr, chunk_size);

#pragma unroll 4
  for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
    float w = static_cast<float>(weight_chunk_ptr[i]);
    float g = static_cast<float>(grad_chunk_ptr[i]);
    float m1 = static_cast<float>(momentum_1_chunk_ptr[i]);
    float m2 = static_cast<float>(momentum_2_chunk_ptr[i]);

    // Compute exponentially-averaged historical gradient.
    m1 = alpha * m1 + (1.f - alpha) * g;

    // Compute exponentially-averaged historical squared gradient.
    m2 = beta * m2 + (1.f - beta) * g * g;

    float denom = _Sqrt(m2) + epsilon;
    w = w - (lr_corrected * m1 / denom);

    // Perform weight decay.
    w = w - (lr * decay * w);

    // Update the new weight and momentums.
    weight_chunk_ptr[i] = static_cast<T_WEIGHT>(w);
    momentum_1_chunk_ptr[i] = static_cast<T_MOMENTUM>(m1);
    momentum_2_chunk_ptr[i] = static_cast<T_MOMENTUM>(m2);
  }
}

template <typename T_WEIGHT, typename T_GRAD, typename T_MOMENTUM>
void AdamWMTAFunctor<T_WEIGHT, T_GRAD, T_MOMENTUM>::operator()(
    cudaStream_t stream,
    ChunkGroup<MTA_ADAMW_GROUP_SIZE> chunks,
    const float alpha,
    const float beta,
    const float epsilon,
    const float lr,
    const float decay,
    const int64_t adam_mode,
    const int64_t correct_bias,
    const int64_t update_count) {
  const int block_count = chunks.chunk_count;
  const int thread_count = ChunkGroup<MTA_ADAMW_GROUP_SIZE>::thread_count_per_block;

  float alpha_correction = 1.f, beta_correction = 1.f;
  float lr_corrected = lr;
  if (correct_bias == 1) {
    // Notes:
    // > there is a minor difference compared with Apex's implementation,
    //   which uses double storing corrections before casting to float passing to kernels.
    // > std::pow(float, int) return double since C++11, so we cast back to float.
    alpha_correction = 1.f - static_cast<float>(std::pow(alpha, update_count));
    beta_correction = 1.f - static_cast<float>(std::pow(beta, update_count));
    lr_corrected *= std::sqrt(beta_correction) / alpha_correction;
  }

  // Currently two kinds of AdamW supported:
  // Mode 0: Pytorch https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW,
  //         bias correction is applied on m and v individually,
  //         weight decay is applied before weight is updated.
  // Mode 1: Huggingface https://github.com/huggingface/transformers/blob/d91841315aab55cf1347f4eb59332858525fad0f/
  //         src/transformers/optimization.py,
  //         bias correction is applied on learning rate, then use lr_corrected for subsequent computations.
  //         weight decay is applied after weight is updated.
  if (adam_mode == 0) {
    AdamWComputeMode0<T_WEIGHT, T_GRAD, T_MOMENTUM><<<block_count, thread_count, 0, stream>>>(
        chunks, alpha, beta, epsilon, lr, alpha_correction, beta_correction, decay);
  } else if (adam_mode == 1) {
    AdamWComputeMode1<T_WEIGHT, T_GRAD, T_MOMENTUM><<<block_count, thread_count, 0, stream>>>(
        chunks, alpha, beta, epsilon, lr, lr_corrected, decay);
  } else {
    ORT_THROW("Unsupported Adamw optimizer mode.");
  }
}

#define INSTANTIATE_ADAMMTA_FUNCTOR(T_WEIGHT, T_GRAD, T_MOMENTUM)           \
  template void AdamWMTAFunctor<T_WEIGHT, T_GRAD, T_MOMENTUM>::operator()(  \
      cudaStream_t stream,                                                  \
      ChunkGroup<MTA_ADAMW_GROUP_SIZE> chunks,                              \
      const float alpha,                                                    \
      const float beta,                                                     \
      const float epsilon,                                                  \
      const float lr,                                                       \
      const float decay,                                                    \
      const int64_t adam_mode,                                              \
      const int64_t correct_bias,                                           \
      const int64_t update_count);                                          \
                                                                            \
  template __global__ void AdamWComputeMode0<T_WEIGHT, T_GRAD, T_MOMENTUM>( \
      ChunkGroup<MTA_ADAMW_GROUP_SIZE> chunks,                              \
      const float alpha,                                                    \
      const float beta,                                                     \
      const float epsilon,                                                  \
      const float lr,                                                       \
      const float alpha_correction,                                         \
      const float beta_correction,                                          \
      const float decay);                                                   \
                                                                            \
  template __global__ void AdamWComputeMode1<T_WEIGHT, T_GRAD, T_MOMENTUM>( \
      ChunkGroup<MTA_ADAMW_GROUP_SIZE> chunks,                              \
      const float alpha,                                                    \
      const float beta,                                                     \
      const float epsilon,                                                  \
      const float lr,                                                       \
      const float lr_corrected,                                             \
      const float decay);

INSTANTIATE_ADAMMTA_FUNCTOR(float, float, float);

#undef INSTANTIATE_ADAMMTA_FUNCTOR

}  // namespace cuda
}  // namespace onnxruntime
