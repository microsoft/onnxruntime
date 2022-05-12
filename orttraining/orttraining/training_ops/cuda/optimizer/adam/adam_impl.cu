// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/optimizer/adam/adam_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/optimizer/common.cuh"
#include "orttraining/training_ops/cuda/optimizer/common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
namespace onnxruntime {
namespace cuda {

template <typename T_WEIGHT, typename T_GRAD, typename T_MOMENTUM>
__device__ void PrepareMTAData(
    const ChunkGroup<MTA_ADAM_GROUP_SIZE>& chunks,
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
__global__ void _AdamOptimizer_mode0(
    ChunkGroup<MTA_ADAM_GROUP_SIZE> chunks,
    const float alpha,
    const float beta,
    const float lambda,
    const float epsilon,
    const float lr,
    const float lr_corrected,
    const float alpha_correction,
    const float beta_correction,
    const float decay) {
  const int block_idx = blockIdx.x;

  T_WEIGHT* weight_chunk_ptr;
  T_GRAD* grad_chunk_ptr;
  T_MOMENTUM* momentum_1_chunk_ptr;
  T_MOMENTUM* momentum_2_chunk_ptr;
  int chunk_size;

  PrepareMTAData(chunks, block_idx, weight_chunk_ptr, grad_chunk_ptr,
                 momentum_1_chunk_ptr, momentum_2_chunk_ptr, chunk_size);

  // A shared constant.
  const float one = 1.0f;
#pragma unroll(4)
  for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
    float w = static_cast<float>(weight_chunk_ptr[i]);
    float g = static_cast<float>(grad_chunk_ptr[i]);
    float m1 = static_cast<float>(momentum_1_chunk_ptr[i]);
    float m2 = static_cast<float>(momentum_2_chunk_ptr[i]);

    w = w - (w * lr * decay);

    // Compute exponentially-averaged historical gradient.
    const float m1o = alpha * m1 + (one - alpha) * g;
    const float m1o_corrected = m1o / alpha_correction;

    // Compute exponentially-averaged historical squared gradient.
    const float m2o = beta * m2 + (one - beta) * g * g;
    const float m2o_corrected = m2o / beta_correction;

    // Compute weight update.
    const float denom = _Sqrt(m2o_corrected) + epsilon;
    const float update = (m1o_corrected / denom) + (lambda * w);

    const float delta = -lr * update;

    // Compute the new weight.
    weight_chunk_ptr[i] = static_cast<T_WEIGHT>(static_cast<float>(weight_chunk_ptr[i]) + delta);

    // Update momentums.
    momentum_1_chunk_ptr[i] = static_cast<T_MOMENTUM>(m1o);
    momentum_2_chunk_ptr[i] = static_cast<T_MOMENTUM>(m2o);

    // m1 = alpha * m1 + (1 - alpha) * g;
    // m2 = beta * m2 + (1 - beta) * g * g;
    // float denom = (sqrtf(m2) / sqrtf(beta_correction)) + epsilon;
    // w = w - (lr * m1) / (alpha_correction * denom);
  }
}

// Huggingface AdamW equivalence.
template <typename T_WEIGHT, typename T_GRAD, typename T_MOMENTUM>
__global__ void _AdamOptimizer_mode1(
    ChunkGroup<MTA_ADAM_GROUP_SIZE> chunks,
    const float alpha,
    const float beta,
    const float lambda,
    const float epsilon,
    const float lr,
    const float lr_corrected,
    const float alpha_correction,
    const float beta_correction,
    const float decay) {
  const int block_idx = blockIdx.x;

  T_WEIGHT* weight_chunk_ptr;
  T_GRAD* grad_chunk_ptr;
  T_MOMENTUM* momentum_1_chunk_ptr;
  T_MOMENTUM* momentum_2_chunk_ptr;
  int chunk_size;

  PrepareMTAData(chunks, block_idx, weight_chunk_ptr, grad_chunk_ptr,
                 momentum_1_chunk_ptr, momentum_2_chunk_ptr, chunk_size);

  // A shared constant.
  const float one = 1.0f;
#pragma unroll(4)
  for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
    float w = static_cast<float>(weight_chunk_ptr[i]);
    float g = static_cast<float>(grad_chunk_ptr[i]);
    float m1 = static_cast<float>(momentum_1_chunk_ptr[i]);
    float m2 = static_cast<float>(momentum_2_chunk_ptr[i]);

    // Compute exponentially-averaged historical gradient.
    const float m1o = alpha * m1 + (one - alpha) * g;

    // Compute exponentially-averaged historical squared gradient.
    const float m2o = beta * m2 + (one - beta) * g * g;

    const float denom = _Sqrt(m2o) + epsilon;

    // Apply bias correction terms on learning rate
    const float step_size = lr * _Sqrt(beta_correction) / alpha_correction;

    // Huggingface updates weights in the following logic:
    // param' = param - step_size * m1o / denom
    // param_out = param' - original_lr * lambda * param'
    // then param_out = param - step_size * m1o / denom - original_lr * lambda * (param - step_size * m1o / denom)
    // so delta = -step_size * m1o / denom - original_lr * lambda * (param - step_size * m1o / denom)
    const float delta = -step_size * m1o / denom - lr * lambda * (w - step_size * m1o / denom);

    // Compute the new weight.
    weight_chunk_ptr[i] = static_cast<T_WEIGHT>(static_cast<float>(weight_chunk_ptr[i]) + delta);

    // Update momentums.
    momentum_1_chunk_ptr[i] = static_cast<T_MOMENTUM>(m1o);
    momentum_2_chunk_ptr[i] = static_cast<T_MOMENTUM>(m2o);

    // m1 = beta1 * m1 + (1 - beta1) * g;
    // m2 = beta2 * m2 + (1 - beta2) * g * g;
    // float denom = sqrtf(m2) + epsilon;
    // w = w - (lr_corrected * m1 / denom);
    // w = w - (lr * decay * w);
  }
}

template <typename T_WEIGHT, typename T_GRAD, typename T_MOMENTUM>
void AdamMTAFunctor<T_WEIGHT, T_GRAD, T_MOMENTUM>::operator()(
    cudaStream_t stream,
    ChunkGroup<MTA_ADAM_GROUP_SIZE> chunks,
    const float alpha,
    const float beta,
    const float lambda,
    const float epsilon,
    float lr,
    float lr_corrected,
    int64_t adam_mode,
    const float decay,
    const int64_t update_count) {
  std::cout << "AdamMTAFunctor<T_WEIGHT, T_GRAD, T_MOMENTUM>::operator(): " << std::endl;
  const int block_count = chunks.chunk_count;
  const int thread_count = ChunkGroup<MTA_ADAM_GROUP_SIZE>::thread_count_per_block;

  // If bias correction coefficients are set to 1s, it's equivalent to disabling bias correction.
  bool do_bias_correction = true;
  const float alpha_correction = do_bias_correction ? onnxruntime::contrib::compute_bias_correction_coefficient(alpha, update_count) : 1.f;
  const float beta_correction = do_bias_correction ? onnxruntime::contrib::compute_bias_correction_coefficient(beta, update_count) : 1.f;

  // Currently two modes of Adamw are supported:
  // Mode 0: Pytorch https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW,
  //         bias correction is applied on m and v individually,
  //         weight decay is applied before weight is updated.
  // Mode 1: Huggingface https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW.,
  //         bias correction is applied on learning rate,
  //         weight decay is applied after weight is updated.
  if (adam_mode == 0) {
    _AdamOptimizer_mode0<T_WEIGHT, T_GRAD, T_MOMENTUM><<<block_count, thread_count, 0, stream>>>(
        chunks, alpha, beta, lambda, epsilon, lr, lr_corrected,
        alpha_correction, beta_correction, decay);
  } else if (adam_mode == 1) {
    _AdamOptimizer_mode1<T_WEIGHT, T_GRAD, T_MOMENTUM><<<block_count, thread_count, 0, stream>>>(
        chunks, alpha, beta, lambda, epsilon, lr, lr_corrected,
        alpha_correction, beta_correction, decay);
  } else {
    ORT_THROW("Unsupported Adamw optimizer mode.");
  }
}

#define INSTANTIATE_ADAMMTA_FUNCTOR(T_WEIGHT, T_GRAD, T_MOMENTUM)              \
  template void AdamMTAFunctor<T_WEIGHT, T_GRAD, T_MOMENTUM>::operator()(      \
      cudaStream_t stream,                                                     \
      ChunkGroup<MTA_ADAM_GROUP_SIZE> chunks,                                  \
      const float alpha,                                                       \
      const float beta,                                                        \
      const float lambda,                                                      \
      const float epsilon,                                                     \
      float lr,                                                                \
      float lr_corrected,                                                      \
      int64_t adam_mode,                                                       \
      const float decay,                                                       \
      const int64_t update_count);                                             \
                                                                               \
  template __global__ void _AdamOptimizer_mode0<T_WEIGHT, T_GRAD, T_MOMENTUM>( \
      ChunkGroup<MTA_ADAM_GROUP_SIZE> chunks,                                  \
      const float alpha,                                                       \
      const float beta,                                                        \
      const float lambda,                                                      \
      const float epsilon,                                                     \
      const float lr,                                                          \
      const float lr_corrected,                                                \
      const float alpha_correction,                                            \
      const float beta_correction,                                             \
      const float decay);                                                      \
                                                                               \
  template __global__ void _AdamOptimizer_mode1<T_WEIGHT, T_GRAD, T_MOMENTUM>( \
      ChunkGroup<MTA_ADAM_GROUP_SIZE> chunks,                                  \
      const float alpha,                                                       \
      const float beta,                                                        \
      const float lambda,                                                      \
      const float epsilon,                                                     \
      const float lr,                                                          \
      const float lr_corrected,                                                \
      const float alpha_correction,                                            \
      const float beta_correction,                                             \
      const float decay);

INSTANTIATE_ADAMMTA_FUNCTOR(float, float, float);

}  // namespace cuda
}  // namespace onnxruntime
