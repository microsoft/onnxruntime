// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/multi_tensor/common.cuh"
#include "core/framework/stream_handles.h"

namespace onnxruntime {
namespace cuda {
// Implementation can be found in cuda file.
// T1's precision should be higher than T2. It's used for
// large tensors. Small tensors should use multi-tensor version
// of this.
template <typename T1, typename T2, typename T3, typename T_GRAD_NORM>
void LambComputeDirection(
    cudaStream_t stream,
    const T1* weights,
    const T2* grads,
    const T3* moment_1,
    const T3* moment_2,
    const T1* loss_scale,
    const T_GRAD_NORM* grad_norm,
    float alpha,
    float beta,
    float lambda,
    float epsilon,
    float max_norm,
    float alpha_correction,
    float beta_correction,
    T2* update_direction,
    T3* moment_1_out,
    T3* moment_2_out,
    size_t count);

// Implementation can be found in cuda file.
// T2's precision should be higher than T1. It's used for
// large tensors. Small tensors should use multi-tensor version
// of this.
template <typename T1, typename T2, typename T3, typename T_MIXED_PRECISION_FP>
void LambUpdate(
    cudaStream_t stream,
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
    size_t count);

// Lamb's stage 1 maps [w, g, m1, m2] to [d, m1_new, m2_new] where
//  w: weight tensor
//  g: gradient (reused to store update direction)
//  m1: 1st momentum
//  m2: 2nd momentum
//  d: update direction
//  m1_new: updated 1st momentum
//  m2_new: updated 2nd momentum
// Because we reuse g to store d, there are only 6 tensors in total and
// therefore the type of chunk_group is ChunkGroup<6>.
//
// Tensor pointers associated with the i-th tensor in this chunk:
//  w: chunk_group.tensor_ptrs[0][i]
//  g (or d): chunk_group.tensor_ptrs[1][i]
//  m1: chunk_group.tensor_ptrs[2][i]
//  m2: chunk_group.tensor_ptrs[3][i]
//  m1_new: chunk_group.tensor_ptrs[4][i]
//  m2_new: chunk_group.tensor_ptrs[5][i]
template <typename T1, typename T2, typename T3, typename T_GRAD_NORM>
struct LambMultiTensorComputeDirectionFunctor {
  void operator()(
      cudaStream_t stream,
      ChunkGroup<6> chunk_group,
      const T1* loss_scale,
      const T_GRAD_NORM* grad_norm,
      const float lambda,
      const float alpha,
      const float beta,
      const float epsilon,
      const float max_norm,
      const float alpha_correction,
      const float beta_correction);
};

// Lamb's reduction maps [w, d] to [w_norm, d_norm] where
//  w: weight tensor
//  d: update direction
//  w_norm: norm of w
//  d_norm: norm of d
// There are 4 distinct tensors in total and therefore the
// type of chunk_group is ChunkGroup<4>.
//
// Tensor pointers associated with the i-th tensor in this chunk:
//  w: chunk_group.tensor_ptrs[0][i]
//  d: chunk_group.tensor_ptrs[1][i]
//  w_norm: chunk_group.tensor_ptrs[2][i]
//  d_norm: chunk_group.tensor_ptrs[3][i]
template <typename TIn1, typename TIn2, typename TOut1, typename TOut2, typename TBuf>
struct LambMultiTensorReductionFunctor {
  void operator()(
      cudaStream_t stream,
      ChunkGroup<4> chunk_group,
      const CudaKernel& kernel,
      void* reduction_buffer,
      size_t reduction_buffer_size,
      onnxruntime::Stream* ort_stream);
};

// Lamb's reduction mapping [w, d] to [w_norm, d_norm] spans multiples thread blocks
//
// This includes any block-index for which it holds
//    i-th tensor-index == chunk_group.block_index_to_tensor_group_index[ block-index ]
// and where i-th tensor-index corresponds to tensor group w(i), d(i), w_norm(i), d_norm(i)
// (see above)
//
// The above span of blocks corresponding i-th tensor will be contiguous.
// To perform an ORDERED reduction across the thread blocks for i-th tensor,
//   the following struct is passed for every tensor.
// It consists of fields:
//   'leading_block' := lowest block-index corresponding i-th tensor
//   'number_blocks' := number block-index "
//   'completed_blocks' := initialized to zero (for internal use)
// Note 'completed_blocks' prevents inter-block reduction until intra-block reduction is complete.
struct LambMultiTensorSyncRangeAndLock {
  int leading_block;
  int number_blocks;
  int completed_blocks;
};

// Lamb's stage 2 maps [w_norm, w_norm, w, d] to [w_new, g_new, w_mixed_precision_new] where
//  w_norm: norm of w
//  d_norm: norm of d
//  w: weight tensor
//  d: update direction
//  w_new: updated weight tensor
//  g_new: updated gradient tensor
//  w_mixed_precision_new: updated weight tensor of mixed-precision type
// There are 7 distinct tensors in total and therefore the
// type of chunk_group is ChunkGroup<7>.
//
// Tensor pointers associated with the i-th tensor in this chunk:
//  w_norm: chunk_group.tensor_ptrs[0][i]
//  d_norm: chunk_group.tensor_ptrs[1][i]
//  w: chunk_group.tensor_ptrs[2][i]
//  d: chunk_group.tensor_ptrs[3][i]
//  w_new: chunk_group.tensor_ptrs[4][i]
//  g_new: chunk_group.tensor_ptrs[5][i]
//  w_mixed_precision_new: chunk_group.tensor_ptrs[6][i]
template <typename T1, typename T2, typename T3, typename T_MIXED_PRECISION_FP>
struct LambMultiTensorUpdateFunctor {
  void operator()(
      cudaStream_t stream,
      ChunkGroup<7> chunk_group,
      const T1* eta,
      const float ratio_min,
      const float ratio_max);
};

}  // namespace cuda
}  // namespace onnxruntime
