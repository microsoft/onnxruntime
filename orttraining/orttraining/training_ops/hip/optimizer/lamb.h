// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/multi_tensor/common.cuh"

namespace onnxruntime {
namespace hip {

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD_NORM>
class LambOptimizer final : public HipKernel {
 public:
  LambOptimizer(const OpKernelInfo& info) : HipKernel(info) {
    alpha_ = info.GetAttrsOrDefault("alpha", std::vector<float>(1024, 0.9f));
    beta_ = info.GetAttrsOrDefault("beta", std::vector<float>(1024, 0.999f));
    lambda_ = info.GetAttrsOrDefault("lambda", std::vector<float>(1024, 0.0f));
    epsilon_ = info.GetAttrsOrDefault("epsilon", std::vector<float>(1024, 1e-6f));
    threshold_ = info.GetAttrsOrDefault("threshold", std::vector<float>(1024, 1.0f));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::vector<float> alpha_;
  std::vector<float> beta_;
  std::vector<float> lambda_;
  std::vector<float> epsilon_;
  std::vector<float> threshold_;
};

// Implementation can be found in hip file, optimizers_impl.cu
// T1's precision should be higher than T2. It's used for 
// large tensors. Small tensors should use multi-tensor version
// of this.
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
    T2* update_direction,
    T3* moment_1_out,
    T3* moment_2_out,
    size_t count);

// Implementation can be found in hip file, optimizers_impl.cu
// T2's precision should be higher than T1. It's used for
// large tensors. Small tensors should use multi-tensor version 
// of this.
template <typename T1, typename T2, typename T3>
void LambUpdate(
    const T1* eta,
    const T2* r_norm,
    const T2* w_norm,
    const T2* weights,
    const T2 threshold,
    const T3* update_direction,
    T2* weights_out,
    T3* gradients_out,
    half* fp16_weights_out,
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
      ChunkGroup<6> chunk_group,
      const T1* loss_scale,
      const T_GRAD_NORM* grad_norm,
      const T1 lambda,
      const T3 alpha,
      const T3 beta,
      const T3 epsilon);
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
  void operator()(ChunkGroup<4> chunk_group);
};

// Lamb's stage 2 maps [w_norm, w_norm, w, d] to [w_new, g_new, w_fp16_new] where
//  w_norm: norm of w
//  d_norm: norm of d
//  w: weight tensor
//  d: update direction
//  w_new: updated weight tensor
//  g_new: updated gradient tensor
//  w_fp16_new: updated weight tensor in half-precision
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
//  w_fp16_new: chunk_group.tensor_ptrs[6][i]
template <typename T1, typename T2, typename T3>
struct LambMultiTensorUpdateFunctor {
  void operator()(
      ChunkGroup<7> chunk_group,
      const T1* eta,
      const T2 threshold);
};

}  // namespace hip
}  // namespace onnxruntime
