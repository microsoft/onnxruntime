// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/multi_tensor/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
void SGDOptimizerImpl(
    const T* eta,
    const T* weights,
    const T* gradients,
    T* weight_out,
    size_t count);

class SGDOptimizer final : public CudaKernel {
 public:
  SGDOptimizer(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

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
    T3* weight_out,
    T4* moment_1_out,
    T4* moment_2_out,
    half* fp16_weights_out,
    size_t count);

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
class AdamOptimizer final : public CudaKernel {
 public:
  AdamOptimizer(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("alpha", &alpha_, 0.9f);
    info.GetAttrOrDefault("beta", &beta_, 0.999f);
    info.GetAttrOrDefault("lambda", &lambda_, 0.0f);
    info.GetAttrOrDefault("epsilon", &epsilon_, 1e-6f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
  float lambda_;
  float epsilon_;
};

// Implementation can be found in cuda file, optimizers_impl.cu
// T1's precision should be higher than T2.
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
    size_t count);

// Implementation can be found in cuda file, optimizers_impl.cu
// T2's precision should be higher than T1.
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
    size_t count);

template <typename T1, typename T2, typename T3, typename T4>
class LambOptimizer final : public CudaKernel {
 public:
  LambOptimizer(const OpKernelInfo& info) : CudaKernel(info) {
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

// Implementation can be found in cuda file, optimizers_impl.cu
template <typename T, typename T_GRAD>
void AccumulateGradientImpl(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count);

template <typename T, typename T_GRAD>
class AccumulateGradient final : public CudaKernel {
 public:
  AccumulateGradient(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class ZeroGradient final : public CudaKernel {
 public:
  ZeroGradient(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

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
template<typename T1, typename T2, typename T3>
struct LambStage1MultiTensorFunctor {
  void operator()(
    ChunkGroup<6> chunk_group,
    const T1 *loss_scale,
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
template<typename TIn1, typename TIn2, typename TOut1, typename TOut2, typename TBuf>
struct LambReductionMultiTensorFunctor {
  void operator()(ChunkGroup<4> chunk_group);
};

// Lamb's stage 2 maps [w_norm, w_norm, w, d] to [w_new, w_fp16_new] where
//  w_norm: norm of w
//  d_norm: norm of d
//  w: weight tensor
//  d: update direction
//  w_new: updated weight tensor
//  w_fp16_new: updated weight tensor in half-precision
// There are 6 distinct tensors in total and therefore the
// type of chunk_group is ChunkGroup<6>.
//
// Tensor pointers associated with the i-th tensor in this chunk:
//  w_norm: chunk_group.tensor_ptrs[0][i]
//  d_norm: chunk_group.tensor_ptrs[1][i]
//  w: chunk_group.tensor_ptrs[2][i]
//  d: chunk_group.tensor_ptrs[3][i]
//  w_new: chunk_group.tensor_ptrs[4][i]
//  w_fp16_new: chunk_group.tensor_ptrs[5][i]
template<typename T1, typename T2, typename T3>
struct LambStage2MultiTensorFunctor {
  void operator()(
    ChunkGroup<6> chunk_group,
    const T1* eta,
    const T2 threshold);
};

}  // namespace cuda
}  // namespace onnxruntime
