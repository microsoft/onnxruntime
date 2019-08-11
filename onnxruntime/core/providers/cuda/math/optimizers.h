// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"

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

template <typename T1, typename T2, typename T3, typename T4>
void AdamOptimizerImpl(
    const T1* eta,
    const T2* update_count,
    const T3* weights,
    const T4* grads,
    const T4* moment_1,
    const T4* moment_2,
    T4 alpha,
    T4 beta,
    T4 lambda,
    T4 epsilon,
    T3* weight_out,
    T4* moment_1_out,
    T4* moment_2_out,
    T2* update_count_out,
    size_t count);

template <typename T1, typename T2, typename T3, typename T4>
class AdamOptimizer final : public CudaKernel {
 public:
  AdamOptimizer(const OpKernelInfo& info): CudaKernel(info) {
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
template <typename T>
void LambComputeDirectionImpl(
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
    size_t count);

// Implementation can be found in cuda file, optimizers_impl.cu
template <typename T>
void LambUpdateImpl(
    const T* eta,
    const T* r_norm,
    const T* w_norm,
    const T* weights,
    const T* update_direction,
    T* weights_out,
    size_t count);

// Implementation can be found in cuda file, optimizers_impl.cu
template <typename T>
void LambScalarL2NormReductionImpl(
    const T* value,
    T* value_out);

class LambOptimizer final : public CudaKernel {
 public:
  LambOptimizer(const OpKernelInfo& info): CudaKernel(info) {
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

}  // namespace cuda
}  // namespace onnxruntime
