// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD, typename T_GRAD_NORM>
void AdamOptimizerImpl(
    const T1* eta,
    const T2 update_count,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    const T3* loss_scale,
    const T_GRAD_NORM* grad_norm,
    const T4 alpha,
    const T4 beta,
    const T4 lambda,
    const T4 epsilon,
    const bool do_bias_correction,
    const int64_t weight_decay_mode,
    T4* moment_1_out,
    T4* moment_2_out,
    T3* weights_out,
    T_GRAD* grads_out,
    half* fp16_weights_out,
    size_t count);

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD, typename T_GRAD_NORM>
class AdamOptimizer final : public CudaKernel {
 public:
  AdamOptimizer(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("alpha", &alpha_, 0.9f);
    info.GetAttrOrDefault("beta", &beta_, 0.999f);
    info.GetAttrOrDefault("lambda", &lambda_, 0.0f);
    info.GetAttrOrDefault("epsilon", &epsilon_, 1e-8f);

    int64_t tmp_flag = static_cast<int64_t>(0);
    ORT_ENFORCE(info.GetAttr<int64_t>("do_bias_correction", &tmp_flag).IsOK(), "Missing/Invalid do_bias_correction");
    ORT_ENFORCE(tmp_flag == 0 || tmp_flag == 1, "do_bias_correction must be either 0 or 1.");
    do_bias_correction_ = tmp_flag != 0 ? true : false;
    info.GetAttrOrDefault("weight_decay_mode", &weight_decay_mode_, static_cast<int64_t>(0));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
  float lambda_;
  float epsilon_;
  bool do_bias_correction_;
  int64_t weight_decay_mode_;
};

}  // namespace cuda
}  // namespace onnxruntime
