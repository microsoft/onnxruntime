// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "orttraining/training_ops/cpu/optimizer/adamw/adamwbase.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class AdamWOptimizer final : public OpKernel, public AdamWOptimizerBase {
 public:
  AdamWOptimizer(const OpKernelInfo& info)
      : OpKernel(info), AdamWOptimizerBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  Status CopyInputTensorToOutputTensor(const Tensor& source_tensor, Tensor& dest_tensor) const override;
  Status AdamWComputeMode0(Tensor& weight, Tensor& gradient, Tensor& momentums_1, Tensor& momentums_2, float lr,
                           float alpha_correction,
                           float beta_correction) const;
  Status AdamWComputeMode1(Tensor& weight, Tensor& gradient, Tensor& momentums_1, Tensor& momentums_2, float lr,
                           float lr_corrected) const;
};

}  // namespace contrib
}  // namespace onnxruntime
