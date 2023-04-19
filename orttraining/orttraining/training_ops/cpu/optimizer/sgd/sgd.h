// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/optimizer/common.h"
#include "orttraining/training_ops/cpu/optimizer/sgd/sgdbase.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class SGDOptimizerV2 final : public OpKernel, public SGDOptimizerV2Base {
 public:
  SGDOptimizerV2(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
