// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace contrib {

template <typename T1, typename T2>
class DropoutGrad final : public OpKernel {
 public:
  DropoutGrad(const OpKernelInfo& info) : OpKernel{info} {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
