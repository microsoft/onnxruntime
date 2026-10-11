// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class EngramGate final : public OpKernel {
 public:
  explicit EngramGate(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  float epsilon_;
};

}  // namespace contrib
}  // namespace onnxruntime
