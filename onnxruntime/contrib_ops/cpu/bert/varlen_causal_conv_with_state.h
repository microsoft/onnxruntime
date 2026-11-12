// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class VarlenCausalConvWithState final : public OpKernel {
 public:
  explicit VarlenCausalConvWithState(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int dilation_;
  int state_update_capacity_;
  std::string activation_;
};

}  // namespace contrib
}  // namespace onnxruntime
