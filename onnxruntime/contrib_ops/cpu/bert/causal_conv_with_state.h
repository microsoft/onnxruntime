// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include <string>

namespace onnxruntime {
namespace contrib {

template <typename T>
class CausalConvWithState final : public OpKernel {
 public:
  CausalConvWithState(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int ndim_;
  std::string activation_;
  // Always 0 on CPU (a state window is CUDA-only), but kept so the shared shape helper in
  // causal_conv_with_state_helper.h is driven the same way on every EP.
  int state_window_;
};

}  // namespace contrib
}  // namespace onnxruntime
