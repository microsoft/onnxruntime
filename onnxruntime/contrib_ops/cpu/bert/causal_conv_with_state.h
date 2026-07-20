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
};

}  // namespace contrib
}  // namespace onnxruntime
