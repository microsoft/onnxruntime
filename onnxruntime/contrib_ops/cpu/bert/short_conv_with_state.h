// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include <string>

namespace onnxruntime {
namespace contrib {

template <typename T>
class ShortConvWithState final : public OpKernel {
 public:
  explicit ShortConvWithState(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  std::string activation_;
  int64_t dilation_;
  float epsilon_;
};

}  // namespace contrib
}  // namespace onnxruntime
