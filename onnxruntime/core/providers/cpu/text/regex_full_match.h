// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "re2/re2.h"

namespace onnxruntime {

class RegexFullMatch final : public OpKernel {
 public:
  explicit RegexFullMatch(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  RE2 re_;
};

}  // namespace onnxruntime
