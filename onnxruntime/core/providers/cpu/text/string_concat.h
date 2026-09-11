// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(DISABLE_STRING_TYPE)

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class StringConcat final : public OpKernel {
 public:
  StringConcat(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime

#endif  // !defined(DISABLE_STRING_TYPE)
