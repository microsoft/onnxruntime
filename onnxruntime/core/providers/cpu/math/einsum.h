// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class Einsum final : public OpKernel {
 public:
  Einsum(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("equation", &equation_).IsOK(), "Missing 'equation' attribute");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::string equation_;
};

}  // namespace onnxruntime
