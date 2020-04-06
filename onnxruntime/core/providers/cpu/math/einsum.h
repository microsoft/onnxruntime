// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/einsum_utils.h"

namespace onnxruntime {

class Einsum final : public OpKernel {
 public:
  Einsum(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("equation", &equation_).IsOK(), "Missing 'equation' attribute");
    einsum_equation_preprocessor_ = onnxruntime::make_unique<EinsumEquationPreprocessor>(equation_);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::string equation_;
  std::unique_ptr<EinsumEquationPreprocessor> einsum_equation_preprocessor_;
};

}  // namespace onnxruntime
