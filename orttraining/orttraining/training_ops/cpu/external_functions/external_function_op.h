// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/external_functions/external_function_registry.h"

namespace onnxruntime {
namespace contrib {

template <bool is_backward>
class ExternalFunctionOpBase final : public OpKernel {
 public:
  ExternalFunctionOpBase(const OpKernelInfo& info) : OpKernel(info) {
    std::string name;
    ORT_THROW_IF_ERROR(info.GetAttr("name", &name));
    external_function_ = external_functions::CreateExternalFunctionKernel(name, info, is_backward);
    ORT_ENFORCE(external_function_);
  }

  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  std::unique_ptr<OpKernel> external_function_;
};

}  // namespace contrib
}  // namespace onnxruntime
