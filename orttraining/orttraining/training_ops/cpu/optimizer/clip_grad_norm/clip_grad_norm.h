// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class InplaceClipGradNorm final : public OpKernel {
 public:
  InplaceClipGradNorm(const OpKernelInfo& info)
      : OpKernel(info) {
    info.GetAttrOrDefault("max_norm", &max_norm_, 1.0f);
    info.GetAttrOrDefault("norm_type", &norm_type_, std::string("fro"));
    ORT_ENFORCE(norm_type_ == "fro", "Given norm type ", norm_type_, " is not supported for InplaceClipGradNorm.");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float max_norm_;
  std::string norm_type_;
};

}  // namespace contrib
}  // namespace onnxruntime
