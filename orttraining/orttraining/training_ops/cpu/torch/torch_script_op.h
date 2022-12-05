// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class TorchScript : public OpKernel {
 public:
  TorchScript(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("key", &key_));
    ORT_THROW_IF_ERROR(info.GetAttr("script", &script_));
  }

  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  int64_t key_;
  std::string script_;
};

}  // namespace contrib
}  // namespace onnxruntime
