// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class DecodeImage final : public OpKernel {
 public:
  DecodeImage(const OpKernelInfo& info) : OpKernel(info);

  Status Compute(OpKernelContext* context) const override;

 private:
  std::string pixel_format_;
};

}  // namespace onnxruntime
