// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace vulkan {

class Relu : public OpKernel {
 public:
  explicit Relu(const OpKernelInfo& info) : OpKernel(info) {
  }
  
  Status Compute(OpKernelContext* ctx) const override;

 private:
};

}  // namespace vulkan
}  // namespace onnxruntime