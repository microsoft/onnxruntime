// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace js {

class JsKernelLookup : public IExecutionProvider::IKernelLookup {
 public:
  JsKernelLookup(const IKernelLookup& orig): orig_(orig) {
  }
  const KernelCreateInfo* LookUpKernel(const Node& node) const override;
 private:
  const IKernelLookup& orig_;
};

}  // namespace js
}  // namespace onnxruntime
