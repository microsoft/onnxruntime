// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/js/js_execution_provider.h"

struct pthreadpool;

namespace onnxruntime {
namespace js {

class JsKernel : public OpKernel {
 public:
  explicit JsKernel(const OpKernelInfo& info)
      : OpKernel(info) {
  }
};
}  // namespace js
}  // namespace onnxruntime
