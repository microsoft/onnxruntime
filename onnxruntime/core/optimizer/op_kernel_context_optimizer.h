// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class OpKernelContextOptimizer : public onnxruntime::OpKernelContext {
 public:
  OpKernelContextOptimizer() : OpKernelContext() {}
};

}  // namespace onnxruntime