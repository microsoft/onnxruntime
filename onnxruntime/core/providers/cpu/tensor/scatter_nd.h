// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}

class ScatterND final : public OpKernel {
 public:
  explicit ScatterND(const OpKernelInfo& info) : OpKernel(info) {
    // 'reduction' attribute was added in opset 16.
    // its default value is 'none' in which case the op behaves the same as before opset 16.
    if (!info.GetAttr<std::string>("reduction", &reduction_).IsOK()) {
      reduction_ = "none";
    }
  }
  Status Compute(OpKernelContext* context) const override;

  static Status ValidateShapes(const TensorShape& input_shape,
                               const TensorShape& indice_shape,
                               const TensorShape& update_shape);
 private:
  std::string reduction_{"none"};
};

}  // namespace onnxruntime
