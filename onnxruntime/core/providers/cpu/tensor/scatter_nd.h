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
  enum class Reduction : int {
    None = 0,
    Add,
    Mul,
    Min,
    Max,
  };

  explicit ScatterND(const OpKernelInfo& info) : OpKernel(info) {
    // 'reduction' attribute was added in opset 16.
    // its default value is 'none' in which case the op behaves the same as before opset 16.
    std::string reduction;
    if (info.GetAttr<std::string>("reduction", &reduction).IsOK()) {
      if (reduction == "add")
        reduction_ = Reduction::Add;
      else if (reduction == "mul")
        reduction_ = Reduction::Mul;
      else if (reduction == "min")
        reduction_ = Reduction::Min;
      else if (reduction == "max")
        reduction_ = Reduction::Max;
    }
  }

  Status Compute(OpKernelContext* context) const override;

  static Status ValidateShapes(const TensorShape& input_shape,
                               const TensorShape& indice_shape,
                               const TensorShape& update_shape);

 private:
  Reduction reduction_{Reduction::None};
};

}  // namespace onnxruntime
