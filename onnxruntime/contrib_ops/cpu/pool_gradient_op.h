// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "op_gradients.h"
#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
namespace contrib {

template <typename T>
class MaxPoolGrad final : public OpKernel {
 public:
  explicit MaxPoolGrad(const OpKernelInfo& info) : OpKernel(info) {
    output_tensor_shapes_ = InferOutputShapes(info);
    ORT_ENFORCE(!output_tensor_shapes_[0].empty());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MaxPoolGrad);
  std::vector<VectorInt64> output_tensor_shapes_;
};

}  // namespace contrib
}  // namespace onnxruntime
