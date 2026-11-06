// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/flatten.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class Flatten final : public WebGpuKernel {
 public:
  explicit Flatten(const OpKernelInfo& info) : WebGpuKernel{info} {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 1);
  }

  Status ComputeInternal(ComputeContext& context) const override {
    const Tensor* input_tensor = context.Input<Tensor>(0);
    const TensorShape& input_shape = input_tensor->Shape();
    int64_t input_rank = input_shape.NumDimensions();

    // Handle negative axis
    int64_t axis = axis_;
    if (axis < 0) {
      axis += input_rank;
    }

    if (axis > input_rank) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Invalid value for axis, must be less than or equal to input_rank");
    }

    int64_t first_dim = input_shape.SizeToDimension(static_cast<size_t>(axis));
    int64_t second_dim = input_shape.SizeFromDimension(static_cast<size_t>(axis));

    TensorShape output_shape({first_dim, second_dim});
    Tensor* output_tensor = context.Output(0, output_shape);

    const void* source = input_tensor->DataRaw();
    void* target = output_tensor->MutableDataRaw();
    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      ORT_RETURN_IF_ERROR(context.CopyTensor(*input_tensor, *output_tensor));
    }

    return Status::OK();
  }

 private:
  int64_t axis_;
};

}  // namespace webgpu
}  // namespace onnxruntime
