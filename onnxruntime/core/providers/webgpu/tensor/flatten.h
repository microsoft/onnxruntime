// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/flatten.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
namespace webgpu {

class Flatten final : public OpKernel {
 public:
  explicit Flatten(const OpKernelInfo& info) : OpKernel{info} {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* input_tensor = context->Input<Tensor>(0);
    if (input_tensor == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Input tensor is not set");
    }
    const TensorShape& input_shape = input_tensor->Shape();
    int64_t input_rank = input_shape.NumDimensions();

    // Handle negative axis
    int64_t axis = axis_;
    if (axis_ < 0) {
      axis += input_rank;
    }

    std::initializer_list<int64_t> output_dims;

    int64_t first_dim = 1;
    for (int64_t i = 0; i < axis; i++) {
      first_dim *= input_shape[i];
    }

    int64_t second_dim = 1;
    for (int64_t i = axis; i < input_rank; i++) {
      second_dim *= input_shape[i];
    }
    output_dims = {first_dim, second_dim};

    TensorShape output_shape(output_dims);
    Tensor* output_tensor = context->Output(0, output_shape);

    const void* source = input_tensor->DataRaw();
    void* target = output_tensor->MutableDataRaw();
    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*input_tensor, *output_tensor));
    }

    return Status::OK();
  }

  private:
    int64_t axis_;
};

}  // namespace webgpu
}  // namespace onnxruntime