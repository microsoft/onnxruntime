// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
namespace webgpu {

class Identity final : public OpKernel {
 public:
  explicit Identity(const OpKernelInfo& info) : OpKernel{info} {
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* input_tensor = context->Input<Tensor>(0);
    if (input_tensor == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    }

    const TensorShape& input_shape = input_tensor->Shape();
    Tensor* output_tensor = context->Output(0, input_shape);

    const void* source = input_tensor->DataRaw();
    void* target = output_tensor->MutableDataRaw();

    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*input_tensor, *output_tensor));
    }

    return Status::OK();
  }
};

}  // namespace webgpu
}  // namespace onnxruntime
