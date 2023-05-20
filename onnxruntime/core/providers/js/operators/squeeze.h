// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/framework/data_transfer_manager.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace onnxruntime {
namespace js {

class Squeeze final : public JsKernel {
 public:
  Squeeze(const OpKernelInfo& info) : JsKernel(info){}

  Status Compute(OpKernelContext* context) const override {
    // Get the input tensor
    const Tensor* inputTensor = context->Input<Tensor>(0);
    if (inputTensor == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Input tensor is not set");
    }
    // Get the shape of the input tensor
    const TensorShape& input_shape = inputTensor->Shape();
    // Get the squeeze tensor
    const Tensor* squeezeTensor = context->Input<Tensor>(1);
    gsl::span<const int64_t> squeeze_data_span;
    // If the squeeze tensor is set, get the dimensions to squeeze
    if (squeezeTensor != nullptr) {
      if (squeezeTensor->Shape().NumDimensions() != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "A squeeze tensor must be a vector tensor, got ", squeezeTensor->Shape().NumDimensions(), " dimensions");
      }
      squeeze_data_span = squeezeTensor->template DataAsSpan<int64_t>();
    }
    // Create a vector for the output tensor dimensions
    std::vector<int64_t> result_dims;
    for (int i = 0; i < static_cast<int>(input_shape.NumDimensions()); i++) {
      if (squeezeTensor != nullptr) {
        // Check if the dimension is in the squeeze tensor
        bool found = false;
        for (int j = 0; j < static_cast<int>(squeeze_data_span.size()); j++) {
          int dim = squeeze_data_span[j];
          if (dim < 0) {
            dim += input_shape.NumDimensions();
          }
          if (dim == i) {
            // The dimension size must be 1 to be squeezed
            if (input_shape[i] != 1) {
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Dimension ", i, " is not of size 1");
            }
            found = true;
            break;
          }
        }
        // If the dimension is not in the squeeze tensor, add it to the output tensor dimensions
        if (!found) {
          result_dims.push_back(input_shape[i]);
        }
      } else {
        // If the squeeze tensor is not set, add the dimension to the output tensor dimensions if it is not 1
        if (input_shape[i] != 1) {
          result_dims.push_back(input_shape[i]);
        }
      }
    }
    // Reshape the input tensor using the squeeze dimensions
    // The code is the same as in reshape.h
    TensorShapeVector shape(result_dims.begin(), result_dims.end());
    ReshapeHelper helper(input_shape, shape, false);
    Tensor* outputTensor = context->Output(0, TensorShape(shape));
    const void* source = inputTensor->DataRaw();
    void* target = outputTensor->MutableDataRaw();
    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*inputTensor, *outputTensor));
    }

    return Status::OK();
  }
};


}  // namespace js
}  // namespace onnxruntime
