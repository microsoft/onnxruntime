// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

// Attributes for ONNX DeformConv (opset 19+). Mirrors CPU for consistency.
// See https://onnx.ai/onnx/operators/onnx__DeformConv.html
struct DeformConvAttributes {
  explicit DeformConvAttributes(const OpKernelInfo& info) {
    Status status = info.GetAttrs("kernel_shape", kernel_shape);
    ORT_ENFORCE(status.IsOK(), "Attribute kernel_shape is not set.");
    status = info.GetAttrs("strides", strides);
    ORT_ENFORCE(status.IsOK(), "Attribute strides is not set.");
    status = info.GetAttrs("pads", pads);
    ORT_ENFORCE(status.IsOK(), "Attribute pads is not set.");
    status = info.GetAttrs("dilations", dilations);
    ORT_ENFORCE(status.IsOK(), "Attribute dilations is not set.");
    group = info.GetAttrOrDefault<int64_t>("group", 1);
    offset_group = info.GetAttrOrDefault<int64_t>("offset_group", 1);
  }

  TensorShapeVector kernel_shape;
  TensorShapeVector strides;
  TensorShapeVector pads;
  TensorShapeVector dilations;
  int64_t group{1};
  int64_t offset_group{1};
};

template <typename T>
class DeformConv final : public CudaKernel {
 public:
  explicit DeformConv(const OpKernelInfo& info) : CudaKernel(info), attrs_(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  DeformConvAttributes attrs_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DeformConv);
};

}  // namespace cuda
}  // namespace onnxruntime
