// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

// Attributes for ONNX DeformConv (opset 19+).
// See https://onnx.ai/onnx/operators/onnx__DeformConv.html
struct DeformConvAttributes {
  explicit DeformConvAttributes(const OpKernelInfo& info) {
    // Optional attributes.
    // If not present, they will be empty/default, and handled in Compute.
    (void)info.GetAttrs("kernel_shape", kernel_shape);
    (void)info.GetAttrs("strides", strides);
    (void)info.GetAttrs("pads", pads);
    (void)info.GetAttrs("dilations", dilations);
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
class DeformConv : public OpKernel {
 public:
  explicit DeformConv(const OpKernelInfo& info) : OpKernel(info), attrs_(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  DeformConvAttributes attrs_;
};

}  // namespace onnxruntime
