// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

std::vector<std::vector<int64_t>> InferOutputShapes(OpKernelInfo info);

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

template <typename T>
class AveragePoolGrad final : public OpKernel {
 public:
  explicit AveragePoolGrad(const OpKernelInfo& info) : OpKernel(info) {
    output_tensor_shapes_ = InferOutputShapes(info);
    ORT_ENFORCE(!output_tensor_shapes_[0].empty());

    ORT_ENFORCE(info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK(),
                "No kernel shape is set.");

    if (!info.GetAttrs<int64_t>("strides", strides_).IsOK() || strides_.empty()) {
      strides_.resize(kernel_shape_.size(), 1);
    }

    if (!info.GetAttrs<int64_t>("pads", pads_).IsOK() || pads_.empty()) {
      pads_.resize(2 * kernel_shape_.size(), 0);
    }

    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("count_include_pad", &temp).IsOK());
    count_include_pad_ = (temp != 0);

    ORT_ENFORCE(strides_.size() == kernel_shape_.size());
    ORT_ENFORCE(pads_.size() == 2 * kernel_shape_.size());
    ORT_ENFORCE(output_tensor_shapes_[0].size() == kernel_shape_.size() + 2);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AveragePoolGrad);
  std::vector<VectorInt64> output_tensor_shapes_;
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;
  bool count_include_pad_{};

  Status Compute1DAveragePoolGrad(OpKernelContext* context) const;
  Status Compute3DAveragePoolGrad(OpKernelContext* context) const;
  Status Compute2DAveragePoolGrad(OpKernelContext* context) const;
};

}  // namespace contrib
}  // namespace onnxruntime
