// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {

template <typename T>
class RoiPool : public OpKernel {
 public:
  RoiPool(const OpKernelInfo& info) : OpKernel(info) {
    std::vector<int64_t> pooled_shape;
    ORT_THROW_IF_ERROR(info.GetAttrs<int64_t>("pooled_shape", pooled_shape));
    ORT_ENFORCE(pooled_shape.size() == 2);

    pooled_height_ = pooled_shape[0];
    pooled_width_ = pooled_shape[1];
    ORT_ENFORCE(pooled_height_ > 0);
    ORT_ENFORCE(pooled_width_ > 0);

    ORT_ENFORCE(info.GetAttr<float>("spatial_scale", &spatial_scale_).IsOK());
    ORT_ENFORCE(spatial_scale_ > 0);
  }

  ~RoiPool() override = default;

  Status Compute(OpKernelContext* context) const override;

 protected:
  int64_t pooled_height_, pooled_width_;
  float spatial_scale_;
};
}  // namespace onnxruntime
