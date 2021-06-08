// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

struct PrepareContext;

class NonMaxSuppressionBase {
 protected:
  explicit NonMaxSuppressionBase(const OpKernelInfo& info) {
    center_point_box_ = info.GetAttrOrDefault<int64_t>("center_point_box", 0);
    ORT_ENFORCE(0 == center_point_box_ || 1 == center_point_box_, "center_point_box only support 0 or 1");
  }

  int64_t GetCenterPointBox() const {
    return center_point_box_;
  }

 public:
  static Status PrepareCompute(OpKernelContext* ctx, PrepareContext& pc);
  static Status GetThresholdsFromInputs(const PrepareContext& pc,
                                        int64_t& max_output_boxes_per_class,
                                        float& iou_threshold,
                                        float& score_threshold);

 private:
  int64_t center_point_box_;
};

class NonMaxSuppression final : public OpKernel, public NonMaxSuppressionBase {
 public:
  explicit NonMaxSuppression(const OpKernelInfo& info) : OpKernel(info), NonMaxSuppressionBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace onnxruntime
