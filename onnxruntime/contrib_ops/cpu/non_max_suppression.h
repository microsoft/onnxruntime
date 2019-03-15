// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
//#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

class NonMaxSuppression final : public OpKernel {
 public:
  NonMaxSuppression(const OpKernelInfo& info) : OpKernel(info) {
    score_threshold_ = info.GetAttrOrDefault<float>("score_threshold", 0.0f);
    iou_threshold_ = info.GetAttrOrDefault<float>("iou_threshold", 0.0f);
    ORT_ENFORCE(iou_threshold_ >= 0 && iou_threshold_ <= 1, "iou_threshold must be in range [0, 1]");
    max_output_boxes_per_batch_ = info.GetAttrOrDefault<int64_t>("max_output_boxes_per_batch", 0);
    center_point_box_ = info.GetAttrOrDefault<int64_t>("center_point_box", 0);
    ORT_ENFORCE(0 == center_point_box_ || 0 == center_point_box_, "center_point_box only support 0 or 1");
    num_batches_ = 0;
    num_classes_ = 0;
    num_boxes_ = 0;
  }

  Status Compute(OpKernelContext* context) const override;

private:
  bool SuppressByIOU(const float* boxes_data, int32_t box_index1, int32_t box_index2) const;
  void MaxMin(const float& lhs, const float& rhs, float& min, float& max) const;
  Status ParepareCompute(const TensorShape& boxes_shape, const TensorShape& scores_shape) const;

private :
  float score_threshold_;
  float iou_threshold_;
  int64_t max_output_boxes_per_batch_;
  int64_t center_point_box_;

  int64_t num_batches_;
  int64_t num_classes_;
  int64_t num_boxes_;
};
}  // namespace contrib
}  // namespace onnxruntime
