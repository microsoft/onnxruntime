// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class NonMaxSuppression final : public OpKernel {
 public:
  NonMaxSuppression(const OpKernelInfo& info) : OpKernel(info) {
    center_point_box_ = info.GetAttrOrDefault<int64_t>("center_point_box", 0);
    ORT_ENFORCE(0 == center_point_box_ || 1 == center_point_box_, "center_point_box only support 0 or 1");
    num_batches_ = 0;
    num_classes_ = 0;
    num_boxes_ = 0;
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool SuppressByIOU(const float* boxes_data, int32_t box_index1, int32_t box_index2, float iou_threshold) const;
  void MaxMin(const float& lhs, const float& rhs, float& min, float& max) const;
  Status ParepareCompute(OpKernelContext* ctx, const TensorShape& boxes_shape, const TensorShape& scores_shape,
                         int32_t& max_output_boxes_per_batch, float& iou_threshold, float& score_threshold, bool& has_score_threshold) const;

 private:
  int64_t center_point_box_;

  int64_t num_batches_;
  int64_t num_classes_;
  int64_t num_boxes_;

  struct selected_index {
    selected_index(int32_t batch_index, int32_t class_index, int32_t box_index)
        : batch_index_(batch_index), class_index_(class_index), box_index_(box_index) {}
    int32_t batch_index_ = 0;
    int32_t class_index_ = 0;
    int32_t box_index_ = 0;
  };
};
}  // namespace contrib
}  // namespace onnxruntime
