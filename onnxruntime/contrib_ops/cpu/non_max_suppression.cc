/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#include "contrib_ops/cpu/non_max_suppression.h"
#include <queue>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    NonMaxSuppression,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder(),
    NonMaxSuppression);

void NonMaxSuppression::MaxMin(const float& lhs, const float& rhs, float& min, float& max) const {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}

bool NonMaxSuppression::SuppressByIOU(const float* boxes_data, int32_t box_index1, int32_t box_index2, float iou_threshold) const {
  float x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max;
  // center_point_box_ only support 0 or 1
  if (0 == center_point_box_) {
    // boxes data format [y1, x1, y2, x2],
    MaxMin(boxes_data[4 * box_index1 + 1], boxes_data[4 * box_index1 + 3], x1_min, x1_max);
    MaxMin(boxes_data[4 * box_index1 + 0], boxes_data[4 * box_index1 + 2], y1_min, y1_max);
    MaxMin(boxes_data[4 * box_index2 + 1], boxes_data[4 * box_index2 + 3], x2_min, x2_max);
    MaxMin(boxes_data[4 * box_index2 + 0], boxes_data[4 * box_index2 + 2], y2_min, y2_max);
  } else {
    // 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
    float box1_width_half = boxes_data[4 * box_index1 + 2] / 2;
    float box1_height_half = boxes_data[4 * box_index1 + 3] / 2;
    float box2_width_half = boxes_data[4 * box_index2 + 2] / 2;
    float box2_height_half = boxes_data[4 * box_index2 + 3] / 2;

    x1_min = boxes_data[4 * box_index1 + 0] - box1_width_half;
    x1_max = boxes_data[4 * box_index1 + 0] + box1_width_half;
    y1_min = boxes_data[4 * box_index1 + 1] - box1_height_half;
    y1_max = boxes_data[4 * box_index1 + 1] + box1_height_half;

    x2_min = boxes_data[4 * box_index2 + 0] - box2_width_half;
    x2_max = boxes_data[4 * box_index2 + 0] + box2_width_half;
    y2_min = boxes_data[4 * box_index2 + 1] - box2_height_half;
    y2_max = boxes_data[4 * box_index2 + 1] + box2_height_half;
  }

  const float intersection_x_min = std::max(x1_min, x2_min);
  const float intersection_y_min = std::max(y1_min, y2_min);
  const float intersection_x_max = std::min(x1_max, x2_max);
  const float intersection_y_max = std::min(y1_max, y2_max);

  const float intersection_area = std::max(intersection_x_max - intersection_x_min, static_cast<float>(0.0)) *
                                  std::max(intersection_y_max - intersection_y_min, static_cast<float>(0.0));

  if (intersection_area <= static_cast<float>(0.0)) {
    return false;
  }

  const float area1 = (x1_max - x1_min) * (y1_max - y1_min);
  const float area2 = (x2_max - x2_min) * (y2_max - y2_min);
  const float union_area = area1 + area2 - intersection_area;

  if (area1 <= static_cast<float>(0.0) || area2 <= static_cast<float>(0.0) || union_area <= static_cast<float>(0.0)) {
    return false;
  }

  const float intersection_over_union = intersection_area / union_area;

  return intersection_over_union > iou_threshold;
}

Status NonMaxSuppression::ParepareCompute(OpKernelContext* ctx, const TensorShape& boxes_shape, const TensorShape& scores_shape,
                                          int32_t& max_output_boxes_per_class, float& iou_threshold, float& score_threshold, bool& has_score_threshold) const {
  ORT_RETURN_IF_NOT(boxes_shape.NumDimensions() == 3, "boxes must be a 3D tensor.");
  ORT_RETURN_IF_NOT(scores_shape.NumDimensions() == 3, "scores must be a 3D tensor.");

  auto boxes_dims = boxes_shape.GetDims();
  auto scores_dims = scores_shape.GetDims();
  ORT_RETURN_IF_NOT(boxes_dims[0] == scores_dims[0], "boxes and scores should have same num_batches.");
  ORT_RETURN_IF_NOT(boxes_dims[1] == scores_dims[2], "boxes and scores should have same spatial_dimention.");
  ORT_RETURN_IF_NOT(boxes_dims[2] == 4, "The most inner dimension in boxes must have 4 data.");

  const_cast<int64_t&>(num_batches_) = boxes_dims[0];
  const_cast<int64_t&>(num_classes_) = scores_dims[1];
  const_cast<int64_t&>(num_boxes_) = boxes_dims[1];

  const Tensor* max_output_boxes_per_class_tensor = ctx->Input<Tensor>(2);
  if (max_output_boxes_per_class_tensor != nullptr) {
    max_output_boxes_per_class = *(max_output_boxes_per_class_tensor->Data<int32_t>());
    ORT_RETURN_IF_NOT(max_output_boxes_per_class > 0, "max_output_boxes_per_class should be greater than 0.");
  }

  const Tensor* iou_threshold_tensor = ctx->Input<Tensor>(3);
  if (iou_threshold_tensor != nullptr) {
    iou_threshold = *(iou_threshold_tensor->Data<float>());
    ORT_RETURN_IF_NOT((iou_threshold >= 0 && iou_threshold <= 1), "iou_threshold must be in range [0, 1].");
  }

  const Tensor* score_threshold_tensor = ctx->Input<Tensor>(4);
  if (score_threshold_tensor != nullptr) {
    has_score_threshold = true;
    score_threshold = *(score_threshold_tensor->Data<float>());
  }

  return Status::OK();
}

Status NonMaxSuppression::Compute(OpKernelContext* ctx) const {
  const Tensor* boxes = ctx->Input<Tensor>(0);
  ORT_ENFORCE(boxes);
  const Tensor* scores = ctx->Input<Tensor>(1);
  ORT_ENFORCE(scores);

  auto& boxes_shape = boxes->Shape();
  auto& scores_shape = scores->Shape();

  int32_t max_output_boxes_per_class = 0;
  float iou_threshold = 0;
  // Not so sure for the value range of score_threshold, so set a bool to indicate whether it has this input
  bool has_score_threshold = false;
  float score_threshold = 0;

  auto ret = ParepareCompute(ctx, boxes_shape, scores_shape, max_output_boxes_per_class,
                             iou_threshold, score_threshold, has_score_threshold);
  ORT_RETURN_IF_NOT(ret.IsOK(), ret.ErrorMessage());

  const float* boxes_data = boxes->Data<float>();
  const float* scores_data = scores->Data<float>();

  struct ScoreIndexPair {
    float score;
    int32_t index;
  };

  auto LessCompare = [](const ScoreIndexPair& lhs, const ScoreIndexPair& rhs) {
    return lhs.score < rhs.score;
  };

  std::vector<selected_index> tmp_selected_indices;
  for (int64_t batch_index = 0; batch_index < num_batches_; ++batch_index) {
    for (int64_t class_index = 0; class_index < num_classes_; ++class_index) {
      int64_t box_score_offset = (batch_index * num_classes_ + class_index) * num_boxes_;
      int64_t box_offset = batch_index * num_classes_ * num_boxes_ * 4;
      // Filter by score_threshold_
      std::priority_queue<ScoreIndexPair, std::deque<ScoreIndexPair>, decltype(LessCompare)> sorted_scores_with_index(LessCompare);
      for (int64_t box_index = 0; box_index < num_boxes_; ++box_index) {
        if (!has_score_threshold || (has_score_threshold && scores_data[box_score_offset + box_index] > score_threshold)) {
          sorted_scores_with_index.emplace(ScoreIndexPair({scores_data[box_score_offset + box_index], static_cast<int32_t>(box_index)}));
        }
      }

      ScoreIndexPair next_top_score;
      std::vector<int32_t> selected_indicies_inside_class;
      // Get the next box with top score, filter by iou_threshold_
      while (!sorted_scores_with_index.empty()) {
        next_top_score = sorted_scores_with_index.top();
        sorted_scores_with_index.pop();

        bool selected = true;
        // Check with existing selected boxes for this class, suppress if exceed the IOU (Intersection Over Union) threshold
        for (int i = 0; i < selected_indicies_inside_class.size(); ++i) {
          if (SuppressByIOU(boxes_data + box_offset, selected_indicies_inside_class[i], next_top_score.index, iou_threshold)) {
            selected = false;
            break;
          }
        }

        if (selected) {
          if (max_output_boxes_per_class > 0 && selected_indicies_inside_class.size() >= max_output_boxes_per_class) {
            break;
          }
          selected_indicies_inside_class.push_back(next_top_score.index);
          tmp_selected_indices.push_back(selected_index(static_cast<int32_t>(batch_index), static_cast<int32_t>(class_index), next_top_score.index));
        }
      }  //while
    }    //for class_index
  }      //for batch_index

  int32_t num_selected = static_cast<int32_t>(tmp_selected_indices.size());
  Tensor* selected_indices = ctx->Output(0, {num_selected, 3});
  ORT_ENFORCE(selected_indices);
  memcpy(selected_indices->MutableData<int32_t>(), tmp_selected_indices.data(), num_selected * sizeof(selected_index));

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
