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

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    NonMaxSuppression,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int32_t>()),
    NonMaxSuppression<float>);

template <typename T>
void NonMaxSuppression<T>::MaxMin(const T& lhs, const T& rhs, T& min, T& max) const {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}

template <typename T>
bool NonMaxSuppression<T>::SuppressByIOU(const T* boxes_data, int32_t box_index1, int32_t box_index2) const {
  T x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max;
  // boxes data [y1, x1, y2, x2],
  MaxMin(boxes_data[4 * box_index1 + 1], boxes_data[4 * box_index1 + 3], x1_min, x1_max);
  MaxMin(boxes_data[4 * box_index1 + 0], boxes_data[4 * box_index1 + 2], y1_min, y1_max);
  MaxMin(boxes_data[4 * box_index2 + 1], boxes_data[4 * box_index2 + 3], x2_min, x2_max);
  MaxMin(boxes_data[4 * box_index2 + 0], boxes_data[4 * box_index2 + 2], y2_min, y2_max);

  const T intersection_x_min = std::max(x1_min, x2_min);
  const T intersection_y_min = std::max(y1_min, y2_min);
  const T intersection_x_max = std::min(x1_max, x2_max);
  const T intersection_y_max = std::min(y1_max, y2_max);

  const T intersection_area = std::max(intersection_x_max - intersection_x_min, static_cast<T>(0.0)) *
                              std::max(intersection_y_max - intersection_y_min, static_cast<T>(0.0));

  if (intersection_area <= static_cast<T>(0.0)) {
    return false;
  }

  const T area1 = (x1_max - x1_min) * (y1_max - y1_min);
  const T area2 = (x2_max - x2_min) * (y2_max - y2_min);
  const T union_area = area1 + area2 - intersection_area;

  if (area1 <= static_cast<T>(0.0) || area2 <= static_cast<T>(0.0) || union_area <= static_cast<T>(0.0)) {
    return false;
  }

  const T intersection_over_union = intersection_area / union_area;

  return intersection_over_union > iou_threshold_;
}

template <typename T>
Status NonMaxSuppression<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* boxes = ctx->Input<Tensor>(0);
  ORT_ENFORCE(boxes);
  const Tensor* scores = ctx->Input<Tensor>(1);
  ORT_ENFORCE(scores);

  const TensorShape& boxes_shape = boxes->Shape();
  auto boxes_dims = boxes_shape.GetDims();
  ORT_RETURN_IF_NOT(boxes_shape.NumDimensions() == 2, "boxes must be a 2D tensor.");
  int64_t num_boxes = boxes_dims[0];
  ORT_RETURN_IF_NOT(boxes_dims[1] == 4, "boxes shape must be a 2D tensor with shape [num_boxes, 4].");

  const TensorShape& scores_shape = scores->Shape();
  ORT_RETURN_IF_NOT(scores_shape.NumDimensions() == 1, "boxes must be a 1D tensor.");
  ORT_RETURN_IF_NOT(scores_shape.GetDims()[0] == num_boxes, "scores and boxes should have same num_boxes.");

  if (max_output_size_ <= 0 || boxes_dims[0] == 0) {
    TensorShape output_shape({0});
    ctx->Output(0, output_shape);
    return Status::OK();
  }

  const T* boxes_data = boxes->Data<T>();
  const T* scores_data = scores->Data<T>();

  struct ScoreIndexPair {
    T score;
    int32_t index;
  };

  auto LessCompare = [](const ScoreIndexPair& lhs, const ScoreIndexPair& rhs) {
    return lhs.score < rhs.score;
  };

  // Filter by score_threshold_
  std::priority_queue<ScoreIndexPair, std::deque<ScoreIndexPair>, decltype(LessCompare)> sorted_scores_with_index(LessCompare);
  for (int32_t i = 0; i < num_boxes; ++i) {
    if (static_cast<float>(scores_data[i]) > score_threshold_) {
      sorted_scores_with_index.emplace(ScoreIndexPair({scores_data[i], i}));
    }
  }

  int num_of_selected = 0;
  std::vector<int32_t> selected_index(max_output_size_, 0);
  ScoreIndexPair next_top_score;

  // Get the next box with top score, filter by iou_threshold_
  while (num_of_selected < max_output_size_ && !sorted_scores_with_index.empty()) {
    next_top_score = sorted_scores_with_index.top();
    sorted_scores_with_index.pop();

    bool selected = true;
    // Check with existing boxes, suppress if exceed the IOU (Intersection Over Union) threshold
    for (int i = num_of_selected - 1; i >= 0; --i) {
      if (SuppressByIOU(boxes_data, selected_index[i], next_top_score.index)) {
        selected = false;
        break;
      }
    }

    if (selected) {
      selected_index[num_of_selected] = next_top_score.index;
      ++num_of_selected;
    }
  }

  int64_t num_to_copy = pad_to_max_output_size_ == 1 ? max_output_size_ : num_of_selected;
  TensorShape output_shape({num_to_copy});
  Tensor* selected_indices = ctx->Output(0, output_shape);
  auto output_data = selected_indices->MutableData<int32_t>();
  memcpy(output_data, selected_index.data(), num_to_copy * sizeof(int32_t));

  TensorShape valid_outputs_shape({1});
  Tensor* valid_outputs = ctx->Output(1, valid_outputs_shape);
  if (valid_outputs) {
    valid_outputs->MutableData<int32_t>()[0] = num_of_selected;
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
