// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {

struct PrepareContext;

namespace cuda {

//void NonMaxSuppressionImpl(const float* boxes, const float* scores,
//                           int64_t max_output_boxes_per_class,
//                           float score_threshold,
//                           float iou_threshold);

}  // namespace cuda
}  // namespace onnxruntime
