// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

// This contains the utility functions which will be used to build a webnn model

#pragma once

#include "core/common/status.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {
namespace webnn {

// Try to see if we can map explicit padding to auto padding for Conv/Pool.
// Since usually use auto padding is more efficient.
common::Status HandleAutoPad(const std::vector<int64_t> input_shape,
                             const int64_t weight_size_y,
                             const int64_t weight_size_x,
                             const std::vector<int64_t>& onnx_pads,
                             const std::vector<int64_t>& onnx_strides,
                             const std::vector<int64_t>& onnx_dilations,
                             AutoPadType auto_pad_type,
                             std::vector<int64_t>& pads_out,
                             bool use_nchw) ORT_MUST_USE_RESULT;

}  // namespace webnn
}  // namespace onnxruntime
