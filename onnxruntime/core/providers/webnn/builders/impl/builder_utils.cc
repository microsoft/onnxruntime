// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/safeint.h>
#include <core/providers/common.h>
#include "core/providers/shared/utils/utils.h"

#include "builder_utils.h"
#include "core/providers/webnn/builders/helper.h"

namespace onnxruntime {
namespace webnn {

common::Status ComputeConvPads(const std::vector<int64_t> input_shape,
                               const int64_t weight_size_y,
                               const int64_t weight_size_x,
                               const std::vector<int64_t>& onnx_pads,
                               const std::vector<int64_t>& onnx_strides,
                               const std::vector<int64_t>& onnx_dilations,
                               AutoPadType auto_pad_type,
                               std::vector<int64_t>& pads_out,
                               bool use_nchw) {
  const int64_t input_size_y = use_nchw ? input_shape[2] : input_shape[1];
  const int64_t input_size_x = use_nchw ? input_shape[3] : input_shape[2];
  const int64_t stride_y = onnx_strides[0];
  const int64_t stride_x = onnx_strides[1];
  const int64_t dilation_y = onnx_dilations[0];
  const int64_t dilation_x = onnx_dilations[1];

  int64_t padding_top = onnx_pads[0];
  int64_t padding_bottom = onnx_pads[2];
  int64_t padding_left = onnx_pads[1];
  int64_t padding_right = onnx_pads[3];

  ORT_RETURN_IF_ERROR(ComputePad(input_size_y,
                                 stride_y, weight_size_y, dilation_y,
                                 auto_pad_type,
                                 padding_top, padding_bottom));
  ORT_RETURN_IF_ERROR(ComputePad(input_size_x,
                                 stride_x, weight_size_x, dilation_x,
                                 auto_pad_type,
                                 padding_left, padding_right));

  pads_out = {padding_top, padding_left, padding_bottom, padding_right};

  return Status::OK();
}

common::Status HandleAutoPad(const std::vector<int64_t> input_shape,
                             const int64_t weight_size_y,
                             const int64_t weight_size_x,
                             const std::vector<int64_t>& onnx_pads,
                             const std::vector<int64_t>& onnx_strides,
                             const std::vector<int64_t>& onnx_dilations,
                             AutoPadType auto_pad_type,
                             std::vector<int64_t>& pads_out,
                             bool use_nchw) {
  if (AutoPadType::SAME_UPPER == auto_pad_type) {
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        AutoPadType::SAME_UPPER, pads_out, use_nchw));
  } else {
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        AutoPadType::SAME_LOWER, pads_out, use_nchw));
  }
  return Status::OK();
}

}  // namespace webnn
}  // namespace onnxruntime
