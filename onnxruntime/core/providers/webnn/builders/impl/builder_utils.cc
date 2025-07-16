// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

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

common::Status ComputeConvTransposePadAndOutputShape(
    const int64_t in_size,
    const int64_t stride,
    const int64_t kernel,
    const int64_t dilation,
    const int64_t adj,
    AutoPadType pad_type,
    int64_t& pad_head,
    int64_t& pad_tail,
    int64_t& out_size) {
  // Output shape is explicitly provided - pad values will have to be computed.
  if (out_size != -1) {
    // total pad
    auto total_pad = ComputeTotalPad(in_size, stride, adj, kernel, dilation, out_size);
    DistributePadding(pad_type, total_pad, pad_head, pad_tail);
    return Status::OK();
  }

  // Output shape is not provided - it needs to be computed along with pad values (if applicable).

  // Compute padding if the auto_pad attribute is SAME_UPPER/SAME_LOWER.
  if (pad_type == AutoPadType::SAME_UPPER || pad_type == AutoPadType::SAME_LOWER) {
    // The ONNX spec says if `auto_pad` attribute is set, pad until the `out_size`
    // is `in_size * stride`.
    auto total_pad = ComputeTotalPad(in_size, stride, adj,
                                     kernel, dilation, /*out_size = */ in_size * stride);
    DistributePadding(pad_type, total_pad, pad_head, pad_tail);
  }

  out_size = (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - pad_head - pad_tail;

  return Status::OK();
}

common::Status ComputeConvTransposePadsAndOutputShape(const std::vector<int64_t> input_shape,
                                                      const int64_t weight_size_y,
                                                      const int64_t weight_size_x,
                                                      const std::vector<int64_t>& onnx_pads,
                                                      const std::vector<int64_t>& onnx_strides,
                                                      const std::vector<int64_t>& onnx_dilations,
                                                      const std::vector<int64_t>& onnx_output_padding,
                                                      AutoPadType auto_pad_type,
                                                      std::vector<int64_t>& pads_out,
                                                      std::vector<int64_t>& output_shape_out,
                                                      bool use_nchw) {
  const int64_t input_size_y = use_nchw ? input_shape[2] : input_shape[1];
  const int64_t input_size_x = use_nchw ? input_shape[3] : input_shape[2];
  const int64_t stride_y = onnx_strides[0];
  const int64_t stride_x = onnx_strides[1];
  const int64_t dilation_y = onnx_dilations[0];
  const int64_t dilation_x = onnx_dilations[1];
  const int64_t output_padding_y = onnx_output_padding[0];
  const int64_t output_padding_x = onnx_output_padding[1];

  int64_t padding_top = onnx_pads[0];
  int64_t padding_bottom = onnx_pads[2];
  int64_t padding_left = onnx_pads[1];
  int64_t padding_right = onnx_pads[3];
  int64_t output_shape_out_y = output_shape_out[0];
  int64_t output_shape_out_x = output_shape_out[1];
  ORT_RETURN_IF_ERROR(ComputeConvTransposePadAndOutputShape(
      input_size_y,
      stride_y,
      weight_size_y,
      dilation_y,
      output_padding_y,
      auto_pad_type,
      padding_top,
      padding_bottom,
      output_shape_out_y));
  ORT_RETURN_IF_ERROR(ComputeConvTransposePadAndOutputShape(
      input_size_x,
      stride_x,
      weight_size_x,
      dilation_x,
      output_padding_x,
      auto_pad_type,
      padding_left,
      padding_right,
      output_shape_out_x));

  // WebNN only needs the height and width of the output shape.
  output_shape_out = {output_shape_out_y, output_shape_out_x};
  pads_out = {padding_top, padding_left, padding_bottom, padding_right};

  return Status::OK();
}

}  // namespace webnn
}  // namespace onnxruntime
