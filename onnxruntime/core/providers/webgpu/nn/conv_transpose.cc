// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "conv.h"
#include "conv_transpose.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/webgpu/nn/conv_backprop_webgpu.h"

namespace onnxruntime {
namespace webgpu {
// kernel shape is the spacial dims of the filter.
// ie. filter shape with batch and channel. kernel shape dimension is 2 less than the filter dimension

template <bool is_channels_last>
Status ConvTranspose<is_channels_last>::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto* filter = context.Input<Tensor>(1);
  TensorShape input_shape = input->Shape();
  TensorShape filter_shape = filter->Shape();
  const InlinedVector<size_t> perm = {2, 3, 0, 1};
  TensorShapeVector local_output_padding(conv_transpose_attrs_.output_padding.begin(), conv_transpose_attrs_.output_padding.end());
  ConvAttributes::ConvPadVector local_pads(conv_transpose_attrs_.pads.begin(), conv_transpose_attrs_.pads.end());
  TensorShapeVector local_dilations(conv_transpose_attrs_.dilations.begin(), conv_transpose_attrs_.dilations.end());
  TensorShapeVector local_strides(conv_transpose_attrs_.strides.begin(), conv_transpose_attrs_.strides.end());
  TensorShapeVector kernel_shape_vector;
  auto rank = input_shape.NumDimensions();
  TensorShape input_spacial_shape = input_shape.Slice(is_channels_last ? 1 : 2, is_channels_last ? rank - 1 : rank);
  local_pads.reserve(2 * (input_spacial_shape.NumDimensions()));
  ORT_RETURN_IF_ERROR(conv_transpose_attrs_.ComputeKernelShape(filter_shape, kernel_shape_vector, false));
  if (local_output_padding.empty()) {
    local_output_padding.resize(kernel_shape_vector.size(), 0);
  }
  if (local_pads.empty()) {
    local_pads.resize(kernel_shape_vector.size() * 2, 0);
  }
  if (local_dilations.empty()) {
    local_dilations.resize(kernel_shape_vector.size(), 1);
  }
  if (local_strides.empty()) {
    local_strides.resize(kernel_shape_vector.size(), 1);
  }
  auto group = conv_transpose_attrs_.group;
  auto num_output_channels = group * filter_shape[1];
  auto batch_size = input_shape[0];
  TensorShapeVector output_shape_vector;
  conv_transpose_attrs_.ComputePadsAndOutputShape(input_spacial_shape, num_output_channels, kernel_shape_vector, local_strides, local_dilations, local_output_padding, batch_size, &local_pads, &output_shape_vector, is_channels_last);
  TensorShape computed_output_shape(output_shape_vector);
  std::vector<uint32_t> strides;
  std::vector<uint32_t> pads;
  std::vector<uint32_t> dilations;
  auto transform_dim = [](int64_t dim) { return static_cast<int32_t>(dim); };
  std::transform(local_pads.begin(), local_pads.end(), std::back_inserter(pads), transform_dim);
  std::transform(local_strides.begin(), local_strides.end(), std::back_inserter(strides), transform_dim);
  std::transform(local_dilations.begin(), local_dilations.end(), std::back_inserter(dilations), transform_dim);

  bool has_bias = context.InputCount() > 2;
  const auto* bias = has_bias ? context.Input<Tensor>(2) : nullptr;
  if (input_shape.NumDimensions() == 3 && filter_shape.NumDimensions() == 3) {
    // ConvTranspose1D
    TensorShapeVector input_shape_vector = input_shape.AsShapeVector();
    TensorShapeVector filter_shape_vector = filter_shape.AsShapeVector();
    input_shape_vector.insert(input_shape_vector.begin() + (is_channels_last ? 1 : 2), 1, 1);
    output_shape_vector.insert(output_shape_vector.begin() + (is_channels_last ? 1 : 2), 1, 1);
    filter_shape_vector.insert(filter_shape_vector.begin() + 2, 1);
    input_shape = TensorShape(input_shape_vector);
    filter_shape = TensorShape(filter_shape_vector);
    pads.insert(pads.begin(), 0);
    pads.insert(pads.begin() + 2, 0);
    strides.insert(strides.begin(), 1);
    dilations.insert(dilations.begin(), 1);
  }
  if (input_shape.NumDimensions() > 4 || filter_shape.NumDimensions() > 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only Conv2d or Conv1d are supported.");
  } else if (input_shape.NumDimensions() < 2 || filter_shape.NumDimensions() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input and kernel tensors must have at least 3 dimensions");
  }
  // Transpose weights
  Tensor transposed_filter;
  ORT_RETURN_IF_ERROR(TransposeKernel(context, filter, filter_shape, &transposed_filter, perm));
  TensorShape output_shape(output_shape_vector);
  TensorShape transposed_filter_shape = transposed_filter.Shape();
  std::vector<const Tensor*> inputs = {input, &transposed_filter};
  std::vector<TensorShape> input_output_shapes = {input_shape, transposed_filter_shape};
  if (has_bias) {
    inputs.push_back(bias);
    input_output_shapes.push_back(bias->Shape());
  }
  uint32_t auto_pad_adjust = conv_transpose_attrs_.auto_pad == AutoPadType::SAME_LOWER ? 1 : 0;
  auto pad0 = conv_transpose_attrs_.auto_pad == AutoPadType::NOTSET ? pads[0] : (pads[0] + pads[2] + auto_pad_adjust) / 2;
  auto pad1 = conv_transpose_attrs_.auto_pad == AutoPadType::NOTSET ? pads[1] : (pads[1] + pads[3] + auto_pad_adjust) / 2;
  Tensor* output = context.Output(0, computed_output_shape);
  input_output_shapes.push_back(output_shape);
  auto program = CreateConvTranspose2DProgram(inputs, {pad0, pad1}, strides, dilations, output, is_channels_last, input_output_shapes, static_cast<uint32_t>(conv_transpose_attrs_.group));
  return context.RunProgram(program);
}

ONNX_OPERATOR_KERNEL_EX(
    ConvTranspose,
    kMSInternalNHWCDomain,
    11,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    ConvTranspose<true>);

ONNX_OPERATOR_KERNEL_EX(
    ConvTranspose,
    kOnnxDomain,
    11,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    ConvTranspose<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ConvTranspose,
    kMSInternalNHWCDomain,
    1, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    ConvTranspose<true>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ConvTranspose,
    kOnnxDomain,
    1, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    ConvTranspose<false>);

}  // namespace webgpu
}  // namespace onnxruntime
