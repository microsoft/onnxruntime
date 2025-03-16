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

template <bool is_channels_last>
Status ConvTranspose<is_channels_last>::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto* kernel = context.Input<Tensor>(1);
  TensorShape input_shape = input->Shape();
  TensorShape kernel_shape = kernel->Shape();
  TensorShapeVector local_output_padding(conv_transpose_attrs_.output_padding.begin(), conv_transpose_attrs_.output_padding.end());
  ConvAttributes::ConvPadVector local_pads(conv_transpose_attrs_.pads.begin(), conv_transpose_attrs_.pads.end());
  TensorShapeVector local_dilations(conv_transpose_attrs_.dilations.begin(), conv_transpose_attrs_.dilations.end());
  TensorShapeVector local_strides(conv_transpose_attrs_.strides.begin(), conv_transpose_attrs_.strides.end());
  if (local_output_padding.empty()) {
    local_output_padding.resize(kernel_shape.NumDimensions(), 0);
  }
  if (local_pads.empty()) {
    local_pads.resize(kernel_shape.NumDimensions() * 2, 0);
  }
  if (local_dilations.empty()) {
    local_dilations.resize(kernel_shape.NumDimensions(), 1);
  }
  if (local_strides.empty()) {
    local_strides.resize(kernel_shape.NumDimensions(), 1);
  }
  auto group = conv_transpose_attrs_.group;
  auto num_output_channels = group * kernel_shape[1];
  auto batch_size = input_shape[0];
  TensorShapeVector output_shape(conv_transpose_attrs_.output_shape.begin(), conv_transpose_attrs_.output_shape.end());
  conv_transpose_attrs_.ComputePadsAndOutputShape(input_shape, num_output_channels, kernel_shape.AsShapeVector(), local_strides, local_dilations, local_output_padding, batch_size, &local_pads, &output_shape, is_channels_last);
  std::vector<uint32_t> strides;
  std::vector<uint32_t> pads;
  std::vector<uint32_t> dilations;
  auto transform_dim = [](int64_t dim) { return static_cast<int32_t>(dim); };
  std::transform(local_pads.begin(), local_pads.end(), std::back_inserter(pads), transform_dim);
  std::transform(local_strides.begin(), local_strides.end(), std::back_inserter(strides), transform_dim);
  std::transform(local_dilations.begin(), local_dilations.end(), std::back_inserter(dilations), transform_dim);

  bool has_bias = context.InputCount() > 2;
  const auto* bias = has_bias ? context.Input<Tensor>(2) : nullptr;

  if (input_shape.NumDimensions() == 3 && kernel_shape.NumDimensions() == 3) {
    // ConvTranspose1D
    TensorShapeVector input_shape_vector = input_shape.AsShapeVector();
    TensorShapeVector kernel_shape_vector = kernel_shape.AsShapeVector();
    input_shape_vector.insert(input_shape_vector.begin() + (is_channels_last ? 1 : 2, 1), 1);
    kernel_shape_vector.insert(kernel_shape_vector.begin() + 1, 1);
    input_shape = TensorShape(input_shape_vector);
    kernel_shape = TensorShape(kernel_shape_vector);
    pads.insert(pads.begin(), 0);
    pads.insert(pads.begin() + 2, 0);
    strides.insert(strides.begin(), 1);
    dilations.insert(dilations.begin(), 1);
  }
  if (input_shape.NumDimensions() > 4 || kernel_shape.NumDimensions() > 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only Conv2d or Conv1d are supported.");
  } else if (input_shape.NumDimensions() < 2 || kernel_shape.NumDimensions() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input and kernel tensors must have at least 3 dimensions");
  }
  // Transpose weights
  Tensor transposed_kernel;
  ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel));
  TensorShape transposed_kernel_shape = transposed_kernel.Shape();
  std::vector<TensorShape> input_output_shapes = {input_shape, transposed_kernel_shape, output_shape};
  std::vector<const Tensor*> inputs = {input, &transposed_kernel};
  if (has_bias) {
    inputs.push_back(bias);
  }
  Tensor* output = context.Output(0, output_shape);
  auto program = CreateConvTranspose2DProgram(inputs, pads, strides, dilations, output, is_channels_last, input_output_shapes, static_cast<uint32_t>(conv_transpose_attrs_.group));
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
