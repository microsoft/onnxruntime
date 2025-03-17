// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/nn/conv2d_mm_webgpu.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/transpose.h"

namespace onnxruntime {
namespace webgpu {

template <bool is_channels_last>
TensorShape Conv<is_channels_last>::ComputeOutputShape(const TensorShape& input_shape, const TensorShape& weight_shape) const {
  auto channel_index = is_channels_last ? input_shape.NumDimensions() - 1 : 1;
  auto batch_size = input_shape[0];
  auto output_channels = weight_shape[0];
  auto kernel_spatial_shape = weight_shape.Slice(2);
  TensorShape dilated_kernel_spatial_shape(kernel_spatial_shape);  // dilated kernel shape
  for (size_t i = 0; i < dilated_kernel_spatial_shape.NumDimensions(); ++i) {
    dilated_kernel_spatial_shape[i] = kernel_spatial_shape[i] + (kernel_spatial_shape[i] - 1) * (conv_attrs_.dilations[i] - 1);
  }
  TensorShape input_spacial_shape_with_pads(input_shape.Slice(2));
  for (size_t i = 0; i < input_spacial_shape_with_pads.NumDimensions(); ++i) {
    input_spacial_shape_with_pads[i] = input_shape[i + 2] + conv_attrs_.pads[i] + conv_attrs_.pads[i + input_spacial_shape_with_pads.NumDimensions()];
  }
  std::vector<int64_t> output_dims(dilated_kernel_spatial_shape.NumDimensions() + 2);
  output_dims[0] = batch_size;
  output_dims[channel_index] = output_channels;
  size_t j = is_channels_last ? 1 : 2;
  for (size_t i = 0; i < dilated_kernel_spatial_shape.NumDimensions(); ++i, ++j) {
    output_dims[j] = (input_spacial_shape_with_pads[i] - dilated_kernel_spatial_shape[i]) / conv_attrs_.strides[i] + 1;
  }
  TensorShape output_shape(output_dims);

  return TensorShape(output_shape);
}

template <bool is_channels_last>
Status Conv<is_channels_last>::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto* kernel = context.Input<Tensor>(1);
  const auto& input_shape = input->Shape();
  const auto& kernel_shape = kernel->Shape();
  auto channel_index = is_channels_last ? input_shape.NumDimensions() - 1 : 1;
  if (input_shape.NumDimensions() > 4 || kernel_shape.NumDimensions() > 4) {
    // Conv3D or higher dimensions
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only Conv2d or Conv1d are supported.");
  } else if (input_shape.NumDimensions() == 4 || kernel_shape.NumDimensions() == 4) {
    // Conv2D
  } else if (input_shape.NumDimensions() == 3 || kernel_shape.NumDimensions() == 3) {
    // Conv1D
    auto input_shape_4d = is_channels_last ? TensorShape({input_shape[0], 1, input_shape[1], input_shape[2]}) : TensorShape({input_shape[0], input_shape[1], 1, input_shape[2]});
    auto kernel_shape_4d = TensorShape({kernel_shape[0], kernel_shape[1], 1, kernel_shape[2]});
    const_cast<Tensor*>(input)->Reshape(input_shape_4d);
    const_cast<Tensor*>(kernel)->Reshape(kernel_shape_4d);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input and kernel tensors must have at least 3 dimensions");
  }
  // Transpose weights
  std::vector<size_t> perm = {2, 3, 1, 0};
  TensorShape transposed_kernel_shape(kernel_shape);
  for (size_t i = 0; i < kernel_shape.NumDimensions(); ++i) {
    transposed_kernel_shape[i] = kernel_shape[perm[i]];
  }
  auto transposed_kernel = context.CreateGPUTensor(kernel->DataType(), transposed_kernel_shape);
  ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context, perm, *input, transposed_kernel));
  // Compute matmul
  auto output_shape = ComputeOutputShape(input->Shape(), kernel->Shape());
  bool has_bias = context.InputCount() > 2;
  auto input_channels = input_shape[is_channels_last ? 3 : 1];
  auto kernel_height = kernel_shape[2];
  auto kernel_width = kernel_shape[3];
  auto output_channels = kernel_shape[channel_index];
  auto output_height = output_shape[is_channels_last ? 1 : 2];
  auto output_width = output_shape[is_channels_last ? 2 : 3];
  auto output = context.Output(0, output_shape);
  auto dim_a_outer = static_cast<uint32_t>(is_channels_last ? output_height * output_width : output_channels);
  auto dim_b_outer = static_cast<uint32_t>(is_channels_last ? output_channels : output_height * output_width);
  auto dim_inner = static_cast<uint32_t>(is_channels_last ? input_channels : kernel_height * kernel_width);
  std::vector<const Tensor*> inputs(context.InputCount());
  inputs[0] = input;
  inputs[1] = &transposed_kernel;
  if (has_bias) {
    inputs[2] = context.Input<Tensor>(2);
  }
  Conv2dMMProgram conv2d_mm_program = CreateConv2dMMProgram(inputs, output, conv_attrs_, dim_a_outer, dim_b_outer, dim_inner, is_channels_last);
  return context.RunProgram(conv2d_mm_program);
}

#define WEBGPU_ONNX_CONV_OPERATOR_KERNEL(VERSION_FROM)                                \
  ONNX_OPERATOR_KERNEL_EX(                                                            \
      Conv,                                                                           \
      kMSInternalNHWCDomain,                                                          \
      VERSION_FROM,                                                                   \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<true>);                                                                    \
                                                                                      \
  ONNX_OPERATOR_KERNEL_EX(                                                            \
      Conv,                                                                           \
      kOnnxDomain,                                                                    \
      VERSION_FROM,                                                                   \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<false>);

#define WEBGPU_ONNX_CONV_OPERATOR_VERSIONED_KERNEL(VERSION_FROM, VERSION_TO)          \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                  \
      Conv,                                                                           \
      kOnnxDomain,                                                                    \
      VERSION_FROM, VERSION_TO,                                                       \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<false>);                                                                   \
                                                                                      \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                  \
      Conv,                                                                           \
      kMSInternalNHWCDomain,                                                          \
      VERSION_FROM, VERSION_TO,                                                       \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<true>);

WEBGPU_ONNX_CONV_OPERATOR_VERSIONED_KERNEL(1, 10)
WEBGPU_ONNX_CONV_OPERATOR_VERSIONED_KERNEL(11, 21)
WEBGPU_ONNX_CONV_OPERATOR_KERNEL(22)

}  // namespace webgpu
}  // namespace onnxruntime
