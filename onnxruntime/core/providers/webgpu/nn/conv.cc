// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/nn/conv2d_mm_webgpu.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/webgpu/nn/grouped_conv.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/matmul.h"
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
  bool has_bias = context.InputCount() > 2;
  const auto* input = context.Input<Tensor>(0);
  const auto* kernel = context.Input<Tensor>(1);
  TensorShape input_shape = input->Shape();
  TensorShape kernel_shape = kernel->Shape();
  auto channel_index = is_channels_last ? input_shape.NumDimensions() - 1 : 1;
  if (input_shape.NumDimensions() > 4 || kernel_shape.NumDimensions() > 4) {
    // Conv3D or higher dimensions
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only Conv2d or Conv1d are supported.");
  } else if (input_shape.NumDimensions() == 4 || kernel_shape.NumDimensions() == 4) {
    // Conv2D
  } else if (input_shape.NumDimensions() == 3 || kernel_shape.NumDimensions() == 3) {
    // Conv1D
    input_shape = is_channels_last ? TensorShape({input_shape[0], 1, input_shape[1], input_shape[2]}) : TensorShape({input_shape[0], input_shape[1], 1, input_shape[2]});
    kernel_shape = TensorShape({kernel_shape[0], kernel_shape[1], 1, kernel_shape[2]});
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input and kernel tensors must have at least 3 dimensions");
  }
  auto output_shape = ComputeOutputShape(input_shape, kernel_shape);
  auto* output = context.Output(0, output_shape);

  if (conv_attrs_.group > 1) {
    std::vector<uint32_t> pads = {static_cast<uint32_t>(conv_attrs_.pads[0]), static_cast<uint32_t>(conv_attrs_.pads[1])};
    std::vector<uint32_t> strides(static_cast<uint32_t>(conv_attrs_.strides[0]), static_cast<uint32_t>(conv_attrs_.strides[1]));
    std::vector<uint32_t> dilations(conv_attrs_.dilations.size());
    auto transform_dim = [](int64_t dim) { return static_cast<int32_t>(dim); };
    // std::transform(conv_attrs_.strides.begin(), conv_attrs_.strides.end(), std::back_inserter(strides), transform_dim);
    std::transform(conv_attrs_.dilations.begin(), conv_attrs_.dilations.end(), std::back_inserter(dilations), transform_dim);
    std::vector<const Tensor*> inputs(context.InputCount());
    inputs[0] = input;
    if (is_channels_last) {
      // Transpose weights
      std::vector<size_t> perm = {2, 3, 1, 0};
      TensorShape transposed_kernel_shape(kernel_shape);
      for (size_t i = 0; i < kernel_shape.NumDimensions(); ++i) {
        transposed_kernel_shape[i] = kernel_shape[perm[i]];
      }
      auto transposed_kernel = context.CreateGPUTensor(kernel->DataType(), transposed_kernel_shape);
      ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context, perm, *kernel, transposed_kernel));
      inputs[1] = &transposed_kernel;
    } else {
      inputs[1] = kernel;
    }
    auto output_channels = output_shape[channel_index];
    auto output_channels_per_group = output_channels / conv_attrs_.group;
    auto components = static_cast<int>(is_channels_last && output_channels_per_group >= 4 ? GetMaxComponents(output_channels) : 1);
    auto output_size = output_shape.Size() / components;
    GroupedConvProgram program(conv_attrs_, has_bias, is_channels_last);
    program.AddInputs({{inputs[0], ProgramTensorMetadataDependency::TypeAndRank, input_shape, components}, {inputs[1], ProgramTensorMetadataDependency::TypeAndRank, kernel_shape, components}})
        .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, components})
        .AddUniformVariables({{static_cast<uint32_t>(output_size)}, {dilations}, {strides}, {pads}, {static_cast<uint32_t>(output_channels_per_group)}, {static_cast<uint32_t>(components)}})
        .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
    return context.RunProgram(program);
  }
  const auto input_height = input_shape[is_channels_last ? 1 : 2];
  const auto input_width = input_shape[is_channels_last ? 2 : 3];
  const auto kernel_height = kernel_shape[2];
  const auto kernel_width = kernel_shape[3];
  const auto output_height = output_shape[is_channels_last ? 1 : 2];
  const auto output_width = output_shape[is_channels_last ? 2 : 3];
  const auto input_channels = input_shape[is_channels_last ? 3 : 1];
  const auto output_channels = kernel_shape[is_channels_last ? 3 : 1];

  const auto same_size = is_channels_last && input_height == output_height && input_width == output_width && conv_attrs_.pads[0] == 0 && conv_attrs_.pads[1] == 0;
  if (same_size || (kernel_height == 1 && kernel_width == 1 && conv_attrs_.pads[0] == 0 && conv_attrs_.pads[1] == 0 && conv_attrs_.strides[0] == 1 && conv_attrs_.strides[1] == 1)) {

    const auto batch = output_shape[0];
    TensorShape input_reshape(input_shape);
    TensorShape kernel_reshape(kernel_shape);
    TensorShape output_reshape(output_shape);
    if (is_channels_last) {
      // Transpose weights
      std::vector<size_t> perm = {2, 3, 1, 0};
      TensorShape transposed_kernel_shape(kernel_shape);
      for (size_t i = 0; i < kernel_shape.NumDimensions(); i++) {
        transposed_kernel_shape[i] = kernel_shape[perm[i]];
      }
      auto transposed_kernel = context.CreateGPUTensor(kernel->DataType(), transposed_kernel_shape);
      ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context,
        perm, *kernel, transposed_kernel));
      if (same_size) {
        const auto shared_dim = input_height * input_width * input_channels;
        input_reshape = TensorShape({1, batch, shared_dim});
        kernel_reshape = TensorShape({1, shared_dim, output_channels});
        output_reshape = TensorShape({1, batch, output_channels});
      } else {
        input_reshape = TensorShape({batch, input_height * input_width * input_channels});
        kernel_reshape = TensorShape({1 * input_channels, output_channels});
        output_reshape = TensorShape({batch, output_height * output_width, output_channels});
      }
    } else {
      input_reshape = TensorShape({batch, input_channels, input_height * input_width});
      kernel_reshape = TensorShape({1, output_channels, input_channels});
      output_reshape = TensorShape({batch, output_channels, output_height * output_width});
    }
    auto N = output_reshape[2];
    auto K = input_reshape[input_reshape.NumDimensions() - 1];
    if (N < 8 && K < 8) {
      const auto components = GetMaxComponents(N);
      const auto a_components = GetMaxComponents(K);
      const auto output_number = GetMaxComponents(output_shape[1]);
      uint32_t output_size = static_cast<uint32_t>(output_shape.Size() / components / output_number);
      const size_t output_rank = output_shape.NumDimensions();
      TensorShape outer_dims = output_rank > 2 ? output_shape.Slice(0, output_rank - 2) : TensorShape({});
      const int64_t batch_size = outer_dims.Size();
      TensorShape output_shape_shader({batch_size, output_shape[1], output_shape[2]});
      MatMulNaiveProgram program(output_size, output_number, has_bias);
      program
          .CacheHint(std::to_string(components), std::to_string(a_components), std::to_string(output_number))
          .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, input_reshape, int(a_components)},
                      {kernel, ProgramTensorMetadataDependency::TypeAndRank, kernel_reshape, int(components)}});
      if (has_bias) {
        const auto* bias = context.Input(2);
        program.AddInput({bias, ProgramTensorMetadataDependency::Rank, 1});
      }
      program
          .AddOutputs({{output, ProgramTensorMetadataDependency::None, output_shape_shader, int(components)}})
          .SetDispatchGroupSize(static_cast<uint32_t>(ceil(output_size / 64)))
          .AddIndices(outer_dims)
          .AddUniformVariables({{output_size}, {static_cast<uint32_t>(output_reshape[1])}, {static_cast<uint32_t>(output_reshape[2])}, {static_cast<uint32_t>(K)}});
      return context.RunProgram(program);
    } else {
      MatMulProgram program = CreateMatMulProgram(context);
      return context.RunProgram(program);
    }
  }
  // Transpose weights
  std::vector<size_t> perm = {2, 3, 1, 0};
  TensorShape transposed_kernel_shape(kernel_shape);
  for (size_t i = 0; i < kernel_shape.NumDimensions(); ++i) {
    transposed_kernel_shape[i] = kernel_shape[perm[i]];
  }
  auto transposed_kernel = context.CreateGPUTensor(kernel->DataType(), transposed_kernel_shape);
  ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context, perm, *kernel, transposed_kernel));
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
