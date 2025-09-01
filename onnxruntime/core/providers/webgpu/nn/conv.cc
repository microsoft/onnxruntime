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

namespace {

inline uint32_t ceil_div(int64_t numerator, int32_t denominator) {
  return static_cast<uint32_t>((numerator + denominator - 1) / denominator);
}

}  // namespace

namespace onnxruntime {
namespace webgpu {

Status TransposeKernel(ComputeContext& context, const Tensor* kernel, const TensorShape& kernel_shape, Tensor* transposed_kernel, const InlinedVector<size_t>& perm) {
  // Transpose weights
  auto rank = kernel_shape.NumDimensions();
  TensorShapeVector transposed_kernel_shape_vector(rank);
  for (size_t i = 0; i < rank; ++i) {
    transposed_kernel_shape_vector[i] = kernel_shape[perm[i]];
  }
  uint32_t output_size = onnxruntime::narrow<uint32_t>(kernel_shape.Size());

  uint32_t dispatch_x = ceil_div(output_size, 64);
  uint32_t dispatch_y = 1;
  uint32_t dispatch_z = 1;

  // This temporary workaround addresses a significant performance bottleneck
  // (10x slower) for the shape (3, 3, 2560, 1280) due to an issue with Intel's
  // GPU drivers. We manually normalize the dispatch group size to restore
  // performance.
  //
  // TODO: Revert this change once the driver issue is fixed.
  if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
    ORT_ENFORCE(rank == static_cast<size_t>(4), "Input tensor must have rank 4.");
    dispatch_x = ceil_div(transposed_kernel_shape_vector[0] * transposed_kernel_shape_vector[1], 2);
    dispatch_y = ceil_div(transposed_kernel_shape_vector[2], 4);
    dispatch_z = ceil_div(transposed_kernel_shape_vector[3], 8);
  }

  TensorShape transposed_kernel_shape(transposed_kernel_shape_vector);
  *transposed_kernel = context.CreateGPUTensor(kernel->DataType(), transposed_kernel_shape);
  bool use_shared = false;
  TransposeProgram program{perm, use_shared};
  program
      .CacheHint(absl::StrJoin(perm, "-"))
      .AddInput({kernel, ProgramTensorMetadataDependency::TypeAndRank, kernel_shape, 1})
      .AddOutput({transposed_kernel, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariable({output_size})
      .SetWorkgroupSize(64)
      .SetDispatchGroupSize(dispatch_x, dispatch_y, dispatch_z);
  return context.RunProgram(program);
}

template <bool is_channels_last, bool is_fused>
Status Conv<is_channels_last, is_fused>::ComputeInternal(ComputeContext& context) const {
  bool has_bias = context.InputCount() > 2;
  const auto* input = context.Input<Tensor>(0);
  const auto* kernel = context.Input<Tensor>(1);
  const auto* bias = has_bias ? context.Input<Tensor>(2) : nullptr;
  TensorShape input_shape = input->Shape();
  TensorShape kernel_shape = kernel->Shape();
  ConvAttributes::ConvPadVector local_pads(conv_attrs_.pads.begin(), conv_attrs_.pads.end());
  TensorShapeVector local_dilations(conv_attrs_.dilations.begin(), conv_attrs_.dilations.end());
  TensorShapeVector local_strides(conv_attrs_.strides.begin(), conv_attrs_.strides.end());
  TensorShapeVector kernel_spacial_shape_vector;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(kernel_shape, kernel_spacial_shape_vector, false));
  if (local_pads.empty()) {
    local_pads.resize(kernel_spacial_shape_vector.size() * 2, 0);
  }
  if (local_dilations.empty()) {
    local_dilations.resize(kernel_spacial_shape_vector.size(), 1);
  }
  if (local_strides.empty()) {
    local_strides.resize(kernel_spacial_shape_vector.size(), 1);
  }
  TensorShapeVector input_shape_vector = input_shape.AsShapeVector();
  auto batch = input_shape[0];
  TensorShapeVector output_shape_vector = {batch};
  TensorShape input_spacial_shape = is_channels_last ? TensorShape(TensorShapeVector(std::next(input_shape_vector.begin()), std::prev(input_shape_vector.end()))) : input_shape.Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_spacial_shape, kernel_spacial_shape_vector, local_strides, local_dilations, local_pads, output_shape_vector));
  auto output_channels = kernel_shape[0];
  if (is_channels_last) {
    output_shape_vector.push_back(output_channels);
  } else {
    output_shape_vector.insert(output_shape_vector.begin() + 1, output_channels);
  }
  auto output_shape = TensorShape(output_shape_vector);
  auto* output = context.Output(0, output_shape);
  std::vector<uint32_t> strides;
  std::vector<uint32_t> pads;
  std::vector<uint32_t> dilations;
  auto transform_dim = [](int64_t dim) { return static_cast<int32_t>(dim); };
  std::transform(local_pads.begin(), local_pads.end(), std::back_inserter(pads), transform_dim);
  std::transform(local_strides.begin(), local_strides.end(), std::back_inserter(strides), transform_dim);
  std::transform(local_dilations.begin(), local_dilations.end(), std::back_inserter(dilations), transform_dim);
  auto rank = input_shape.NumDimensions();
  const InlinedVector<size_t> perm = {2, 3, 1, 0};
  if (rank > 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only Conv1d and Conv2d are supported.");
  } else if (rank == 4) {
    // Conv2D
  } else if (rank == 3) {
    // Conv1D
    TensorShapeVector kernel_shape_vector = kernel_shape.AsShapeVector();
    input_shape_vector.insert(input_shape_vector.begin() + (is_channels_last ? 1 : 2), 1, 1);
    output_shape_vector.insert(output_shape_vector.begin() + (is_channels_last ? 1 : 2), 1, 1);
    kernel_shape_vector.insert(kernel_shape_vector.begin() + 2, 1);
    input_shape = TensorShape(input_shape_vector);
    kernel_shape = TensorShape(kernel_shape_vector);
    pads.insert(pads.begin(), 0);
    pads.insert(pads.begin() + 2, 0);
    strides.insert(strides.begin(), 1);
    dilations.insert(dilations.begin(), 1);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input and kernel tensors must have at least 3 dimensions");
  }
  std::vector<const Tensor*> inputs(has_bias ? 3 : 2);
  inputs[0] = input;
  inputs[1] = kernel;
  if (has_bias) {
    inputs[2] = bias;
  }
  std::vector<TensorShape> modified_input_output_shapes = {input_shape, kernel_shape};
  if (has_bias) {
    modified_input_output_shapes.push_back(bias->Shape());
  }
  modified_input_output_shapes.push_back(TensorShape(output_shape_vector));
  uint32_t auto_pad_adjust = conv_attrs_.auto_pad == AutoPadType::SAME_LOWER ? 1 : 0;
  auto pad0 = conv_attrs_.auto_pad == AutoPadType::NOTSET ? pads[0] : (pads[0] + pads[2] + auto_pad_adjust) / 2;
  auto pad1 = conv_attrs_.auto_pad == AutoPadType::NOTSET ? pads[1] : (pads[1] + pads[3] + auto_pad_adjust) / 2;
  std::vector<uint32_t> updated_pads{pad0, pad1};
  if (conv_attrs_.group > 1) {
    Tensor transposed_kernel;
    if (is_channels_last) {
      ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
      inputs[1] = &transposed_kernel;
      modified_input_output_shapes[1] = transposed_kernel.Shape();
    }
    auto output_channels_per_group = output_channels / conv_attrs_.group;
    auto components = static_cast<int>(is_channels_last && output_channels_per_group >= 4 ? GetMaxComponents(output_channels) : 1);
    auto output_size = output_shape.Size() / components;
    GroupedConvProgram program(activation_, has_bias, is_channels_last);
    auto reduced_kernel_shape = ReduceShapeByComponents(modified_input_output_shapes[1], components);
    auto reduced_output_shape = ReduceShapeByComponents(modified_input_output_shapes[has_bias ? 3 : 2], components);
    program.CacheHint(activation_.ToString(), std::to_string(components), std::to_string(is_channels_last))
        .AddInput({inputs[0], ProgramTensorMetadataDependency::TypeAndRank, modified_input_output_shapes[0], 1})
        .AddInput({inputs[1], ProgramTensorMetadataDependency::TypeAndRank, reduced_kernel_shape, components})
        .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, reduced_output_shape, components})
        .AddUniformVariables({{static_cast<uint32_t>(output_size)}, {dilations}, {strides}, {updated_pads}, {static_cast<uint32_t>(output_channels_per_group)}, {static_cast<uint32_t>(components)}})
        .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
    if (has_bias) {
      auto reduced_bias_shape = ReduceShapeByComponents(modified_input_output_shapes[2], components);
      program.AddInput({inputs[2], ProgramTensorMetadataDependency::TypeAndRank, reduced_bias_shape, components});
    }
    return context.RunProgram(program);
  }
  const auto input_height = input_shape[is_channels_last ? 1 : 2];
  const auto input_width = input_shape[is_channels_last ? 2 : 3];
  const auto input_channels = input_shape[is_channels_last ? 3 : 1];
  const auto kernel_height = kernel_shape[2];
  const auto kernel_width = kernel_shape[3];
  const auto output_height = output_shape_vector[is_channels_last ? 1 : 2];
  const auto output_width = output_shape_vector[is_channels_last ? 2 : 3];

  const auto same_size = is_channels_last && input_height == kernel_height && input_width == kernel_width && pads[0] == 0 && pads[1] == 0;
  if (same_size || (kernel_height == 1 && kernel_width == 1 && pads[0] == 0 && pads[1] == 0 && strides[0] == 1 && strides[1] == 1)) {
    Tensor transposed_kernel;
    TensorShape input_reshape;
    TensorShape kernel_reshape;
    TensorShape matmul_output_shape;
    std::vector<const Tensor*> matmul_inputs;
    std::vector<TensorShape> matmul_input_reshapes;
    if (is_channels_last) {
      // Transpose weights

      ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
      inputs[1] = &transposed_kernel;
      if (same_size) {
        const auto shared_dim = input_height * input_width * input_channels;
        input_reshape = TensorShape({1, batch, shared_dim});
        kernel_reshape = TensorShape({1, shared_dim, output_channels});
        matmul_output_shape = TensorShape({1, batch, output_channels});
      } else {
        input_reshape = TensorShape({batch, input_height * input_width, input_channels});
        kernel_reshape = TensorShape({1, input_channels, output_channels});
        matmul_output_shape = TensorShape({batch, output_height * output_width, output_channels});
      }
      matmul_inputs.push_back(input);
      matmul_inputs.push_back(&transposed_kernel);
      matmul_input_reshapes.push_back(input_reshape);
      matmul_input_reshapes.push_back(kernel_reshape);
    } else {
      input_reshape = TensorShape({batch, input_channels, input_height * input_width});
      kernel_reshape = TensorShape({1, output_channels, input_channels});
      matmul_output_shape = TensorShape({batch, output_channels, output_height * output_width});
      matmul_inputs.push_back(kernel);
      matmul_inputs.push_back(input);
      matmul_input_reshapes.push_back(kernel_reshape);
      matmul_input_reshapes.push_back(input_reshape);
    }
    if (has_bias) {
      matmul_inputs.push_back(bias);
    }
    auto N = matmul_output_shape[2];
    auto matmul_first_input_numdims = matmul_input_reshapes[0].NumDimensions();
    auto K = matmul_input_reshapes[0].GetDims()[matmul_first_input_numdims - 1];
    if (N < 8 && K < 8) {
      const auto components = GetMaxComponents(N);
      const auto a_components = GetMaxComponents(K);
      const auto output_number = GetMaxComponents(output_shape[1]);
      uint32_t output_size = static_cast<uint32_t>(output_shape.Size() / components / output_number);
      const size_t output_rank = matmul_output_shape.NumDimensions();
      TensorShape outer_dims = output_rank > 2 ? matmul_output_shape.Slice(0, output_rank - 2) : TensorShape({});
      MatMulNaiveProgram program(activation_, output_rank, output_number, has_bias, is_channels_last);
      program
          .CacheHint(std::to_string(components), std::to_string(a_components), std::to_string(output_number))
          .AddInputs({{matmul_inputs[0], ProgramTensorMetadataDependency::TypeAndRank, ReduceShapeByComponents(matmul_input_reshapes[0], a_components), int(a_components)},
                      {matmul_inputs[1], ProgramTensorMetadataDependency::TypeAndRank, ReduceShapeByComponents(matmul_input_reshapes[1], components), int(components)}});
      if (has_bias) {
        program.AddInput({bias, ProgramTensorMetadataDependency::Rank, ReduceShapeByComponents(bias->Shape(), components), components});
      }
      program
          .AddOutputs({{output, ProgramTensorMetadataDependency::None, ReduceShapeByComponents(matmul_output_shape, components), int(components)}})
          .SetDispatchGroupSize(static_cast<uint32_t>((output_size + 63) / 64))
          .AddIndices(outer_dims)
          .AddUniformVariables({{output_size}, {static_cast<uint32_t>(matmul_output_shape[1])}, {static_cast<uint32_t>(matmul_output_shape[2])}, {static_cast<uint32_t>(K)}});
      return context.RunProgram(program);
    } else {
      MatMulProgram program = CreateMatMulProgram(activation_, matmul_inputs, output, is_channels_last, matmul_input_reshapes[0], matmul_input_reshapes[1]);
      return context.RunProgram(program);
    }
  }
  // Transpose weights
  Tensor transposed_kernel;
  ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
  auto dim_a_outer = static_cast<uint32_t>(is_channels_last ? output_height * output_width : output_channels);
  auto dim_b_outer = static_cast<uint32_t>(is_channels_last ? output_channels : output_height * output_width);
  auto dim_inner = static_cast<uint32_t>(kernel_height * kernel_width * input_channels);
  inputs[1] = &transposed_kernel;
  TensorShape transposed_kernel_shape = transposed_kernel.Shape();
  modified_input_output_shapes[1] = transposed_kernel.Shape();
  Conv2dMMProgram conv2d_mm_program = CreateConv2dMMProgram(activation_, inputs, pads, strides, dilations, output, dim_a_outer, dim_b_outer, dim_inner, is_channels_last, modified_input_output_shapes);
  return context.RunProgram(conv2d_mm_program);
}

// Explicit template instantiation for FusedConv
template class Conv<false, false>;
template class Conv<false, true>;
template class Conv<true, false>;
template class Conv<true, true>;

#define WEBGPU_ONNX_CONV_OPERATOR_KERNEL(VERSION_FROM)                                \
  ONNX_OPERATOR_KERNEL_EX(                                                            \
      Conv,                                                                           \
      kMSInternalNHWCDomain,                                                          \
      VERSION_FROM,                                                                   \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<true, true>);                                                              \
                                                                                      \
  ONNX_OPERATOR_KERNEL_EX(                                                            \
      Conv,                                                                           \
      kOnnxDomain,                                                                    \
      VERSION_FROM,                                                                   \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<false, false>);

#define WEBGPU_ONNX_CONV_OPERATOR_VERSIONED_KERNEL(VERSION_FROM, VERSION_TO)          \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                  \
      Conv,                                                                           \
      kOnnxDomain,                                                                    \
      VERSION_FROM, VERSION_TO,                                                       \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<false, false>);                                                            \
                                                                                      \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                  \
      Conv,                                                                           \
      kMSInternalNHWCDomain,                                                          \
      VERSION_FROM, VERSION_TO,                                                       \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<true, true>);

WEBGPU_ONNX_CONV_OPERATOR_VERSIONED_KERNEL(1, 10)
WEBGPU_ONNX_CONV_OPERATOR_VERSIONED_KERNEL(11, 21)
WEBGPU_ONNX_CONV_OPERATOR_KERNEL(22)

}  // namespace webgpu
}  // namespace onnxruntime
