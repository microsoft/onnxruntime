// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/nn/conv2d_mm.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/webgpu/nn/grouped_conv.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/matmul.h"

namespace onnxruntime {
namespace webgpu {

// Get the preferred kernel format for Conv operator
// Returns format descriptor string (e.g., "hwio", "ABcd16a4b"), or empty string if no transformation needed
Status ConvGetPreferredKernelFormat(const Node& node, int input_index, std::string& format_descriptor) {
  // Conv operator - kernel is input index 1
  // Conv signature: [X, W, B?] where X=activations, W=kernel, B=optional bias
  if (input_index != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No format transformation needed");
  }

  // Get kernel shape and dtype from NodeArg
  const auto& input_defs = node.InputDefs();
  if (input_index >= static_cast<int>(input_defs.size())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid input index");
  }

  const auto* kernel_arg = input_defs[input_index];
  if (!kernel_arg || !kernel_arg->Exists()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Kernel input does not exist");
  }

  // Get shape
  const auto* shape_proto = kernel_arg->Shape();
  if (!shape_proto) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Kernel shape is unknown");
  }

  TensorShapeVector dims;
  for (const auto& dim : shape_proto->dim()) {
    if (dim.has_dim_value()) {
      dims.push_back(dim.dim_value());
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Kernel has dynamic shape");
    }
  }

  // Conv kernels must be 4D: [O, I, H, W] (or 3D for Conv1D which gets expanded to 4D)
  if (dims.size() != 4 && dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No format transformation needed");
  }

  // Get data type
  const auto* type_proto = kernel_arg->TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Kernel type is unknown");
  }

  auto elem_type_enum = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type_proto->tensor_type().elem_type());
  const auto* kernel_dtype = DataTypeImpl::TensorTypeFromONNXEnum(elem_type_enum)->GetElementType();

  // Only support float32 and float16
  const bool is_float32 = (kernel_dtype == DataTypeImpl::GetType<float>());
  const bool is_float16 = (kernel_dtype == DataTypeImpl::GetType<MLFloat16>());

  if (!is_float32 && !is_float16) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No format transformation needed");
  }

  // Check if this is channels_last (NHWC) layout by checking the domain
  const bool is_channels_last = (node.Domain() == kMSInternalNHWCDomain);

  // Get group attribute to match Conv execution logic
  int64_t group = 1;
  const auto& attributes = node.GetAttributes();
  auto group_attr = attributes.find("group");
  if (group_attr != attributes.end()) {
    group = group_attr->second.i();
  }

  // Get kernel spatial dimensions for MatMul optimization path check
  const int64_t kernel_height = dims.size() == 4 ? dims[2] : 1;  // Conv1D has no H dim
  const int64_t kernel_width = dims.size() == 4 ? dims[3] : dims[2];

  // Get input shape to check same_size condition
  const auto* input_arg = input_defs[0];
  int64_t input_height = -1;
  int64_t input_width = -1;
  if (input_arg && input_arg->Exists()) {
    const auto* input_shape_proto = input_arg->Shape();
    if (input_shape_proto && input_shape_proto->dim_size() >= 3) {
      if (is_channels_last) {
        // For channels_last: [N, H, W, C] or [N, W, C] for Conv1D
        if (input_shape_proto->dim_size() == 4) {
          if (input_shape_proto->dim(1).has_dim_value()) {
            input_height = input_shape_proto->dim(1).dim_value();
          }
          if (input_shape_proto->dim(2).has_dim_value()) {
            input_width = input_shape_proto->dim(2).dim_value();
          }
        } else if (input_shape_proto->dim_size() == 3) {
          // Conv1D
          input_height = 1;
          if (input_shape_proto->dim(1).has_dim_value()) {
            input_width = input_shape_proto->dim(1).dim_value();
          }
        }
      } else {
        // For channels_first: [N, C, H, W] or [N, C, W] for Conv1D
        if (input_shape_proto->dim_size() == 4) {
          if (input_shape_proto->dim(2).has_dim_value()) {
            input_height = input_shape_proto->dim(2).dim_value();
          }
          if (input_shape_proto->dim(3).has_dim_value()) {
            input_width = input_shape_proto->dim(3).dim_value();
          }
        } else if (input_shape_proto->dim_size() == 3) {
          // Conv1D
          input_height = 1;
          if (input_shape_proto->dim(2).has_dim_value()) {
            input_width = input_shape_proto->dim(2).dim_value();
          }
        }
      }
    }
  }

  // Get pads and strides attributes
  std::vector<int64_t> pads;
  auto pads_attr = attributes.find("pads");
  if (pads_attr != attributes.end()) {
    pads.assign(pads_attr->second.ints().begin(), pads_attr->second.ints().end());
  }

  std::vector<int64_t> strides_vec;
  auto strides_attr = attributes.find("strides");
  if (strides_attr != attributes.end()) {
    strides_vec.assign(strides_attr->second.ints().begin(), strides_attr->second.ints().end());
  }

  // Default pads and strides if not specified
  if (pads.empty()) {
    pads.resize(dims.size() == 4 ? 4 : 2, 0);  // 4 for 2D conv, 2 for 1D conv
  }
  if (strides_vec.empty()) {
    strides_vec.resize(dims.size() == 4 ? 2 : 1, 1);
  }

  // Analyze execution paths to determine if kernel needs pre-transformation:

  // Path 1: Grouped convolution (group > 1)
  //   - Only transposes when is_channels_last
  //   - channels_first: no transpose
  if (group > 1) {
    if (is_channels_last) {
      format_descriptor = "hwio";
      return Status::OK();
    } else {
      // channels_first grouped conv doesn't transpose
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No format transformation needed");
    }
  }

  // Path 2: MatMul optimization (same_size or 1x1 conv conditions)
  //   - channels_last: transposes
  //   - channels_first: does NOT transpose

  const bool same_size = (input_height > 0 && input_width > 0 && input_height == kernel_height &&
                          input_width == kernel_width && pads[0] == 0 && pads[1] == 0);

  const bool is_1x1_conv =
      (kernel_height == 1 && kernel_width == 1 && pads[0] == 0 && pads[1] == 0 && strides_vec.size() > 0 &&
       strides_vec[0] == 1 && (strides_vec.size() == 1 || strides_vec[1] == 1));

  if (same_size || is_1x1_conv) {
    if (is_channels_last) {
      // MatMul optimization transposes for channels_last
      format_descriptor = "hwio";
      return Status::OK();
    } else {
      // MatMul optimization does NOT transpose for channels_first
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No format transformation needed");
    }
  }

  // Path 3: General convolution (fallback path)
  //   - ALWAYS transposes regardless of is_channels_last
  //   - Both channels_last AND channels_first transpose here
  format_descriptor = "hwio";
  return Status::OK();

  // TODO: Add shape-based heuristics for blocked format in the future:
  // const int64_t O = dims[0];  // output channels
  // const int64_t I = dims[1];  // input channels
  // if (O >= 16 && I >= 4 && minimal_padding_overhead) {
  //   format_descriptor = "ABcd16a4b";  // 16x4 blocks on O and I dims
  //   return Status::OK();
  // }
}

Status TransposeKernel(ComputeContext& context, const Tensor* kernel, const TensorShape& kernel_shape, Tensor* transposed_kernel, const InlinedVector<size_t>& perm) {
  // Transpose weights
  auto rank = kernel_shape.NumDimensions();
  TensorShapeVector transposed_kernel_shape_vector(rank);
  for (size_t i = 0; i < rank; ++i) {
    transposed_kernel_shape_vector[i] = kernel_shape[perm[i]];
  }
  TensorShape transposed_kernel_shape(transposed_kernel_shape_vector);
  *transposed_kernel = context.CreateGPUTensor(kernel->DataType(), transposed_kernel_shape);
  const Tensor reshaped_kernel(kernel->DataType(), kernel_shape, const_cast<void*>(kernel->DataRaw()), kernel->Location());
  return Transpose::DoTranspose(context, perm, reshaped_kernel, *transposed_kernel);
}

template <bool is_channels_last, bool is_fused>
Status Conv<is_channels_last, is_fused>::ComputeInternal(ComputeContext& context) const {
  bool has_bias = context.InputCount() > 2;
  const auto* input = context.Input<Tensor>(0);
  const auto* kernel = context.Input<Tensor>(1);
  const auto* bias = has_bias ? context.Input<Tensor>(2) : nullptr;
  TensorShape input_shape = input->Shape();
  TensorShape kernel_shape = kernel->Shape();

  // Check if kernel is pre-transformed to hwio format
  const bool is_kernel_hwio = (kernel->GetFormatDescriptor() == "hwio");

  // If kernel is pre-transformed to hwio format, we need to get the logical oihw shape
  // for computing kernel spatial dimensions and output channels
  // hwio format: [H, W, I, O] -> oihw format: [O, I, H, W]
  if (is_kernel_hwio) {
    // Convert hwio shape back to oihw for dimension calculations
    const auto& hwio_shape = kernel_shape.GetDims();
    if (hwio_shape.size() == 4) {
      // hwio -> oihw: permutation is {3, 2, 0, 1}
      kernel_shape = TensorShape({hwio_shape[3], hwio_shape[2], hwio_shape[0], hwio_shape[1]});
    }
  }

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
      // Check if kernel is already in hwio format (pre-transformed)
      if (is_kernel_hwio) {
        // Kernel is already in hwio format, use it directly
        inputs[1] = kernel;
        modified_input_output_shapes[1] = kernel->Shape();  // Use actual tensor shape (hwio)
      } else {
        // Need to transpose kernel from oihw to hwio at runtime
        ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
        inputs[1] = &transposed_kernel;
        modified_input_output_shapes[1] = transposed_kernel.Shape();
      }
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
      // Check if kernel is already in hwio format (pre-transformed)
      const Tensor* kernel_to_use = kernel;
      if (is_kernel_hwio) {
        // Kernel is already in hwio format, use it directly
        kernel_to_use = kernel;
      } else {
        // Need to transpose kernel from oihw to hwio at runtime
        ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
        kernel_to_use = &transposed_kernel;
      }
      inputs[1] = kernel_to_use;
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
      matmul_inputs.push_back(kernel_to_use);
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
  // General Conv path - transpose weights if needed
  Tensor transposed_kernel;
  const Tensor* kernel_to_use = kernel;
  TensorShape kernel_to_use_shape;

  // Check if kernel is already in hwio format (pre-transformed)
  if (is_kernel_hwio) {
    // Kernel is already in hwio format, use it directly
    kernel_to_use = kernel;
    kernel_to_use_shape = kernel->Shape();  // Use actual tensor shape (hwio)
  } else {
    // Need to transpose kernel from oihw to hwio at runtime
    ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
    kernel_to_use = &transposed_kernel;
    kernel_to_use_shape = transposed_kernel.Shape();
  }

  auto dim_a_outer = static_cast<uint32_t>(is_channels_last ? output_height * output_width : output_channels);
  auto dim_b_outer = static_cast<uint32_t>(is_channels_last ? output_channels : output_height * output_width);
  auto dim_inner = static_cast<uint32_t>(kernel_height * kernel_width * input_channels);
  inputs[1] = kernel_to_use;
  modified_input_output_shapes[1] = kernel_to_use_shape;
  Conv2dMMProgram conv2d_mm_program =
      CreateConv2dMMProgram(activation_, inputs, pads, strides, dilations, output, dim_a_outer, dim_b_outer, dim_inner,
                            is_channels_last, modified_input_output_shapes);
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
