// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/tensor/concat.h"

#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_CONCAT_VERSIONED_KERNEL(start, end)            \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                          \
      Concat,                                                 \
      kOnnxDomain,                                            \
      start,                                                  \
      end,                                                    \
      kWebGpuExecutionProvider,                               \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", WebGpuSupportedNumberTypes()), \
      Concat);

#define WEBGPU_CONCAT_KERNEL(version)                         \
  ONNX_OPERATOR_KERNEL_EX(                                    \
      Concat,                                                 \
      kOnnxDomain,                                            \
      version,                                                \
      kWebGpuExecutionProvider,                               \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", WebGpuSupportedNumberTypes()), \
      Concat);

WEBGPU_CONCAT_VERSIONED_KERNEL(1, 3)
WEBGPU_CONCAT_VERSIONED_KERNEL(4, 10)
WEBGPU_CONCAT_VERSIONED_KERNEL(11, 12)
WEBGPU_CONCAT_KERNEL(13)

// #define RESHAPE_TO_3D
#define UNROLL_INPUT_INDEX

#ifdef UNROLL_INPUT_INDEX
void AppendCalculateInputIndexFunction(std::ostream& os, size_t input_count) {
  os << "fn calculate_input_index(index: u32) -> u32 {\n";
  // workaround Android issue with a `for` loop by manually unrolling it.
  for (size_t i = 0; i < input_count - 1; ++i) {
    os << "  if (index < " << GetElementAt("uniforms.size_in_concat_axis", i, input_count) << ") {\n"
       << "    return " << i << ";\n"
       << "  }\n";
  }
  os << "  return " << input_count - 1 << ";\n"
     << "}\n";
}
#else
void AppendCalculateInputIndexFunction(std::ostream& os, size_t input_count) {
  os << "fn calculate_input_index(index: u32) -> u32 {\n"
     << "  for (var i = 0u; i < " << input_count << "; i = i + 1u) {\n"
     << "    if (index < " << GetElementAt("uniforms.size_in_concat_axis", "i", input_count) << ") {\n"
     << "      return i;\n"
     << "    }\n"
     << "  }\n"
     << "  return " << input_count << ";\n"
     << "}\n";
}
#endif

void AppendAssignOutputDataFunction(std::ostream& os, gsl::span<const ShaderVariableHelper*> inputs,
                                    const ShaderVariableHelper& output) {
  os << "fn assign_output_data(global_idx: u32, input_index: u32, indices: output_indices_t) {\n";
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i == 0) {
      os << "  if (input_index == 0u) {\n";
    } else if (i == inputs.size() - 1) {
      os << "  } else {\n";
    } else {
      os << "  } else if (input_index == " << i << "u) {\n";
    }
    os << "     " << output.SetByOffset("global_idx", inputs[i]->GetByIndices("indices")) << ";\n";
  }
  os << "  }\n"
        "}\n";
}

Status ConcatProgram::GenerateShaderCode(ShaderHelper& shader) const {
  size_t input_count = Inputs().size();
  std::vector<const ShaderVariableHelper*> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back(&shader.AddInput("input_" + std::to_string(i),
                                      ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias));
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  // add implementation of fn calculate_input_index
  AppendCalculateInputIndexFunction(shader.AdditionalImplementation(), input_count);
  // add implementation of fn assign_output_data
  AppendAssignOutputDataFunction(shader.AdditionalImplementation(), inputs, output);

  auto axis = axis_;

#ifdef RESHAPE_TO_3D
  const auto rank = Inputs()[0].tensor->Shape().NumDimensions();
  if (rank > 3) {
    // if rank > 3 we reshape to 2D if axis is first or last dim, and 3D otherwise.
    // if axis is fist dim it's unchanged (axis_ != 0) check above
    // if last dim adjust from rank - 1 to 1
    // otherwise it's the middle dim of the 3D shape so also 1
    if (axis != 0) {
      axis = 1;
    }
  }
#endif

  const std::string size_in_concat_axis = GetElementAt("uniforms.size_in_concat_axis", "input_index - 1", input_count);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
      << "  var output_location = " << output.OffsetToIndices("global_idx") << ";\n"
      << "  let axis_dim_value = " << output.IndicesGet("output_location", axis) << ";\n"
      << "  let input_index = calculate_input_index(axis_dim_value);\n"
      << "  if (input_index != 0u) {\n"
      << "    " << output.IndicesSet("output_location", axis, "axis_dim_value - " + size_in_concat_axis) << ";\n"
      << "  }\n"
         "  assign_output_data(global_idx, input_index, output_location);\n";

  return Status::OK();
}

Status Concat::ComputeInternal(ComputeContext& context) const {
  int input_count = context.InputCount();
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(context.Input<Tensor>(i));
  }

  Prepare prepare;
  ORT_RETURN_IF_ERROR(PrepareForCompute(&context.KernelContext(), input_tensors, prepare));
  if (prepare.output_num_elements == 0) {
    return Status::OK();
  }

#ifdef RESHAPE_TO_3D
  const auto rank = prepare.output_tensor->Shape().NumDimensions();
  const bool reshape_to_3D = rank > 3;

  const auto reshape_to_3D_func = [&](const TensorShape& shape) -> TensorShape {
    // can do 2D if axis is 0 or -1. use prepare.axis as it has been adjusted to be positive
    if (prepare.axis == 0) {
      return TensorShape{shape[0], shape.SizeFromDimension(1)};
    } else if (prepare.axis == rank - 1) {
      return TensorShape{shape.SizeToDimension(rank - 1), shape[rank - 1]};
    } else {
      return TensorShape{shape.SizeToDimension(prepare.axis),
                         shape[prepare.axis],
                         shape.SizeFromDimension(prepare.axis + 1)};
    }
  };
#endif

  uint32_t output_size = gsl::narrow_cast<int32_t>(prepare.output_tensor->Shape().Size());

  ConcatProgram program{prepare.axis};

  std::vector<uint32_t> sizes_in_concat_axis;
  sizes_in_concat_axis.reserve(input_count);
  uint32_t sum = 0;
  for (int i = 0; i < input_count; ++i) {
    const auto& input = prepare.inputs[i];
    if (input.tensor->Shape().Size() == 0) {
      continue;
    }

#ifdef RESHAPE_TO_3D
    if (reshape_to_3D) {
      program.AddInput({input.tensor, ProgramTensorMetadataDependency::TypeAndRank,
                        reshape_to_3D_func(input.tensor->Shape())});
    } else {
      program.AddInput({input.tensor, ProgramTensorMetadataDependency::TypeAndRank});
    }
#else
    program.AddInput({input.tensor, ProgramTensorMetadataDependency::TypeAndRank});
#endif

    auto axis_size = input.tensor->Shape()[prepare.axis];
    sum += static_cast<uint32_t>(axis_size);
    sizes_in_concat_axis.push_back(sum);
  }

  size_t non_empty_input_count = sizes_in_concat_axis.size();

  if (non_empty_input_count + 1 > context.DeviceLimits().maxStorageBuffersPerShaderStage) {
    // TODO: support when input_count + 1 > maxStorageBuffersPerShaderStage, by raising the limit or
    // run the program in multiple passes.
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The number of storage buffer (input=",
                           input_count, ", output=1) exceeds the limit (",
                           context.DeviceLimits().maxStorageBuffersPerShaderStage, ") of the device.");
  }

#ifdef RESHAPE_TO_3D
  ProgramOutput output = reshape_to_3D ? ProgramOutput{prepare.output_tensor,
                                                       ProgramTensorMetadataDependency::None,
                                                       reshape_to_3D_func(prepare.output_tensor->Shape())}
                                       : ProgramOutput{prepare.output_tensor};
#else
  ProgramOutput output = ProgramOutput{prepare.output_tensor};
#endif

  program.CacheHint(absl::StrJoin(std::make_tuple(non_empty_input_count, prepare.axis), ","))
      .AddOutput(std::move(output))
      .SetDispatchGroupSize((prepare.output_num_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({gsl::span<const uint32_t>(sizes_in_concat_axis.data(), sizes_in_concat_axis.size()),
                            output_size});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
