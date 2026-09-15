// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/tensor/concat.h"

#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

// Concat is a pure data-movement op: elements are copied, never interpreted in shader arithmetic,
// so enabling int64 is safe. int64 (stored as vec2<u32>) is copied losslessly via the raw
// storage-word path in AppendAssignOutputDataFunction, preserving the full 64-bit value instead of
// the truncating i32 value type used by arithmetic kernels.
KernelCreateInfo CreateConcatVersionedKernelInfo(int start_version, int end_version, bool enable_int64) {
  std::vector<MLDataType> type_constraints = GetOpTypeConstraints(enable_int64, /*enable_bool=*/false);

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Concat>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Concat")
          .SetDomain(kOnnxDomain)
          .SinceVersion(start_version, end_version)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", std::move(type_constraints))
          .Build(),
      kernel_create_fn};
}

KernelCreateInfo CreateConcatKernelInfo(int since_version, bool enable_int64) {
  std::vector<MLDataType> type_constraints = GetOpTypeConstraints(enable_int64, /*enable_bool=*/false);

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Concat>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Concat")
          .SetDomain(kOnnxDomain)
          .SinceVersion(since_version)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", std::move(type_constraints))
          .Build(),
      kernel_create_fn};
}

void AppendCalculateInputIndexFunction(OStringStream& os, size_t input_count) {
  os << "fn calculate_input_index(global_idx: u32) -> u32 {\n"
     << "  for (var i = 1u; i < " << input_count << "; i = i + 1u) {\n"
     << "    if (global_idx < " << GetElementAt("uniforms.offsets", "i", input_count) << ") {\n"
     << "      return i - 1;\n"
     << "    }\n"
     << "  }\n"
     << "  return " << input_count - 1 << ";\n"
     << "}\n";
}

void AppendAssignOutputDataFunction(OStringStream& os, gsl::span<const ShaderVariableHelper*> inputs, const ShaderVariableHelper& output, size_t axis, size_t input_count, bool is_int64) {
  os << "fn assign_output_data(global_idx: u32, input_index: u32) {\n";
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i == 0) {
      os << "  if (input_index == 0u) {\n";
    } else if (i == inputs.size() - 1) {
      os << "  } else {\n";
    } else {
      os << "  } else if (input_index == " << i << "u) {\n";
    }
    std::string offset = GetElementAt("uniforms.offsets", "input_index", input_count);
    std::string concat_axis_offset = GetElementAt("uniforms.sizes_in_concat_axis", std::to_string(i), input_count);
    std::string output_indices_axis = "output_indices" + (inputs[i]->Rank() > 1 ? "[" + std::to_string(axis) + "]" : "");
    std::string input_offset = "global_idx - " + offset;
    os << "     var output_indices = " << inputs[i]->OffsetToIndices(input_offset) << ";\n"
       << "     " << output_indices_axis << " += " << concat_axis_offset << ";\n";
    if (is_int64) {
      // int64 is stored as vec2<u32>. Copy the raw storage bits so the full 64-bit value is
      // preserved; the value-type accessors would truncate it to i32.
      os << "     let output_offset = " << output.IndicesToOffset("output_indices") << ";\n"
         << "     " << output.SetByOffset("output_offset", inputs[i]->GetByOffset(input_offset, /*use_storage_type=*/true), /*use_storage_type=*/true) << "\n";
    } else {
      os << "     " << output.SetByIndices("output_indices", inputs[i]->GetByOffset(input_offset)) << "\n";
    }
  }
  os << "  }\n"
        "}\n";
}

Status ConcatProgram::GenerateShaderCode(ShaderHelper& shader) const {
  size_t input_count = Inputs().size();
  std::vector<const ShaderVariableHelper*> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back(&shader.AddInput("input_" + std::to_string(i), ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias));
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  AppendCalculateInputIndexFunction(shader.AdditionalImplementation(), input_count);
  AppendAssignOutputDataFunction(shader.AdditionalImplementation(), inputs, output, axis_, input_count, is_int64_);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let input_index = calculate_input_index(global_idx);\n"
                            << "assign_output_data(global_idx, input_index);\n";

  return Status::OK();
}

Status Concat::ComputeInternal(ComputeContext& context) const {
  uint32_t input_count = context.InputCount();
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  for (uint32_t i = 0; i < input_count; ++i) {
    input_tensors.push_back(context.Input<Tensor>(i));
  }

  Prepare prepare;
  ORT_RETURN_IF_ERROR(PrepareForComputeImpl(&context.KernelContext(), input_tensors, prepare));
  if (prepare.output_num_elements == 0) {
    return Status::OK();
  }

  uint32_t axis = static_cast<uint32_t>(prepare.axis);
  uint32_t max_inputs_per_concat = context.DeviceLimits().maxStorageBuffersPerShaderStage - 1;
  bool is_int64 = prepare.output_tensor->DataType() == DataTypeImpl::GetType<int64_t>();

  uint32_t input_index = 0;
  uint32_t cumulative_size_in_concat_axis = 0;

  while (input_index < input_count) {
    ConcatProgram program{axis, is_int64};
    uint32_t num_inputs_this_concat = std::min(max_inputs_per_concat, input_count - input_index);

    std::vector<uint32_t> offsets;
    offsets.reserve(num_inputs_this_concat + 1);
    offsets.push_back(0);

    std::vector<uint32_t> sizes_in_concat_axis;
    sizes_in_concat_axis.reserve(num_inputs_this_concat + 1);
    sizes_in_concat_axis.push_back(cumulative_size_in_concat_axis);

    uint32_t output_size = 0;
    for (uint32_t i = 0; i < num_inputs_this_concat; i++) {
      auto& input = prepare.inputs[input_index + i];
      if (input.tensor->Shape().Size() == 0) {
        continue;
      }
      program.AddInput({input.tensor, ProgramTensorMetadataDependency::TypeAndRank});

      uint32_t size = onnxruntime::narrow<int32_t>(input.tensor->Shape().Size());
      uint32_t axis_size = static_cast<uint32_t>(input.tensor->Shape()[axis]);

      output_size += size;
      offsets.push_back(output_size);
      cumulative_size_in_concat_axis += axis_size;
      sizes_in_concat_axis.push_back(cumulative_size_in_concat_axis);
    }

    offsets.pop_back();
    sizes_in_concat_axis.pop_back();

    program.CacheHint(absl::StrJoin(std::make_tuple(num_inputs_this_concat, prepare.axis), ","))
        .AddOutputs({prepare.output_tensor})
        .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({gsl::span<const uint32_t>(offsets.data(), offsets.size()), gsl::span<const uint32_t>(sizes_in_concat_axis.data(), sizes_in_concat_axis.size()), output_size});
    ORT_RETURN_IF_ERROR(context.RunProgram(program));

    input_index += num_inputs_this_concat;
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
