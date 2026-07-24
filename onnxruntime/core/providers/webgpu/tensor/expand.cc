// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"

#include "core/providers/webgpu/tensor/expand.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status ExpandProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size");
  // bool and uint8 are both 1-byte-per-element types that are packed 4-per-u32 in the storage buffer,
  // so they share the same broadcast logic and differ only in how a single element is extracted from
  // and assembled into the packed word.
  const bool is_bool = Inputs()[0].var_type == ProgramVariableDataType::Boolx4;
  const bool is_uint8 = Inputs()[0].var_type == ProgramVariableDataType::Uint8x4;
  if (is_bool || is_uint8) {
    const auto& input_indices = shader.AddIndices("input_indices");
    const auto& output_indices = shader.AddIndices("output_indices");

    // Extract the single element located at element offset "off".
    auto get_element = [&](const std::string& off) -> std::string {
      if (is_bool) {
        return input.GetByOffset(off + " / 4") + "[" + off + " % 4]";
      }
      // Shift the byte at "off % 4" down to the low 8 bits of the packed u32.
      return "((" + input.GetByOffset(off + " / 4") + " >> ((" + off + " % 4) * 8u)) & 0xFFu)";
    };
    // Broadcast a single element into all 4 lanes of a packed word.
    auto splat_element = [&](const std::string& e) -> std::string {
      return is_bool ? ("vec4<bool>(" + e + ")") : ("(" + e + " * 0x01010101u)");
    };
    // Assemble a packed word from 4 elements.
    auto pack_elements = [&](const std::string& e0, const std::string& e1, const std::string& e2, const std::string& e3) -> std::string {
      if (is_bool) {
        return "vec4<bool>(" + e0 + ", " + e1 + ", " + e2 + ", " + e3 + ")";
      }
      return "(" + e0 + " | (" + e1 + " << 8u) | (" + e2 + " << 16u) | (" + e3 + " << 24u))";
    };

    if (input_last_dim_divisible_by_4_) {
      // The last dims of input shape and output shape are all divisible by 4, so a whole packed word
      // maps directly to a whole packed word.
      shader.MainFunctionBody() << "  let output_indices = " << output_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "  let input_offset = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << output.SetByOffset("global_idx", input.GetByOffset("input_offset / 4"));
    } else if (output_last_dim_divisible_by_4_) {
      // The last dim of output shape is divisible by 4, and the last dim of input shape is 1.
      shader.MainFunctionBody() << "  let output_indices = " << output_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "  let input_offset = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << "  let value = " << splat_element(get_element("input_offset")) << ";\n"
                                << "  " << output.SetByOffset("global_idx", "value");
    } else {
      shader.MainFunctionBody() << "  var output_indices = " << output_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "  let input_offset_0 = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << "  output_indices = " << output_indices.OffsetToIndices("global_idx * 4 + 1") << ";\n"
                                << "  let input_offset_1 = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << "  output_indices = " << output_indices.OffsetToIndices("global_idx * 4 + 2") << ";\n"
                                << "  let input_offset_2 = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << "  output_indices = " << output_indices.OffsetToIndices("global_idx * 4 + 3") << ";\n"
                                << "  let input_offset_3 = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << "  let value = "
                                << pack_elements(get_element("input_offset_0"), get_element("input_offset_1"),
                                                 get_element("input_offset_2"), get_element("input_offset_3"))
                                << ";\n"
                                << output.SetByOffset("global_idx", "value");
    }
    return Status::OK();
  }
  if (input.NumComponents() != output.NumComponents()) {
    const auto& output_indices = shader.AddIndices("output_indices");
    shader.MainFunctionBody() << "  let output_indices = " << output_indices.OffsetToIndices("global_idx * 4") << ";\n"
                              << "  let input_offset = " << input.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n  "
                              << "  let value = vec4<input_value_t>(" << input.GetByOffset("input_offset") << ");\n"
                              << output.SetByOffset("global_idx", "value");
  } else {
    shader.MainFunctionBody() << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                              << "  let input_offset = " << input.BroadcastedIndicesToOffset("output_indices", output) << ";\n  "
                              << output.SetByOffset("global_idx", input.GetByOffset("input_offset"));
  }
  return Status::OK();
}

Status Expand::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const auto* input_shape_tensor = context.Input(1);

  auto output_dims = input_shape_tensor->DataAsSpan<int64_t>();
  TensorShape output_shape{};
  TensorShape input_shape = input_tensor->Shape();
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), input_shape, output_dims, output_shape));

  auto* output_tensor = context.Output(0, output_shape);

  bool is_int64 = input_tensor->DataType() == DataTypeImpl::GetType<int64_t>();
  // bool and uint8 are 1-byte-per-element types that are not valid storage buffer types, so we pack
  // 4 of them into a single `u32` and handle them with a dedicated shader path.
  bool is_bool = input_tensor->DataType() == DataTypeImpl::GetType<bool>();
  bool is_uint8 = input_tensor->DataType() == DataTypeImpl::GetType<uint8_t>();
  bool is_packed_byte = is_bool || is_uint8;
  bool input_last_dim_divisible_by_4 = (!(input_shape.IsScalar() || is_int64)) && (input_shape[input_shape.NumDimensions() - 1] % 4 == 0);
  bool output_last_dim_divisible_by_4 = (!(output_shape.IsScalar() || is_int64)) && (output_shape[output_shape.NumDimensions() - 1] % 4 == 0);
  const int components_i = (is_packed_byte || input_last_dim_divisible_by_4) ? 4 : 1;
  const int components_o = (is_packed_byte || output_last_dim_divisible_by_4) ? 4 : 1;
  uint32_t data_size = onnxruntime::narrow<uint32_t>((output_shape.Size() + components_o - 1) / components_o);
  if (data_size == 0) {
    return Status::OK();
  }
  ExpandProgram program{input_last_dim_divisible_by_4, output_last_dim_divisible_by_4};
  program.SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {data_size},
      });
  if (is_packed_byte) {
    program.CacheHint(std::to_string(static_cast<int>(input_last_dim_divisible_by_4)), std::to_string(static_cast<int>(output_last_dim_divisible_by_4)))
        .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, ProgramInput::Flatten, components_i}})
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, {data_size}, components_o}})
        .AddIndices(std::move(input_shape));
  } else {
    program.AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, components_i}})
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, components_o}});
  }
  if (is_packed_byte || components_i != components_o) {
    program.AddIndices(std::move(output_shape));
  }
  return context.RunProgram(program);
}

template <int StartVersion, int EndVersion>
KernelCreateInfo CreateExpandVersionedKernelInfo(bool enable_int64) {
  std::vector<MLDataType> type_constraints = GetOpTypeConstraints(enable_int64, true);
  type_constraints.push_back(DataTypeImpl::GetTensorType<uint8_t>());

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Expand>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Expand")
          .SetDomain(kOnnxDomain)
          .SinceVersion(StartVersion, EndVersion)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", type_constraints)
          .InputMemoryType(OrtMemTypeCPU, 1)
          .Build(),
      kernel_create_fn};
}

template <int SinceVersion>
KernelCreateInfo CreateExpandKernelInfo(bool enable_int64) {
  std::vector<MLDataType> type_constraints = GetOpTypeConstraints(enable_int64, true);
  type_constraints.push_back(DataTypeImpl::GetTensorType<uint8_t>());

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Expand>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Expand")
          .SetDomain(kOnnxDomain)
          .SinceVersion(SinceVersion)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", type_constraints)
          .InputMemoryType(OrtMemTypeCPU, 1)
          .Build(),
      kernel_create_fn};
}

// Explicit template instantiations
template KernelCreateInfo CreateExpandVersionedKernelInfo<8, 12>(bool);
template KernelCreateInfo CreateExpandKernelInfo<13>(bool);

}  // namespace webgpu
}  // namespace onnxruntime
