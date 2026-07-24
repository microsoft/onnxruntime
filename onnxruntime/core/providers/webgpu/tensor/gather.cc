// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/gather.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status GatherProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& data = shader.AddInput("data", ShaderUsage::UseIndicesTypeAlias);
  const auto& indices = shader.AddInput("input_indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);

  const auto& data_indices = shader.AddIndices("data_indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& output_indices = shader.AddIndices("output_indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  bool is_bool = Inputs()[0].var_type == ProgramVariableDataType::Boolx4;
  bool is_uint8 = Inputs()[0].var_type == ProgramVariableDataType::Uint8x4;
  bool pack_as_bytes = is_bool || is_uint8;
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size")
                            << "  var idx : input_indices_value_t;\n"
                            << "  var output_indices : output_indices_indices_t;\n"
                            << "  var indices_indices : input_indices_indices_t;\n"
                            << "  var data_indices : data_indices_indices_t;\n"
                            << "  var value : output_value_t;\n"
                            << "  var data_offset : u32;\n";
  for (int comp = 0; comp < (pack_as_bytes ? 4 : 1); comp++) {
    if (pack_as_bytes) {
      shader.MainFunctionBody() << "  if (" << comp << "u + 4u * global_idx < uniforms.output_size) {\n";
    }
    shader.MainFunctionBody() << "  output_indices = " << output_indices.OffsetToIndices(pack_as_bytes ? (std::to_string(comp) + " + 4 * global_idx") : "global_idx") << ";\n";

    for (int i = 0; i < indices.Rank(); i++) {
      shader.MainFunctionBody() << "  " << indices.IndicesSet("indices_indices", i, output_indices.IndicesGet("output_indices", axis_ + i)) << ";\n";
    }

    shader.MainFunctionBody() << "  idx = " << indices.GetByIndices("indices_indices") << ";\n"
                              << "  if (idx < 0) {\n"
                              << "    idx = idx + input_indices_value_t(" << data_indices.IndicesGet("uniforms.data_indices_shape", axis_) << ");\n"
                              << "  }\n";

    for (int i = 0, j = 0; i < data_indices.Rank(); i++) {
      if (static_cast<uint32_t>(i) == axis_) {
        shader.MainFunctionBody() << "  " << data_indices.IndicesSet("data_indices", i, "u32(idx)") << ";\n";
        j += indices.Rank();
      } else {
        shader.MainFunctionBody() << "  " << data_indices.IndicesSet("data_indices", i, output_indices.IndicesGet("output_indices", j)) << ";\n";
        j++;
      }
    }

    shader.MainFunctionBody() << "  data_offset = " << data_indices.IndicesToOffset("data_indices") << ";\n";
    if (is_bool) {
      shader.MainFunctionBody() << "  value[" << comp << "] = " << data.GetByOffset("data_offset / 4") << "[data_offset % 4];\n";
    } else if (is_uint8) {
      // Extract one byte from the packed u32: shift right by (data_offset % 4) * 8 bits, then mask.
      // Accumulate into output packed u32 by shifting into the correct byte position.
      shader.MainFunctionBody() << "  value = value | (((" << data.GetByOffset("data_offset / 4u")
                                << " >> ((data_offset % 4u) * 8u)) & 0xFFu) << (" << comp << "u * 8u));\n";
    } else {
      shader.MainFunctionBody() << "  value = " << data.GetByOffset("data_offset") << ";\n";
    }
    if (pack_as_bytes) {
      shader.MainFunctionBody() << "  }\n";
    }
  }

  shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", "value");

  return Status::OK();
}

Status Gather::ComputeInternal(ComputeContext& context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForComputeImpl(&context.KernelContext(), p));
  uint32_t data_size = onnxruntime::narrow<uint32_t>(p.output_tensor->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  bool pack_as_bytes = p.input_tensor->DataType() == DataTypeImpl::GetType<bool>() ||
                       p.input_tensor->DataType() == DataTypeImpl::GetType<uint8_t>();
  uint32_t output_size = data_size;
  if (pack_as_bytes) {
    // Shader packs four 1-byte elements into one u32 (4 components per thread).
    data_size = (data_size + 3) / 4;
  }

  uint32_t axis = static_cast<uint32_t>(p.axis);
  GatherProgram program{axis};
  program
      .AddInputs({{p.input_tensor, ProgramTensorMetadataDependency::TypeAndRank, ProgramInput::Flatten, (pack_as_bytes ? 4 : 1)},
                  {p.indices_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({p.output_tensor, ProgramTensorMetadataDependency::Rank, {data_size}, (pack_as_bytes ? 4 : 1)})
      .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(axis))
      .AddIndices(p.input_tensor->Shape())
      .AddIndices(p.output_tensor->Shape())
      .AddUniformVariables({{data_size}, {output_size}});
  return context.RunProgram(program);
}

#define WEBGPU_GATHER_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE)                                                                              \
  ONNX_OPERATOR_KERNEL_EX(                                                                                                                      \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,                                                                                  \
      KernelDefBuilder().TypeConstraint("T", TYPE).TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>()), \
      KERNEL_CLASS);

#define WEBGPU_GATHER_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE)                                                   \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                                                            \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,                                                                 \
      KernelDefBuilder().TypeConstraint("T", TYPE).TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>()), \
      KERNEL_CLASS);

WEBGPU_GATHER_VERSIONED_KERNEL(Gather, 1, 10, Gather, BuildKernelDefConstraintsFromTypeList<TypeList<float, MLFloat16, int32_t, uint32_t, bool, uint8_t>>())
WEBGPU_GATHER_VERSIONED_KERNEL(Gather, 11, 12, Gather, BuildKernelDefConstraintsFromTypeList<TypeList<float, MLFloat16, int32_t, uint32_t, bool, uint8_t>>())
WEBGPU_GATHER_KERNEL(Gather, 13, Gather, BuildKernelDefConstraintsFromTypeList<TypeList<float, MLFloat16, int32_t, uint32_t, bool, uint8_t>>())

}  // namespace webgpu
}  // namespace onnxruntime
