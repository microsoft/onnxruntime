// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/gather.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_kernel.h"
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
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size");
  if (pack_as_bytes) {
    // bool and uint8 both pack four 1-byte elements per thread into one u32 storage word.
    // The accumulator holds one element per lane and is handed to SetByOffset unpacked, because
    // SetByOffset performs the byte packing for Uint8x4 (and Boolx4) itself. Do not pre-pack
    // into a u32 here: SetByOffset would splat that scalar across all four lanes, mask each to
    // its low byte and re-pack, so every output word would be element 0's byte repeated four
    // times. uint8 needs an explicit vec4<u32> because output_value_t for Uint8x4 is the packed
    // u32 storage type, unlike Boolx4's vec4<bool>.
    shader.MainFunctionBody() << (is_uint8 ? "  var value : vec4<u32> = vec4<u32>(0u);\n"
                                           : "  var value : output_value_t;\n");
    for (int comp = 0; comp < 4; comp++) {
      shader.MainFunctionBody() << "  if (" << comp << "u + 4u * global_idx < uniforms.output_size) {\n"
                                << "    var output_indices : output_indices_indices_t;\n"
                                << "    var indices_indices : input_indices_indices_t;\n"
                                << "    var data_indices : data_indices_indices_t;\n"
                                << "    var idx : input_indices_value_t;\n";
      shader.MainFunctionBody() << "    output_indices = " << output_indices.OffsetToIndices(std::to_string(comp) + " + 4 * global_idx") << ";\n";

      for (int i = 0; i < indices.Rank(); i++) {
        shader.MainFunctionBody() << "    " << indices.IndicesSet("indices_indices", i, output_indices.IndicesGet("output_indices", axis_ + i)) << ";\n";
      }

      shader.MainFunctionBody() << "    idx = " << indices.GetByIndices("indices_indices") << ";\n"
                                << "    if (idx < 0) {\n"
                                << "      idx = idx + input_indices_value_t(" << data_indices.IndicesGet("uniforms.data_indices_shape", axis_) << ");\n"
                                << "    }\n";

      for (int i = 0, j = 0; i < data_indices.Rank(); i++) {
        if (static_cast<uint32_t>(i) == axis_) {
          shader.MainFunctionBody() << "    " << data_indices.IndicesSet("data_indices", i, "u32(idx)") << ";\n";
          j += indices.Rank();
        } else {
          shader.MainFunctionBody() << "    " << data_indices.IndicesSet("data_indices", i, output_indices.IndicesGet("output_indices", j)) << ";\n";
          j++;
        }
      }

      shader.MainFunctionBody() << "    let data_offset = " << data_indices.IndicesToOffset("data_indices") << ";\n";
      if (is_bool) {
        shader.MainFunctionBody() << "    value[" << comp << "] = " << data.GetByOffset("data_offset / 4") << "[data_offset % 4];\n";
      } else {
        shader.MainFunctionBody() << "    value[" << comp << "] = unpack4xU8(" << data.GetByOffset("data_offset / 4u")
                                  << ")[data_offset % 4u];\n";
      }
      shader.MainFunctionBody() << "  }\n";
    }
    shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", "value");
  } else {
    shader.MainFunctionBody() << "  var idx : input_indices_value_t;\n"
                              << "  var output_indices : output_indices_indices_t;\n"
                              << "  var indices_indices : input_indices_indices_t;\n"
                              << "  var data_indices : data_indices_indices_t;\n"
                              << "  var data_offset : u32;\n";
    shader.MainFunctionBody() << "  output_indices = " << output_indices.OffsetToIndices("global_idx") << ";\n";

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
    // use_storage_type is only honored for Int64/Uint64 by GetByOffset/SetByOffset; for every
    // other type it is ignored, so passing is_int64_ directly is equivalent to the plain
    // value-type access when is_int64_ is false. For int64 it copies the raw vec2<u32> storage
    // bits so the full 64-bit value is preserved instead of being truncated to i32.
    shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", data.GetByOffset("data_offset", is_int64_), is_int64_);
  }

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

  bool is_int64 = p.input_tensor->DataType() == DataTypeImpl::GetType<int64_t>();
  uint32_t axis = static_cast<uint32_t>(p.axis);
  GatherProgram program{axis, is_int64};
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

// Gather is a pure data-movement op: elements are copied, never interpreted in shader arithmetic,
// so enabling int64 is safe. int64 (stored as vec2<u32>) is copied losslessly via the raw
// storage-word path in GenerateShaderCode, preserving the full 64-bit value instead of the
// truncating i32 value type used by arithmetic kernels. The base type set already includes bool and
// uint8 (packed 4-per-u32), which int64 support leaves untouched.
static std::vector<MLDataType> GetGatherTypeConstraints(bool enable_int64) {
  std::vector<MLDataType> type_constraints = WebGpuSupportedNumberBoolAndUint8Types();
  if (enable_int64) {
    type_constraints.push_back(DataTypeImpl::GetTensorType<int64_t>());
  }
  return type_constraints;
}

KernelCreateInfo CreateGatherVersionedKernelInfo(int start_version, int end_version, bool enable_int64) {
  std::vector<MLDataType> type_constraints = GetGatherTypeConstraints(enable_int64);

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Gather>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Gather")
          .SetDomain(kOnnxDomain)
          .SinceVersion(start_version, end_version)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", std::move(type_constraints))
          .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
          .Build(),
      kernel_create_fn};
}

KernelCreateInfo CreateGatherKernelInfo(int since_version, bool enable_int64) {
  std::vector<MLDataType> type_constraints = GetGatherTypeConstraints(enable_int64);

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Gather>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Gather")
          .SetDomain(kOnnxDomain)
          .SinceVersion(since_version)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", std::move(type_constraints))
          .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
          .Build(),
      kernel_create_fn};
}

}  // namespace webgpu
}  // namespace onnxruntime
