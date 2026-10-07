// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/program_cache_key.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

namespace {
std::string CalculateStructuredKey(const ProgramBase& program,
                                   std::span<const uint32_t> inputs_segments,
                                   std::span<const uint32_t> outputs_segments,
                                   const void* generator_type) {
  std::string key;
  key.reserve(128);
  key.push_back('\1');
  // The token identifies a static generator and its immutable metadata in this process.
  // This is not a persistent cache or an encoding of a configuration pointer.
  AppendConfigValue(key, reinterpret_cast<uintptr_t>(generator_type));
  program.AppendSpecializationKey(key);
  AppendConfigValue(key, program.WorkgroupSizeX());
  AppendConfigValue(key, program.WorkgroupSizeY());
  AppendConfigValue(key, program.WorkgroupSizeZ());
  AppendConfigValue(key, program.SubgroupSize());
  AppendConfigValue(key, program.IndirectDispatchTensor() != nullptr);
  AppendConfigValue(key, program.OverridableConstants().size());
  for (const auto& value : program.OverridableConstants()) {
    AppendConfigValue(key, value.has_value);
    if (value.has_value) {
      AppendConfigValue(key, value.type);
      switch (value.type) {
        case ProgramConstantDataType::Float32:
          AppendConfigValue(key, value.f32);
          break;
        case ProgramConstantDataType::Float16:
          AppendConfigValue(key, value.f16.val);
          break;
        case ProgramConstantDataType::Int32:
          AppendConfigValue(key, value.i32);
          break;
        case ProgramConstantDataType::Uint32:
          AppendConfigValue(key, value.u32);
          break;
        case ProgramConstantDataType::Bool:
          AppendConfigValue(key, value.boolean);
          break;
      }
    }
  }
  AppendConfigValue(key, program.UniformVariables().size());
  for (const auto& uniform : program.UniformVariables()) {
    AppendConfigValue(key, uniform.length);
    if (uniform.length != 0) AppendConfigValue(key, uniform.data_type);
  }
  const auto append_tensor = [&](const auto& tensor, uint32_t segments, size_t owner) {
    AppendConfigValue(key, tensor.var_type);
    AppendConfigValue(key, tensor.tensor->GetElementType());
    const auto& shape = tensor.use_override_shape ? tensor.override_shape : tensor.tensor->Shape();
    AppendConfigValue(key, shape.NumDimensions());
    const bool static_shape =
        (tensor.dependency & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape;
    AppendConfigValue(key, static_shape);
    if (static_shape) {
      AppendConfigValue(key, shape.GetDims());
    }
    AppendConfigValue(key, segments);
    AppendConfigValue(key, tensor.is_buffer_view);
    if (tensor.is_buffer_view) {
      AppendConfigValue(key, owner);
    } else {
      AppendConfigValue(key, tensor.buffer_offset_in_elements);
    }
  };
  AppendConfigValue(key, program.Inputs().size());
  for (size_t i = 0; i < program.Inputs().size(); ++i) {
    append_tensor(program.Inputs()[i], inputs_segments[i], program.InputBufferOwner(i));
  }
  AppendConfigValue(key, program.Outputs().size());
  for (size_t i = 0; i < program.Outputs().size(); ++i) {
    append_tensor(program.Outputs()[i], outputs_segments[i], program.OutputBufferOwner(i));
    AppendConfigValue(key, program.Outputs()[i].is_atomic);
  }
  AppendConfigValue(key, program.Indices().size());
  for (const auto& indices : program.Indices()) AppendConfigValue(key, indices.NumDimensions());
  return key;
}

}  // namespace

std::string ProgramCacheKeyForLogging(std::string_view key) {
  if (key.empty() || key[0] != '\1') return std::string{key};
  constexpr char hex[] = "0123456789abcdef";
  std::string result{"configured:"};
  for (unsigned char byte : key) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 15]);
  }
  return result;
}

std::string CalculateProgramCacheKey(const ProgramBase& program,
                                     std::span<uint32_t> inputs_segments,
                                     std::span<uint32_t> outputs_segments) {
  return CalculateStructuredKey(program, inputs_segments, outputs_segments, program.StructuredKeyType());
}

}  // namespace webgpu
}  // namespace onnxruntime
