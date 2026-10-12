// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/program_cache_key.h"

#include <atomic>
#include "core/providers/webgpu/shader_metadata.h"

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
  ShaderProgramMetadata{program}.AppendTo(key);
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
  for (size_t i = 0; i < program.Inputs().size(); ++i) {
    ShaderTensorMetadata{program.Inputs()[i], inputs_segments[i], program.InputBufferOwner(i)}.AppendTo(key);
  }
  for (size_t i = 0; i < program.Outputs().size(); ++i) {
    ShaderTensorMetadata{program.Outputs()[i], outputs_segments[i], program.OutputBufferOwner(i)}.AppendTo(key);
  }
  AppendConfigValue(key, program.Indices().size());
  for (const auto& indices : program.Indices()) AppendConfigValue(key, indices.NumDimensions());
  return key;
}

}  // namespace

std::string ShaderDiagnosticIds::Format(const std::string& key) const {
  static std::atomic<uint64_t> next_id{1};
  uint64_t id;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [entry, inserted] = ids_.try_emplace(key, 0);
    if (inserted) entry->second = next_id.fetch_add(1, std::memory_order_relaxed);
    id = entry->second;
  }
  return "shader:" + std::to_string(id);
}

std::string CalculateProgramCacheKey(const ProgramBase& program,
                                     std::span<uint32_t> inputs_segments,
                                     std::span<uint32_t> outputs_segments) {
  return CalculateStructuredKey(program, inputs_segments, outputs_segments, program.StructuredKeyType());
}

}  // namespace webgpu
}  // namespace onnxruntime
