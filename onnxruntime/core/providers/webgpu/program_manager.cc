// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/safeint.h"

#include "core/common/common.h"
#include "core/common/logging/logging.h"

#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

ProgramArtifact::ProgramArtifact(const ProgramBase& program, wgpu::ComputePipeline&& compute_pipeline)
    : name{program.Name()}, compute_pipeline{compute_pipeline} {
  // prepare uniform info
  size_t current_offset = 0;
  for (const auto& uniform : program.UniformVariables()) {
    bool is_f16 = uniform.data_type == ProgramUniformVariableDataType::Float16;
    size_t length = uniform.length;
    size_t element_size = ProgramUniformVariableDataTypeSize[static_cast<int>(uniform.data_type)];
    // https://www.w3.org/TR/WGSL/#alignof
    size_t base_alignment = is_f16
                                ? (length > 4 ? 16 : length > 2 ? 8
                                                                : length * element_size)
                                : (length > 2 ? 16 : length * element_size);
    size_t struct_size = is_f16 && length <= 4 ? length * element_size : 16;

    current_offset = (current_offset + base_alignment - 1) / base_alignment * base_alignment;
    uniforms.push_back({uniform.data_type, current_offset, length});

    // For non-float16 type, when length > 4, the uniform variable is of type array<vec4<i32|u32|f32>,N>, where
    // N = ceil(data.length / 4) and SizeOf(vec4<i32|u32|f32>) = 16. The total byte length is N * SizeOf(vec4<i32|u32|f32>).
    // For float16 type, when length > 4, the uniform variable is of type array<mat2x4<f16>,N>, where
    // N = ceil(data.length / 8) and SizeOf(mat2x4<f16>) = 16. The total byte length is N * SizeOf(mat2x4<f16>).
    size_t element_per_struct = is_f16 ? 8 : 4;
    current_offset +=
        length > 4 ? (length + element_per_struct - 1) / element_per_struct * struct_size : length * element_size;
  }

  // Meet alignment of struct here: https://www.w3.org/TR/WGSL/#alignment-and-size. For simplicity, set
  // max_alignment_of_field to 16 since the underlying buffer has been rounded up to 16.
  const int max_alignment_of_field = 16;
  uniform_total_size = (current_offset + max_alignment_of_field - 1) / max_alignment_of_field * max_alignment_of_field;
}

Status ProgramManager::NormalizeDispatchGroupSize(uint32_t& x, uint32_t& y, uint32_t& z) const {
  ORT_RETURN_IF(x == 0 || y == 0 || z == 0, "Invalid dispatch group size (", x, ", ", y, ", ", z, ")");

  auto limit_per_dimension = limits_.maxComputeWorkgroupsPerDimension;
  if (x > limit_per_dimension || y > limit_per_dimension || z > limit_per_dimension) {
    auto size = static_cast<double>(x) * static_cast<double>(y) * static_cast<double>(z);
    SafeInt<uint32_t> dispatch_avg = std::ceil(std::sqrt(size));
    if (dispatch_avg > limit_per_dimension) {
      dispatch_avg = std::ceil(std::cbrt(size));
      ORT_RETURN_IF(dispatch_avg > limit_per_dimension, "The dispatch group size exceeds WebGPU maximum.");
      x = y = z = dispatch_avg;
    } else {
      x = y = dispatch_avg;
      z = 1;
    }
  }
  return Status::OK();
}

Status ProgramManager::Build(const ProgramBase& program,
                             const ProgramMetadata& program_metadata,
#ifndef NDEBUG  // if debug build
                             const std::string& program_key,
#endif
                             uint32_t normalized_dispatch_x,
                             uint32_t normalized_dispatch_y,
                             uint32_t normalized_dispatch_z,
                             wgpu::ComputePipeline& compute_pipeline) const {
  ShaderHelper shader_helper{program,
                             program_metadata,
                             device_,
                             limits_,
                             normalized_dispatch_x,
                             normalized_dispatch_y,
                             normalized_dispatch_z};
  ORT_RETURN_IF_ERROR(shader_helper.Init());

  ORT_RETURN_IF_ERROR(program.GenerateShaderCode(shader_helper));

  // code is a large std::string that contains the final shader code
  auto code = shader_helper.GetFinalSourceCode();

  LOGS_DEFAULT(VERBOSE) << "\n=== WebGPU Shader code [" << program.Name()
#ifndef NDEBUG  // if debug build
                        << ", Key=\"" << program_key << "\""
#endif
                        << "] Start ===\n\n"
                        << code
                        << "\n=== WebGPU Shader code [" << program.Name()
#ifndef NDEBUG  // if debug build
                        << ", Key=\"" << program_key << "\""
#endif
                        << "] End ===\n";

  wgpu::ShaderModuleWGSLDescriptor wgsl_descriptor{};
  wgsl_descriptor.code = code.c_str();

  wgpu::ShaderModuleDescriptor descriptor{};
  descriptor.nextInChain = &wgsl_descriptor;

  auto shader_module = device_.CreateShaderModule(&descriptor);

  // process overridable constants if available
  size_t constant_count = program.OverridableConstants().size();

  // making a copy of the constant names is required because they are stored as std::string_view in the program
  // metadata. A value of std::string_view is not guaranteed to be a C-stlye string (null-terminated) and hence
  // cannot be used directly in the WebGPU API (which expects a const char*).
  std::vector<std::string> constant_names;
  constant_names.reserve(constant_count);
  std::vector<wgpu::ConstantEntry> constant_entries;
  constant_entries.reserve(constant_count);
  for (size_t i = 0; i < constant_count; ++i) {
    const auto& constant_override = program.OverridableConstants()[i];
    const auto& constant_def = program_metadata.overridable_constants[i];

    if (constant_override.has_value) {
      double value = 0;
      switch (constant_override.type) {
        case ProgramConstantDataType::Bool:
          value = constant_override.boolean ? 1 : 0;
          break;
        case ProgramConstantDataType::Float16:
          // convert f16(MLFloat16) -> f32(float) -> f64(double)
          // because the value of a constant must be a double in WebGPU API, it is expensive to use f16 overridable constants.
          value = constant_override.f16.ToFloat();
          break;
        case ProgramConstantDataType::Float32:
          value = constant_override.f32;
          break;
        case ProgramConstantDataType::Int32:
          value = constant_override.i32;
          break;
        case ProgramConstantDataType::Uint32:
          value = constant_override.u32;
          break;
      }

      const auto& name_string = constant_names.emplace_back(constant_def.name);
      wgpu::ConstantEntry entry{};
      entry.key = name_string.c_str();
      entry.value = value;
      constant_entries.push_back(std::move(entry));
    }
  }

  wgpu::ProgrammableStageDescriptor compute_stage{};
  compute_stage.module = shader_module;
  compute_stage.entryPoint = "main";
  if (!constant_entries.empty()) {
    compute_stage.constants = constant_entries.data();
    compute_stage.constantCount = constant_entries.size();
  }

  wgpu::ComputePipelineDescriptor pipeline_descriptor{};
  pipeline_descriptor.compute = compute_stage;
#ifndef NDEBUG  // if debug build
  pipeline_descriptor.label = program.Name().c_str();
#endif

  compute_pipeline = device_.CreateComputePipeline(&pipeline_descriptor);

  return Status();
}

const ProgramArtifact* ProgramManager::Get(const std::string& key) const {
  auto result = programs_.find(key);
  if (result != programs_.end()) {
    return &result->second;
  }

  return nullptr;
}

const ProgramArtifact* ProgramManager::Set(const std::string& key, ProgramArtifact&& program) {
  return &(programs_.emplace(key, std::move(program)).first->second);
}

}  // namespace webgpu
}  // namespace onnxruntime
