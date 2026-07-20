// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <fstream>
#include <memory>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/platform/env_var.h"

#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

ProgramArtifact::ProgramArtifact(std::string program_name,
                                 wgpu::ComputePipeline&& compute_pipeline,
                                 wgpu::BindGroupLayout&& bind_group_layout,
                                 std::vector<int>&& shape_uniform_ranks)
    : name{std::move(program_name)},
      compute_pipeline{std::move(compute_pipeline)},
      bind_group_layout{std::move(bind_group_layout)},
      shape_uniform_ranks{std::move(shape_uniform_ranks)} {}

ProgramManager::ProgramManager(WebGpuContext& webgpu_context)
    : webgpu_context_{webgpu_context} {
  if (std::string dump_file_path = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_EP_SHADER_DUMP_FILE");
      !dump_file_path.empty()) {
    auto dump_file = std::make_shared<std::ofstream>(dump_file_path.c_str(), std::ios::app);
    shader_dump_fn_ = [dump_file = std::move(dump_file)](std::string_view shader_content) {
      *dump_file << shader_content << "\n";
    };
  }
}

Status ProgramManager::NormalizeDispatchGroupSize(uint32_t& x, uint32_t& y, uint32_t& z) const {
  ORT_RETURN_IF(x == 0 || y == 0 || z == 0, "Invalid dispatch group size (", x, ", ", y, ", ", z, ")");

  auto limit_per_dimension = webgpu_context_.DeviceLimits().maxComputeWorkgroupsPerDimension;
  if (x > limit_per_dimension || y > limit_per_dimension || z > limit_per_dimension) {
    double size = static_cast<double>(x) * static_cast<double>(y) * static_cast<double>(z);
    double dispatch_avg = std::ceil(std::sqrt(size));
    if (dispatch_avg > limit_per_dimension) {
      dispatch_avg = std::ceil(std::cbrt(size));
      ORT_RETURN_IF(dispatch_avg > limit_per_dimension, "The dispatch group size exceeds WebGPU maximum.");
      x = y = z = static_cast<uint32_t>(dispatch_avg);
    } else {
      x = y = static_cast<uint32_t>(dispatch_avg);
      z = 1;
    }
  }
  return Status::OK();
}

Status ProgramManager::CalculateSegmentsForInputsAndOutputs(const ProgramBase& program, std::vector<uint32_t>& inputs_segments, std::vector<uint32_t>& outputs_segments) const {
  inputs_segments.resize(program.Inputs().size(), 1);
  outputs_segments.resize(program.Outputs().size(), 1);

  const uint64_t maxStorageBufferBindingSize = webgpu_context_.DeviceLimits().maxStorageBufferBindingSize;

  // Inputs
  for (size_t i = 0; i < program.Inputs().size(); ++i) {
    const auto& input = program.Inputs()[i];
    if (input.tensor && input.tensor->SizeInBytes() > maxStorageBufferBindingSize) {
      uint32_t segments = static_cast<uint32_t>((input.tensor->SizeInBytes() + maxStorageBufferBindingSize - 1) / maxStorageBufferBindingSize);
      inputs_segments[i] = segments;
    }
  }
  // Outputs
  for (size_t i = 0; i < program.Outputs().size(); ++i) {
    const auto& output = program.Outputs()[i];
    if (output.tensor && output.tensor->SizeInBytes() > maxStorageBufferBindingSize) {
      uint32_t segments = static_cast<uint32_t>((output.tensor->SizeInBytes() + maxStorageBufferBindingSize - 1) / maxStorageBufferBindingSize);
      outputs_segments[i] = segments;
    }
  }
  return Status::OK();
}

wgpu::PipelineLayout ProgramManager::CreatePipelineLayout(const ProgramBase& program,
                                                          std::span<const uint32_t> inputs_segments,
                                                          std::span<const uint32_t> outputs_segments,
                                                          std::span<const int> shape_uniform_ranks,
                                                          wgpu::BindGroupLayout& bind_group_layout) const {
  size_t storage_binding_count = 0;
  for (uint32_t segments : inputs_segments) {
    storage_binding_count += segments;
  }
  for (uint32_t segments : outputs_segments) {
    storage_binding_count += segments;
  }
  const bool has_uniform_binding =
      std::any_of(shape_uniform_ranks.begin(), shape_uniform_ranks.end(), [](int rank) { return rank > 0; }) ||
      std::any_of(program.UniformVariables().cbegin(), program.UniformVariables().cend(),
                  [](const ProgramUniformVariableValue& uniform) { return uniform.length > 0; });

  std::vector<wgpu::BindGroupLayoutEntry> bind_group_layout_entries;
  bind_group_layout_entries.reserve(storage_binding_count + (has_uniform_binding ? 1 : 0));
  uint32_t binding = 0;
  auto append_buffer_bindings = [&bind_group_layout_entries, &binding](std::span<const uint32_t> segments,
                                                                       wgpu::BufferBindingType type) {
    for (uint32_t segment_count : segments) {
      for (uint32_t segment = 0; segment < segment_count; ++segment) {
        wgpu::BindGroupLayoutEntry entry{};
        entry.binding = binding++;
        entry.visibility = wgpu::ShaderStage::Compute;
        entry.buffer.type = type;
        bind_group_layout_entries.push_back(entry);
      }
    }
  };
  append_buffer_bindings(inputs_segments, wgpu::BufferBindingType::ReadOnlyStorage);
  append_buffer_bindings(outputs_segments, wgpu::BufferBindingType::Storage);
  if (has_uniform_binding) {
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = binding++;
    entry.visibility = wgpu::ShaderStage::Compute;
    entry.buffer.type = wgpu::BufferBindingType::Uniform;
    bind_group_layout_entries.push_back(entry);
  }

  ORT_ENFORCE(binding < webgpu_context_.DeviceLimits().maxBindingsPerBindGroup,
              "Number of bind group entries (", binding,
              ") exceeds device limit (", webgpu_context_.DeviceLimits().maxBindingsPerBindGroup, ").");

  wgpu::BindGroupLayoutDescriptor bind_group_layout_descriptor{};
  bind_group_layout_descriptor.label = program.Name().c_str();
  bind_group_layout_descriptor.entryCount = bind_group_layout_entries.size();
  bind_group_layout_descriptor.entries = bind_group_layout_entries.data();
  bind_group_layout = webgpu_context_.Device().CreateBindGroupLayout(&bind_group_layout_descriptor);

  wgpu::PipelineLayoutDescriptor pipeline_layout_descriptor{};
  pipeline_layout_descriptor.bindGroupLayoutCount = 1;
  pipeline_layout_descriptor.bindGroupLayouts = &bind_group_layout;
  return webgpu_context_.Device().CreatePipelineLayout(&pipeline_layout_descriptor);
}

Status ProgramManager::Build(const ProgramBase& program,
                             const ProgramMetadata& program_metadata,
                             const std::span<uint32_t> inputs_segments,
                             const std::span<uint32_t> outputs_segments,
                             const std::string& program_key,
                             uint32_t normalized_dispatch_x,
                             uint32_t normalized_dispatch_y,
                             uint32_t normalized_dispatch_z,
                             wgpu::ComputePipeline& compute_pipeline,
                             wgpu::BindGroupLayout& bind_group_layout,
                             std::vector<int>& shape_uniform_ranks,
                             wgpu::Future& future,
                             std::unique_ptr<PipelineCallbackContext>& callback_context) const {
  auto& device = webgpu_context_.Device();
  ShaderHelper shader_helper{program,
                             program_metadata,
                             webgpu_context_,
                             inputs_segments,
                             outputs_segments,
                             normalized_dispatch_x,
                             normalized_dispatch_y,
                             normalized_dispatch_z};
  ORT_RETURN_IF_ERROR(shader_helper.Init());

  ORT_RETURN_IF_ERROR(program.GenerateShaderCode(shader_helper));

  // Add indirect buffer as the last shader input when using indirect dispatch.
  if (program.IndirectDispatchTensor() != nullptr) {
    shader_helper.AddInput("indirect_buffer", ShaderUsage::None);
  }

  ORT_RETURN_IF_ERROR(shader_helper.ValidateShapeForInputs());
  ORT_RETURN_IF_ERROR(shader_helper.ValidateShapeForOutputs());
  ORT_RETURN_IF_ERROR(shader_helper.ValidateIndices());

  // code is a large std::string that contains the final shader code
  std::string code;
  ORT_RETURN_IF_ERROR(shader_helper.GenerateSourceCode(code, shape_uniform_ranks));
  auto pipeline_layout = CreatePipelineLayout(program, inputs_segments, outputs_segments,
                                              shape_uniform_ranks, bind_group_layout);

  // Dump shader code, if requested. It is dumped to `shader_dump_fn_` if set or VERBOSE logging otherwise.
  {
    const auto shader_content = [&program, &program_key, &code]() {
      return MakeString("\n=== WebGPU Shader code [", program.Name(),
                        ", Key=\"", program_key, "\"",
                        "] Start ===\n\n",
                        code,
                        "\n=== WebGPU Shader code [", program.Name(),
                        ", Key=\"", program_key, "\"",
                        "] End ===\n");
    };

    if (shader_dump_fn_) {
      shader_dump_fn_(shader_content());
    } else {
      LOGS_DEFAULT(VERBOSE) << shader_content();
    }
  }

  wgpu::ShaderSourceWGSL wgsl_source{};
  wgsl_source.code = code.c_str();

  wgpu::ShaderModuleDescriptor descriptor{};
  descriptor.nextInChain = &wgsl_source;
  descriptor.label = program.Name().c_str();

  auto shader_module = device.CreateShaderModule(&descriptor);

  // TODO: a new cache hierarchy for constants.
  //
  // Explaination:
  // Currently, we use Uniforms for dynamic data. This helps to reduce the number of program artifacts.
  //
  // "dynamic data" here means the data the determined at runtime, such as the shape of the input tensor.
  //
  // However, some programs may not necessarily depend on dynamic data. For example, "Clip" may depend on the value of "min" and "max".
  // We are using uniforms for the value of "min" and "max" in the current implementation, but usually "min" and "max" are determined
  // earlier because they are either from Attributes or from the initializers of the model.
  //
  // Questions:
  // - can we use one instance of ShaderModule to create multiple ComputePipeline?
  // - is there any benefit to do so compared to the current implementation?
  //

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

  wgpu::ComputeState compute_state{};
  compute_state.module = shader_module;
  compute_state.entryPoint = "main";
  if (!constant_entries.empty()) {
    compute_state.constants = constant_entries.data();
    compute_state.constantCount = constant_entries.size();
  }

  wgpu::ComputePipelineDescriptor pipeline_descriptor{};
  pipeline_descriptor.layout = pipeline_layout;
  pipeline_descriptor.compute = compute_state;
#ifndef NDEBUG  // if debug build
  pipeline_descriptor.label = program.Name().c_str();
#endif

  auto pipeline_callback =
      [](wgpu::CreatePipelineAsyncStatus status, wgpu::ComputePipeline pipeline, wgpu::StringView message,
         PipelineCallbackContext* context) noexcept {
        if (status == wgpu::CreatePipelineAsyncStatus::Success) {
          context->pipeline = std::move(pipeline);
        } else {
          context->status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                            "Failed to create a WebGPU compute pipeline: ",
                                            std::string_view{message});
        }
      };

  callback_context = std::make_unique<PipelineCallbackContext>(compute_pipeline, Status{});
  future = device.CreateComputePipelineAsync(
      &pipeline_descriptor,
      wgpu::CallbackMode::WaitAnyOnly,
      pipeline_callback,
      callback_context.get());
  return Status::OK();
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
