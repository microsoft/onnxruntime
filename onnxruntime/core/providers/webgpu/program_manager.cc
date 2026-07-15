// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string_view>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/platform/env_var.h"

#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/program_manager_helpers.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

namespace detail {

// Returns the 1-based `line_num`-th line of `code` (without trailing CR/LF), or an empty view
// if the line does not exist.
static std::string_view GetSourceLine(std::string_view code, uint64_t line_num) {
  if (line_num == 0) {
    return {};
  }
  size_t start = 0;
  uint64_t remaining = line_num;
  while (remaining > 1) {
    size_t next = code.find('\n', start);
    if (next == std::string_view::npos) {
      return {};
    }
    start = next + 1;
    --remaining;
  }
  size_t end = code.find('\n', start);
  if (end == std::string_view::npos) {
    end = code.size();
  }
  while (end > start && code[end - 1] == '\r') {
    --end;
  }
  return code.substr(start, end - start);
}

std::string FormatShaderCompilationInfo(std::string_view code, const wgpu::CompilationInfo* info) {
  if (info == nullptr || info->messageCount == 0) {
    return {};
  }

  std::ostringstream oss;
  for (size_t i = 0; i < info->messageCount; ++i) {
    const auto& msg = info->messages[i];
    const char* type_str = "info";
    switch (msg.type) {
      case wgpu::CompilationMessageType::Error:
        type_str = "error";
        break;
      case wgpu::CompilationMessageType::Warning:
        type_str = "warning";
        break;
      case wgpu::CompilationMessageType::Info:
        type_str = "info";
        break;
    }
    oss << "  " << type_str << " at line " << msg.lineNum << ":" << msg.linePos << ": "
        << std::string_view{msg.message} << "\n";

    auto src_line = GetSourceLine(code, msg.lineNum);
    if (!src_line.empty()) {
      oss << "    | " << src_line << "\n";
      oss << "    | ";
      // linePos is 1-based; clamp to line length.
      size_t col = (msg.linePos > 0) ? static_cast<size_t>(msg.linePos - 1) : 0;
      col = std::min(col, src_line.size());
      for (size_t c = 0; c < col; ++c) {
        oss << (src_line[c] == '\t' ? '\t' : ' ');
      }
      size_t caret_len = (msg.length > 0) ? static_cast<size_t>(msg.length) : 1;
      if (col < src_line.size()) {
        caret_len = std::min(caret_len, src_line.size() - col);
      }
      if (caret_len == 0) {
        caret_len = 1;
      }
      oss << '^';
      for (size_t c = 1; c < caret_len; ++c) {
        oss << '~';
      }
      oss << "\n";
    }
  }
  return oss.str();
}

std::string AnnotateShaderWithLineNumbers(std::string_view code) {
  std::ostringstream oss;
  uint64_t line_num = 1;
  size_t start = 0;
  while (start < code.size()) {
    size_t end = code.find('\n', start);
    size_t print_end = (end == std::string_view::npos) ? code.size() : end;
    while (print_end > start && code[print_end - 1] == '\r') {
      --print_end;
    }
    oss << std::setw(5) << line_num << " | " << code.substr(start, print_end - start) << "\n";
    if (end == std::string_view::npos) {
      return oss.str();
    }
    start = end + 1;
    ++line_num;
  }
  // Empty input still renders a single "line 1" entry, so that an empty shader is not silently
  // formatted as an empty string.
  if (line_num == 1) {
    oss << std::setw(5) << 1 << " | \n";
  }
  return oss.str();
}

}  // namespace detail

ProgramArtifact::ProgramArtifact(const ProgramBase& program, wgpu::ComputePipeline&& compute_pipeline, std::vector<int>&& shape_uniform_ranks)
    : name{program.Name()},
      compute_pipeline{compute_pipeline},
      shape_uniform_ranks{shape_uniform_ranks} {}

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

Status ProgramManager::Build(const ProgramBase& program,
                             const ProgramMetadata& program_metadata,
                             const std::span<uint32_t> inputs_segments,
                             const std::span<uint32_t> outputs_segments,
                             const std::string& program_key,
                             uint32_t normalized_dispatch_x,
                             uint32_t normalized_dispatch_y,
                             uint32_t normalized_dispatch_z,
                             wgpu::ComputePipeline& compute_pipeline,
                             std::vector<int>& shape_uniform_ranks) const {
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
  pipeline_descriptor.compute = compute_state;
#ifndef NDEBUG  // if debug build
  pipeline_descriptor.label = program.Name().c_str();
#endif

  struct CreateComputePipelineContext {
    wgpu::ComputePipeline& pipeline;
    Status status;
  } create_pipeline_context{compute_pipeline, {}};

  ORT_RETURN_IF_ERROR(
      webgpu_context_.Wait(
          device.CreateComputePipelineAsync(
              &pipeline_descriptor,
              wgpu::CallbackMode::WaitAnyOnly,
              // Note: Don't throw from a Dawn callback.
              [](wgpu::CreatePipelineAsyncStatus status, wgpu::ComputePipeline pipeline, wgpu::StringView message,
                 CreateComputePipelineContext* context) noexcept {
                if (status == wgpu::CreatePipelineAsyncStatus::Success) {
                  context->pipeline = std::move(pipeline);
                } else {
                  context->status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create a WebGPU compute pipeline: ",
                                                    std::string_view{message});
                }
              },
              &create_pipeline_context)));

  if (!create_pipeline_context.status.IsOK()) {
    // Retrieve structured shader compilation diagnostics so we can report line numbers and the
    // offending source lines. This is invaluable when a generated WGSL shader fails to compile.
    std::string formatted_compilation_info;
    auto info_status = webgpu_context_.Wait(shader_module.GetCompilationInfo(
        wgpu::CallbackMode::WaitAnyOnly,
        // Note: Don't throw from a Dawn callback. `code` and `formatted_compilation_info` live
        // until Wait() returns, so capturing by reference is safe.
        [&code, &formatted_compilation_info](wgpu::CompilationInfoRequestStatus status,
                                             wgpu::CompilationInfo const* info) noexcept {
          if (status == wgpu::CompilationInfoRequestStatus::Success) {
            formatted_compilation_info = detail::FormatShaderCompilationInfo(code, info);
          }
        }));

    std::ostringstream oss;
    oss << create_pipeline_context.status.ErrorMessage()
        << "\n  program: \"" << program.Name() << "\""
        << "\n  key:     \"" << program_key << "\"";
    if (!info_status.IsOK()) {
      oss << "\n  (failed to retrieve shader compilation info: " << info_status.ErrorMessage() << ")";
    }
    if (!formatted_compilation_info.empty()) {
      oss << "\nShader compilation messages:\n"
          << formatted_compilation_info
          << "Annotated WGSL source:\n"
          << detail::AnnotateShaderWithLineNumbers(code);
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, oss.str());
  }

  return create_pipeline_context.status;
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
