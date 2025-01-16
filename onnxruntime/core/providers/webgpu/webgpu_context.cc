// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <cmath>

#if !defined(__wasm__)
#include "dawn/dawn_proc.h"
#if !defined(USE_EXTERNAL_DAWN)
#include "dawn/native/DawnNative.h"
#endif
#endif

#include "core/common/common.h"
#include "core/common/path_string.h"
#include "core/platform/env.h"

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/program_cache_key.h"
#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/string_macros.h"

namespace onnxruntime {
namespace webgpu {

void WebGpuContext::Initialize(const WebGpuBufferCacheConfig& buffer_cache_config, int backend_type) {
  std::call_once(init_flag_, [this, &buffer_cache_config, backend_type]() {
    // Create wgpu::Adapter
    if (adapter_ == nullptr) {
#if !defined(__wasm__) && defined(_MSC_VER) && defined(DAWN_ENABLE_D3D12) && !defined(USE_EXTERNAL_DAWN)
      // If we are using the D3D12 backend on Windows and the build does not use external Dawn, dxil.dll and dxcompiler.dll are required.
      //
      // Dawn will try to load them later, but if they are in the different directory to the executable, it may fail to find them.
      // To avoid this issue, we try to load them from the same directory as current module (usually onnxruntime.dll).
      auto runtime_path = Env::Default().GetRuntimePath();
      if (!runtime_path.empty()) {
        Status status;
        void* module_handle = nullptr;

        PathString dxil_path = runtime_path + ToPathString(L"dxil.dll");
        status = Env::Default().LoadDynamicLibrary(dxil_path, false, &module_handle);
        if (status.IsOK() && module_handle != nullptr) {
          modules_.Add(dxil_path, module_handle);
        }

        PathString dxcompiler_path = runtime_path + ToPathString(L"dxcompiler.dll");
        status = Env::Default().LoadDynamicLibrary(dxcompiler_path, false, &module_handle);
        if (status.IsOK() && module_handle != nullptr) {
          modules_.Add(dxcompiler_path, module_handle);
        }
      }
#endif

      wgpu::RequestAdapterOptions req_adapter_options = {};
      req_adapter_options.backendType = static_cast<wgpu::BackendType>(backend_type);
      req_adapter_options.powerPreference = wgpu::PowerPreference::HighPerformance;

#if !defined(__wasm__)
      auto enabled_adapter_toggles = GetEnabledAdapterToggles();

      wgpu::DawnTogglesDescriptor adapter_toggles_desc = {};
      adapter_toggles_desc.enabledToggleCount = enabled_adapter_toggles.size();
      adapter_toggles_desc.enabledToggles = enabled_adapter_toggles.data();

      req_adapter_options.nextInChain = &adapter_toggles_desc;
#endif

      ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(instance_.RequestAdapter(
                                                                     &req_adapter_options,
                                                                     wgpu::CallbackMode::WaitAnyOnly,
                                                                     [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message, wgpu::Adapter* ptr) {
                                                                       ORT_ENFORCE(status == wgpu::RequestAdapterStatus::Success, "Failed to get a WebGPU adapter: ", std::string_view{message});
                                                                       *ptr = adapter;
                                                                     },
                                                                     &adapter_),
                                                                 UINT64_MAX));
      ORT_ENFORCE(adapter_ != nullptr, "Failed to get a WebGPU adapter.");
    }

    // Create wgpu::Device
    if (device_ == nullptr) {
      wgpu::DeviceDescriptor device_desc = {};

#if !defined(__wasm__)
      wgpu::DawnTogglesDescriptor device_toggles_desc = {};
      device_desc.nextInChain = &device_toggles_desc;

      auto enabled_device_toggles = GetEnabledDeviceToggles();
      device_toggles_desc.enabledToggleCount = enabled_device_toggles.size();
      device_toggles_desc.enabledToggles = enabled_device_toggles.data();

      auto disabled_device_toggles = GetDisabledDeviceToggles();
      device_toggles_desc.disabledToggleCount = disabled_device_toggles.size();
      device_toggles_desc.disabledToggles = disabled_device_toggles.data();
#endif

      std::vector<wgpu::FeatureName> required_features = GetAvailableRequiredFeatures(adapter_);
      if (required_features.size() > 0) {
        device_desc.requiredFeatures = required_features.data();
        device_desc.requiredFeatureCount = required_features.size();
      }
      wgpu::RequiredLimits required_limits = GetRequiredLimits(adapter_);
      device_desc.requiredLimits = &required_limits;

      // TODO: revise temporary error handling
      device_desc.SetUncapturedErrorCallback([](const wgpu::Device& /*device*/, wgpu::ErrorType type, const char* message) {
        LOGS_DEFAULT(ERROR) << "WebGPU device error(" << int(type) << "): " << message;
      });
      // TODO: revise temporary device lost handling
      device_desc.SetDeviceLostCallback(wgpu::CallbackMode::AllowSpontaneous, [](const wgpu::Device& /*device*/, wgpu::DeviceLostReason reason, const char* message) {
        LOGS_DEFAULT(INFO) << "WebGPU device lost (" << int(reason) << "): " << message;
      });

      ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(adapter_.RequestDevice(
                                                                     &device_desc,
                                                                     wgpu::CallbackMode::WaitAnyOnly,
                                                                     [](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message, wgpu::Device* ptr) {
                                                                       ORT_ENFORCE(status == wgpu::RequestDeviceStatus::Success, "Failed to get a WebGPU device: ", std::string_view{message});
                                                                       *ptr = device;
                                                                     },
                                                                     &device_),
                                                                 UINT64_MAX));
      ORT_ENFORCE(device_ != nullptr, "Failed to get a WebGPU device.");
    }

    // cache adapter info
    ORT_ENFORCE(Adapter().GetInfo(&adapter_info_));
    // cache device limits
    wgpu::SupportedLimits device_supported_limits;
    ORT_ENFORCE(Device().GetLimits(&device_supported_limits));
    device_limits_ = device_supported_limits.limits;

    // create buffer manager
    buffer_mgr_ = BufferManagerFactory::Create(*this,
                                               buffer_cache_config.storage.mode,
                                               buffer_cache_config.uniform.mode,
                                               buffer_cache_config.query_resolve.mode);

    // create program manager
    program_mgr_ = std::make_unique<ProgramManager>(Device(), DeviceLimits());

    // set query type
#if !defined(__wasm__)
    if (device_.HasFeature(wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses)) {
      query_type_ = TimestampQueryType::InsidePasses;
    } else
#endif
        if (device_.HasFeature(wgpu::FeatureName::TimestampQuery)) {
      query_type_ = TimestampQueryType::AtPasses;
    } else {
      query_type_ = TimestampQueryType::None;
    }
  });
}

Status WebGpuContext::Wait(wgpu::Future f) {
  auto status = instance_.WaitAny(f, UINT64_MAX);
  if (status == wgpu::WaitStatus::Success) {
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to wait for the operation:", uint32_t(status));
}

Status WebGpuContext::Run(ComputeContext& context, const ProgramBase& program) {
  const auto& inputs = program.Inputs();
  const auto& outputs = program.Outputs();

  if (outputs.size() == 0) {
    return Status::OK();
  }

  if (ValidationMode() >= ValidationMode::Basic) {
    ORT_ENFORCE(std::all_of(inputs.begin(), inputs.end(), [](const ProgramInput& input) {
                  const auto* tensor = input.tensor;
                  return tensor != nullptr &&
                         tensor->Location().mem_type == OrtMemType::OrtMemTypeDefault &&
                         tensor->Location().device.Type() == OrtDevice::GPU &&
                         !strcmp(tensor->Location().name, WEBGPU_BUFFER);
                }),
                "All inputs must be tensors on WebGPU buffers.");

    ORT_ENFORCE(std::all_of(outputs.begin(), outputs.end(), [](const ProgramOutput& output) {
                  const auto* tensor = output.tensor;
                  return tensor != nullptr &&
                         tensor->Location().mem_type == OrtMemType::OrtMemTypeDefault &&
                         tensor->Location().device.Type() == OrtDevice::GPU &&
                         !strcmp(tensor->Location().name, WEBGPU_BUFFER);
                }),
                "All outputs must be tensors on WebGPU buffers.");
  }

  const ProgramMetadata& metadata = program.Metadata();

  // validate program metadata
  if (ValidationMode() >= ValidationMode::Basic) {
    const auto& [constants, overridable_constants, uniform_variables] = metadata;

    // check overridable constants
    ORT_RETURN_IF(program.OverridableConstants().size() != overridable_constants.size(),
                  "Size of overridable constants mismatch in program \"", program.Name(),
                  "\", Expected: ", overridable_constants.size(),
                  ", Actual: ", program.OverridableConstants().size());

    if (ValidationMode() >= ValidationMode::Full) {
      size_t num_overridable_constants = program.OverridableConstants().size();
      for (size_t i = 0; i < num_overridable_constants; ++i) {
        const auto& override_value = program.OverridableConstants()[i];
        const auto& definition = overridable_constants[i];
        ORT_RETURN_IF(override_value.has_value && override_value.type != definition.type,
                      "Overridable override_value[", i, "] (", definition.name, ") data type mismatch in program \"", program.Name(),
                      "\", Expected: ", definition.type,
                      ", Actual: ", override_value.type);
        ORT_RETURN_IF(!override_value.has_value && !definition.has_default_value,
                      "Overridable override_value[", i, "] (", definition.name, ") no override_value specified in program \"", program.Name(),
                      "\"");
      }
    }

    // check uniform variables
    ORT_RETURN_IF(program.UniformVariables().size() != uniform_variables.size(),
                  "Size of uniform_value variables mismatch in program \"", program.Name(),
                  "\", Expected: ", uniform_variables.size(),
                  ", Actual: ", program.UniformVariables().size());

    if (ValidationMode() >= ValidationMode::Full) {
      size_t num_uniform_variables = program.UniformVariables().size();
      for (size_t i = 0; i < num_uniform_variables; ++i) {
        const auto& uniform_value = program.UniformVariables()[i];
        const auto& definition = uniform_variables[i];
        ORT_RETURN_IF(uniform_value.length > 0 && uniform_value.data_type != definition.data_type,
                      "Uniform variable[", i, "] (", definition.name, ") data type mismatch in program \"", program.Name(),
                      "\", Expected: ", definition.data_type,
                      ", Actual: ", uniform_value.data_type);
      }
    }
  }

  uint32_t x = program.DispatchGroupSizeX();
  uint32_t y = program.DispatchGroupSizeY();
  uint32_t z = program.DispatchGroupSizeZ();
  ORT_RETURN_IF_ERROR(program_mgr_->NormalizeDispatchGroupSize(x, y, z));

  bool is_1d_dispatch = (y == 1 && z == 1);

  auto key = CalculateProgramCacheKey(program, is_1d_dispatch);

  if (is_profiling_) {
    PendingKernelInfo pending_kernel_info(context.KernelContext().GetNodeName(),
                                          context.KernelContext().GetOpType(),
                                          program.Name(),
                                          key,
                                          inputs,
                                          outputs);
    pending_kernels_.emplace_back(std::move(pending_kernel_info));
  }

  LOGS(context.Logger(), INFO) << "Starting program \"" << key << "\" (" << x << ", " << y << ", " << z << ")";

  const auto* program_artifact = program_mgr_->Get(key);
  if (program_artifact == nullptr) {
    wgpu::ComputePipeline compute_pipeline;
    std::vector<int> shape_uniform_ranks;
    auto status = program_mgr_->Build(program,
                                      metadata,
#ifndef NDEBUG  // if debug build
                                      key,
#endif
                                      x,
                                      y,
                                      z,
                                      compute_pipeline,
                                      shape_uniform_ranks);
    ORT_RETURN_IF_ERROR(status);
    program_artifact = program_mgr_->Set(key, ProgramArtifact{program,
                                                              std::move(compute_pipeline),
                                                              std::move(shape_uniform_ranks)});
#ifndef NDEBUG  // if debug build
    ORT_ENFORCE(program_artifact != nullptr, "Program artifact should not be nullptr.");
#endif
  }

  // prepare shape uniforms for shader variables (if any) and user defined uniforms
  std::vector<ProgramUniformVariableValue> shape_uniforms;
  shape_uniforms.reserve(program_artifact->shape_uniform_ranks.size() * 2);
  if (ValidationMode() >= ValidationMode::Basic) {
    ORT_RETURN_IF_NOT(program_artifact->shape_uniform_ranks.size() == inputs.size() + outputs.size() + program.Indices().size(),
                      "Invalid program artifact: variable size (", program_artifact->shape_uniform_ranks.size(),
                      ") does not match current program (input: ", inputs.size(),
                      ", output: ", outputs.size(),
                      ", indices: ", program.Indices().size(), ")");
  }

  auto append_shape_uniforms = [&shape_uniforms, program_artifact](size_t i, const TensorShape& shape) {
    if (program_artifact->shape_uniform_ranks[i] > 0) {
      size_t expected_rank = static_cast<size_t>(program_artifact->shape_uniform_ranks[i]);
      ORT_RETURN_IF(expected_rank != shape.NumDimensions(),
                    "Invalid program artifact: variable[", i, "] rank mismatch. Expected: ", expected_rank,
                    ", Actual: ", shape.NumDimensions());

      std::vector<uint32_t> dims(expected_rank);
      std::vector<uint32_t> stride(expected_rank - 1);
      for (size_t j = 0; j < expected_rank; ++j) {
        dims[j] = gsl::narrow<uint32_t>(shape[j]);
        if (j < expected_rank - 1) {
          stride[j] = gsl::narrow<uint32_t>(shape.SizeFromDimension(j + 1));
        }
      }

      shape_uniforms.emplace_back(gsl::make_span(dims));
      if (expected_rank > 1) {
        shape_uniforms.emplace_back(gsl::make_span(stride));
      }
    }
    return Status::OK();
  };

  for (size_t i = 0; i < inputs.size(); i++) {
    ORT_RETURN_IF_ERROR(append_shape_uniforms(i,
                                              inputs[i].use_override_shape ? inputs[i].override_shape : inputs[i].tensor->Shape()));
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    ORT_RETURN_IF_ERROR(append_shape_uniforms(i + inputs.size(),
                                              outputs[i].use_override_shape ? outputs[i].override_shape : outputs[i].tensor->Shape()));
  }
  for (size_t i = 0; i < program.Indices().size(); i++) {
    ORT_RETURN_IF_ERROR(append_shape_uniforms(i + inputs.size() + outputs.size(), program.Indices()[i]));
  }

  const size_t uniform_count = shape_uniforms.size() + program.UniformVariables().size();
  size_t current_offset = 0;
  std::vector<std::tuple<const ProgramUniformVariableValue&, size_t>> uniform_and_offsets;
  uniform_and_offsets.reserve(uniform_count);
  for (size_t i = 0; i < uniform_count; i++) {
    const auto& uniform = i < shape_uniforms.size() ? shape_uniforms[i]
                                                    : program.UniformVariables()[i - shape_uniforms.size()];
    size_t length = uniform.length;
    if (length == 0) {  // skip zero-length uniform
      continue;
    }

    bool is_f16 = uniform.data_type == ProgramUniformVariableDataType::Float16;

    size_t element_size = ProgramUniformVariableDataTypeSize[static_cast<int>(uniform.data_type)];
    // https://www.w3.org/TR/WGSL/#alignof
    size_t base_alignment = is_f16
                                ? (length > 4 ? 16 : length > 2 ? 8
                                                                : length * element_size)
                                : (length > 2 ? 16 : length * element_size);
    size_t struct_size = is_f16 && length <= 4 ? length * element_size : 16;

    current_offset = (current_offset + base_alignment - 1) / base_alignment * base_alignment;
    uniform_and_offsets.emplace_back(uniform, current_offset);

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
  constexpr size_t max_alignment_of_field = 16;
  const size_t uniform_buffer_total_size = (current_offset + max_alignment_of_field - 1) / max_alignment_of_field * max_alignment_of_field;

  WGPUBuffer uniform_buffer = nullptr;
  if (uniform_buffer_total_size > 0) {
    std::vector<uint8_t> uniform_data_buffer(uniform_buffer_total_size);

    for (auto const& [uniform, offset] : uniform_and_offsets) {
      memcpy(uniform_data_buffer.data() + offset, uniform.data.data(), uniform.data.size());
    }

    uniform_buffer = buffer_mgr_->Create(uniform_buffer_total_size, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform);
    device_.GetQueue().WriteBuffer(uniform_buffer, 0, uniform_data_buffer.data(), uniform_buffer_total_size);
  }

  const auto& compute_pass_encoder = GetComputePassEncoder();

  WriteTimestamp(num_pending_dispatches_ * 2);

  uint32_t entry_index = 0;
  std::vector<wgpu::BindGroupEntry> bind_group_entries;
  for (const auto& input : inputs) {
    bind_group_entries.push_back({nullptr, entry_index++, reinterpret_cast<WGPUBuffer>(const_cast<void*>(input.tensor->DataRaw()))});
  }
  for (const auto& output : outputs) {
    bind_group_entries.push_back({nullptr, entry_index++, reinterpret_cast<WGPUBuffer>(output.tensor->MutableDataRaw())});
  }
  if (uniform_buffer) {
    bind_group_entries.push_back({nullptr, entry_index++, uniform_buffer});
  }

  wgpu::BindGroupDescriptor bind_group_desc{};
  bind_group_desc.layout = program_artifact->compute_pipeline.GetBindGroupLayout(0);
  bind_group_desc.entryCount = bind_group_entries.size();
  bind_group_desc.entries = bind_group_entries.data();
  bind_group_desc.label = program_artifact->name.c_str();

  auto bind_group = Device().CreateBindGroup(&bind_group_desc);

  // TODO support graph capture

  compute_pass_encoder.SetPipeline(program_artifact->compute_pipeline);
  compute_pass_encoder.SetBindGroup(0, bind_group);
  compute_pass_encoder.DispatchWorkgroups(x, y, z);

  if (uniform_buffer) {
    buffer_mgr_->Release(uniform_buffer);
  }

  WriteTimestamp(num_pending_dispatches_ * 2 + 1);

  ++num_pending_dispatches_;

  if (num_pending_dispatches_ >= max_num_pending_dispatches_ ||
      (is_profiling_ && query_type_ == TimestampQueryType::AtPasses)) {
    EndComputePass();
  }
  if (num_pending_dispatches_ >= max_num_pending_dispatches_) {
    Flush();
    num_pending_dispatches_ = 0;
  }

  return Status::OK();
}

std::vector<const char*> WebGpuContext::GetEnabledAdapterToggles() const {
  // See the description of all the toggles in toggles.cpp
  // "use_dxc" for Shader Model 6+ features (e.g. float16)
  // "allow_unsafe_apis" for chromium experimental features
  constexpr const char* toggles[] = {
      "use_dxc",
      "allow_unsafe_apis",
  };
  return std::vector<const char*>(std::begin(toggles), std::end(toggles));
}

std::vector<const char*> WebGpuContext::GetEnabledDeviceToggles() const {
  // Enable / disable other toggles that may affect the performance.
  // Other toggles that may be useful: "dump_shaders", "disable_symbol_renaming"
  constexpr const char* toggles[] = {
      "skip_validation",  // only use "skip_validation" when ValidationMode is set to "Disabled"
      "disable_robustness",
      "d3d_disable_ieee_strictness",
  };
  return std::vector<const char*>(ValidationMode() >= ValidationMode::WGPUOnly
                                      ? std::begin(toggles) + 1
                                      : std::begin(toggles),
                                  std::end(toggles));
}

std::vector<const char*> WebGpuContext::GetDisabledDeviceToggles() const {
  constexpr const char* toggles[] = {
      "lazy_clear_resource_on_first_use",
      "timestamp_quantization",
  };
  return std::vector<const char*>(std::begin(toggles), std::end(toggles));
}

std::vector<wgpu::FeatureName> WebGpuContext::GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) const {
  std::vector<wgpu::FeatureName> required_features;
  constexpr wgpu::FeatureName features[]{
#if !defined(__wasm__)
      wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses,
#endif
      wgpu::FeatureName::TimestampQuery,
      wgpu::FeatureName::ShaderF16,
      wgpu::FeatureName::Subgroups,
      wgpu::FeatureName::SubgroupsF16};
  for (auto feature : features) {
    if (adapter.HasFeature(feature)) {
      required_features.push_back(feature);
    }
  }
  return required_features;
}

wgpu::RequiredLimits WebGpuContext::GetRequiredLimits(const wgpu::Adapter& adapter) const {
  wgpu::RequiredLimits required_limits{};
  wgpu::SupportedLimits adapter_limits;
  ORT_ENFORCE(adapter.GetLimits(&adapter_limits));

  required_limits.limits.maxBindGroups = adapter_limits.limits.maxBindGroups;
  required_limits.limits.maxComputeWorkgroupStorageSize = adapter_limits.limits.maxComputeWorkgroupStorageSize;
  required_limits.limits.maxComputeWorkgroupsPerDimension = adapter_limits.limits.maxComputeWorkgroupsPerDimension;
  required_limits.limits.maxStorageBufferBindingSize = adapter_limits.limits.maxStorageBufferBindingSize;
  required_limits.limits.maxBufferSize = adapter_limits.limits.maxBufferSize;
  required_limits.limits.maxComputeInvocationsPerWorkgroup = adapter_limits.limits.maxComputeInvocationsPerWorkgroup;
  required_limits.limits.maxComputeWorkgroupSizeX = adapter_limits.limits.maxComputeWorkgroupSizeX;
  required_limits.limits.maxComputeWorkgroupSizeY = adapter_limits.limits.maxComputeWorkgroupSizeY;
  required_limits.limits.maxComputeWorkgroupSizeZ = adapter_limits.limits.maxComputeWorkgroupSizeZ;

  return required_limits;
}

void WebGpuContext::WriteTimestamp(uint32_t query_index) {
  if (!is_profiling_ || query_type_ != TimestampQueryType::InsidePasses) {
    return;
  }

  const auto& compute_pass_encoder = GetComputePassEncoder();
  compute_pass_encoder.WriteTimestamp(query_set_, query_index);
}

void WebGpuContext::StartProfiling() {
  if (query_type_ == TimestampQueryType::None) {
    return;
  }

  is_profiling_ = true;

  const uint32_t query_count = max_num_pending_dispatches_ * 2;

  if (!query_set_) {
    // Create query set
    wgpu::QuerySetDescriptor querySetDescriptor;
    querySetDescriptor.count = query_count;
    querySetDescriptor.type = wgpu::QueryType::Timestamp;
    query_set_ = device_.CreateQuerySet(&querySetDescriptor);
  }

  if (!query_resolve_buffer_) {
    // Create resolve buffer
    wgpu::BufferDescriptor bufferDescriptor;
    bufferDescriptor.size = query_count * sizeof(uint64_t);
    bufferDescriptor.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc |
                             wgpu::BufferUsage::CopyDst;
    query_resolve_buffer_ = device_.CreateBuffer(&bufferDescriptor);
  }
}

void WebGpuContext::CollectProfilingData(profiling::Events& events) {
  if (!pending_queries_.empty()) {
    for (const auto& pending_query : pending_queries_) {
      const auto& pending_kernels = pending_query.kernels;
      const auto& query_read_buffer = pending_query.query_buffer;

      ORT_ENFORCE(Wait(query_read_buffer.MapAsync(wgpu::MapMode::Read,
                                                  0,
                                                  static_cast<size_t>(query_read_buffer.GetSize()),
                                                  wgpu::CallbackMode::WaitAnyOnly,
                                                  [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                                                    ORT_ENFORCE(status == wgpu::MapAsyncStatus::Success, "Failed to download data from buffer: ", std::string_view{message});
                                                  })) == Status::OK());
      auto mapped_data = static_cast<const uint64_t*>(query_read_buffer.GetConstMappedRange());

      for (size_t i = 0; i < pending_kernels.size(); i++) {
        const PendingKernelInfo& pending_kernel_info = pending_kernels[i];
        const auto& inputs = pending_kernel_info.inputs;
        const auto& outputs = pending_kernel_info.outputs;

        SS(shapes, 128);
        for (size_t s = 0; s < inputs.size(); s++) {
          const auto& input = inputs[s];
          shapes << "inputs[" << s << "] = " << input.override_shape.ToString() << " ";
        }
        for (size_t s = 0; s < outputs.size(); s++) {
          const auto& output = outputs[s];
          shapes << "outputs[" << s << "] = " << output.override_shape.ToString() << " ";
        }

        if (gpu_timestamp_offset_ == 0) {
          gpu_timestamp_offset_ = mapped_data[i * 2];
          // TODO: apply CPU-GPU time offset so that timestamps are aligned
        }
        uint64_t start_time = mapped_data[i * 2] - gpu_timestamp_offset_;
        uint64_t end_time = mapped_data[i * 2 + 1] - gpu_timestamp_offset_;

        const std::unordered_map<std::string, std::string>& event_args = {
            {"shapes", SS_GET(shapes)},
            {"cache_key", pending_kernel_info.cache_key},
        };

        profiling::EventRecord event(profiling::API_EVENT,
                                     -1,
                                     -1,
                                     pending_kernel_info.name,
                                     static_cast<int64_t>(std::round(start_time / 1000.0)),
                                     static_cast<int64_t>(std::round((end_time - start_time) / 1000.0)),
                                     event_args);
        events.emplace_back(std::move(event));
      }

      query_read_buffer.Unmap();
      query_read_buffer.Destroy();
    }

    pending_queries_.clear();
  }

  is_profiling_ = false;
}

void WebGpuContext::EndProfiling(TimePoint /* tp */, profiling::Events& events, profiling::Events& cached_events) {
  // This function is called when no active inference is ongoing.
  ORT_ENFORCE(!is_profiling_, "Profiling is ongoing in an inference run.");

  if (query_type_ != TimestampQueryType::None) {
    // No pending kernels or queries should be present at this point. They should have been collected in CollectProfilingData.
    ORT_ENFORCE(pending_kernels_.empty() && pending_queries_.empty(), "Pending kernels or queries are not empty.");

    events.insert(events.end(),
                  std::make_move_iterator(cached_events.begin()),
                  std::make_move_iterator(cached_events.end()));

    cached_events.clear();
  } else {
    LOGS_DEFAULT(WARNING) << "TimestampQuery is not supported in this device.";
  }
}

void WebGpuContext::Flush() {
  if (!current_command_encoder_) {
    return;
  }

  EndComputePass();

  if (is_profiling_ && num_pending_dispatches_ > 0) {
    uint32_t query_count = num_pending_dispatches_ * 2;
    current_command_encoder_.ResolveQuerySet(
        query_set_,
        0,
        query_count,
        query_resolve_buffer_,
        0);

    wgpu::BufferDescriptor bufferDescriptor;
    bufferDescriptor.size = query_count * sizeof(uint64_t);
    bufferDescriptor.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer query_read_buffer = device_.CreateBuffer(&bufferDescriptor);

    current_command_encoder_.CopyBufferToBuffer(
        query_resolve_buffer_,
        0,
        query_read_buffer,
        0,
        query_count * sizeof(uint64_t));

    pending_queries_.emplace_back(std::move(pending_kernels_), query_read_buffer);
    pending_kernels_.clear();
  }

  auto command_buffer = current_command_encoder_.Finish();
  Device().GetQueue().Submit(1, &command_buffer);
  BufferManager().RefreshPendingBuffers();
  current_command_encoder_ = nullptr;
  num_pending_dispatches_ = 0;
}

std::unordered_map<int32_t, WebGpuContextFactory::WebGpuContextInfo> WebGpuContextFactory::contexts_;
std::mutex WebGpuContextFactory::mutex_;
std::once_flag WebGpuContextFactory::init_default_flag_;
wgpu::Instance WebGpuContextFactory::default_instance_;

WebGpuContext& WebGpuContextFactory::CreateContext(const WebGpuContextConfig& config) {
  const int context_id = config.context_id;
  WGPUInstance instance = config.instance;
  WGPUAdapter adapter = config.adapter;
  WGPUDevice device = config.device;

  if (context_id == 0) {
    // context ID is preserved for the default context. User cannot use context ID 0 as a custom context.
    ORT_ENFORCE(instance == nullptr && adapter == nullptr && device == nullptr,
                "WebGPU EP default context (contextId=0) must not have custom WebGPU instance, adapter or device.");

    std::call_once(init_default_flag_, [
#if !defined(__wasm__)
                                           dawn_proc_table = config.dawn_proc_table
#endif
    ]() {
    // Step.1 - setup dawn proc table (only for non-WASM build)

#if !defined(__wasm__)
      const DawnProcTable* dawn_procs = reinterpret_cast<const DawnProcTable*>(dawn_proc_table);
#if defined(BUILD_DAWN_MONOLITHIC_LIBRARY)
      ORT_ENFORCE(dawn_procs == nullptr, "setting DawnProcTable is not allowed when dynamically linked to webgpu_dawn.");
#else
#if !defined(USE_EXTERNAL_DAWN)
      if (dawn_procs == nullptr) {
        dawn_procs = &dawn::native::GetProcs();
      }
#else
      ORT_ENFORCE(dawn_procs != nullptr, "DawnProcTable must be provided.");
#endif
      dawnProcSetProcs(dawn_procs);
#endif
#endif

      // Step.2 - Create wgpu::Instance
#if !defined(__wasm__)
      wgpu::InstanceDescriptor instance_desc{};
      instance_desc.features.timedWaitAnyEnable = true;
      default_instance_ = wgpu::CreateInstance(&instance_desc);
#else
      default_instance_ = wgpu::CreateInstance(nullptr);
#endif

      ORT_ENFORCE(default_instance_ != nullptr, "Failed to create wgpu::Instance.");
    });
    instance = default_instance_.Get();
  } else {
    // for context ID > 0, user must provide custom WebGPU instance, adapter and device.
    ORT_ENFORCE(instance != nullptr && adapter != nullptr && device != nullptr,
                "WebGPU EP custom context (contextId>0) must have custom WebGPU instance, adapter and device.");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = contexts_.find(context_id);
  if (it == contexts_.end()) {
    GSL_SUPPRESS(r.11)
    auto context = std::unique_ptr<WebGpuContext>(new WebGpuContext(instance, adapter, device, config.validation_mode));
    it = contexts_.emplace(context_id, WebGpuContextFactory::WebGpuContextInfo{std::move(context), 0}).first;
  } else if (context_id != 0) {
    ORT_ENFORCE(it->second.context->instance_.Get() == instance &&
                    it->second.context->adapter_.Get() == adapter &&
                    it->second.context->device_.Get() == device,
                "WebGPU EP context ID ", context_id, " is already created with different WebGPU instance, adapter or device.");
  }
  it->second.ref_count++;
  return *it->second.context;
}

WebGpuContext& WebGpuContextFactory::GetContext(int context_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = contexts_.find(context_id);
  ORT_ENFORCE(it != contexts_.end(), "WebGPU EP context ID ", context_id, " is not found.");

  return *it->second.context;
}

void WebGpuContextFactory::ReleaseContext(int context_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = contexts_.find(context_id);
  ORT_ENFORCE(it != contexts_.end(), "WebGPU EP context ID ", context_id, " is not found.");

  if (--it->second.ref_count == 0) {
    contexts_.erase(it);
  }
}

void WebGpuContextFactory::Cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);
  contexts_.clear();
  default_instance_ = nullptr;
}

void CleanupWebGpuContexts() {
  WebGpuContextFactory::Cleanup();
}

}  // namespace webgpu
}  // namespace onnxruntime
