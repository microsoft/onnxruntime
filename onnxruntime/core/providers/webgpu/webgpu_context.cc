// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <cmath>

#include "core/common/common.h"

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/program_cache_key.h"
#include "core/providers/webgpu/program_manager.h"

namespace onnxruntime {
namespace webgpu {

namespace {

std::vector<const char*> GetEnabledAdapterToggles() {
  // See the description of all the toggles in toggles.cpp
  // "use_dxc" for Shader Model 6+ features (e.g. float16)
  // "allow_unsafe_apis" for chromium experimental features
  constexpr const char* toggles[] = {
      "use_dxc",
      "allow_unsafe_apis",
  };
  return std::vector<const char*>(std::begin(toggles), std::end(toggles));
}

std::vector<const char*> GetEnabledDeviceToggles() {
  // Enable / disable other toggles that may affect the performance.
  // Other toggles that may be useful: "dump_shaders", "disable_symbol_renaming"
  constexpr const char* toggles[] = {
      "skip_validation",
      "disable_robustness",
      "disable_workgroup_init",
      "d3d_disable_ieee_strictness",
  };
  return std::vector<const char*>(std::begin(toggles), std::end(toggles));
}

std::vector<const char*> GetDisabledDeviceToggles() {
  constexpr const char* toggles[] = {
      "lazy_clear_resource_on_first_use",
  };
  return std::vector<const char*>(std::begin(toggles), std::end(toggles));
}

std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) {
  std::vector<wgpu::FeatureName> required_features;
  constexpr wgpu::FeatureName features[]{
      wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses,
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

wgpu::RequiredLimits GetRequiredLimits(const wgpu::Adapter& adapter) {
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

}  // namespace

void WebGpuContext::Initialize(const WebGpuExecutionProviderInfo& webgpu_ep_info) {
  std::call_once(init_flag_, [this, &webgpu_ep_info]() {
    // Initialization.Step.1 - Create wgpu::Instance
    if (instance_ == nullptr) {
      wgpu::InstanceDescriptor instance_desc{};
      instance_desc.features.timedWaitAnyEnable = true;
      instance_ = wgpu::CreateInstance(&instance_desc);

      ORT_ENFORCE(instance_ != nullptr, "Failed to create wgpu::Instance.");
    }

    // Initialization.Step.2 - Create wgpu::Adapter
    if (adapter_ == nullptr) {
      wgpu::RequestAdapterOptions req_adapter_options = {};
      wgpu::DawnTogglesDescriptor adapter_toggles_desc = {};
      req_adapter_options.nextInChain = &adapter_toggles_desc;

      auto enabled_adapter_toggles = GetEnabledAdapterToggles();
      adapter_toggles_desc.enabledToggleCount = enabled_adapter_toggles.size();
      adapter_toggles_desc.enabledToggles = enabled_adapter_toggles.data();

      wgpu::RequestAdapterCallbackInfo req_adapter_callback_info = {};
      req_adapter_callback_info.mode = wgpu::CallbackMode::WaitAnyOnly;
      req_adapter_callback_info.callback = [](WGPURequestAdapterStatus status,
                                              WGPUAdapter adapter, const char* message,
                                              void* userdata) {
        ORT_ENFORCE(status == WGPURequestAdapterStatus_Success, "Failed to get a WebGPU adapter: ", message);
        *static_cast<wgpu::Adapter*>(userdata) = wgpu::Adapter::Acquire(adapter);
      };
      req_adapter_callback_info.userdata = &adapter_;
      ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(instance_.RequestAdapter(&req_adapter_options, req_adapter_callback_info), UINT64_MAX));
      ORT_ENFORCE(adapter_ != nullptr, "Failed to get a WebGPU adapter.");
    }

    // Initialization.Step.3 - Create wgpu::Device
    if (device_ == nullptr) {
      wgpu::DeviceDescriptor device_desc = {};
      wgpu::DawnTogglesDescriptor device_toggles_desc = {};
      device_desc.nextInChain = &device_toggles_desc;

      auto enabled_device_toggles = GetEnabledDeviceToggles();
      device_toggles_desc.enabledToggleCount = enabled_device_toggles.size();
      device_toggles_desc.enabledToggles = enabled_device_toggles.data();

      auto disabled_device_toggles = GetDisabledDeviceToggles();
      device_toggles_desc.disabledToggleCount = disabled_device_toggles.size();
      device_toggles_desc.disabledToggles = disabled_device_toggles.data();

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
        // cannot use ORT logger because it may be already destroyed
        std::cerr << "WebGPU device lost (" << int(reason) << "): " << message;
      });

      wgpu::RequestDeviceCallbackInfo req_device_callback_info = {};
      req_device_callback_info.mode = wgpu::CallbackMode::WaitAnyOnly;
      req_device_callback_info.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* userdata) {
        ORT_ENFORCE(status == WGPURequestDeviceStatus_Success, "Failed to get a WebGPU device: ", message);
        *static_cast<wgpu::Device*>(userdata) = wgpu::Device::Acquire(device);
      };
      req_device_callback_info.userdata = &device_;
      ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(adapter_.RequestDevice(&device_desc, req_device_callback_info), UINT64_MAX));
      ORT_ENFORCE(device_ != nullptr, "Failed to get a WebGPU device.");
    }

    // cache adapter info
    ORT_ENFORCE(Adapter().GetInfo(&adapter_info_));
    // cache device limits
    wgpu::SupportedLimits device_supported_limits;
    ORT_ENFORCE(Device().GetLimits(&device_supported_limits));
    device_limits_ = device_supported_limits.limits;

    // create buffer manager
    buffer_mgr_ = BufferManagerFactory::Create(*this, webgpu_ep_info.storage_buffer_cache_mode, webgpu_ep_info.uniform_buffer_cache_mode, webgpu_ep_info.query_resolve_buffer_cache_mode);

    // create program manager
    program_mgr_ = std::make_unique<ProgramManager>(Device(), DeviceLimits());
  });
}

Status WebGpuContext::Wait(wgpu::Future f) {
  auto status = instance_.WaitAny(f, UINT64_MAX);
  if (status == wgpu::WaitStatus::Success) {
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to wait for the operation:", uint32_t(status));
}

Status WebGpuContext::Run(const ComputeContext& context, const ProgramBase& program) {
  const auto& inputs = program.Inputs();
  const auto& outputs = program.Outputs();

#ifndef NDEBUG  // if debug build
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
#endif

  if (outputs.size() == 0) {
    return Status::OK();
  }

  const ProgramMetadata metadata = program.GetMetadata();

  // validate program metadata
  {
    const auto& [constants, overridable_constants, uniform_variables] = metadata;

    // check overridable constants
    ORT_RETURN_IF(program.OverridableConstants().size() != overridable_constants.size(),
                  "Size of overridable constants mismatch in program \"", program.Name(),
                  "\", Expected: ", overridable_constants.size(),
                  ", Actual: ", program.OverridableConstants().size());
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

    // check uniform variables
    ORT_RETURN_IF(program.UniformVariables().size() != uniform_variables.size(),
                  "Size of uniform_value variables mismatch in program \"", program.Name(),
                  "\", Expected: ", uniform_variables.size(),
                  ", Actual: ", program.UniformVariables().size());
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

  uint32_t x = program.DispatchGroupSizeX();
  uint32_t y = program.DispatchGroupSizeY();
  uint32_t z = program.DispatchGroupSizeZ();
  ORT_RETURN_IF_ERROR(program_mgr_->NormalizeDispatchGroupSize(x, y, z));

  bool is_1d_dispatch = (y == 1 && z == 1);

  auto key = CalculateProgramCacheKey(program, is_1d_dispatch);

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
  ORT_RETURN_IF_NOT(program_artifact->shape_uniform_ranks.size() == inputs.size() + outputs.size(),
                    "Invalid program artifact: variable size (", program_artifact->shape_uniform_ranks.size(),
                    ") does not match current program (input: ", inputs.size(), ", output: ", outputs.size(), ")");
  for (size_t i = 0; i < program_artifact->shape_uniform_ranks.size(); ++i) {
    SafeInt<int> expected_rank = program_artifact->shape_uniform_ranks[i];
    if (expected_rank > 0) {
      const auto& shape = i < inputs.size() ? (inputs[i].use_override_shape ? inputs[i].override_shape
                                                                            : inputs[i].tensor->Shape())
                                            : (outputs[i - inputs.size()].use_override_shape ? outputs[i - inputs.size()].override_shape
                                                                                             : outputs[i - inputs.size()].tensor->Shape());
      ORT_RETURN_IF(expected_rank != shape.NumDimensions(),
                    "Invalid program artifact: variable[", i, "] rank mismatch. Expected: ", (int)expected_rank,
                    ", Actual: ", shape.NumDimensions());

      std::vector<uint32_t> dims(expected_rank);
      std::vector<uint32_t> stride(expected_rank - 1);
      for (size_t j = 0; j < expected_rank; ++j) {
        dims[j] = SafeInt<uint32_t>(shape[j]);
        if (j < expected_rank - 1) {
          stride[j] = SafeInt<uint32_t>(shape.SizeFromDimension(j + 1));
        }
      }

      shape_uniforms.emplace_back(gsl::make_span(dims));
      if (expected_rank > 1) {
        shape_uniforms.emplace_back(gsl::make_span(stride));
      }
    }
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
  const size_t max_alignment_of_field = 16;
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

  // TODO: write timestamp query

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

  // TODO: write timestamp query

  ++num_pending_dispatches_;

  // if (querytype == at-passes) { EndComputePass(); }

  if (num_pending_dispatches_ >= max_num_pending_dispatches_) {
    Flush();
  }

  return Status::OK();
}

std::unordered_map<int32_t, std::unique_ptr<WebGpuContext>> WebGpuContextFactory::contexts_;
std::mutex WebGpuContextFactory::mutex_;

WebGpuContext& WebGpuContextFactory::CreateContext(int context_id, WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device) {
  if (context_id == 0) {
    // context ID is preserved for the default context. User cannot use context ID 0 as a custom context.
    ORT_ENFORCE(instance == nullptr && adapter == nullptr && device == nullptr,
                "WebGPU EP default context (contextId=0) must not have custom WebGPU instance, adapter or device.");
  } else {
    // for context ID > 0, user must provide custom WebGPU instance, adapter and device.
    ORT_ENFORCE(instance != nullptr && adapter != nullptr && device != nullptr,
                "WebGPU EP custom context (contextId>0) must have custom WebGPU instance, adapter and device.");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = contexts_.find(context_id);
  if (it == contexts_.end()) {
    auto context = std::unique_ptr<WebGpuContext>(new WebGpuContext(instance, adapter, device));
    it = contexts_.emplace(context_id, std::move(context)).first;
  } else if (context_id != 0) {
    ORT_ENFORCE(it->second->instance_.Get() == instance && it->second->adapter_.Get() == adapter && it->second->device_.Get() == device,
                "WebGPU EP context ID ", context_id, " is already created with different WebGPU instance, adapter or device.");
  }
  return *it->second;
}

WebGpuContext& WebGpuContextFactory::GetContext(int context_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = contexts_.find(context_id);
  ORT_ENFORCE(it != contexts_.end(), "WebGPU EP context ID ", context_id, " is not found.");

  return *it->second;
}

}  // namespace webgpu
}  // namespace onnxruntime
