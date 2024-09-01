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

std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) {
  std::vector<wgpu::FeatureName> required_features;
  constexpr wgpu::FeatureName features[]{
      wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses,
      wgpu::FeatureName::TimestampQuery,
      wgpu::FeatureName::ShaderF16};
  for (auto feature : features) {
    if (adapter.HasFeature(feature)) {
      required_features.push_back(feature);
    }
  }
  return required_features;
}

wgpu::RequiredLimits GetAvailableRequiredLimits(const wgpu::Adapter& adapter) {
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
      std::vector<wgpu::FeatureName> required_features = GetAvailableRequiredFeatures(adapter_);
      if (required_features.size() > 0) {
        device_desc.requiredFeatures = required_features.data();
      }
      wgpu::RequiredLimits required_limits = GetAvailableRequiredLimits(adapter_);
      device_desc.requiredLimits = &required_limits;

      // TODO: temporary error handling
      device_desc.SetUncapturedErrorCallback([](const wgpu::Device& /*device*/, wgpu::ErrorType type, const char* message) {
        LOGS_DEFAULT(ERROR) << "WebGPU device error(" << int(type) << "): " << message;
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
                       tensor->Location().name == WEBGPU_BUFFER;
              }),
              "All inputs must be tensors on WebGPU buffers.");

  ORT_ENFORCE(std::all_of(outputs.begin(), outputs.end(), [](Tensor* tensor) {
                return tensor != nullptr &&
                       tensor->Location().mem_type == OrtMemType::OrtMemTypeDefault &&
                       tensor->Location().device.Type() == OrtDevice::GPU &&
                       tensor->Location().name == WEBGPU_BUFFER;
              }),
              "All outputs must be tensors on WebGPU buffers.");
#endif

  if (outputs.size() == 0) {
    return Status::OK();
  }

  ProgramMetadata metadata = program.GetMetadata();

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
    auto status = program_mgr_->Build(program,
                                      metadata,
#ifndef NDEBUG  // if debug build
                                      key,
#endif
                                      x,
                                      y,
                                      z,
                                      compute_pipeline);
    ORT_RETURN_IF_ERROR(status);
    program_artifact = program_mgr_->Set(key, ProgramArtifact{program, std::move(compute_pipeline)});
#ifndef NDEBUG  // if debug build
    ORT_ENFORCE(program_artifact != nullptr, "Program artifact should not be nullptr.");
#endif
  }

  WGPUBuffer uniform_buffer = nullptr;
  auto uniform_buffer_size = program_artifact->uniform_total_size;
  if (uniform_buffer_size > 0) {
    auto num_uniforms = program.UniformVariables().size();
    ORT_ENFORCE(program_artifact->uniforms.size() == num_uniforms,
                "Uniforms size mismatch. Artifact: ", program_artifact->uniforms.size(), ", Current: ", num_uniforms);

    std::vector<uint8_t> uniform_data(uniform_buffer_size);

    for (size_t i = 0; i < num_uniforms; ++i) {
      const auto& uniform = program.UniformVariables()[i];
      const auto& artifact_uniform = program_artifact->uniforms[i];

      ORT_ENFORCE(uniform.data_type == artifact_uniform.data_type,
                  "Uniform[", i, "] data type mismatch. Artifact: ", artifact_uniform.data_type,
                  ", Current: ", uniform.data_type);
      ORT_ENFORCE(uniform.length == artifact_uniform.length,
                  "Uniform[", i, "] elements number mismatch. Artifact: ", artifact_uniform.length, ", Current: ", uniform.length);
      ORT_ENFORCE(uniform.data.size() == artifact_uniform.length * ProgramUniformVariableDataTypeSize[static_cast<int>(uniform.data_type)],
                  "Uniform[", i, "] data size mismatch. Artifact: ", artifact_uniform.length * ProgramUniformVariableDataTypeSize[static_cast<int>(uniform.data_type)],
                  ", Current: ", uniform.data.size());

      auto offset = artifact_uniform.offset;
      auto size = uniform.data.size();
      memcpy(uniform_data.data() + offset, uniform.data.data(), size);
    }

    uniform_buffer = buffer_mgr_->Create(uniform_buffer_size, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform);
    device_.GetQueue().WriteBuffer(uniform_buffer, 0, uniform_data.data(), uniform_buffer_size);
  }

  const auto& compute_pass_encoder = GetComputePassEncoder();

  // TODO: write timestamp query

  uint32_t entry_index = 0;
  std::vector<wgpu::BindGroupEntry> bind_group_entries;
  for (const auto& input : inputs) {
    bind_group_entries.push_back({nullptr, entry_index++, reinterpret_cast<WGPUBuffer>(const_cast<void*>(input.tensor->DataRaw()))});
  }
  for (const auto& output : outputs) {
    bind_group_entries.push_back({nullptr, entry_index++, reinterpret_cast<WGPUBuffer>(output->MutableDataRaw())});
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
