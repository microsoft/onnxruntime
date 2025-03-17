// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include <webgpu/webgpu_cpp.h>

#include "core/common/common.h"
#include "core/framework/library_handles.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/program_manager.h"

namespace onnxruntime {
class Tensor;

namespace webgpu {
class WebGpuContext;
class ComputeContext;
class ProgramBase;

struct WebGpuContextConfig {
  int context_id;
  WGPUInstance instance;
  WGPUAdapter adapter;
  WGPUDevice device;
  const void* dawn_proc_table;
  ValidationMode validation_mode;
};

struct WebGpuBufferCacheConfig {
  struct ConfigEntry {
    BufferCacheMode mode;
    std::string config_string;
  };
  ConfigEntry storage;
  ConfigEntry uniform;
  ConfigEntry query_resolve;
  ConfigEntry default_entry;
};

class WebGpuContextFactory {
 public:
  struct WebGpuContextInfo {
    std::unique_ptr<WebGpuContext> context;
    int ref_count;
  };

  static WebGpuContext& CreateContext(const WebGpuContextConfig& config);
  static WebGpuContext& GetContext(int context_id);

  static void ReleaseContext(int context_id);

  static void Cleanup();

 private:
  WebGpuContextFactory() {}

  static std::unordered_map<int32_t, WebGpuContextInfo> contexts_;
  static std::mutex mutex_;
  static std::once_flag init_default_flag_;
  static wgpu::Instance default_instance_;
};

// Class WebGpuContext includes all necessary resources for the context.
class WebGpuContext final {
 public:
  void Initialize(const WebGpuBufferCacheConfig& buffer_cache_config, int backend_type);

  Status Wait(wgpu::Future f);

  const wgpu::Adapter& Adapter() const { return adapter_; }
  const wgpu::Device& Device() const { return device_; }

  const wgpu::AdapterInfo& AdapterInfo() const { return adapter_info_; }
  const wgpu::Limits& DeviceLimits() const { return device_limits_; }

  const wgpu::CommandEncoder& GetCommandEncoder() {
    if (!current_command_encoder_) {
      current_command_encoder_ = device_.CreateCommandEncoder();
    }
    return current_command_encoder_;
  }

  const wgpu::ComputePassEncoder& GetComputePassEncoder() {
    if (!current_compute_pass_encoder_) {
      auto& command_encoder = GetCommandEncoder();

      wgpu::ComputePassDescriptor compute_pass_desc{};

      if (is_profiling_ && query_type_ == TimestampQueryType::AtPasses) {
        wgpu::ComputePassTimestampWrites timestampWrites = {
            query_set_, num_pending_dispatches_ * 2, num_pending_dispatches_ * 2 + 1};
        compute_pass_desc.timestampWrites = &timestampWrites;
      }

      current_compute_pass_encoder_ = command_encoder.BeginComputePass(&compute_pass_desc);
    }
    return current_compute_pass_encoder_;
  }

  void EndComputePass() {
    if (current_compute_pass_encoder_) {
      current_compute_pass_encoder_.End();
      current_compute_pass_encoder_ = nullptr;
    }
  }

  void Flush();

  webgpu::BufferManager& BufferManager() const { return *buffer_mgr_; }

  inline webgpu::ValidationMode ValidationMode() const {
    return validation_mode_;
  }

  void StartProfiling();
  void CollectProfilingData(profiling::Events& events);
  void EndProfiling(TimePoint, profiling::Events& events, profiling::Events& cached_events);

  Status Run(ComputeContext& context, const ProgramBase& program);

 private:
  enum class TimestampQueryType {
    None = 0,
    InsidePasses,
    AtPasses
  };

  WebGpuContext(WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device, webgpu::ValidationMode validation_mode)
      : instance_{instance}, adapter_{adapter}, device_{device}, validation_mode_{validation_mode}, query_type_{TimestampQueryType::None} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuContext);

  std::vector<const char*> GetEnabledAdapterToggles() const;
  std::vector<const char*> GetEnabledDeviceToggles() const;
  std::vector<const char*> GetDisabledDeviceToggles() const;
  std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) const;
  wgpu::RequiredLimits GetRequiredLimits(const wgpu::Adapter& adapter) const;
  void WriteTimestamp(uint32_t query_index);

  struct PendingKernelInfo {
    PendingKernelInfo(std::string_view kernel_name,
                      std::string_view kernel_type,
                      std::string_view program_name,
                      std::string_view cache_key,
                      const std::vector<ProgramInput>& inputs,
                      const std::vector<ProgramOutput>& outputs)
        : name{absl::StrJoin({kernel_name, kernel_type, program_name}, "&")}, cache_key{cache_key}, inputs{inputs}, outputs{outputs} {}

    PendingKernelInfo(PendingKernelInfo&&) = default;
    PendingKernelInfo& operator=(PendingKernelInfo&&) = default;
    ORT_DISALLOW_COPY_AND_ASSIGNMENT(PendingKernelInfo);

    std::string name;
    std::string cache_key;
    std::vector<ProgramInput> inputs;
    std::vector<ProgramOutput> outputs;
  };

  struct PendingQueryInfo {
    PendingQueryInfo(std::vector<PendingKernelInfo>&& kernels, wgpu::Buffer query_buffer)
        : kernels{std::move(kernels)}, query_buffer{query_buffer} {}

    PendingQueryInfo(PendingQueryInfo&&) = default;
    PendingQueryInfo& operator=(PendingQueryInfo&&) = default;
    ORT_DISALLOW_COPY_AND_ASSIGNMENT(PendingQueryInfo);

    std::vector<PendingKernelInfo> kernels;
    wgpu::Buffer query_buffer;
  };

  friend class WebGpuContextFactory;

  std::once_flag init_flag_;

  LibraryHandles modules_;

  wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;

  webgpu::ValidationMode validation_mode_;

  wgpu::AdapterInfo adapter_info_;
  wgpu::Limits device_limits_;

  wgpu::CommandEncoder current_command_encoder_;
  wgpu::ComputePassEncoder current_compute_pass_encoder_;

  std::unique_ptr<webgpu::BufferManager> buffer_mgr_;
  std::unique_ptr<ProgramManager> program_mgr_;

  uint32_t num_pending_dispatches_ = 0;
  const uint32_t max_num_pending_dispatches_ = 16;

  // profiling
  TimestampQueryType query_type_;
  wgpu::QuerySet query_set_;
  wgpu::Buffer query_resolve_buffer_;

  // info of kernels pending submission for a single batch
  std::vector<PendingKernelInfo> pending_kernels_;
  // info of queries pending appending to profiling events
  std::vector<PendingQueryInfo> pending_queries_;

  uint64_t gpu_timestamp_offset_ = 0;
  bool is_profiling_ = false;
};

}  // namespace webgpu
}  // namespace onnxruntime
