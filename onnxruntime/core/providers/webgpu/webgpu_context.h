// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include "core/providers/webgpu/webgpu_external_header.h"

#include "core/common/common.h"
#include "core/framework/library_handles.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/program_manager.h"

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
#include "core/providers/webgpu/webgpu_pix_frame_generator.h"
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

namespace onnxruntime {
class Tensor;

namespace webgpu {
class WebGpuContext;
class ComputeContext;
class ProgramBase;

// Definition for CapturedCommandInfo in the webgpu namespace
struct CapturedCommandInfo {
  wgpu::ComputePipeline compute_pipeline;
  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;
  std::array<uint32_t, 3> dispatch_group;
};

struct WebGpuContextConfig {
  int context_id;
  WGPUInstance instance;
  WGPUDevice device;
  const void* dawn_proc_table;
  ValidationMode validation_mode;
  bool preserve_device;
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
  void Initialize(const WebGpuBufferCacheConfig& buffer_cache_config, int backend_type, bool enable_pix_capture);

  Status Wait(wgpu::Future f);

  const wgpu::Device& Device() const { return device_; }

  const wgpu::AdapterInfo& AdapterInfo() const { return adapter_info_; }
  const wgpu::Limits& DeviceLimits() const { return device_limits_; }
  bool DeviceHasFeature(wgpu::FeatureName feature) const { return device_features_.find(feature) != device_features_.end(); }
#if !defined(__wasm__)
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& SubgroupMatrixConfigs() const { return subgroup_matrix_configs_; }
#endif

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
        wgpu::PassTimestampWrites timestampWrites = {
            nullptr,
            query_set_,
            num_pending_dispatches_ * 2,
            num_pending_dispatches_ * 2 + 1};
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
  void CaptureBegin(std::vector<webgpu::CapturedCommandInfo>* captured_commands);
  void CaptureEnd();
  void Replay(const std::vector<webgpu::CapturedCommandInfo>& captured_commands);
  void ReleaseGraphResources(std::vector<webgpu::CapturedCommandInfo>& captured_commands);

  void Flush();

  webgpu::BufferManager& BufferManager() const { return *buffer_mgr_; }
  void SetExternalBufferManager(webgpu::BufferManager* buffer_mgr);

  inline webgpu::ValidationMode ValidationMode() const {
    return validation_mode_;
  }

  void StartProfiling();
  void CollectProfilingData(profiling::Events& events);
  void EndProfiling(TimePoint, profiling::Events& events, profiling::Events& cached_events);

  //
  // Push error scope.
  //
  // This is useful only when "skip_validation" is not set.
  //
  void PushErrorScope();

  //
  // Pop error scope.
  //
  // This is useful only when "skip_validation" is not set.
  //
  Status PopErrorScope();

  Status Run(ComputeContext& context, const ProgramBase& program);
  void OnRunEnd();

 private:
  enum class TimestampQueryType {
    None = 0,
    InsidePasses,
    AtPasses
  };

  WebGpuContext(WGPUInstance instance, WGPUDevice device, webgpu::ValidationMode validation_mode, bool preserve_device)
      : instance_{instance}, device_{device}, validation_mode_{validation_mode}, query_type_{TimestampQueryType::None}, preserve_device_{preserve_device} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuContext);

  void LaunchComputePipeline(const wgpu::ComputePassEncoder& compute_pass_encoder,
                             const std::vector<WGPUBuffer>& bind_buffers,
                             const ProgramArtifact& program_artifact,
                             uint32_t x, uint32_t y, uint32_t z);

  std::vector<const char*> GetEnabledAdapterToggles() const;
  std::vector<const char*> GetEnabledDeviceToggles() const;
  std::vector<const char*> GetDisabledDeviceToggles() const;
  std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) const;
  wgpu::Limits GetRequiredLimits(const wgpu::Adapter& adapter) const;
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
  wgpu::Device device_;

  webgpu::ValidationMode validation_mode_;

  wgpu::Queue device_queue_;
  wgpu::AdapterInfo adapter_info_;
  wgpu::Limits device_limits_;
  std::unordered_set<wgpu::FeatureName> device_features_;
#if !defined(__wasm__)
  wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs_;
#endif

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
  bool preserve_device_;
  SessionState session_status_{SessionState::Default};

  // External vector to store captured commands, owned by EP
  std::vector<webgpu::CapturedCommandInfo>* external_captured_commands_ = nullptr;  // External buffer manager for graph mode, owned by EP
  webgpu::BufferManager* external_buffer_mgr_ = nullptr;

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
  std::unique_ptr<WebGpuPIXFrameGenerator> pix_frame_generator_ = nullptr;
#endif  // ENABLE_PIX_FOR_WEBGPU_EP
};

}  // namespace webgpu
}  // namespace onnxruntime
