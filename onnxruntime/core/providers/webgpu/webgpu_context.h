// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <map>
#include <memory>
#include <mutex>

#include <webgpu/webgpu_cpp.h>

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/program_manager.h"

namespace onnxruntime {
class Tensor;

namespace webgpu {
class WebGpuContext;
class ComputeContext;
class ProgramBase;

class WebGpuContextFactory {
 public:
  static WebGpuContext& CreateContext(int context_id,
                                      WGPUInstance instance,
                                      WGPUAdapter adapter,
                                      WGPUDevice device,
                                      ValidationMode validation_mode);
  static WebGpuContext& GetContext(int context_id);

 private:
  WebGpuContextFactory() {}

  static std::unordered_map<int32_t, std::unique_ptr<WebGpuContext>> contexts_;
  static OrtMutex mutex_;
};

// Class WebGpuContext includes all necessary resources for the context.
class WebGpuContext final {
 public:
  void Initialize(const WebGpuExecutionProviderInfo& webgpu_ep_info);

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

      if (query_type_ == TimestampQueryType::AtPasses) {
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

  void Flush() {
    if (!current_command_encoder_) {
      return;
    }

    EndComputePass();

    if (query_type_ != TimestampQueryType::None) {
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

      pending_queries_info_.emplace(pending_queries_buffers_.size(), pending_kernels_);
      pending_queries_buffers_.push_back(query_read_buffer);
      pending_kernels_.clear();
    }

    auto command_buffer = current_command_encoder_.Finish();
    Device().GetQueue().Submit(1, &command_buffer);
    BufferManager().RefreshPendingBuffers();
    current_command_encoder_ = nullptr;
    num_pending_dispatches_ = 0;
  }

  webgpu::BufferManager& BufferManager() const { return *buffer_mgr_; }

  inline webgpu::ValidationMode ValidationMode() const {
    return validation_mode_;
  }

  void StartProfiling(TimePoint);
  void EndProfiling(TimePoint, profiling::Events&);

  Status Run(const ComputeContext& context, const ProgramBase& program);

 private:
  enum class TimestampQueryType {
    None = 0,
    InsidePasses,
    AtPasses
  };

  WebGpuContext(WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device, webgpu::ValidationMode validation_mode)
      : query_type_{TimestampQueryType::None}, instance_{instance}, adapter_{adapter}, device_{device}, validation_mode_{validation_mode} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuContext);

  std::vector<const char*> GetEnabledAdapterToggles() const;
  std::vector<const char*> GetEnabledDeviceToggles() const;
  std::vector<const char*> GetDisabledDeviceToggles() const;
  std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) const;
  wgpu::RequiredLimits GetRequiredLimits(const wgpu::Adapter& adapter) const;
  void WriteTimestamp(uint32_t query_index);

  TimestampQueryType query_type_;
  uint64_t query_time_base_;
  wgpu::QuerySet query_set_;
  wgpu::Buffer query_resolve_buffer_;

  struct PendingKernelInfo {
    PendingKernelInfo(std::string program_name, std::vector<ProgramInput> inputs, std::vector<ProgramOutput> outputs)
        : program_name{program_name}, inputs{inputs}, outputs{outputs} {}
    std::string program_name;
    std::vector<ProgramInput> inputs;
    std::vector<ProgramOutput> outputs;
  };
  // info of kernels pending submission for a single batch
  std::vector<PendingKernelInfo> pending_kernels_;
  std::map<size_t, std::vector<PendingKernelInfo>> pending_queries_info_;
  std::vector<wgpu::Buffer> pending_queries_buffers_;

  std::once_flag init_flag_;

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
  friend class WebGpuContextFactory;

  uint32_t num_pending_dispatches_ = 0;
  const uint32_t max_num_pending_dispatches_ = 16;
};

}  // namespace webgpu
}  // namespace onnxruntime
