// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <memory>
#include <mutex>

#include <webgpu/webgpu_cpp.h>

#include "core/common/common.h"
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
  static WebGpuContext& CreateContext(int context_id, WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device);
  static WebGpuContext& GetContext(int context_id);

 private:
  WebGpuContextFactory() {}

  static std::unordered_map<int32_t, std::unique_ptr<WebGpuContext>> contexts_;
  static std::mutex mutex_;
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

      // TODO: add support for GPU Query

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

    // TODO: add support for GPU Query

    auto command_buffer = current_command_encoder_.Finish();
    Device().GetQueue().Submit(1, &command_buffer);
    BufferManager().RefreshPendingBuffers();
    current_command_encoder_ = nullptr;
  }

  webgpu::BufferManager& BufferManager() const { return *buffer_mgr_; }

  Status Run(const ComputeContext& context, const ProgramBase& program);

 private:
  WebGpuContext(WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device) : instance_{instance}, adapter_{adapter}, device_{device} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuContext);

  std::once_flag init_flag_;

  wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;

  wgpu::AdapterInfo adapter_info_;
  wgpu::Limits device_limits_;

  wgpu::CommandEncoder current_command_encoder_;
  wgpu::ComputePassEncoder current_compute_pass_encoder_;

  std::unique_ptr<webgpu::BufferManager> buffer_mgr_;
  std::unique_ptr<ProgramManager> program_mgr_;
  friend class WebGpuContextFactory;

  int num_pending_dispatches_ = 0;
  const int max_num_pending_dispatches_ = 16;
};

}  // namespace webgpu
}  // namespace onnxruntime
