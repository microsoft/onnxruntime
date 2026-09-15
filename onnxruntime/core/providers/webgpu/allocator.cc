// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <utility>

#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

GpuBufferAllocator::GpuBufferAllocator(
    std::function<const BufferManager&()> buffer_manager_getter,
    std::function<CommandRecordingState&()> recording_getter,
    bool is_read_only_allocator,
    std::function<bool()> should_submit_zero_initialize)
    : IAllocator(
          OrtMemoryInfo(WEBGPU_BUFFER,
                        is_read_only_allocator ? OrtAllocatorType::OrtReadOnlyAllocator
                                               : OrtAllocatorType::OrtDeviceAllocator,
                        WebGpuDevice,
                        OrtMemTypeDefault)),
      buffer_manager_getter_{std::move(buffer_manager_getter)},
      recording_getter_{std::move(recording_getter)},
      should_submit_zero_initialize_{std::move(should_submit_zero_initialize)},
      mapped_at_creation_{is_read_only_allocator && buffer_manager_getter_().SupportsUMA()},
      initialize_to_zero_{!is_read_only_allocator} {
}

void* GpuBufferAllocator::Alloc(size_t size) {
  auto& recording = recording_getter_();
  std::lock_guard<std::recursive_mutex> lock{recording.mutex};
  return Allocate(size, should_submit_zero_initialize_ && should_submit_zero_initialize_());
}

void* GpuBufferAllocator::Allocate(size_t size, bool submit_zero_initialize) {
  if (size == 0) {
    return nullptr;
  }

  auto& recording = recording_getter_();
  std::lock_guard<std::recursive_mutex> lock{recording.mutex};
  stats_.num_allocs++;

  wgpu::BufferUsage usage = mapped_at_creation_ ? wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapWrite
                                                : wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;

  return buffer_manager_getter_().Create(recording, size, usage, initialize_to_zero_,
                                         submit_zero_initialize);
}

#if defined(ORT_USE_EP_API_ADAPTERS)
void* GpuBufferAllocator::AllocOnStream(size_t size, Stream* stream) {
  if (stream == nullptr) {
    return Alloc(size);
  }
  ORT_ENFORCE(&GetWebGpuStreamCommandState(reinterpret_cast<OrtSyncStream*>(stream)) == &recording_getter_(),
              "WebGPU allocator and stream belong to different Sessions.");
  return Allocate(size, false);
}

namespace {

struct WebGpuSessionAllocator final : OrtAllocator {
  explicit WebGpuSessionAllocator(AllocatorPtr impl) : OrtAllocator{}, impl_{std::move(impl)} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = [](OrtAllocator* allocator, void* buffer) noexcept {
      static_cast<WebGpuSessionAllocator*>(allocator)->impl_->Free(buffer);
    };
    Info = [](const OrtAllocator* allocator) noexcept -> const OrtMemoryInfo* {
      return &static_cast<const WebGpuSessionAllocator*>(allocator)->impl_->Info();
    };
    if (impl_->IsStreamAware()) {
      AllocOnStream = [](OrtAllocator* allocator, size_t size, OrtSyncStream* stream) noexcept -> void* {
        ORT_TRY {
          return static_cast<WebGpuSessionAllocator*>(allocator)->impl_->AllocOnStream(
              size, reinterpret_cast<Stream*>(stream));
        }
        ORT_CATCH(...) { return nullptr; }
      };
    }
  }

  static void* ORT_API_CALL AllocImpl(OrtAllocator* allocator, size_t size) noexcept {
    ORT_TRY { return static_cast<WebGpuSessionAllocator*>(allocator)->impl_->Alloc(size); }
    ORT_CATCH(...) { return nullptr; }
  }

  AllocatorPtr impl_;
};

}  // namespace

OrtAllocator* CreateWebGpuSessionAllocator(AllocatorPtr allocator) {
  return new WebGpuSessionAllocator(std::move(allocator));
}

bool TryReleaseWebGpuSessionAllocator(OrtAllocator* allocator) {
  if (allocator->Alloc != WebGpuSessionAllocator::AllocImpl) {
    return false;
  }
  delete static_cast<WebGpuSessionAllocator*>(allocator);
  return true;
}
#endif

void GpuBufferAllocator::Free(void* p) {
  if (p != nullptr) {
    auto& recording = recording_getter_();
    std::lock_guard<std::recursive_mutex> lock{recording.mutex};
    buffer_manager_getter_().Release(recording, static_cast<WGPUBuffer>(p));
    stats_.num_allocs--;
  }
}

void GpuBufferAllocator::GetStats(AllocatorStats* stats) {
  auto& recording = recording_getter_();
  std::lock_guard<std::recursive_mutex> lock{recording.mutex};
  *stats = stats_;
}

ExternalGpuBufferAllocator::ExternalGpuBufferAllocator(std::shared_ptr<WebGpuContext> context)
    : IAllocator(OrtMemoryInfo(WEBGPU_BUFFER,
                               OrtAllocatorType::OrtDeviceAllocator,
                               WebGpuDevice,
                               OrtMemTypeDefault)),
      context_{std::move(context)},
      command_state_{std::make_unique<CommandRecordingState>()} {
}

ExternalGpuBufferAllocator::~ExternalGpuBufferAllocator() = default;

void* ExternalGpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  std::lock_guard<std::recursive_mutex> lock{command_state_->mutex};
  ++stats_.num_allocs;
  constexpr wgpu::BufferUsage usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;
  return context_->BufferManager().Create(*command_state_, size, usage,
                                          true, true);
}

void ExternalGpuBufferAllocator::Free(void* p) {
  if (p == nullptr) {
    return;
  }

  std::lock_guard<std::recursive_mutex> lock{command_state_->mutex};
  context_->BufferManager().Release(*command_state_, static_cast<WGPUBuffer>(p));
  --stats_.num_allocs;
}

void ExternalGpuBufferAllocator::GetStats(AllocatorStats* stats) {
  std::lock_guard<std::recursive_mutex> lock{command_state_->mutex};
  *stats = stats_;
}

WebGpuNoOpAllocator::WebGpuNoOpAllocator(bool is_read_only_allocator)
    : IAllocator(
          OrtMemoryInfo(WEBGPU_BUFFER,
                        is_read_only_allocator ? OrtAllocatorType::OrtReadOnlyAllocator
                                               : OrtAllocatorType::OrtDeviceAllocator,
                        WebGpuDevice,
                        OrtMemTypeDefault)) {
}

void* WebGpuNoOpAllocator::Alloc(size_t /*size*/) {
  ORT_THROW("WebGPU EP device-free context must not allocate device memory.");
}

void WebGpuNoOpAllocator::Free(void* /*p*/) {
}

AllocatorPtr CreateWebGpuAllocator(bool device_free,
                                   std::function<const BufferManager&()> buffer_manager_getter,
                                   std::function<CommandRecordingState&()> recording_getter,
                                   bool is_read_only_allocator,
                                   std::function<bool()> should_submit_zero_initialize) {
  if (device_free) {
    return std::make_shared<WebGpuNoOpAllocator>(is_read_only_allocator);
  }
  return std::make_shared<GpuBufferAllocator>(std::move(buffer_manager_getter), std::move(recording_getter),
                                              is_read_only_allocator,
                                              std::move(should_submit_zero_initialize));
}

}  // namespace webgpu
}  // namespace onnxruntime
