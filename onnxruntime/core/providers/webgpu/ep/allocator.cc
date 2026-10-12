// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "allocator.h"

#include <utility>

#include "core/common/logging/logging.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/ep/sync_stream.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

// A matching Session stream orders deferred clears with subsequent copies/kernels in the same
// recording. Null streams use Alloc's policy; streams from another Session are rejected.
void* GpuBufferAllocator::AllocOnStream(size_t size, Stream* stream) {
  if (stream == nullptr) {
    return Alloc(size);
  }
  ORT_ENFORCE(&ep::GetWebGpuStreamCommandState(reinterpret_cast<OrtSyncStream*>(stream)) == &recording_getter_(),
              "WebGPU allocator and stream belong to different Sessions.");
  return Allocate(size, false);
}

namespace ep {
namespace {

// Wraps an existing Session allocator, not another buffer pool or recording.
// Unlike the Env's adapter::Allocator, this wrapper also forwards AllocOnStream.
struct WebGpuSessionAllocator final : OrtAllocator {
  explicit WebGpuSessionAllocator(AllocatorPtr impl) : OrtAllocator{}, impl_{std::move(impl)} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = [](OrtAllocator* allocator, void* buffer) noexcept {
      // Do not retry or directly release after an exception: Free may have transferred ownership.
      ORT_TRY {
        static_cast<WebGpuSessionAllocator*>(allocator)->impl_->Free(buffer);
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          LOGS_DEFAULT(ERROR) << "WebGPU Session allocator Free failed: " << ex.what();
        });
      }
      ORT_CATCH(...) {
        LOGS_DEFAULT(ERROR) << "WebGPU Session allocator Free failed with an unknown exception.";
      }
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

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
