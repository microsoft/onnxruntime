// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace windows {
namespace d3d12 {

class D3D12FileBufferLoader {
 public:
  struct Config {
    uint32_t upload_slot_count = 12;
    uint64_t upload_slot_size = 16ull * 1024ull * 1024ull;
    uint64_t max_destination_heap_size = 512ull * 1024ull * 1024ull;
  };

  struct FileRange {
    std::wstring path;
    uint64_t offset = 0;
    uint64_t length = 0;
  };

  struct CancellationToken {
    using IsCancellationRequestedFn = bool (*)(void* state) noexcept;

    IsCancellationRequestedFn is_cancellation_requested = nullptr;
    void* state = nullptr;

    bool IsCancellationRequested() const noexcept {
      return is_cancellation_requested != nullptr &&
             is_cancellation_requested(state);
    }
  };

  struct Buffer {
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    uint64_t size = 0;
  };

  struct Batch {
    std::vector<Microsoft::WRL::ComPtr<ID3D12Heap>> heaps;
    std::vector<Buffer> buffers;

    void Clear() noexcept {
      buffers.clear();
      heaps.clear();
    }
  };

  static common::Status Create(
      ID3D12Device* device,
      std::unique_ptr<D3D12FileBufferLoader>& loader,
      const Config& config = {}) noexcept;

  ~D3D12FileBufferLoader();

  D3D12FileBufferLoader(const D3D12FileBufferLoader&) = delete;
  D3D12FileBufferLoader& operator=(const D3D12FileBufferLoader&) = delete;

  // The returned buffers correspond one-to-one with ranges and preserve input order.
  // Each buffer is in D3D12_RESOURCE_STATE_COMMON when this method succeeds.
  common::Status Load(
      const std::vector<FileRange>& ranges,
      Batch& result,
      const CancellationToken& cancellation = {}) noexcept;

 private:
  struct Impl;

  explicit D3D12FileBufferLoader(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace d3d12
}  // namespace windows
}  // namespace onnxruntime
