// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <d3d12.h>
#include <wil/Resource.h>
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
  struct FileReadRegion;
  struct PreparedFile;
  struct UploadSlot;
  struct AllocationResult;

  D3D12FileBufferLoader(
      ID3D12Device* device,
      const Config& config) noexcept;

  common::Status Initialize();
  common::Status LoadInternal(
      const std::vector<FileRange>& ranges,
      Batch& result,
      const CancellationToken& cancellation);
  common::Status AllocateDestinations(
      const std::vector<uint64_t>& sizes,
      Batch& batch);
  common::Status PrepareFiles(
      const std::vector<FileRange>& ranges,
      const CancellationToken& cancellation,
      std::vector<PreparedFile>& files);
  common::Status PrepareSlots(
      uint64_t alignment,
      const CancellationToken& cancellation);
  common::Status IssueRead(
      HANDLE file,
      UploadSlot& slot,
      uint64_t offset,
      DWORD size);
  common::Status CompleteRead(
      uint64_t file_size,
      UploadSlot& slot,
      DWORD& bytes_read);
  template <typename EnsureAllocationFn>
  common::Status ReadRegion(
      PreparedFile& file,
      const FileReadRegion& region,
      const std::vector<FileRange>& ranges,
      Batch& batch,
      bool& allocation_ready,
      EnsureAllocationFn& ensure_allocation,
      const CancellationToken& cancellation,
      uint64_t& last_submitted_fence);
  common::Status SubmitCopies(
      UploadSlot& slot,
      uint64_t chunk_begin,
      uint64_t bytes,
      const std::vector<size_t>& range_indices,
      const std::vector<FileRange>& ranges,
      Batch& batch,
      uint64_t& last_submitted_fence);
  common::Status TransitionToCommon(
      Batch& batch,
      const CancellationToken& cancellation,
      uint64_t& last_submitted_fence);
  common::Status WaitForFence(
      uint64_t value,
      const CancellationToken& cancellation);
  common::Status SignalSubmittedWork(uint64_t value);
  void WaitForFenceUncancelled(uint64_t value) noexcept;
  void DrainActiveReads() noexcept;

  Microsoft::WRL::ComPtr<ID3D12Device> device_;
  Config config_;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> copy_queue_;
  Microsoft::WRL::ComPtr<ID3D12Fence> copy_fence_;
  wil::unique_handle fence_event_;
  std::vector<UploadSlot> slots_;
  Batch untracked_batch_;
  uint64_t next_fence_value_ = 0;
  bool retain_untracked_submission_ = false;
  std::mutex load_mutex_;
};

}  // namespace d3d12
}  // namespace windows
}  // namespace onnxruntime
