// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/d3d12_file_loader/d3d12_file_buffer_loader.h"

#include <algorithm>
#include <array>
#include <future>
#include <iterator>
#include <limits>
#include <mutex>
#include <system_error>
#include <unordered_map>
#include <utility>

#include <gsl/gsl>
#include <wil/Resource.h>

#include "core/common/common.h"

namespace onnxruntime {
namespace windows {
namespace d3d12 {

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint64_t kBufferAlignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
// The cancellation token is callback-based and has no waitable handle, so a
// short timed wait keeps cancellation responsive without adding noticeable
// fence-completion tail latency.
constexpr DWORD kCancellationPollMilliseconds = 1;

common::Status HResultError(const char* operation, HRESULT hr) {
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL, operation, " failed with HRESULT ",
      static_cast<uint32_t>(hr));
}

common::Status Win32Error(const char* operation, DWORD error) {
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL, operation, " failed, error code ", error, " - ",
      std::system_category().message(error));
}

common::Status CancelledStatus() {
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, MODEL_LOAD_CANCELED,
      "D3D12 file buffer loading was canceled.");
}

uint64_t AlignDown(uint64_t value, uint64_t alignment) noexcept {
  return value - value % alignment;
}

bool TryAlignUp(uint64_t value, uint64_t alignment, uint64_t& result) noexcept {
  if (alignment == 0 ||
      value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    return false;
  }
  result = AlignDown(value + alignment - 1, alignment);
  return true;
}

D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type) noexcept {
  D3D12_HEAP_PROPERTIES properties{};
  properties.Type = type;
  properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  properties.CreationNodeMask = 1;
  properties.VisibleNodeMask = 1;
  return properties;
}

D3D12_RESOURCE_DESC BufferDescription(
    uint64_t size, D3D12_RESOURCE_FLAGS flags) noexcept {
  D3D12_RESOURCE_DESC description{};
  description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  description.Width = size;
  description.Height = 1;
  description.DepthOrArraySize = 1;
  description.MipLevels = 1;
  description.Format = DXGI_FORMAT_UNKNOWN;
  description.SampleDesc.Count = 1;
  description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  description.Flags = flags;
  return description;
}

}  // namespace

struct D3D12FileBufferLoader::Impl {
  struct FileReadRegion {
    uint64_t begin = 0;
    uint64_t end = 0;
  };

  struct PreparedFile {
    std::wstring path;
    wil::unique_hfile file;
    uint64_t size = 0;
    uint64_t alignment = 0;
    std::vector<size_t> range_indices;
    std::vector<FileReadRegion> regions;
  };

  struct UploadSlot {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    wil::unique_handle read_event;
    void* mapped = nullptr;
    OVERLAPPED overlapped{};
    uint64_t file_offset = 0;
    DWORD requested = 0;
    uint64_t fence_value = 0;
    bool read_active = false;
  };

  struct AllocationResult {
    common::Status status;
    Batch batch;
  };

  Impl(ID3D12Device* device, const Config& config);
  ~Impl();

  common::Status Initialize();
  common::Status Load(
      const std::vector<FileRange>& ranges,
      Batch& result,
      const CancellationToken& cancellation);

 private:
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
      HANDLE file,
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
  void WaitForFenceUncancelled(uint64_t value) noexcept;
  void DrainActiveReads(HANDLE file) noexcept;

  ComPtr<ID3D12Device> device_;
  Config config_;
  ComPtr<ID3D12CommandQueue> copy_queue_;
  ComPtr<ID3D12Fence> copy_fence_;
  wil::unique_handle fence_event_;
  std::vector<UploadSlot> slots_;
  uint64_t next_fence_value_ = 0;
  std::mutex load_mutex_;
};

D3D12FileBufferLoader::Impl::Impl(
    ID3D12Device* device, const Config& config)
    : device_(device), config_(config) {
}

D3D12FileBufferLoader::Impl::~Impl() {
  for (auto& slot : slots_) {
    if (slot.resource && slot.mapped != nullptr) {
      slot.resource->Unmap(0, nullptr);
      slot.mapped = nullptr;
    }
  }
}

common::Status D3D12FileBufferLoader::Impl::Initialize() {
  if (device_ == nullptr) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "D3D12FileBufferLoader requires a non-null D3D12 device.");
  }
  if (config_.upload_slot_count < 2 ||
      config_.upload_slot_count > MAXIMUM_WAIT_OBJECTS) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "upload_slot_count must be between 2 and ", MAXIMUM_WAIT_OBJECTS, ".");
  }
  if (config_.upload_slot_size == 0 ||
      config_.upload_slot_size > std::numeric_limits<DWORD>::max() ||
      config_.upload_slot_size % kBufferAlignment != 0) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "upload_slot_size must be a non-zero 64 KiB multiple no larger than DWORD_MAX.");
  }
  if (config_.max_destination_heap_size == 0 ||
      config_.max_destination_heap_size > 512ull * 1024ull * 1024ull ||
      config_.max_destination_heap_size % kBufferAlignment != 0) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "max_destination_heap_size must be a non-zero 64 KiB multiple no larger than 512 MiB.");
  }

  D3D12_COMMAND_QUEUE_DESC queue_description{};
  queue_description.Type = D3D12_COMMAND_LIST_TYPE_COPY;
  HRESULT hr = device_->CreateCommandQueue(
      &queue_description, IID_PPV_ARGS(&copy_queue_));
  if (FAILED(hr)) {
    return HResultError("ID3D12Device::CreateCommandQueue", hr);
  }

  hr = device_->CreateFence(
      0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copy_fence_));
  if (FAILED(hr)) {
    return HResultError("ID3D12Device::CreateFence", hr);
  }

  fence_event_.reset(CreateEventExW(
      nullptr, nullptr, 0, EVENT_MODIFY_STATE | SYNCHRONIZE));
  if (!fence_event_) {
    return Win32Error("CreateEventExW(copy fence)", GetLastError());
  }

  const auto upload_heap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
  const auto upload_description =
      BufferDescription(config_.upload_slot_size, D3D12_RESOURCE_FLAG_NONE);
  slots_.resize(config_.upload_slot_count);
  for (auto& slot : slots_) {
    hr = device_->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &upload_description,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&slot.resource));
    if (FAILED(hr)) {
      return HResultError(
          "ID3D12Device::CreateCommittedResource(upload)", hr);
    }

    hr = slot.resource->Map(0, nullptr, &slot.mapped);
    if (FAILED(hr)) {
      return HResultError("ID3D12Resource::Map(upload)", hr);
    }

    hr = device_->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&slot.allocator));
    if (FAILED(hr)) {
      return HResultError(
          "ID3D12Device::CreateCommandAllocator", hr);
    }

    hr = device_->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_COPY, slot.allocator.Get(), nullptr,
        IID_PPV_ARGS(&slot.command_list));
    if (FAILED(hr)) {
      return HResultError("ID3D12Device::CreateCommandList", hr);
    }
    hr = slot.command_list->Close();
    if (FAILED(hr)) {
      return HResultError(
          "ID3D12GraphicsCommandList::Close(initial)", hr);
    }

    slot.read_event.reset(CreateEventExW(
        nullptr, nullptr, CREATE_EVENT_MANUAL_RESET,
        EVENT_MODIFY_STATE | SYNCHRONIZE));
    if (!slot.read_event) {
      return Win32Error("CreateEventExW(read)", GetLastError());
    }
  }

  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::Impl::Load(
    const std::vector<FileRange>& ranges,
    Batch& result,
    const CancellationToken& cancellation) {
  std::lock_guard<std::mutex> lock(load_mutex_);
  result.Clear();

  if (ranges.empty()) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "At least one file range is required.");
  }
  if (cancellation.IsCancellationRequested()) {
    return CancelledStatus();
  }

  std::vector<uint64_t> sizes;
  sizes.reserve(ranges.size());
  for (size_t index = 0; index < ranges.size(); ++index) {
    const auto& range = ranges[index];
    if (range.path.empty()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "File range ", index, " has an empty path.");
    }
    if (range.length == 0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "File range ", index, " has zero length.");
    }
    if (range.offset >
        std::numeric_limits<uint64_t>::max() - range.length) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "File range ", index, " overflows uint64_t.");
    }
    sizes.push_back(range.length);
  }

  // Destination allocation can run while the CPU opens files and discovers
  // the sector alignment required for unbuffered reads.
  std::future<AllocationResult> allocation_future = std::async(
      std::launch::async, [this, sizes = std::move(sizes)]() noexcept {
        AllocationResult allocation;
        try {
          allocation.status =
              AllocateDestinations(sizes, allocation.batch);
        } catch (const std::exception& ex) {
          allocation.status = ORT_MAKE_STATUS(
              ONNXRUNTIME, RUNTIME_EXCEPTION,
              "Destination allocation failed: ", ex.what());
        } catch (...) {
          allocation.status = ORT_MAKE_STATUS(
              ONNXRUNTIME, RUNTIME_EXCEPTION,
              "Destination allocation failed with an unknown exception.");
        }
        return allocation;
      });

  std::vector<PreparedFile> files;
  HANDLE active_file = INVALID_HANDLE_VALUE;
  uint64_t last_submitted_fence = 0;
  bool load_succeeded = false;
  // Resources referenced by in-flight I/O or GPU work must remain alive on
  // every error and cancellation path.
  auto cleanup = gsl::finally([&]() noexcept {
    if (!load_succeeded) {
      DrainActiveReads(active_file);
      WaitForFenceUncancelled(last_submitted_fence);
    }
    if (allocation_future.valid()) {
      try {
        (void)allocation_future.get();
      } catch (...) {
      }
    }
  });

  ORT_RETURN_IF_ERROR(PrepareFiles(ranges, cancellation, files));

  uint64_t maximum_alignment = 1;
  for (const auto& file : files) {
    maximum_alignment = std::max(maximum_alignment, file.alignment);
  }
  ORT_RETURN_IF_ERROR(PrepareSlots(maximum_alignment, cancellation));

  Batch loaded_batch;
  bool allocation_ready = false;
  auto ensure_allocation = [&]() -> common::Status {
    if (allocation_ready) {
      return common::Status::OK();
    }
    AllocationResult allocation = allocation_future.get();
    if (!allocation.status.IsOK()) {
      return allocation.status;
    }
    loaded_batch = std::move(allocation.batch);
    allocation_ready = true;
    return common::Status::OK();
  };

  for (auto& file : files) {
    active_file = file.file.get();
    for (const auto& region : file.regions) {
      const auto status = ReadRegion(
          file, region, ranges, loaded_batch, allocation_ready,
          ensure_allocation, cancellation, last_submitted_fence);
      if (!status.IsOK()) {
        return status;
      }
    }
    active_file = INVALID_HANDLE_VALUE;
  }

  ORT_RETURN_IF_ERROR(ensure_allocation());
  ORT_RETURN_IF_ERROR(
      WaitForFence(last_submitted_fence, cancellation));
  ORT_RETURN_IF_ERROR(
      TransitionToCommon(loaded_batch, cancellation, last_submitted_fence));

  result = std::move(loaded_batch);
  load_succeeded = true;
  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::Impl::AllocateDestinations(
    const std::vector<uint64_t>& sizes,
    Batch& batch) {
  struct Placement {
    bool committed = false;
    size_t heap_index = 0;
    uint64_t offset = 0;
    D3D12_RESOURCE_DESC description{};
  };

  std::vector<Placement> placements;
  placements.reserve(sizes.size());
  std::vector<uint64_t> heap_sizes;

  for (size_t index = 0; index < sizes.size(); ++index) {
    uint64_t resource_size = 0;
    if (!TryAlignUp(sizes[index], 16, resource_size)) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Destination ", index, " size overflows while aligning for Dawn.");
    }
    const auto description = BufferDescription(
        resource_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto allocation_info =
        device_->GetResourceAllocationInfo(0, 1, &description);
    if (allocation_info.SizeInBytes ==
        std::numeric_limits<uint64_t>::max()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "GetResourceAllocationInfo failed for destination ", index, ".");
    }
    if (allocation_info.SizeInBytes > config_.max_destination_heap_size) {
      placements.push_back({true, 0, 0, description});
      continue;
    }

    if (heap_sizes.empty()) {
      heap_sizes.push_back(0);
    }
    size_t heap_index = heap_sizes.size() - 1;
    uint64_t offset = 0;
    if (!TryAlignUp(
            heap_sizes.back(), allocation_info.Alignment, offset)) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "Destination placement offset overflow.");
    }
    if (heap_sizes.back() != 0 &&
        offset >
            config_.max_destination_heap_size -
                allocation_info.SizeInBytes) {
      heap_sizes.push_back(0);
      heap_index = heap_sizes.size() - 1;
      offset = 0;
    }

    placements.push_back({false, heap_index, offset, description});
    heap_sizes[heap_index] =
        offset + allocation_info.SizeInBytes;
  }

  batch.heaps.resize(heap_sizes.size());
  for (size_t index = 0; index < heap_sizes.size(); ++index) {
    uint64_t heap_size = 0;
    if (!TryAlignUp(
            heap_sizes[index], kBufferAlignment, heap_size) ||
        heap_size > config_.max_destination_heap_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "Destination heap size is invalid.");
    }

    D3D12_HEAP_DESC heap_description{};
    heap_description.SizeInBytes = heap_size;
    heap_description.Properties =
        HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    heap_description.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    const HRESULT hr = device_->CreateHeap(
        &heap_description, IID_PPV_ARGS(&batch.heaps[index]));
    if (FAILED(hr)) {
      return HResultError("ID3D12Device::CreateHeap", hr);
    }
  }

  batch.buffers.resize(sizes.size());
  for (size_t index = 0; index < placements.size(); ++index) {
    const auto& placement = placements[index];
    HRESULT hr = S_OK;
    if (placement.committed) {
      const auto heap_properties = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
      hr = device_->CreateCommittedResource(
          &heap_properties, D3D12_HEAP_FLAG_NONE, &placement.description,
          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
          IID_PPV_ARGS(&batch.buffers[index].resource));
    } else {
      hr = device_->CreatePlacedResource(
          batch.heaps[placement.heap_index].Get(),
          placement.offset, &placement.description,
          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
          IID_PPV_ARGS(&batch.buffers[index].resource));
    }
    if (FAILED(hr)) {
      return HResultError(
          placement.committed
              ? "ID3D12Device::CreateCommittedResource(destination)"
              : "ID3D12Device::CreatePlacedResource",
          hr);
    }
    batch.buffers[index].size = sizes[index];
  }

  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::Impl::PrepareFiles(
    const std::vector<FileRange>& ranges,
    const CancellationToken& cancellation,
    std::vector<PreparedFile>& files) {
  // Group ranges by source file so each file is opened once. Later, aligned
  // overlapping ranges are merged to avoid redundant unbuffered reads.
  std::unordered_map<std::wstring, size_t> file_indices;
  file_indices.reserve(ranges.size());
  for (size_t range_index = 0;
       range_index < ranges.size(); ++range_index) {
    const auto& range = ranges[range_index];
    const auto [file_index_it, inserted] =
        file_indices.try_emplace(range.path, files.size());
    if (inserted) {
      PreparedFile file;
      file.path = range.path;
      files.push_back(std::move(file));
    }
    files[file_index_it->second].range_indices.push_back(range_index);
  }

  for (auto& file : files) {
    if (cancellation.IsCancellationRequested()) {
      return CancelledStatus();
    }

    file.file.reset(CreateFileW(
        file.path.c_str(), GENERIC_READ, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED |
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr));
    if (!file.file) {
      return Win32Error(
          "CreateFileW(unbuffered)", GetLastError());
    }

    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file.file.get(), &file_size)) {
      return Win32Error("GetFileSizeEx", GetLastError());
    }
    if (file_size.QuadPart < 0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL, "Source file has a negative size.");
    }
    file.size = static_cast<uint64_t>(file_size.QuadPart);

    FILE_STORAGE_INFO storage_info{};
    if (!GetFileInformationByHandleEx(
            file.file.get(), FileStorageInfo,
            &storage_info, sizeof(storage_info))) {
      return Win32Error(
          "GetFileInformationByHandleEx(FileStorageInfo)",
          GetLastError());
    }
    file.alignment = std::max<uint64_t>(
        storage_info.LogicalBytesPerSector,
        storage_info.PhysicalBytesPerSectorForPerformance);
    if (file.alignment == 0 ||
        (file.alignment & (file.alignment - 1)) != 0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "Storage alignment is not a non-zero power of two.");
    }
    if (config_.upload_slot_size % file.alignment != 0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "upload_slot_size does not satisfy source file alignment.");
    }

    std::sort(
        file.range_indices.begin(), file.range_indices.end(),
        [&](size_t left, size_t right) {
          return ranges[left].offset < ranges[right].offset;
        });

    for (size_t range_index : file.range_indices) {
      const auto& range = ranges[range_index];
      if (range.offset > file.size ||
          range.length > file.size - range.offset) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "File range ", range_index,
            " exceeds its source file.");
      }

      uint64_t aligned_end = 0;
      uint64_t aligned_file_size = 0;
      if (!TryAlignUp(
              range.offset + range.length,
              file.alignment, aligned_end) ||
          !TryAlignUp(
              file.size, file.alignment,
              aligned_file_size)) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "Aligned file range overflow.");
      }
      FileReadRegion region{
          AlignDown(range.offset, file.alignment),
          std::min(aligned_end, aligned_file_size)};
      if (!file.regions.empty() &&
          region.begin <= file.regions.back().end) {
        file.regions.back().end =
            std::max(file.regions.back().end, region.end);
      } else {
        file.regions.push_back(region);
      }
    }
  }

  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::Impl::PrepareSlots(
    uint64_t alignment,
    const CancellationToken& cancellation) {
  for (auto& slot : slots_) {
    ORT_RETURN_IF_ERROR(
        WaitForFence(slot.fence_value, cancellation));
    if (reinterpret_cast<uintptr_t>(slot.mapped) %
            alignment !=
        0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "Mapped upload buffer does not satisfy source file alignment.");
    }
    slot.overlapped = {};
    slot.file_offset = 0;
    slot.requested = 0;
    slot.fence_value = 0;
    slot.read_active = false;
  }
  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::Impl::IssueRead(
    HANDLE file,
    UploadSlot& slot,
    uint64_t offset,
    DWORD size) {
  if (!ResetEvent(slot.read_event.get())) {
    return Win32Error("ResetEvent(read)", GetLastError());
  }
  slot.overlapped = {};
  slot.overlapped.Offset = static_cast<DWORD>(offset);
  slot.overlapped.OffsetHigh =
      static_cast<DWORD>(offset >> 32);
  slot.overlapped.hEvent = slot.read_event.get();
  slot.file_offset = offset;
  slot.requested = size;
  slot.read_active = true;
  if (!ReadFile(
          file, slot.mapped, size, nullptr,
          &slot.overlapped)) {
    const DWORD error = GetLastError();
    if (error != ERROR_IO_PENDING) {
      slot.read_active = false;
      return Win32Error("ReadFile(unbuffered)", error);
    }
  }
  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::Impl::CompleteRead(
    HANDLE file,
    uint64_t file_size,
    UploadSlot& slot,
    DWORD& bytes_read) {
  if (!GetOverlappedResult(
          file, &slot.overlapped,
          &bytes_read, FALSE)) {
    return Win32Error(
        "GetOverlappedResult", GetLastError());
  }
  slot.read_active = false;
  const uint64_t required_bytes =
      std::min<uint64_t>(
          slot.requested,
          file_size - slot.file_offset);
  if (bytes_read < required_bytes) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "Unbuffered read ended before the required file range.");
  }
  return common::Status::OK();
}

template <typename EnsureAllocationFn>
common::Status D3D12FileBufferLoader::Impl::ReadRegion(
    PreparedFile& file,
    const FileReadRegion& region,
    const std::vector<FileRange>& ranges,
    Batch& batch,
    bool& allocation_ready,
    EnsureAllocationFn& ensure_allocation,
    const CancellationToken& cancellation,
    uint64_t& last_submitted_fence) {
  uint64_t next_offset = region.begin;

  // Each upload slot moves through read-active, copy-in-flight, and reusable
  // states. Keep all eligible slots busy without reusing one before its
  // previous GPU copy reaches the fence.
  auto issue_available = [&]() -> common::Status {
    for (auto& slot : slots_) {
      if (next_offset >= region.end) {
        break;
      }
      if (slot.read_active) {
        continue;
      }
      if (slot.fence_value != 0 &&
          copy_fence_->GetCompletedValue() <
              slot.fence_value) {
        continue;
      }
      if (cancellation.IsCancellationRequested()) {
        return CancelledStatus();
      }
      const DWORD bytes = static_cast<DWORD>(
          std::min<uint64_t>(
              config_.upload_slot_size,
              region.end - next_offset));
      ORT_RETURN_IF_ERROR(
          IssueRead(
              file.file.get(), slot,
              next_offset, bytes));
      next_offset += bytes;
    }
    return common::Status::OK();
  };

  ORT_RETURN_IF_ERROR(issue_available());
  while (true) {
    std::array<HANDLE, MAXIMUM_WAIT_OBJECTS> handles{};
    std::array<size_t, MAXIMUM_WAIT_OBJECTS> slot_indices{};
    DWORD handle_count = 0;
    for (size_t index = 0;
         index < slots_.size(); ++index) {
      if (slots_[index].read_active) {
        handles[handle_count] =
            slots_[index].read_event.get();
        slot_indices[handle_count] = index;
        ++handle_count;
      }
    }

    if (handle_count == 0) {
      if (next_offset >= region.end) {
        break;
      }
      uint64_t next_fence =
          std::numeric_limits<uint64_t>::max();
      for (const auto& slot : slots_) {
        if (!slot.read_active &&
            slot.fence_value != 0) {
          next_fence =
              std::min(next_fence, slot.fence_value);
        }
      }
      if (next_fence ==
          std::numeric_limits<uint64_t>::max()) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, FAIL,
            "Event-driven read scheduler made no progress.");
      }
      ORT_RETURN_IF_ERROR(
          WaitForFence(next_fence, cancellation));
      ORT_RETURN_IF_ERROR(issue_available());
      continue;
    }

    const DWORD wait_result = WaitForMultipleObjects(
        handle_count, handles.data(), FALSE,
        kCancellationPollMilliseconds);
    if (wait_result == WAIT_TIMEOUT) {
      if (cancellation.IsCancellationRequested()) {
        return CancelledStatus();
      }
      continue;
    }
    if (wait_result < WAIT_OBJECT_0 ||
        wait_result >= WAIT_OBJECT_0 + handle_count) {
      return Win32Error(
          "WaitForMultipleObjects(read)",
          GetLastError());
    }

    auto& slot =
        slots_[slot_indices[wait_result - WAIT_OBJECT_0]];
    DWORD bytes_read = 0;
    ORT_RETURN_IF_ERROR(
        CompleteRead(
            file.file.get(), file.size,
            slot, bytes_read));

    if (!allocation_ready) {
      ORT_RETURN_IF_ERROR(ensure_allocation());
    }
    ORT_RETURN_IF_ERROR(
        SubmitCopies(
            slot, slot.file_offset, bytes_read,
            file.range_indices, ranges, batch,
            last_submitted_fence));
    ORT_RETURN_IF_ERROR(issue_available());
  }

  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::Impl::SubmitCopies(
    UploadSlot& slot,
    uint64_t chunk_begin,
    uint64_t bytes,
    const std::vector<size_t>& range_indices,
    const std::vector<FileRange>& ranges,
    Batch& batch,
    uint64_t& last_submitted_fence) {
  HRESULT hr = slot.allocator->Reset();
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12CommandAllocator::Reset", hr);
  }
  hr = slot.command_list->Reset(
      slot.allocator.Get(), nullptr);
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12GraphicsCommandList::Reset", hr);
  }

  const uint64_t chunk_end = chunk_begin + bytes;
  for (size_t range_index : range_indices) {
    const auto& range = ranges[range_index];
    const uint64_t range_end =
        range.offset + range.length;
    const uint64_t copy_begin =
        std::max(chunk_begin, range.offset);
    const uint64_t copy_end =
        std::min(chunk_end, range_end);
    if (copy_begin < copy_end) {
      slot.command_list->CopyBufferRegion(
          batch.buffers[range_index].resource.Get(),
          copy_begin - range.offset,
          slot.resource.Get(),
          copy_begin - chunk_begin,
          copy_end - copy_begin);
    }

    // Dawn buffer sizes are 16-byte aligned. Initialize the tail by
    // repeating source bytes so no padding remains uninitialized.
    const uint64_t aligned_size =
        (range.length + 15) & ~uint64_t{15};
    for (uint64_t padding_offset = range.length;
         padding_offset < aligned_size;) {
      const uint64_t source_relative_offset =
          (padding_offset - range.length) % range.length;
      const uint64_t source_offset =
          range.offset + source_relative_offset;
      if (source_offset >= chunk_begin &&
          source_offset < chunk_end) {
        const uint64_t copy_size = std::min(
            {aligned_size - padding_offset,
             range.length - source_relative_offset,
             chunk_end - source_offset});
        slot.command_list->CopyBufferRegion(
            batch.buffers[range_index].resource.Get(),
            padding_offset,
            slot.resource.Get(),
            source_offset - chunk_begin,
            copy_size);
        padding_offset += copy_size;
      } else {
        ++padding_offset;
      }
    }
  }

  hr = slot.command_list->Close();
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12GraphicsCommandList::Close", hr);
  }
  ID3D12CommandList* command_lists[] = {
      slot.command_list.Get()};
  copy_queue_->ExecuteCommandLists(1, command_lists);

  slot.fence_value = ++next_fence_value_;
  last_submitted_fence = slot.fence_value;
  hr = copy_queue_->Signal(
      copy_fence_.Get(), slot.fence_value);
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12CommandQueue::Signal", hr);
  }
  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::Impl::TransitionToCommon(
    Batch& batch,
    const CancellationToken& cancellation,
    uint64_t& last_submitted_fence) {
  if (batch.buffers.empty()) {
    return common::Status::OK();
  }

  auto& slot = slots_.front();
  HRESULT hr = slot.allocator->Reset();
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12CommandAllocator::Reset(transition)", hr);
  }
  hr = slot.command_list->Reset(
      slot.allocator.Get(), nullptr);
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12GraphicsCommandList::Reset(transition)", hr);
  }

  std::vector<D3D12_RESOURCE_BARRIER> barriers;
  barriers.reserve(batch.buffers.size());
  for (const auto& buffer : batch.buffers) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type =
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource =
        buffer.resource.Get();
    barrier.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore =
        D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter =
        D3D12_RESOURCE_STATE_COMMON;
    barriers.push_back(barrier);
  }
  slot.command_list->ResourceBarrier(
      static_cast<UINT>(barriers.size()),
      barriers.data());

  hr = slot.command_list->Close();
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12GraphicsCommandList::Close(transition)", hr);
  }
  ID3D12CommandList* command_lists[] = {
      slot.command_list.Get()};
  copy_queue_->ExecuteCommandLists(1, command_lists);

  last_submitted_fence = ++next_fence_value_;
  hr = copy_queue_->Signal(
      copy_fence_.Get(), last_submitted_fence);
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12CommandQueue::Signal(transition)", hr);
  }
  return WaitForFence(
      last_submitted_fence, cancellation);
}

common::Status D3D12FileBufferLoader::Impl::WaitForFence(
    uint64_t value,
    const CancellationToken& cancellation) {
  if (value == 0 ||
      copy_fence_->GetCompletedValue() >= value) {
    return common::Status::OK();
  }
  const HRESULT hr =
      copy_fence_->SetEventOnCompletion(
          value, fence_event_.get());
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12Fence::SetEventOnCompletion", hr);
  }

  while (true) {
    const DWORD wait_result = WaitForSingleObject(
        fence_event_.get(),
        kCancellationPollMilliseconds);
    if (wait_result == WAIT_OBJECT_0) {
      return common::Status::OK();
    }
    if (wait_result != WAIT_TIMEOUT) {
      return Win32Error(
          "WaitForSingleObject(copy fence)",
          GetLastError());
    }
    if (cancellation.IsCancellationRequested()) {
      return CancelledStatus();
    }
  }
}

void D3D12FileBufferLoader::Impl::WaitForFenceUncancelled(
    uint64_t value) noexcept {
  if (value == 0 ||
      copy_fence_->GetCompletedValue() >= value) {
    return;
  }
  if (SUCCEEDED(copy_fence_->SetEventOnCompletion(
          value, fence_event_.get())) &&
      WaitForSingleObject(
          fence_event_.get(), INFINITE) ==
          WAIT_OBJECT_0) {
    return;
  }

  while (true) {
    const uint64_t completed =
        copy_fence_->GetCompletedValue();
    if (completed >= value ||
        completed == std::numeric_limits<uint64_t>::max()) {
      return;
    }
    Sleep(1);
  }
}

void D3D12FileBufferLoader::Impl::DrainActiveReads(HANDLE file) noexcept {
  if (file == nullptr ||
      file == INVALID_HANDLE_VALUE) {
    return;
  }
  for (auto& slot : slots_) {
    if (!slot.read_active) {
      continue;
    }
    (void)CancelIoEx(file, &slot.overlapped);
  }
  for (auto& slot : slots_) {
    if (!slot.read_active) {
      continue;
    }
    DWORD bytes_read = 0;
    (void)GetOverlappedResult(
        file, &slot.overlapped,
        &bytes_read, TRUE);
    slot.read_active = false;
  }
}

D3D12FileBufferLoader::D3D12FileBufferLoader(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}

D3D12FileBufferLoader::~D3D12FileBufferLoader() = default;

common::Status D3D12FileBufferLoader::Create(
    ID3D12Device* device,
    std::unique_ptr<D3D12FileBufferLoader>& loader,
    const Config& config) noexcept {
  loader.reset();
  try {
    auto impl = std::make_unique<Impl>(device, config);
    ORT_RETURN_IF_ERROR(impl->Initialize());
    loader.reset(
        new D3D12FileBufferLoader(std::move(impl)));
    return common::Status::OK();
  } catch (const std::exception& ex) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, RUNTIME_EXCEPTION,
        "D3D12FileBufferLoader::Create failed: ",
        ex.what());
  } catch (...) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, RUNTIME_EXCEPTION,
        "D3D12FileBufferLoader::Create failed with an unknown exception.");
  }
}

common::Status D3D12FileBufferLoader::Load(
    const std::vector<FileRange>& ranges,
    Batch& result,
    const CancellationToken& cancellation) noexcept {
  if (!impl_) {
    result.Clear();
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "D3D12FileBufferLoader is not initialized.");
  }
  try {
    return impl_->Load(ranges, result, cancellation);
  } catch (const std::exception& ex) {
    result.Clear();
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, RUNTIME_EXCEPTION,
        "D3D12FileBufferLoader::Load failed: ",
        ex.what());
  } catch (...) {
    result.Clear();
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, RUNTIME_EXCEPTION,
        "D3D12FileBufferLoader::Load failed with an unknown exception.");
  }
}

}  // namespace d3d12
}  // namespace windows
}  // namespace onnxruntime
