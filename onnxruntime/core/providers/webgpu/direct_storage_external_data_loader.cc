// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)

#include <Windows.h>
#include <d3d12.h>
#include <dstorage.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dawn/native/D3D12Backend.h"

#include "core/common/logging/logging.h"
#include "core/framework/tensor.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/direct_storage_external_data_loader.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

using Microsoft::WRL::ComPtr;

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kMaxRequestSize = 64ull * 1024ull * 1024ull;
constexpr uint32_t kMaxAllocationWorkers = 4;
constexpr uint64_t kCancellationTag = 1;

constexpr uint32_t GrowQueueCapacity(uint32_t capacity) {
  return capacity > DSTORAGE_MAX_QUEUE_CAPACITY / 2
             ? DSTORAGE_MAX_QUEUE_CAPACITY
             : capacity * 2;
}

static_assert(GrowQueueCapacity(32768) == DSTORAGE_MAX_QUEUE_CAPACITY);

double Milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

common::Status HResultStatus(const char* operation, HRESULT result) {
  std::ostringstream message;
  message << operation << " failed with HRESULT 0x" << std::hex
          << static_cast<uint32_t>(result) << ".";
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, message.str());
}

struct TensorKey {
  std::filesystem::path path;
  std::string name;
  uint64_t offset;
  size_t length;

  bool operator<(const TensorKey& other) const {
    if (path != other.path) {
      return path.native() < other.path.native();
    }
    if (name != other.name) {
      return name < other.name;
    }
    if (offset != other.offset) {
      return offset < other.offset;
    }
    return length < other.length;
  }
};

struct PreparedTensor {
  TensorKey key;
  ComPtr<ID3D12Resource> resource;
  wgpu::SharedBufferMemory memory;
  wgpu::Buffer buffer;
  bool access_started = false;
  bool claimed = false;
};

struct DirectStorageBatch {
  struct FileInfo {
    std::filesystem::path canonical_path;
    size_t length;
  };

  std::vector<PreparedTensor> tensors;
  std::map<TensorKey, size_t> tensors_by_key;
  std::map<std::filesystem::path, FileInfo> files;
  size_t total_bytes = 0;
  size_t request_count = 0;
  bool finalized = false;
};

struct DirectStorageLoadMetrics {
  double initialization_ms = 0.0;
  double allocation_ms = 0.0;
  double preparation_ms = 0.0;
  double enqueue_ms = 0.0;
  double io_ms = 0.0;
  uint32_t allocation_workers = 0;
};

struct ScopedHandle {
  HANDLE value = nullptr;

  ~ScopedHandle() {
    if (value != nullptr) {
      CloseHandle(value);
    }
  }

  ScopedHandle() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedHandle);
};

common::Status LoadBatchToD3D12(
    DirectStorageBatch& batch,
    ID3D12Device* d3d_device,
    const std::function<bool()>& is_cancelled,
    DirectStorageLoadMetrics& metrics) {
  ORT_RETURN_IF_NOT(d3d_device != nullptr, "A D3D12 device is required.");
  if (is_cancelled && is_cancelled()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                           "DirectStorage initializer loading was canceled.");
  }
  ORT_RETURN_IF(batch.request_count >
                    static_cast<size_t>(DSTORAGE_MAX_QUEUE_CAPACITY) - 2,
                "DirectStorage initializer batch requires ", batch.request_count,
                " requests, exceeding one queue's capacity.");

  const auto preparation_start = Clock::now();
  const auto allocation_start = Clock::now();
  Clock::time_point allocation_end;
  metrics.allocation_workers =
      std::min(kMaxAllocationWorkers, static_cast<uint32_t>(batch.tensors.size()));

  std::future<HRESULT> allocation_future;
  try {
    allocation_future = std::async(std::launch::async, [&]() {
      std::atomic<size_t> next_index{0};
      std::atomic<HRESULT> allocation_result{S_OK};
      const auto allocate = [&]() {
        while (SUCCEEDED(allocation_result.load(std::memory_order_relaxed))) {
          const size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
          if (index >= batch.tensors.size()) {
            return;
          }
          if (batch.tensors[index].key.length == 0) {
            continue;
          }

          const uint64_t resource_size =
              (static_cast<uint64_t>(batch.tensors[index].key.length) + 15) &
              ~uint64_t{15};
          D3D12_HEAP_PROPERTIES heap_properties{};
          heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
          D3D12_RESOURCE_DESC resource_descriptor{};
          resource_descriptor.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
          resource_descriptor.Width = resource_size;
          resource_descriptor.Height = 1;
          resource_descriptor.DepthOrArraySize = 1;
          resource_descriptor.MipLevels = 1;
          resource_descriptor.SampleDesc.Count = 1;
          resource_descriptor.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
          resource_descriptor.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

          const HRESULT result = d3d_device->CreateCommittedResource(
              &heap_properties, D3D12_HEAP_FLAG_NONE, &resource_descriptor,
              D3D12_RESOURCE_STATE_COMMON, nullptr,
              IID_PPV_ARGS(&batch.tensors[index].resource));
          if (FAILED(result)) {
            HRESULT expected = S_OK;
            allocation_result.compare_exchange_strong(
                expected, result, std::memory_order_relaxed);
            return;
          }
        }
      };

      std::vector<std::thread> workers;
      workers.reserve(metrics.allocation_workers - 1);
      try {
        for (uint32_t worker = 1; worker < metrics.allocation_workers; ++worker) {
          workers.emplace_back(allocate);
        }
      } catch (...) {
        allocation_result.store(E_FAIL, std::memory_order_relaxed);
      }
      allocate();
      for (auto& worker : workers) {
        worker.join();
      }
      allocation_end = Clock::now();
      return allocation_result.load(std::memory_order_relaxed);
    });
  } catch (const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to start DirectStorage allocation workers: ",
                           ex.what());
  }

  const auto initialization_start = Clock::now();
  ComPtr<IDStorageFactory> factory;
  ComPtr<IDStorageQueue> queue;
  ComPtr<IDStorageStatusArray> status_array;
  ComPtr<ID3D12Fence> completion_fence;
  ScopedHandle completion_event;
  std::map<std::filesystem::path, ComPtr<IDStorageFile>> files;
  common::Status initialization_status = common::Status::OK();

  HRESULT result = DStorageGetFactory(IID_PPV_ARGS(&factory));
  if (FAILED(result)) {
    initialization_status = HResultStatus("DStorageGetFactory", result);
  }
  if (initialization_status.IsOK()) {
    result = factory->SetStagingBufferSize(static_cast<UINT32>(kMaxRequestSize));
    if (FAILED(result)) {
      initialization_status =
          HResultStatus("IDStorageFactory::SetStagingBufferSize", result);
    }
  }
  if (initialization_status.IsOK()) {
    for (const auto& tensor : batch.tensors) {
      const auto path = tensor.key.path.lexically_normal();
      if (files.find(path) != files.end()) {
        continue;
      }
      ComPtr<IDStorageFile> file;
      result = factory->OpenFile(path.c_str(), IID_PPV_ARGS(&file));
      if (FAILED(result)) {
        initialization_status =
            HResultStatus("IDStorageFactory::OpenFile", result);
        break;
      }
      files.emplace(path, std::move(file));
    }
  }

  uint32_t queue_capacity = DSTORAGE_MIN_QUEUE_CAPACITY;
  const size_t required_capacity = batch.request_count + 2;
  while (queue_capacity < required_capacity) {
    queue_capacity = GrowQueueCapacity(queue_capacity);
  }
  if (initialization_status.IsOK()) {
    DSTORAGE_QUEUE_DESC queue_descriptor{};
    queue_descriptor.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queue_descriptor.Capacity = static_cast<UINT16>(queue_capacity);
    queue_descriptor.Priority = DSTORAGE_PRIORITY_NORMAL;
    queue_descriptor.Name = "ORT WebGPU external initializers";
    queue_descriptor.Device = d3d_device;
    result = factory->CreateQueue(&queue_descriptor, IID_PPV_ARGS(&queue));
    if (FAILED(result)) {
      initialization_status =
          HResultStatus("IDStorageFactory::CreateQueue", result);
    }
  }
  if (initialization_status.IsOK()) {
    result = factory->CreateStatusArray(
        1, "ORT WebGPU external initializers", IID_PPV_ARGS(&status_array));
    if (FAILED(result)) {
      initialization_status =
          HResultStatus("IDStorageFactory::CreateStatusArray", result);
    }
  }
  if (initialization_status.IsOK()) {
    result = d3d_device->CreateFence(
        0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion_fence));
    if (FAILED(result)) {
      initialization_status = HResultStatus("ID3D12Device::CreateFence", result);
    }
  }
  if (initialization_status.IsOK()) {
    completion_event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (completion_event.value == nullptr) {
      initialization_status = HResultStatus(
          "CreateEventW", HRESULT_FROM_WIN32(GetLastError()));
    }
  }
  const auto initialization_end = Clock::now();

  HRESULT allocation_result = E_FAIL;
  try {
    allocation_result = allocation_future.get();
  } catch (const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "D3D12 allocation workers failed: ", ex.what());
  }
  ORT_RETURN_IF_ERROR(initialization_status);
  ORT_RETURN_IF_ERROR(FAILED(allocation_result)
                          ? HResultStatus(
                                "ID3D12Device::CreateCommittedResource",
                                allocation_result)
                          : common::Status::OK());
  const auto preparation_end = Clock::now();

  const auto enqueue_start = Clock::now();
  for (auto& tensor : batch.tensors) {
    IDStorageFile* file =
        files.at(tensor.key.path.lexically_normal()).Get();
    for (uint64_t tensor_offset = 0; tensor_offset < tensor.key.length;
         tensor_offset += kMaxRequestSize) {
      const UINT32 request_size = static_cast<UINT32>(
          std::min(kMaxRequestSize,
                   static_cast<uint64_t>(tensor.key.length) - tensor_offset));
      DSTORAGE_REQUEST request{};
      request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
      request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
      request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
      request.Source.File.Source = file;
      request.Source.File.Offset = tensor.key.offset + tensor_offset;
      request.Source.File.Size = request_size;
      request.Destination.Buffer.Resource = tensor.resource.Get();
      request.Destination.Buffer.Offset = tensor_offset;
      request.Destination.Buffer.Size = request_size;
      request.CancellationTag = kCancellationTag;
      request.Name = tensor.key.name.c_str();
      queue->EnqueueRequest(&request);
    }
  }
  queue->EnqueueStatus(status_array.Get(), 0);
  constexpr uint64_t kCompletionValue = 1;
  queue->EnqueueSignal(completion_fence.Get(), kCompletionValue);
  const auto enqueue_end = Clock::now();

  result = completion_fence->SetEventOnCompletion(
      kCompletionValue, completion_event.value);
  ORT_RETURN_IF_ERROR(FAILED(result)
                          ? HResultStatus(
                                "ID3D12Fence::SetEventOnCompletion", result)
                          : common::Status::OK());
  const auto io_start = Clock::now();
  queue->Submit();
  bool cancelled = false;
  DWORD wait_result = WAIT_TIMEOUT;
  while (wait_result == WAIT_TIMEOUT) {
    wait_result = WaitForSingleObject(completion_event.value, 10);
    if (!cancelled && is_cancelled && is_cancelled()) {
      queue->CancelRequestsWithTag(UINT64_MAX, kCancellationTag);
      cancelled = true;
    }
  }
  if (wait_result != WAIT_OBJECT_0) {
    while (completion_fence->GetCompletedValue() < kCompletionValue) {
      Sleep(1);
    }
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "WaitForSingleObject failed while waiting for DirectStorage: ",
        wait_result, ".");
  }
  const auto io_end = Clock::now();
  if (cancelled) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, MODEL_LOAD_CANCELED,
        "Loading DirectStorage external weights was canceled due to user request.");
  }
  result = status_array->GetHResult(0);
  ORT_RETURN_IF_ERROR(
      FAILED(result) ? HResultStatus("DirectStorage request", result)
                     : common::Status::OK());

  metrics.initialization_ms =
      Milliseconds(initialization_end - initialization_start);
  metrics.allocation_ms = Milliseconds(allocation_end - allocation_start);
  metrics.preparation_ms = Milliseconds(preparation_end - preparation_start);
  metrics.enqueue_ms = Milliseconds(enqueue_end - enqueue_start);
  metrics.io_ms = Milliseconds(io_end - io_start);
  return common::Status::OK();
}

TensorKey MakeKey(const std::filesystem::path& path, std::string_view name,
                  uint64_t offset, size_t length) {
  return {path, std::string{name}, offset, length};
}

common::Status PrepareTensorForBatch(
    DirectStorageBatch& batch,
    const Env& env,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length) {
  ORT_RETURN_IF(data_offset < 0, "DirectStorage initializer \"", tensor_name,
                "\" has a negative file offset.");

  const size_t length = static_cast<size_t>(data_length);
  ORT_RETURN_IF(length > std::numeric_limits<uint64_t>::max() - 3,
                "DirectStorage initializer \"", tensor_name,
                "\" is too large to align.");

  const uint64_t offset = static_cast<uint64_t>(data_offset);
  ORT_RETURN_IF(offset > std::numeric_limits<uint64_t>::max() -
                             static_cast<uint64_t>(length),
                "DirectStorage initializer \"", tensor_name,
                "\" file range overflows.");

  const auto file_key = data_file_path.lexically_normal();
  auto file_iterator = batch.files.find(file_key);
  if (file_iterator == batch.files.end()) {
    size_t file_length = 0;
    ORT_RETURN_IF_ERROR(env.GetFileLength(data_file_path.c_str(), file_length));
    std::error_code path_error;
    const auto normalized_path =
        std::filesystem::weakly_canonical(data_file_path, path_error);
    ORT_RETURN_IF(path_error,
                  "Failed to canonicalize DirectStorage data file \"",
                  data_file_path.string(), "\": ", path_error.message());
    file_iterator =
        batch.files.emplace(
                       file_key, DirectStorageBatch::FileInfo{normalized_path, file_length})
            .first;
  }

  const size_t file_length = file_iterator->second.length;
  ORT_RETURN_IF(offset > static_cast<uint64_t>(file_length) ||
                    static_cast<uint64_t>(length) >
                        static_cast<uint64_t>(file_length) - offset,
                "DirectStorage initializer \"", tensor_name,
                "\" range [", offset, ", ", offset + length,
                ") exceeds file size ", file_length, " for \"",
                data_file_path.string(), "\".");

  TensorKey key = MakeKey(
      file_iterator->second.canonical_path, tensor_name, offset, length);
  ORT_RETURN_IF(batch.tensors_by_key.find(key) != batch.tensors_by_key.end(),
                "Duplicate DirectStorage initializer: ", tensor_name);

  const size_t requests = length / kMaxRequestSize +
                          (length % kMaxRequestSize != 0 ? 1 : 0);
  ORT_RETURN_IF(batch.request_count >
                    std::numeric_limits<size_t>::max() - requests,
                "DirectStorage request count overflow.");
  ORT_RETURN_IF(batch.total_bytes >
                    std::numeric_limits<size_t>::max() - length,
                "DirectStorage initializer byte count overflow.");

  const size_t index = batch.tensors.size();
  batch.tensors.push_back({std::move(key)});
  batch.tensors_by_key.emplace(batch.tensors.back().key, index);
  batch.request_count += requests;
  batch.total_bytes += length;
  return common::Status::OK();
}

struct ImportedAllocation {
  ComPtr<ID3D12Resource> resource;
  wgpu::SharedBufferMemory memory;
  wgpu::Buffer buffer;
  bool access_started = false;
};

void EndAccessNoThrow(ImportedAllocation& allocation) noexcept {
  if (!allocation.access_started || !allocation.memory || !allocation.buffer) {
    return;
  }

  try {
    wgpu::SharedBufferMemoryEndAccessState end_state{};
    allocation.memory.EndAccess(allocation.buffer, &end_state);
    allocation.access_started = false;
  } catch (...) {
    // Cleanup paths, including allocator Free, must not propagate Dawn failures.
  }
}

}  // namespace

common::Status CheckDirectStorageExternalWeightsSupport(WebGpuContext& context) {
  context.WaitForStartInitializeComplete();
  ORT_RETURN_IF_NOT(context.RequestedBackendType() == wgpu::BackendType::D3D12,
                    "DirectStorage external weights require the Dawn D3D12 backend.");
  if (!context.PipelinedWeightLoadingEnabled()) {
    ORT_RETURN_IF_NOT(
        context.DirectStorageSharedResourceFeaturesAvailable(),
        "DirectStorage external weights require Dawn SharedBufferMemoryD3D12Resource and "
        "SharedFenceDXGISharedHandle features.");
  }

  ID3D12Device* d3d_device = context.DirectStorageD3D12Device();
  ORT_RETURN_IF_NOT(d3d_device != nullptr,
                    "Failed to create a D3D12 device for DirectStorage.");

  ComPtr<IDStorageFactory> factory;
  HRESULT result = DStorageGetFactory(IID_PPV_ARGS(&factory));
  return FAILED(result)
             ? HResultStatus("DStorageGetFactory", result)
             : common::Status::OK();
}

common::Status ResolveWeightLoadAccelerationMode(
    WeightLoadAccelerationMode mode,
    const common::Status& support_status,
    bool& enabled) {
  enabled = false;
  if (!IsWeightLoadAccelerationEnabled(mode)) {
    return common::Status::OK();
  }
  if (support_status.IsOK()) {
    enabled = true;
    return common::Status::OK();
  }
  if (IsWeightLoadAccelerationRequired(mode)) {
    return support_status;
  }
  return common::Status::OK();
}

struct DirectStorageInitializerState::Impl {
  std::mutex mutex;
  IAllocator* allocator = nullptr;
  std::unordered_map<WGPUBuffer, std::unique_ptr<ImportedAllocation>> imported_allocations;
};

DirectStorageInitializerState::DirectStorageInitializerState() = default;

DirectStorageInitializerState::~DirectStorageInitializerState() {
  if (!impl_) {
    return;
  }

  std::vector<std::unique_ptr<ImportedAllocation>> allocations;
  {
    std::lock_guard<std::mutex> lock{impl_->mutex};
    allocations.reserve(impl_->imported_allocations.size());
    for (auto& entry : impl_->imported_allocations) {
      allocations.push_back(std::move(entry.second));
    }
    impl_->imported_allocations.clear();
  }
  for (auto& allocation : allocations) {
    EndAccessNoThrow(*allocation);
  }
}

class DirectStorageWebGpuAllocator final : public IAllocator {
 public:
  DirectStorageWebGpuAllocator(WebGpuContext& context,
                               std::shared_ptr<DirectStorageInitializerState> state)
      : IAllocator(OrtMemoryInfo(WEBGPU_BUFFER,
                                 OrtAllocatorType::OrtReadOnlyAllocator,
                                 WebGpuDevice,
                                 OrtMemTypeDefault)),
        context_{context},
        state_{std::move(state)} {
  }

  void* Alloc(size_t size) override {
    if (size == 0) {
      return nullptr;
    }

    const auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                       wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;
    return context_.InitializerBufferManager().Create(size, usage);
  }

  void Free(void* p) override {
    if (p == nullptr) {
      return;
    }

    std::unique_ptr<ImportedAllocation> imported;
    {
      std::lock_guard<std::mutex> lock{state_->impl_->mutex};
      const auto iterator =
          state_->impl_->imported_allocations.find(static_cast<WGPUBuffer>(p));
      if (iterator != state_->impl_->imported_allocations.end()) {
        imported = std::move(iterator->second);
        state_->impl_->imported_allocations.erase(iterator);
      }
    }

    if (imported) {
      EndAccessNoThrow(*imported);
      return;
    }

    context_.InitializerBufferManager().Release(static_cast<WGPUBuffer>(p));
  }

 private:
  WebGpuContext& context_;
  std::shared_ptr<DirectStorageInitializerState> state_;
};

AllocatorPtr CreateDirectStorageWebGpuAllocator(
    WebGpuContext& context, std::shared_ptr<DirectStorageInitializerState>& out_state) {
  auto state = std::shared_ptr<DirectStorageInitializerState>(
      new DirectStorageInitializerState());
  state->impl_ = std::make_unique<DirectStorageInitializerState::Impl>();
  auto allocator =
      std::make_shared<DirectStorageWebGpuAllocator>(context, state);
  state->impl_->allocator = allocator.get();
  out_state = state;
  return allocator;
}

struct DirectStorageExternalDataLoader::Impl {
  Impl(WebGpuContext& context_in,
       std::shared_ptr<DirectStorageInitializerState> state_in,
       WeightLoadAccelerationMode mode_in)
      : context{context_in},
        state{std::move(state_in)},
        mode{mode_in} {
  }

  void ResolveSupport() const {
    std::call_once(support_once, [this]() {
      auto support_status = CheckDirectStorageExternalWeightsSupport(context);
      if (support_status.IsOK() &&
          mode == WeightLoadAccelerationMode::RequiredPipelined &&
          !context.PipelinedWeightLoadingEnabled()) {
        support_status = ORT_MAKE_STATUS(
            ONNXRUNTIME, FAIL,
            "weightLoadAcceleration=required-pipelined requires the WebGPU "
            "context to be initialized with pipelined weight loading.");
      }
      resolved_status =
          ResolveWeightLoadAccelerationMode(mode, support_status, enabled);
      if (!enabled) {
        context.ContinueInitialize();
      }
      if (resolved_status.IsOK() && !enabled &&
          !IsWeightLoadAccelerationRequired(mode)) {
        LOGS_DEFAULT(WARNING)
            << "DirectStorage external weights are unavailable; using the ordinary "
               "WebGPU initializer loading path. Reason: "
            << support_status.ErrorMessage();
      }
    });
  }

  WebGpuContext& context;
  std::shared_ptr<DirectStorageInitializerState> state;
  WeightLoadAccelerationMode mode;
  mutable std::once_flag support_once;
  mutable common::Status resolved_status;
  mutable bool enabled = false;
  mutable std::unique_ptr<DirectStorageBatch> preload_batch;
  mutable DirectStorageLoadMetrics preload_metrics;
  mutable std::future<common::Status> preload_future;
  mutable std::unique_ptr<DirectStorageBatch> batch;
};

DirectStorageExternalDataLoader::DirectStorageExternalDataLoader(
    WebGpuContext& context,
    std::shared_ptr<DirectStorageInitializerState> state,
    WeightLoadAccelerationMode mode)
    : impl_{std::make_unique<Impl>(context, std::move(state), mode)} {
  ORT_ENFORCE(impl_->state != nullptr && impl_->state->impl_ != nullptr,
              "DirectStorage allocator state is required.");
}

DirectStorageExternalDataLoader::~DirectStorageExternalDataLoader() {
  AbortLoad();
}

bool DirectStorageExternalDataLoader::CanLoad(
    const OrtMemoryInfo& target_memory_info) const {
  impl_->ResolveSupport();
  return target_memory_info.device == WebGpuDevice &&
         target_memory_info.name == WEBGPU_BUFFER &&
         impl_->enabled;
}

bool DirectStorageExternalDataLoader::CreatesTensorForDevice(
    const OrtDevice& target_device) const {
  impl_->ResolveSupport();
  return target_device == WebGpuDevice &&
         (impl_->enabled ||
          IsWeightLoadAccelerationRequired(impl_->mode));
}

bool DirectStorageExternalDataLoader::SupportsPreload() const {
  return true;
}

common::Status DirectStorageExternalDataLoader::BeginPreload() const {
  impl_->ResolveSupport();
  ORT_RETURN_IF_ERROR(impl_->resolved_status);
  if (!impl_->enabled) {
    return common::Status::OK();
  }
  impl_->preload_batch = std::make_unique<DirectStorageBatch>();
  return common::Status::OK();
}

common::Status DirectStorageExternalDataLoader::PreloadTensor(
    const Env& env,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length) const {
  if (!impl_->enabled) {
    return common::Status::OK();
  }
  ORT_RETURN_IF_NOT(impl_->preload_batch != nullptr,
                    "DirectStorage preload batch has not been started.");
  return PrepareTensorForBatch(*impl_->preload_batch, env, data_file_path,
                               tensor_name, data_offset, data_length);
}

common::Status DirectStorageExternalDataLoader::FinalizePreload(
    const std::function<bool()>& is_cancelled) const {
  if (!impl_->enabled) {
    return common::Status::OK();
  }
  ORT_RETURN_IF_NOT(impl_->preload_batch != nullptr,
                    "DirectStorage preload batch has not been started.");
  if (impl_->preload_batch->request_count == 0) {
    impl_->context.ContinueInitialize();
    return common::Status::OK();
  }

  ComPtr<ID3D12Device> d3d_device = impl_->context.DirectStorageD3D12Device();
  ORT_RETURN_IF_NOT(d3d_device != nullptr,
                    "Failed to access the DirectStorage D3D12 device.");
  try {
    impl_->preload_future = std::async(
        std::launch::async,
        [batch = impl_->preload_batch.get(), &metrics = impl_->preload_metrics,
         d3d_device, is_cancelled]() {
          return LoadBatchToD3D12(
              *batch, d3d_device.Get(), is_cancelled, metrics);
        });
  } catch (const std::exception& ex) {
    const auto status = ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "Failed to start the DirectStorage initializer preload: ", ex.what());
    if (!IsWeightLoadAccelerationRequired(impl_->mode)) {
      LOGS_DEFAULT(WARNING)
          << status.ErrorMessage()
          << " Using the ordinary WebGPU initializer path.";
      impl_->enabled = false;
      impl_->preload_batch.reset();
      impl_->context.ContinueInitialize();
      return common::Status::OK();
    }
    impl_->context.ContinueInitialize();
    return status;
  }
  impl_->context.ContinueInitialize();
  return common::Status::OK();
}

common::Status DirectStorageExternalDataLoader::BeginLoad() const {
  impl_->batch.reset();
  impl_->ResolveSupport();
  ORT_RETURN_IF_ERROR(impl_->resolved_status);
  if (!impl_->enabled) {
    return common::Status::OK();
  }
  impl_->batch = std::make_unique<DirectStorageBatch>();
  return common::Status::OK();
}

common::Status DirectStorageExternalDataLoader::PrepareTensor(
    const Env& env,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length) const {
  ORT_RETURN_IF_NOT(impl_->batch != nullptr && !impl_->batch->finalized,
                    "DirectStorage initializer batch has not been started.");
  return PrepareTensorForBatch(*impl_->batch, env, data_file_path, tensor_name,
                               data_offset, data_length);
}

common::Status DirectStorageExternalDataLoader::FinalizeLoad(
    const std::function<bool()>& is_cancelled) const {
  if (!impl_->enabled) {
    return common::Status::OK();
  }
  ORT_RETURN_IF_NOT(impl_->batch != nullptr && !impl_->batch->finalized,
                    "DirectStorage initializer batch has not been started.");
  auto& batch = *impl_->batch;
  const auto fail_or_fallback =
      [this](const common::Status& status) -> common::Status {
    if (IsWeightLoadAccelerationRequired(impl_->mode) ||
        status.Code() == common::MODEL_LOAD_CANCELED) {
      return status;
    }
    LOGS_DEFAULT(WARNING)
        << "DirectStorage initializer loading failed; using the ordinary WebGPU "
           "initializer path: "
        << status.ErrorMessage();
    impl_->enabled = false;
    AbortLoad();
    impl_->context.WaitForInitializeComplete();
    return common::Status::OK();
  };
  if (batch.request_count == 0) {
    common::Status preload_status = common::Status::OK();
    if (impl_->preload_future.valid()) {
      preload_status = impl_->preload_future.get();
    }
    impl_->preload_batch.reset();
    impl_->context.ContinueInitialize();
    if (!preload_status.IsOK()) {
      return fail_or_fallback(preload_status);
    }
    impl_->context.WaitForInitializeComplete();
    batch.finalized = true;
    return common::Status::OK();
  }

  ComPtr<ID3D12Device> d3d_device = impl_->context.DirectStorageD3D12Device();
  if (d3d_device == nullptr) {
    return fail_or_fallback(ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "Failed to access the DirectStorage D3D12 device."));
  }

  DirectStorageLoadMetrics load_metrics;
  bool used_preload = false;
  if (impl_->preload_batch && impl_->preload_future.valid()) {
    const auto preload_status = impl_->preload_future.get();
    if (!preload_status.IsOK()) {
      return fail_or_fallback(preload_status);
    }
    used_preload = std::all_of(
        batch.tensors.begin(), batch.tensors.end(),
        [&](const PreparedTensor& tensor) {
          return impl_->preload_batch->tensors_by_key.find(tensor.key) !=
                 impl_->preload_batch->tensors_by_key.end();
        });
    if (used_preload) {
      for (auto& tensor : batch.tensors) {
        auto& preloaded_tensor = impl_->preload_batch->tensors.at(
            impl_->preload_batch->tensors_by_key.at(tensor.key));
        tensor.resource = std::move(preloaded_tensor.resource);
      }
      load_metrics = impl_->preload_metrics;
    }
    impl_->preload_batch.reset();
  }

  if (!used_preload) {
    const auto load_status =
        LoadBatchToD3D12(batch, d3d_device.Get(), is_cancelled, load_metrics);
    if (!load_status.IsOK()) {
      return fail_or_fallback(load_status);
    }
  }
  impl_->context.WaitForInitializeComplete();

  const auto import_start = Clock::now();
  const auto import_status = [&]() -> common::Status {
    ORT_RETURN_IF_NOT(
        impl_->context.DirectStorageSharedResourceFeaturesAvailable(),
        "DirectStorage external weights require Dawn "
        "SharedBufferMemoryD3D12Resource and SharedFenceDXGISharedHandle "
        "features.");
    ComPtr<ID3D12Device> dawn_d3d_device =
        dawn::native::d3d12::GetD3D12Device(impl_->context.Device().Get());
    ORT_RETURN_IF_NOT(dawn_d3d_device.Get() == d3d_device.Get(),
                      "DirectStorage and Dawn used different D3D12 devices.");

    for (auto& tensor : batch.tensors) {
      if (tensor.key.length == 0) {
        continue;
      }
      dawn::native::d3d12::SharedBufferMemoryD3D12ResourceDescriptor
          resource_descriptor;
      resource_descriptor.resource = tensor.resource;
      wgpu::SharedBufferMemoryDescriptor memory_descriptor{};
      memory_descriptor.label = tensor.key.name.c_str();
      memory_descriptor.nextInChain = &resource_descriptor;
      tensor.memory =
          impl_->context.Device().ImportSharedBufferMemory(&memory_descriptor);
      ORT_RETURN_IF_NOT(tensor.memory, "Failed to import DirectStorage initializer \"",
                        tensor.key.name, "\" into Dawn.");

      wgpu::SharedBufferMemoryProperties properties{};
      ORT_RETURN_IF_NOT(tensor.memory.GetProperties(&properties) == wgpu::Status::Success,
                        "Failed to query imported DirectStorage initializer \"",
                        tensor.key.name, "\".");
      ORT_RETURN_IF(properties.size <
                        static_cast<uint64_t>(tensor.key.length),
                    "Imported DirectStorage initializer \"", tensor.key.name,
                    "\" is smaller than its tensor payload.");

      wgpu::BufferDescriptor buffer_descriptor{};
      buffer_descriptor.label = tensor.key.name.c_str();
      buffer_descriptor.size = properties.size;
      buffer_descriptor.usage = wgpu::BufferUsage::Storage |
                                wgpu::BufferUsage::CopySrc |
                                wgpu::BufferUsage::CopyDst;
      tensor.buffer = tensor.memory.CreateBuffer(&buffer_descriptor);
      ORT_RETURN_IF_NOT(tensor.buffer,
                        "Failed to create buffer for DirectStorage initializer \"",
                        tensor.key.name, "\".");

      wgpu::SharedBufferMemoryBeginAccessDescriptor access_descriptor{};
      access_descriptor.initialized = true;
      ORT_RETURN_IF_NOT(tensor.memory.BeginAccess(
                            tensor.buffer, &access_descriptor) == wgpu::Status::Success,
                        "Failed to begin Dawn access for DirectStorage initializer \"",
                        tensor.key.name, "\".");
      tensor.access_started = true;
    }
    return common::Status::OK();
  }();
  if (!import_status.IsOK()) {
    return fail_or_fallback(import_status);
  }
  const auto import_end = Clock::now();
  batch.finalized = true;

  LOGS_DEFAULT(INFO)
      << "WebGPU DirectStorage external initializer load: "
      << "preloaded=" << used_preload << ", "
      << "DirectStorage initialization=" << load_metrics.initialization_ms << " ms, "
      << "D3D12 allocation=" << load_metrics.allocation_ms << " ms, "
      << "overlapped preparation=" << load_metrics.preparation_ms << " ms, "
      << "enqueue=" << load_metrics.enqueue_ms << " ms, "
      << "file-to-GPU=" << load_metrics.io_ms << " ms, "
      << "Dawn import/access=" << Milliseconds(import_end - import_start)
      << " ms, bytes=" << batch.total_bytes
      << ", tensors=" << batch.tensors.size()
      << ", requests=" << batch.request_count
      << ", allocation_workers=" << load_metrics.allocation_workers << ".";
  return common::Status::OK();
}

void DirectStorageExternalDataLoader::AbortLoad() const noexcept {
  if (impl_) {
    impl_->context.ContinueInitialize();
  }
  if (!impl_) {
    return;
  }

  try {
    if (impl_->preload_future.valid()) {
      impl_->preload_future.wait();
    }
    impl_->preload_batch.reset();
    if (impl_->batch) {
      for (auto& tensor : impl_->batch->tensors) {
        if (!tensor.claimed && tensor.access_started) {
          ImportedAllocation allocation;
          allocation.resource = std::move(tensor.resource);
          allocation.memory = std::move(tensor.memory);
          allocation.buffer = std::move(tensor.buffer);
          allocation.access_started = true;
          EndAccessNoThrow(allocation);
          tensor.access_started = false;
        }
      }
      impl_->batch.reset();
    }
  } catch (...) {
    // Abort is best-effort and is required not to throw.
  }
}

common::Status DirectStorageExternalDataLoader::LoadTensor(
    const Env&,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length,
    const std::shared_ptr<IAllocator>& allocator,
    Tensor& tensor) const {
  ORT_RETURN_IF_NOT(impl_->batch != nullptr && impl_->batch->finalized,
                    "DirectStorage initializer batch has not been finalized.");
  ORT_RETURN_IF(data_offset < 0,
                "DirectStorage initializer has a negative file offset.");
  ORT_RETURN_IF_NOT(allocator != nullptr,
                    "DirectStorage initializer requires its device allocator.");
  ORT_RETURN_IF_NOT(allocator.get() == impl_->state->impl_->allocator,
                    "DirectStorage initializer was passed a different allocator.");

  const size_t length = static_cast<size_t>(data_length);
  const auto file_iterator =
      impl_->batch->files.find(data_file_path.lexically_normal());
  ORT_RETURN_IF(file_iterator == impl_->batch->files.end(),
                "No prepared DirectStorage data file matches \"",
                data_file_path.string(), "\".");
  const TensorKey key =
      MakeKey(file_iterator->second.canonical_path, tensor_name,
              static_cast<uint64_t>(data_offset), length);
  const auto iterator = impl_->batch->tensors_by_key.find(key);
  ORT_RETURN_IF(iterator == impl_->batch->tensors_by_key.end(),
                "No prepared DirectStorage initializer matches \"",
                tensor_name, "\" and the requested file range.");

  auto& prepared = impl_->batch->tensors[iterator->second];
  ORT_RETURN_IF(prepared.claimed,
                "DirectStorage initializer \"", tensor_name,
                "\" has already been consumed.");
  ORT_RETURN_IF(length != tensor.SizeInBytes(),
                "DirectStorage initializer \"", tensor_name,
                "\" length does not match the placeholder tensor.");
  if (length == 0) {
    prepared.claimed = true;
    tensor = Tensor{tensor.DataType(), tensor.Shape(), nullptr, allocator};
    return common::Status::OK();
  }
  ORT_RETURN_IF_NOT(prepared.buffer && prepared.access_started,
                    "DirectStorage initializer \"", tensor_name,
                    "\" does not have an imported buffer.");

  auto imported = std::make_unique<ImportedAllocation>();
  imported->resource = std::move(prepared.resource);
  imported->memory = std::move(prepared.memory);
  imported->buffer = std::move(prepared.buffer);
  imported->access_started = prepared.access_started;
  WGPUBuffer buffer = imported->buffer.Get();

  {
    std::lock_guard<std::mutex> lock{impl_->state->impl_->mutex};
    ORT_RETURN_IF(impl_->state->impl_->imported_allocations.find(buffer) !=
                      impl_->state->impl_->imported_allocations.end(),
                  "DirectStorage imported buffer was registered twice.");
    impl_->state->impl_->imported_allocations.emplace(
        buffer, std::move(imported));
  }

  prepared.access_started = false;
  prepared.claimed = true;
  tensor = Tensor{tensor.DataType(), tensor.Shape(), buffer, allocator};
  return common::Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
