// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)

#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/framework/tensor.h"
#include "core/platform/windows/d3d12_file_loader/d3d12_file_buffer_loader.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/d3d12_external_data_loader.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

using Microsoft::WRL::ComPtr;
using D3D12FileBufferLoader = windows::d3d12::D3D12FileBufferLoader;

namespace {

using Clock = std::chrono::steady_clock;

struct SharedBufferMemoryD3D12ResourceDescriptor : wgpu::ChainedStruct {
  SharedBufferMemoryD3D12ResourceDescriptor() {
    sType = static_cast<wgpu::SType>(
        WGPUSType_SharedBufferMemoryD3D12ResourceDescriptor);
  }

  ComPtr<ID3D12Resource> resource;
};

double Milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

struct TensorKey {
  std::filesystem::path path;
  std::string name;
  uint64_t offset;
  size_t length;

  bool operator<(const TensorKey& other) const {
    if (path < other.path) {
      return true;
    }
    if (other.path < path) {
      return false;
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
  std::shared_ptr<D3D12FileBufferLoader::Batch> heap_owner;
  wgpu::SharedBufferMemory memory;
  wgpu::Buffer buffer;
  bool access_started = false;
  bool claimed = false;
};

struct D3D12AcceleratedBatch {
  struct FileInfo {
    std::filesystem::path canonical_path;
    size_t length;
  };

  std::vector<PreparedTensor> tensors;
  std::map<TensorKey, size_t> tensors_by_key;
  std::map<std::filesystem::path, FileInfo> files;
  size_t total_bytes = 0;
  size_t load_range_count = 0;
  bool finalized = false;
};

struct D3D12AcceleratedLoadMetrics {
  double load_ms = 0.0;
  size_t loaded_ranges = 0;
};

struct CancellationState {
  const std::function<bool()>* external = nullptr;
  const std::atomic<bool>* abort_requested = nullptr;
};

bool InvokeCancellationNoThrow(
    const std::function<bool()>& is_canceled) noexcept {
  if (!is_canceled) {
    return false;
  }
  ORT_TRY {
    return is_canceled();
  }
  ORT_CATCH(...) {
    return true;
  }
}

bool IsCancellationRequested(void* opaque) noexcept {
  const auto& state = *static_cast<const CancellationState*>(opaque);
  if (state.abort_requested != nullptr &&
      state.abort_requested->load(std::memory_order_relaxed)) {
    return true;
  }
  if (state.external == nullptr || !*state.external) {
    return false;
  }
  return InvokeCancellationNoThrow(*state.external);
}

common::Status LoadBatchToD3D12(
    D3D12AcceleratedBatch& batch,
    D3D12FileBufferLoader& loader,
    const std::function<bool()>& is_canceled,
    const std::atomic<bool>& abort_requested,
    D3D12AcceleratedLoadMetrics& metrics) {
  if (abort_requested.load(std::memory_order_relaxed) ||
      InvokeCancellationNoThrow(is_canceled)) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, MODEL_LOAD_CANCELED,
        "D3D12 accelerated initializer loading was canceled.");
  }

  std::vector<D3D12FileBufferLoader::FileRange> ranges;
  std::vector<size_t> tensor_indices;
  ranges.reserve(batch.load_range_count);
  tensor_indices.reserve(batch.load_range_count);
  for (size_t index = 0; index < batch.tensors.size(); ++index) {
    const auto& tensor = batch.tensors[index];
    if (tensor.key.length == 0) {
      continue;
    }
    ranges.push_back(
        {tensor.key.path.native(),
         tensor.key.offset,
         static_cast<uint64_t>(tensor.key.length)});
    tensor_indices.push_back(index);
  }

  ORT_RETURN_IF_NOT(
      ranges.size() == batch.load_range_count,
      "D3D12 accelerated initializer range count is inconsistent.");
  if (ranges.empty()) {
    return common::Status::OK();
  }

  auto loaded = std::make_shared<D3D12FileBufferLoader::Batch>();
  const CancellationState cancellation_state{
      &is_canceled, &abort_requested};
  const D3D12FileBufferLoader::CancellationToken cancellation{
      IsCancellationRequested,
      const_cast<CancellationState*>(&cancellation_state)};

  const auto load_start = Clock::now();
  ORT_RETURN_IF_ERROR(loader.Load(ranges, *loaded, cancellation));
  const auto load_end = Clock::now();
  ORT_RETURN_IF_NOT(
      loaded->buffers.size() == tensor_indices.size(),
      "D3D12 file loader returned an unexpected buffer count.");

  for (size_t index = 0; index < tensor_indices.size(); ++index) {
    auto& tensor = batch.tensors[tensor_indices[index]];
    tensor.resource = loaded->buffers[index].resource;
    tensor.heap_owner = loaded;
  }

  metrics.load_ms = Milliseconds(load_end - load_start);
  metrics.loaded_ranges = ranges.size();
  return common::Status::OK();
}

TensorKey MakeKey(const std::filesystem::path& path,
                  std::string_view name,
                  uint64_t offset,
                  size_t length) {
  return {path, std::string{name}, offset, length};
}

common::Status PrepareTensorForBatch(
    D3D12AcceleratedBatch& batch,
    const Env& env,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length) {
  ORT_RETURN_IF(
      data_offset < 0,
      "D3D12 accelerated initializer \"", tensor_name,
      "\" has a negative file offset.");

  const size_t length = static_cast<size_t>(data_length);
  const uint64_t offset = static_cast<uint64_t>(data_offset);
  ORT_RETURN_IF(
      offset > std::numeric_limits<uint64_t>::max() -
                   static_cast<uint64_t>(length),
      "D3D12 accelerated initializer \"", tensor_name,
      "\" file range overflows.");

  const auto file_key = data_file_path.lexically_normal();
  auto file_iterator = batch.files.find(file_key);
  if (file_iterator == batch.files.end()) {
    size_t file_length = 0;
    ORT_RETURN_IF_ERROR(
        env.GetFileLength(data_file_path.c_str(), file_length));
    std::error_code path_error;
    const auto normalized_path =
        std::filesystem::weakly_canonical(
            data_file_path, path_error);
    ORT_RETURN_IF(
        path_error,
        "Failed to canonicalize D3D12 data file \"",
        data_file_path.string(), "\": ",
        path_error.message());
    file_iterator =
        batch.files
            .emplace(
                file_key,
                D3D12AcceleratedBatch::FileInfo{
                    normalized_path, file_length})
            .first;
  }

  const size_t file_length = file_iterator->second.length;
  ORT_RETURN_IF(
      offset > static_cast<uint64_t>(file_length) ||
          static_cast<uint64_t>(length) >
              static_cast<uint64_t>(file_length) - offset,
      "D3D12 accelerated initializer \"", tensor_name,
      "\" range [", offset, ", ", offset + length,
      ") exceeds file size ", file_length, " for \"",
      data_file_path.string(), "\".");

  TensorKey key = MakeKey(
      file_iterator->second.canonical_path,
      tensor_name, offset, length);
  ORT_RETURN_IF(
      batch.tensors_by_key.find(key) !=
          batch.tensors_by_key.end(),
      "Duplicate D3D12 accelerated initializer \"",
      tensor_name, "\" from \"", key.path.string(),
      "\" range [", offset, ", ", offset + length, ").");
  ORT_RETURN_IF(
      batch.total_bytes >
          std::numeric_limits<size_t>::max() - length,
      "D3D12 accelerated initializer byte count overflow.");

  const size_t index = batch.tensors.size();
  batch.tensors.push_back({std::move(key)});
  batch.tensors_by_key.emplace(
      batch.tensors.back().key, index);
  batch.total_bytes += length;
  if (length != 0) {
    ++batch.load_range_count;
  }
  return common::Status::OK();
}

struct ImportedAllocation {
  ComPtr<ID3D12Resource> resource;
  std::shared_ptr<D3D12FileBufferLoader::Batch> heap_owner;
  wgpu::SharedBufferMemory memory;
  wgpu::Buffer buffer;
  bool access_started = false;
};

void EndAccessNoThrow(
    ImportedAllocation& allocation) noexcept {
  if (!allocation.access_started ||
      !allocation.memory || !allocation.buffer) {
    return;
  }

  ORT_TRY {
    wgpu::SharedBufferMemoryEndAccessState end_state{};
    allocation.memory.EndAccess(
        allocation.buffer, &end_state);
    allocation.access_started = false;
  }
  ORT_CATCH(...) {
    // Cleanup paths, including allocator Free, must not propagate Dawn failures.
  }
}

}  // namespace

common::Status CheckD3D12AcceleratedExternalWeightsSupport(
    const WebGpuContext& context) {
  ORT_RETURN_IF_NOT(
      context.SelectedBackendType() ==
          wgpu::BackendType::D3D12,
      "D3D12 accelerated external weights require the Dawn D3D12 backend.");
  ORT_RETURN_IF_NOT(
      context.D3D12SharedResourceFeaturesAvailable(),
      "D3D12 accelerated external weights require Dawn "
      "SharedBufferMemoryD3D12Resource and SharedFenceDXGISharedHandle features.");

  ORT_RETURN_IF_NOT(
      context.WeightLoadingD3D12Device() != nullptr,
      "Failed to create a D3D12 device for accelerated file loading.");
  return common::Status::OK();
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

struct D3D12AcceleratedInitializerState::Impl {
  std::mutex mutex;
  IAllocator* allocator = nullptr;
  std::unordered_map<
      WGPUBuffer,
      std::unique_ptr<ImportedAllocation>>
      imported_allocations;
};

D3D12AcceleratedInitializerState::
    D3D12AcceleratedInitializerState() = default;

D3D12AcceleratedInitializerState::
    ~D3D12AcceleratedInitializerState() {
  if (!impl_) {
    return;
  }

  std::vector<std::unique_ptr<ImportedAllocation>>
      allocations;
  {
    std::lock_guard<std::mutex> lock{impl_->mutex};
    allocations.reserve(
        impl_->imported_allocations.size());
    for (auto& entry : impl_->imported_allocations) {
      allocations.push_back(std::move(entry.second));
    }
    impl_->imported_allocations.clear();
  }
  for (auto& allocation : allocations) {
    EndAccessNoThrow(*allocation);
  }
}

class D3D12AcceleratedWebGpuAllocator final
    : public IAllocator {
 public:
  D3D12AcceleratedWebGpuAllocator(
      WebGpuContext& context,
      std::function<CommandRecordingState&()>
          recording_getter,
      std::shared_ptr<D3D12AcceleratedInitializerState>
          state)
      : IAllocator(
            OrtMemoryInfo(
                WEBGPU_BUFFER,
                OrtAllocatorType::OrtReadOnlyAllocator,
                WebGpuDevice,
                OrtMemTypeDefault)),
        context_{context},
        recording_getter_{
            std::move(recording_getter)},
        state_{std::move(state)} {
  }

  void* Alloc(size_t size) override {
    if (size == 0) {
      return nullptr;
    }

    const auto usage =
        wgpu::BufferUsage::Storage |
        wgpu::BufferUsage::CopySrc |
        wgpu::BufferUsage::CopyDst |
        wgpu::BufferUsage::Indirect;
    auto& recording = recording_getter_();
    std::lock_guard<std::recursive_mutex> lock{
        recording.mutex};
    return context_.InitializerBufferManager().Create(
        recording, size, usage);
  }

  void Free(void* p) override {
    if (p == nullptr) {
      return;
    }

    auto& recording = recording_getter_();
    std::lock_guard<std::recursive_mutex>
        recording_lock{recording.mutex};
    std::unique_ptr<ImportedAllocation> imported;
    {
      std::lock_guard<std::mutex> lock{
          state_->impl_->mutex};
      const auto iterator =
          state_->impl_->imported_allocations.find(
              static_cast<WGPUBuffer>(p));
      if (iterator !=
          state_->impl_->imported_allocations.end()) {
        imported = std::move(iterator->second);
        state_->impl_->imported_allocations.erase(
            iterator);
      }
    }

    if (imported) {
      if (recording.has_unsubmitted_work) {
        std::shared_ptr<ImportedAllocation> deferred{
            imported.release(),
            [](ImportedAllocation* allocation) {
              EndAccessNoThrow(*allocation);
              delete allocation;
            }};
        recording.pending_release_callbacks.emplace_back(
            [deferred = std::move(deferred)]() mutable {
              deferred.reset();
            });
      } else {
        EndAccessNoThrow(*imported);
      }
      return;
    }

    context_.InitializerBufferManager().Release(
        recording, static_cast<WGPUBuffer>(p));
  }

 private:
  WebGpuContext& context_;
  std::function<CommandRecordingState&()>
      recording_getter_;
  std::shared_ptr<D3D12AcceleratedInitializerState>
      state_;
};

AllocatorPtr CreateD3D12AcceleratedWebGpuAllocator(
    WebGpuContext& context,
    std::function<CommandRecordingState&()>
        recording_getter,
    std::shared_ptr<D3D12AcceleratedInitializerState>&
        out_state) {
  auto state =
      std::shared_ptr<D3D12AcceleratedInitializerState>(
          new D3D12AcceleratedInitializerState());
  state->impl_ =
      std::make_unique<
          D3D12AcceleratedInitializerState::Impl>();
  auto allocator =
      std::make_shared<
          D3D12AcceleratedWebGpuAllocator>(
          context, std::move(recording_getter), state);
  state->impl_->allocator = allocator.get();
  out_state = state;
  return allocator;
}

struct D3D12AcceleratedExternalDataLoader::Impl {
  Impl(
      WebGpuContext& context_in,
      std::shared_ptr<D3D12AcceleratedInitializerState>
          state_in,
      WeightLoadAccelerationMode mode_in)
      : context{context_in},
        state{std::move(state_in)},
        mode{mode_in} {
  }

  void ResolveSupport() const {
    std::call_once(support_once, [this]() {
      common::Status support_status =
          common::Status::OK();
      if (IsWeightLoadAccelerationEnabled(mode)) {
        support_status =
            CheckD3D12AcceleratedExternalWeightsSupport(
                context);
      }
      resolved_status =
          ResolveWeightLoadAccelerationMode(
              mode, support_status, enabled);
      if (resolved_status.IsOK() && !enabled &&
          mode == WeightLoadAccelerationMode::Preferred) {
        LOGS_DEFAULT(WARNING)
            << "D3D12 accelerated external weights are unavailable; "
               "using the ordinary WebGPU initializer loading path. "
               "Reason: "
            << support_status.ErrorMessage();
      }
    });
  }

  common::Status EnsureFileLoader() const {
    if (file_loader) {
      return common::Status::OK();
    }
    return D3D12FileBufferLoader::Create(
        context.WeightLoadingD3D12Device(), file_loader);
  }

  WebGpuContext& context;
  std::shared_ptr<D3D12AcceleratedInitializerState>
      state;
  WeightLoadAccelerationMode mode;
  mutable std::once_flag support_once;
  mutable common::Status resolved_status;
  mutable bool enabled = false;
  mutable std::atomic<bool> abort_requested{false};
  mutable std::unique_ptr<D3D12FileBufferLoader>
      file_loader;
  mutable std::unique_ptr<D3D12AcceleratedBatch> batch;
};

D3D12AcceleratedExternalDataLoader::
    D3D12AcceleratedExternalDataLoader(
        WebGpuContext& context,
        std::shared_ptr<
            D3D12AcceleratedInitializerState>
            state,
        WeightLoadAccelerationMode mode)
    : impl_{std::make_unique<Impl>(
          context, std::move(state), mode)} {
  ORT_ENFORCE(
      impl_->state != nullptr &&
          impl_->state->impl_ != nullptr,
      "D3D12 accelerated allocator state is required.");
}

D3D12AcceleratedExternalDataLoader::
    ~D3D12AcceleratedExternalDataLoader() {
  AbortLoad();
}

bool D3D12AcceleratedExternalDataLoader::CanLoad(
    const OrtMemoryInfo& target_memory_info) const {
  impl_->ResolveSupport();
  return target_memory_info.device == WebGpuDevice &&
         target_memory_info.name == WEBGPU_BUFFER &&
         impl_->enabled;
}

bool D3D12AcceleratedExternalDataLoader::
    SupportsDataType(
        int32_t tensor_data_type) const {
  return tensor_data_type !=
         ONNX_NAMESPACE::TensorProto_DataType_BOOL;
}

bool D3D12AcceleratedExternalDataLoader::
    CreatesTensorForDevice(
        const OrtDevice& target_device) const {
  impl_->ResolveSupport();
  return target_device == WebGpuDevice &&
         (impl_->enabled ||
          IsWeightLoadAccelerationRequired(impl_->mode));
}

common::Status
D3D12AcceleratedExternalDataLoader::BeginLoad() const {
  AbortLoad();
  impl_->ResolveSupport();
  ORT_RETURN_IF_ERROR(impl_->resolved_status);
  if (!impl_->enabled) {
    return common::Status::OK();
  }
  impl_->abort_requested.store(
      false, std::memory_order_relaxed);
  impl_->batch =
      std::make_unique<D3D12AcceleratedBatch>();
  return common::Status::OK();
}

common::Status
D3D12AcceleratedExternalDataLoader::PrepareTensor(
    const Env& env,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length) const {
  ORT_RETURN_IF_NOT(
      impl_->batch != nullptr &&
          !impl_->batch->finalized,
      "D3D12 accelerated initializer batch has not been started.");
  return PrepareTensorForBatch(
      *impl_->batch, env, data_file_path,
      tensor_name, data_offset, data_length);
}

common::Status
D3D12AcceleratedExternalDataLoader::FinalizeLoad(
    const std::function<bool()>& is_canceled) const {
  if (!impl_->enabled) {
    return common::Status::OK();
  }
  ORT_RETURN_IF_NOT(
      impl_->batch != nullptr &&
          !impl_->batch->finalized,
      "D3D12 accelerated initializer batch has not been started.");
  auto& batch = *impl_->batch;
  const auto fail_or_fallback =
      [this, &is_canceled](
          const common::Status& status)
      -> common::Status {
    if (status.Code() ==
            common::MODEL_LOAD_CANCELED ||
        InvokeCancellationNoThrow(is_canceled)) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, MODEL_LOAD_CANCELED,
          "D3D12 accelerated initializer loading was canceled.");
    }
    if (IsWeightLoadAccelerationRequired(
            impl_->mode)) {
      return status;
    }
    LOGS_DEFAULT(WARNING)
        << "D3D12 accelerated initializer loading failed; "
           "using the ordinary WebGPU initializer path: "
        << status.ErrorMessage();
    impl_->enabled = false;
    AbortLoad();
    return common::Status::OK();
  };

  if (batch.load_range_count == 0) {
    if (InvokeCancellationNoThrow(is_canceled)) {
      return fail_or_fallback(ORT_MAKE_STATUS(
          ONNXRUNTIME, MODEL_LOAD_CANCELED,
          "D3D12 accelerated initializer loading was canceled."));
    }
    batch.finalized = true;
    return common::Status::OK();
  }

  D3D12AcceleratedLoadMetrics load_metrics;
  const auto loader_status = impl_->EnsureFileLoader();
  if (!loader_status.IsOK()) {
    return fail_or_fallback(loader_status);
  }
  const auto load_status = LoadBatchToD3D12(
      batch, *impl_->file_loader, is_canceled,
      impl_->abort_requested, load_metrics);
  if (!load_status.IsOK()) {
    return fail_or_fallback(load_status);
  }

  if (InvokeCancellationNoThrow(is_canceled)) {
    return fail_or_fallback(ORT_MAKE_STATUS(
        ONNXRUNTIME, MODEL_LOAD_CANCELED,
        "D3D12 accelerated initializer loading was canceled."));
  }

  const auto import_start = Clock::now();
  const auto import_status =
      [&]() -> common::Status {
    ORT_RETURN_IF_NOT(
        impl_->context.D3D12SharedResourceFeaturesAvailable(),
        "D3D12 accelerated external weights require Dawn "
        "SharedBufferMemoryD3D12Resource and SharedFenceDXGISharedHandle features.");
    for (auto& tensor : batch.tensors) {
      if (InvokeCancellationNoThrow(is_canceled)) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, MODEL_LOAD_CANCELED,
            "D3D12 accelerated initializer loading was canceled.");
      }
      if (tensor.key.length == 0) {
        continue;
      }

      SharedBufferMemoryD3D12ResourceDescriptor
          resource_descriptor;
      resource_descriptor.resource = tensor.resource;
      wgpu::SharedBufferMemoryDescriptor
          memory_descriptor{};
      memory_descriptor.label =
          tensor.key.name.c_str();
      memory_descriptor.nextInChain =
          &resource_descriptor;
      tensor.memory =
          impl_->context.Device()
              .ImportSharedBufferMemory(
                  &memory_descriptor);
      ORT_RETURN_IF_NOT(
          tensor.memory,
          "Failed to import D3D12 accelerated initializer \"",
          tensor.key.name, "\" into Dawn.");

      wgpu::SharedBufferMemoryProperties properties{};
      ORT_RETURN_IF_NOT(
          tensor.memory.GetProperties(&properties) ==
              wgpu::Status::Success,
          "Failed to query imported D3D12 accelerated initializer \"",
          tensor.key.name, "\".");
      ORT_RETURN_IF(
          properties.size <
              static_cast<uint64_t>(
                  tensor.key.length),
          "Imported D3D12 accelerated initializer \"",
          tensor.key.name,
          "\" is smaller than its tensor payload.");

      wgpu::BufferDescriptor buffer_descriptor{};
      buffer_descriptor.label =
          tensor.key.name.c_str();
      // Expose the initialized 16-byte-aligned resource so WebGPU's
      // 4-byte-normalized copies can access tensor tail padding safely. The
      // Tensor retains the logical payload length.
      buffer_descriptor.size = properties.size;
      buffer_descriptor.usage =
          wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopySrc |
          wgpu::BufferUsage::CopyDst;
      tensor.buffer = tensor.memory.CreateBuffer(
          &buffer_descriptor);
      ORT_RETURN_IF_NOT(
          tensor.buffer,
          "Failed to create buffer for D3D12 accelerated initializer \"",
          tensor.key.name, "\".");

      wgpu::SharedBufferMemoryBeginAccessDescriptor
          access_descriptor{};
      access_descriptor.initialized = true;
      ORT_RETURN_IF_NOT(
          tensor.memory.BeginAccess(
              tensor.buffer,
              &access_descriptor) ==
              wgpu::Status::Success,
          "Failed to begin Dawn access for D3D12 accelerated initializer \"",
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

  LOGS_DEFAULT(VERBOSE)
      << "WebGPU D3D12 accelerated external initializer load: "
      << "file-to-D3D12=" << load_metrics.load_ms
      << " ms, Dawn import/access="
      << Milliseconds(import_end - import_start)
      << " ms, bytes=" << batch.total_bytes
      << ", tensors=" << batch.tensors.size()
      << ", loaded_ranges="
      << load_metrics.loaded_ranges << ".";
  return common::Status::OK();
}

void D3D12AcceleratedExternalDataLoader::AbortLoad()
    const noexcept {
  if (impl_) {
    impl_->abort_requested.store(
        true, std::memory_order_relaxed);
  }
  if (!impl_) {
    return;
  }

  ORT_TRY {
    if (impl_->batch) {
      for (auto& tensor :
           impl_->batch->tensors) {
        if (!tensor.claimed &&
            tensor.access_started) {
          ImportedAllocation allocation;
          allocation.resource =
              std::move(tensor.resource);
          allocation.heap_owner =
              std::move(tensor.heap_owner);
          allocation.memory =
              std::move(tensor.memory);
          allocation.buffer =
              std::move(tensor.buffer);
          allocation.access_started = true;
          EndAccessNoThrow(allocation);
          tensor.access_started = false;
        }
      }
      impl_->batch.reset();
    }
  }
  ORT_CATCH(...) {
    // Abort is best-effort and is required not to throw.
  }
}

common::Status
D3D12AcceleratedExternalDataLoader::LoadTensor(
    const Env&,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length,
    const std::shared_ptr<IAllocator>& allocator,
    Tensor& tensor) const {
  ORT_RETURN_IF_NOT(
      impl_->batch != nullptr &&
          impl_->batch->finalized,
      "D3D12 accelerated initializer batch has not been finalized.");
  ORT_RETURN_IF(
      data_offset < 0,
      "D3D12 accelerated initializer has a negative file offset.");
  ORT_RETURN_IF_NOT(
      allocator != nullptr,
      "D3D12 accelerated initializer requires its device allocator.");
  ORT_RETURN_IF_NOT(
      allocator.get() ==
          impl_->state->impl_->allocator,
      "D3D12 accelerated initializer was passed a different allocator.");

  const size_t length =
      static_cast<size_t>(data_length);
  const auto file_iterator =
      impl_->batch->files.find(
          data_file_path.lexically_normal());
  ORT_RETURN_IF(
      file_iterator == impl_->batch->files.end(),
      "No prepared D3D12 data file matches \"",
      data_file_path.string(), "\".");
  const TensorKey key = MakeKey(
      file_iterator->second.canonical_path,
      tensor_name,
      static_cast<uint64_t>(data_offset),
      length);
  const auto iterator =
      impl_->batch->tensors_by_key.find(key);
  ORT_RETURN_IF(
      iterator ==
          impl_->batch->tensors_by_key.end(),
      "No prepared D3D12 accelerated initializer matches \"",
      tensor_name,
      "\" and the requested file range.");

  auto& prepared =
      impl_->batch->tensors[iterator->second];
  ORT_RETURN_IF(
      prepared.claimed,
      "D3D12 accelerated initializer \"",
      tensor_name,
      "\" has already been consumed.");
  ORT_RETURN_IF(
      length != tensor.SizeInBytes(),
      "D3D12 accelerated initializer \"",
      tensor_name,
      "\" length does not match the placeholder tensor.");
  if (length == 0) {
    prepared.claimed = true;
    tensor = Tensor{
        tensor.DataType(), tensor.Shape(),
        nullptr, allocator};
    return common::Status::OK();
  }
  ORT_RETURN_IF_NOT(
      prepared.buffer &&
          prepared.access_started,
      "D3D12 accelerated initializer \"",
      tensor_name,
      "\" does not have an imported buffer.");

  auto imported =
      std::make_unique<ImportedAllocation>();
  imported->resource =
      std::move(prepared.resource);
  imported->heap_owner =
      std::move(prepared.heap_owner);
  imported->memory =
      std::move(prepared.memory);
  imported->buffer =
      std::move(prepared.buffer);
  imported->access_started =
      prepared.access_started;
  WGPUBuffer buffer = imported->buffer.Get();

  {
    std::lock_guard<std::mutex> lock{
        impl_->state->impl_->mutex};
    ORT_RETURN_IF(
        impl_->state->impl_->imported_allocations
                .find(buffer) !=
            impl_->state->impl_->imported_allocations
                .end(),
        "D3D12 accelerated imported buffer was registered twice.");
    impl_->state->impl_->imported_allocations.emplace(
        buffer, std::move(imported));
  }

  prepared.access_started = false;
  prepared.claimed = true;
  tensor = Tensor{
      tensor.DataType(), tensor.Shape(),
      buffer, allocator};
  return common::Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
