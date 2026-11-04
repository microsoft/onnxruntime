// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// provider_api.h must be first to set SHARED_PROVIDER
#include "core/providers/shared_library/provider_api.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(USE_CUDA_MINIMAL)

#include "core/providers/cuda/cuda_external_data_loader.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <exception>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_external_data_loader_thread_pool.h"

namespace onnxruntime {
namespace cuda {
namespace {

class CudaDeviceGuard {
 public:
  common::Status SetDevice(int device_id) {
    CUDA_RETURN_IF_ERROR(cudaGetDevice(&previous_device_));
    if (previous_device_ != device_id) {
      CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
      restore_device_ = true;
    }
    return Status::OK();
  }

  ~CudaDeviceGuard() {
    if (restore_device_) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device_)));
    }
  }

 private:
  int previous_device_{-1};
  bool restore_device_{false};
};

common::Status ReadChunk(const RandomAccessFile& file, FileOffsetType offset, size_t length, void* buffer,
                         size_t configured_reader_count,
                         std::unique_ptr<ExternalDataLoaderThreadPool>& reader_pool) {
  const size_t reader_count =
      length >= kExternalDataLoaderParallelReadThreshold ? configured_reader_count : 1;
  common::Status status = Status::OK();
  ORT_TRY {
    if (reader_count <= 1) {
      return file.Read(offset, gsl::span<char>{static_cast<char*>(buffer), length});
    }
    if (!reader_pool || reader_pool->Size() != reader_count) {
      reader_pool.reset();
      reader_pool = std::make_unique<ExternalDataLoaderThreadPool>(reader_count);
    }
    return reader_pool->Run([&](size_t reader) {
      const size_t reader_length = length / reader_count;
      const size_t remainder = length % reader_count;
      const size_t begin = reader_length * reader + std::min(reader, remainder);
      const size_t end = begin + reader_length + (reader < remainder ? 1 : 0);
      return file.Read(offset + begin,
                       gsl::span<char>{static_cast<char*>(buffer) + begin, end - begin});
    });
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to read external data: ", ex.what());
    });
  }
  ORT_CATCH(...) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to read external data: unknown exception");
  }

  return status;
}

common::Status LoadTaskWithPageableBuffer(const ExternalDataLoadTask& task,
                                          size_t single_buffer_size,
                                          size_t configured_reader_count,
                                          std::unique_ptr<ExternalDataLoaderThreadPool>& reader_pool) {
  InlinedVector<uint8_t> buffer(std::min(single_buffer_size, task.length));

  for (size_t offset = 0; offset < task.length;) {
    const size_t chunk_size = std::min(buffer.size(), task.length - offset);
    ORT_RETURN_IF_ERROR(
        ReadChunk(*task.file, task.file_offset + static_cast<FileOffsetType>(offset),
                  chunk_size, buffer.data(), configured_reader_count, reader_pool));

    TransformExternalDataStagingBuffer(buffer.data(), chunk_size, task.is_bool, task.element_size);

    CUDA_RETURN_IF_ERROR(
        cudaMemcpy(task.destination + offset, buffer.data(), chunk_size, cudaMemcpyHostToDevice));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
    offset += chunk_size;
  }

  return Status::OK();
}

}  // namespace

ExternalDataLoader::ExternalDataLoader(int device_id, size_t reading_thread_count,
                                       AllocatePinnedBufferFn allocate_pinned_buffer,
                                       CreateStreamFn create_stream)
    : ExternalDataLoader(device_id,
                         reading_thread_count,
                         ExternalDataLoaderOptions::kDefaultPinnedMemoryBytes,
                         ExternalDataLoaderOptions::kDefaultChunkBytes,
                         /*reading_threads_explicit=*/false,
                         /*pinned_memory_bytes_explicit=*/false,
                         /*chunk_bytes_explicit=*/false,
                         ExternalDataLoaderOptions::kDefaultMinSizeBytes,
                         /*min_size_bytes_explicit=*/false,
                         allocate_pinned_buffer,
                         create_stream) {}

ExternalDataLoader::ExternalDataLoader(int device_id,
                                       size_t reading_thread_count,
                                       size_t pinned_memory_bytes,
                                       size_t chunk_bytes,
                                       bool reading_threads_explicit,
                                       bool pinned_memory_bytes_explicit,
                                       bool chunk_bytes_explicit,
                                       size_t min_size_bytes,
                                       bool min_size_bytes_explicit,
                                       AllocatePinnedBufferFn allocate_pinned_buffer,
                                       CreateStreamFn create_stream)
    : device_id_(device_id),
      reading_thread_count_(reading_thread_count),
      batch_reading_thread_count_(ResolveExternalDataBatchThreadCount(reading_thread_count, reading_threads_explicit)),
      pinned_memory_bytes_(pinned_memory_bytes),
      chunk_bytes_(chunk_bytes),
      min_size_bytes_(min_size_bytes),
      reading_threads_explicit_(reading_threads_explicit),
      pinned_memory_bytes_explicit_(pinned_memory_bytes_explicit),
      chunk_bytes_explicit_(chunk_bytes_explicit),
      min_size_bytes_explicit_(min_size_bytes_explicit),
      allocate_pinned_buffer_(allocate_pinned_buffer),
      create_stream_(create_stream) {}

ExternalDataLoader::~ExternalDataLoader() {
  reader_pool_.reset();
  ReleaseResources();
}

bool ExternalDataLoader::CanLoad(const OrtMemoryInfo& target_memory_info) const {
  const OrtDevice& device = target_memory_info.device;
  return device.Type() == OrtDevice::GPU &&
         device.MemType() == OrtDevice::MemType::DEFAULT &&
         device.Vendor() == OrtDevice::VendorIds::NVIDIA &&
         device.Id() == device_id_;
}

bool ExternalDataLoader::IsEnabled() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return reading_thread_count_ > 0;
}

common::Status ExternalDataLoader::Configure(const ExternalDataLoaderOptions& options) const {
  std::lock_guard<std::mutex> lock(mutex_);
  bool changed = false;
  if (options.reading_threads.has_value() && !reading_threads_explicit_) {
    ORT_RETURN_IF_ERROR(ExternalDataLoaderOptions::ValidateReadingThreads(
        *options.reading_threads, "session.external_data_loader.threads"));
    reading_thread_count_ = *options.reading_threads;
    batch_reading_thread_count_ = *options.reading_threads;
    reading_threads_explicit_ = true;
    changed = true;
  } else if (options.auto_reading_threads.has_value() && !reading_threads_explicit_ &&
             reading_thread_count_ == ExternalDataLoaderOptions::kDefaultReadingThreads) {
    ORT_RETURN_IF_ERROR(ExternalDataLoaderOptions::ValidateReadingThreads(
        *options.auto_reading_threads, "session.external_data_loader.threads"));
    if (reading_thread_count_ != *options.auto_reading_threads ||
        batch_reading_thread_count_ != *options.auto_reading_threads) {
      reading_thread_count_ = *options.auto_reading_threads;
      batch_reading_thread_count_ = *options.auto_reading_threads;
      changed = true;
    }
  }
  if (options.pinned_memory_bytes.has_value() && !pinned_memory_bytes_explicit_) {
    ORT_RETURN_IF_ERROR(ExternalDataLoaderOptions::ValidatePinnedMemoryBytes(
        *options.pinned_memory_bytes, "session.external_data_loader.pinned_memory_bytes"));
    pinned_memory_bytes_ = *options.pinned_memory_bytes;
    pinned_memory_bytes_explicit_ = true;
    changed = true;
  }
  if (options.chunk_bytes.has_value() && !chunk_bytes_explicit_) {
    ORT_RETURN_IF_ERROR(ExternalDataLoaderOptions::ValidateChunkBytes(
        *options.chunk_bytes, "session.external_data_loader.chunk_bytes"));
    chunk_bytes_ = *options.chunk_bytes;
    chunk_bytes_explicit_ = true;
  }
  if (options.min_size_bytes.has_value() && !min_size_bytes_explicit_) {
    ORT_RETURN_IF_ERROR(ExternalDataLoaderOptions::ValidateMinSizeBytes(
        *options.min_size_bytes, "session.external_data_loader.min_size_bytes"));
    min_size_bytes_ = *options.min_size_bytes;
    min_size_bytes_explicit_ = true;
  }
  if (changed && buffers_[0] != nullptr) {
    ReleaseResources();
  }
  return Status::OK();
}

common::Status ExternalDataLoader::BeginLoad() const {
  std::lock_guard<std::mutex> lock(mutex_);
  in_batch_load_ = true;
  pending_tasks_.clear();
  pending_bytes_ = 0;
  flushed_bytes_ = 0;
  expected_total_bytes_ = 0;
  return Status::OK();
}

void ExternalDataLoader::EndLoad() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  in_batch_load_ = false;
  pending_tasks_.clear();
  pending_bytes_ = 0;
  flushed_bytes_ = 0;
  expected_total_bytes_ = 0;
}

void ExternalDataLoader::SetExpectedLoadBytes(size_t total_bytes) const {
  std::lock_guard<std::mutex> lock(mutex_);
  expected_total_bytes_ = total_bytes;
}

common::Status ExternalDataLoader::FinalizeLoad(const std::function<bool()>& is_canceled) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pending_tasks_.empty()) {
    return Status::OK();
  }

  auto tasks = std::move(pending_tasks_);
  const size_t total_pending_bytes = pending_bytes_;
  // Gate on the whole transaction so early flushes (e.g. to bound open files) don't force the pageable path.
  flushed_bytes_ += total_pending_bytes;
  const size_t gate_bytes = std::max(flushed_bytes_, expected_total_bytes_);
  pending_tasks_.clear();
  pending_bytes_ = 0;

  CudaDeviceGuard device_guard;
  ORT_RETURN_IF_ERROR(device_guard.SetDevice(device_id_));
  const size_t single_buffer_size = pinned_memory_bytes_ / 2;

  const auto load_all_pageable = [&](size_t reader_threads) -> common::Status {
    for (const auto& task : tasks) {
      if (is_canceled && is_canceled()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                               "Saving session state weights is canceled due to user request.");
      }
      ORT_RETURN_IF_ERROR(LoadTaskWithPageableBuffer(task, single_buffer_size, reader_threads, reader_pool_));
    }
    return Status::OK();
  };

  if (reading_thread_count_ == 0 || gate_bytes < min_size_bytes_) {
    return load_all_pageable(1);
  }

  if (!EnsureResources().IsOK()) {
    return load_all_pageable(reading_thread_count_);
  }

  return ExecuteTasks(tasks, /*allow_small_parallel=*/true, is_canceled);
}

common::Status ExternalDataLoader::EnsureResources() const {
  if (buffers_[0] != nullptr) {
    return Status::OK();
  }

  const size_t single_buffer_size = pinned_memory_bytes_ / 2;
  for (size_t i = 0; i < buffers_.size(); ++i) {
    auto status = CUDA_CALL(allocate_pinned_buffer_(&buffers_[i], single_buffer_size));
    if (!status.IsOK()) {
      ReleaseResources();
      return status;
    }
  }

  const size_t max_threads = std::max(reading_thread_count_, batch_reading_thread_count_);
  const size_t stream_count = 2 * std::max<size_t>(1, max_threads);
  streams_.assign(stream_count, nullptr);
  for (size_t i = 0; i < stream_count; ++i) {
    auto status = CUDA_CALL(create_stream_(&streams_[i], cudaStreamNonBlocking));
    if (!status.IsOK()) {
      ReleaseResources();
      return status;
    }
  }

  return Status::OK();
}

void ExternalDataLoader::ReleaseResources() const noexcept {
  int previous_device = -1;
  const bool restore_device =
      cudaGetDevice(&previous_device) == cudaSuccess &&
      previous_device != device_id_ &&
      cudaSetDevice(device_id_) == cudaSuccess;

  for (auto& stream : streams_) {
    if (stream != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(stream)));
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(stream)));
      stream = nullptr;
    }
  }
  streams_.clear();

  for (auto& buffer : buffers_) {
    if (buffer != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaFreeHost(buffer)));
      buffer = nullptr;
    }
  }

  if (restore_device) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device)));
  }
}

common::Status ExternalDataLoader::ExecuteTasks(gsl::span<const ExternalDataLoadTask> tasks,
                                                bool allow_small_parallel,
                                                const std::function<bool()>& is_canceled) const {
  const size_t single_buffer_size = pinned_memory_bytes_ / 2;
  return ExecuteExternalDataLoadTasks(
      tasks,
      allow_small_parallel,
      reading_thread_count_,
      batch_reading_thread_count_,
      single_buffer_size,
      chunk_bytes_,
      buffers_,
      reader_pool_,
      is_canceled,
      [this](size_t /*worker_id*/) -> common::Status {
        return CUDA_CALL(cudaSetDevice(device_id_));
      },
      [this](size_t stream_index) -> common::Status {
        return CUDA_CALL(cudaStreamSynchronize(streams_[stream_index]));
      },
      [this](void* dst_device, void* src_pinned, size_t bytes, size_t stream_index) -> common::Status {
        return CUDA_CALL(
            cudaMemcpyAsync(dst_device, src_pinned, bytes, cudaMemcpyHostToDevice, streams_[stream_index]));
      });
}

common::Status ExternalDataLoader::LoadTensor(const RandomAccessFile& file,
                                              FileOffsetType data_offset,
                                              SafeInt<size_t> data_length,
                                              Tensor& tensor) const {
  ORT_RETURN_IF_NOT(CanLoad(tensor.Location()), "Unsupported tensor location: ",
                    tensor.Location().ToString());
  ORT_RETURN_IF(tensor.IsDataTypeString(),
                "String external initializers cannot be loaded into CUDA memory.");

  const size_t length = data_length;
  ORT_RETURN_IF_NOT(length == tensor.SizeInBytes(), "External data length does not match tensor size.");

  uint64_t file_length = 0;
  ORT_RETURN_IF_ERROR(file.GetLength(file_length));
  const SafeInt<FileOffsetType> end_offset = SafeInt<FileOffsetType>(data_offset) + length;
  ORT_RETURN_IF(data_offset < 0 || end_offset > file_length,
                "External data range is outside the file.");
  if (length == 0) {
    return Status::OK();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const ExternalDataLoadTask task{
      &file,
      data_offset,
      static_cast<uint8_t*>(tensor.MutableDataRaw()),
      length,
      tensor.DataType()->Size(),
      tensor.IsDataType<bool>()};

  if (in_batch_load_) {
    pending_tasks_.push_back(task);
    pending_bytes_ += length;
    return Status::OK();
  }

  CudaDeviceGuard device_guard;
  ORT_RETURN_IF_ERROR(device_guard.SetDevice(device_id_));
  const size_t single_buffer_size = pinned_memory_bytes_ / 2;
  if (reading_thread_count_ == 0) {
    return LoadTaskWithPageableBuffer(task, single_buffer_size, 1, reader_pool_);
  }

  const auto resource_status = EnsureResources();
  if (!resource_status.IsOK()) {
    return LoadTaskWithPageableBuffer(task, single_buffer_size, reading_thread_count_, reader_pool_);
  }

  return ExecuteTasks(gsl::span<const ExternalDataLoadTask>(&task, 1), /*allow_small_parallel=*/false, {});
}

}  // namespace cuda
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(USE_CUDA_MINIMAL)
