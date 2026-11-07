// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && !defined(USE_CUDA_MINIMAL)

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/framework/external_data_loader.h"
#include "core/framework/external_data_loader_thread_pool.h"
#include "cuda_pch.h"

namespace onnxruntime {
namespace cuda {

inline constexpr size_t kExternalDataLoaderBufferSize =
    ExternalDataLoaderOptions::kDefaultPinnedMemoryBytes / 2;
using ::onnxruntime::kExternalDataLoaderParallelReadThreshold;
inline constexpr size_t kExternalDataLoaderMaxChunkSize = ExternalDataLoaderOptions::kDefaultChunkBytes;

/**
 * Loads large external initializers into CUDA memory through two reusable pinned host buffers.
 *
 * When reading_thread_count > 1, each worker thread owns a dedicated 4 KiB-aligned slice in both
 * pinned buffers and a pair of non-blocking CUDA streams, pulling chunks from a lock-free work
 * queue to overlap multi-threaded file reads with asynchronous H2D DMA transfers across tensors.
 */
class ExternalDataLoader final : public IExternalDataLoader {
 public:
  using AllocatePinnedBufferFn = cudaError_t (*)(void**, size_t);
  using CreateStreamFn = cudaError_t (*)(cudaStream_t*, unsigned int);

  ExternalDataLoader(int device_id, size_t reading_thread_count,
                     AllocatePinnedBufferFn allocate_pinned_buffer = cudaMallocHost,
                     CreateStreamFn create_stream = cudaStreamCreateWithFlags);

  ExternalDataLoader(int device_id,
                     size_t reading_thread_count,
                     size_t pinned_memory_bytes,
                     size_t chunk_bytes,
                     bool reading_threads_explicit = false,
                     bool pinned_memory_bytes_explicit = false,
                     bool chunk_bytes_explicit = false,
                     size_t min_size_bytes = ExternalDataLoaderOptions::kDefaultMinSizeBytes,
                     bool min_size_bytes_explicit = false,
                     AllocatePinnedBufferFn allocate_pinned_buffer = cudaMallocHost,
                     CreateStreamFn create_stream = cudaStreamCreateWithFlags);

  ~ExternalDataLoader() override;

  bool CanLoad(const OrtMemoryInfo& target_memory_info) const override;

  bool IsEnabled() const noexcept override;

  common::Status Configure(const ExternalDataLoaderOptions& options) const override;

  common::Status BeginLoad() const override;

  void SetExpectedLoadBytes(size_t total_bytes) const override;

  common::Status FinalizeLoad(const std::function<bool()>& is_canceled = {}) const override;

  void EndLoad() const noexcept override;

  common::Status LoadTensor(const RandomAccessFile& file,
                            FileOffsetType data_offset,
                            SafeInt<size_t> data_length,
                            Tensor& tensor) const override;

 private:
  common::Status EnsureResources() const;
  void ReleaseResources() const noexcept;
  common::Status ExecuteTasks(gsl::span<const ExternalDataLoadTask> tasks,
                              bool allow_small_parallel,
                              const std::function<bool()>& is_canceled) const;

  int device_id_;
  mutable std::mutex mutex_;
  mutable std::array<void*, 2> buffers_{};
  mutable InlinedVector<cudaStream_t> streams_;
  mutable size_t reading_thread_count_;
  mutable size_t batch_reading_thread_count_;
  mutable size_t pinned_memory_bytes_;
  mutable size_t chunk_bytes_;
  mutable size_t min_size_bytes_;
  mutable bool reading_threads_explicit_{false};
  mutable bool pinned_memory_bytes_explicit_{false};
  mutable bool chunk_bytes_explicit_{false};
  mutable bool min_size_bytes_explicit_{false};
  const AllocatePinnedBufferFn allocate_pinned_buffer_;
  const CreateStreamFn create_stream_;
  mutable std::unique_ptr<ExternalDataLoaderThreadPool> reader_pool_;
  mutable bool in_batch_load_{false};
  mutable std::vector<ExternalDataLoadTask> pending_tasks_;
  mutable size_t pending_bytes_{0};
  // Bytes already flushed by earlier FinalizeLoad() calls and the total reported by the framework.
  mutable size_t flushed_bytes_{0};
  mutable size_t expected_total_bytes_{0};
};

}  // namespace cuda
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(USE_CUDA_MINIMAL)
