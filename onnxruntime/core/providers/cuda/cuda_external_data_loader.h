// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <memory>
#include <mutex>

#include "core/framework/external_data_loader.h"
#include "cuda_pch.h"

namespace onnxruntime {
namespace cuda {

inline constexpr size_t kExternalDataLoaderBufferSize = 64 * 1024 * 1024;
inline constexpr size_t kExternalDataLoaderParallelReadThreshold = 16 * 1024 * 1024;

class ExternalDataLoaderThreadPool;

/**
 * Loads large external initializers into CUDA memory through two reusable pinned host buffers.
 *
 * Data path:
 *
 *                           External-data file on local NVMe
 *                                        |
 *                                        | NVMe DMA / block I/O
 *                                        v
 *                           Linux kernel page-cache pages
 *                                        |
 *                                        | CPU copies performed by read()
 *                                        | Parallel reads fill each block
 *                                        |
 *                      +-----------------+-----------------+
 *                      |                                   |
 *                      v                                   v
 *           +-----------------------+           +-----------------------+
 *           | pinned buffer 0, 64M  |           | pinned buffer 1, 64M  |
 *           | CPU fills block N     |           | CPU fills block N + 1 |
 *           +-----------------------+           +-----------------------+
 *                      |                                   |
 *                      | cudaMemcpyAsync                   | cudaMemcpyAsync
 *                      |                                   |
 *                      +-----------------+-----------------+
 *                                        |
 *                                        v
 *                        CUDA BFC Arena initializer buffer
 *                                        |
 *                                        | optional CUDA prepack,
 *                                        | transpose and unpack kernels
 *                                        v
 *                            Final prepared CUDA weights
 *
 * Timeline:
 *
 *   CPU reads block N + 1 into buffer 1
 *          || concurrently with
 *   PCIe DMA transfers block N from buffer 0
 *
 *   CPU reads block N + 2 into buffer 0
 *          || concurrently with
 *   PCIe DMA transfers block N + 1 from buffer 1
 */
class ExternalDataLoader final : public IExternalDataLoader {
 public:
  using AllocatePinnedBufferFn = cudaError_t (*)(void**, size_t);
  using CreateStreamFn = cudaError_t (*)(cudaStream_t*, unsigned int);

  ExternalDataLoader(int device_id, size_t reading_thread_count,
                     AllocatePinnedBufferFn allocate_pinned_buffer = cudaMallocHost,
                     CreateStreamFn create_stream = cudaStreamCreateWithFlags);
  ~ExternalDataLoader() override;

  bool CanLoad(const OrtMemoryInfo& target_memory_info) const override;

  common::Status LoadTensor(const Env& env,
                            const std::filesystem::path& data_file_path,
                            FileOffsetType data_offset,
                            SafeInt<size_t> data_length,
                            Tensor& tensor) const override;

 private:
  common::Status EnsureResources() const;
  void ReleaseResources() const noexcept;

  int device_id_;
  mutable std::mutex mutex_;
  mutable std::array<void*, 2> buffers_{};
  mutable std::array<cudaStream_t, 2> streams_{};
  const size_t reading_thread_count_;
  const AllocatePinnedBufferFn allocate_pinned_buffer_;
  const CreateStreamFn create_stream_;
  mutable std::unique_ptr<ExternalDataLoaderThreadPool> reader_pool_;
};

}  // namespace cuda
}  // namespace onnxruntime
