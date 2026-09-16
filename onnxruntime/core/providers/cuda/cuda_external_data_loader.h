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
