// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <mutex>

#include "core/framework/external_data_loader.h"
#include "cuda_pch.h"

namespace onnxruntime {
namespace cuda {

class ExternalDataLoader final : public IExternalDataLoader {
 public:
  ExternalDataLoader(int device_id, size_t reading_thread_count);
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
};

}  // namespace cuda
}  // namespace onnxruntime
