// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

#include "core/common/common.h"
#include "core/common/status.h"

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class Tensor;
#endif

namespace cuda {

class DirectStorageLoader {
 public:
  virtual ~DirectStorageLoader() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DirectStorageLoader);

  virtual common::Status Load(const std::filesystem::path& path, void* validated_file_handle,
                              int64_t data_offset, size_t data_length, Tensor& tensor) = 0;

  static common::Status Create(int device_id, std::unique_ptr<DirectStorageLoader>& loader);

 protected:
  DirectStorageLoader() = default;
};

}  // namespace cuda
}  // namespace onnxruntime
