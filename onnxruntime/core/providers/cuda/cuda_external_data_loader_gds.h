// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

#include "core/common/status.h"

namespace onnxruntime {
class Tensor;

namespace cuda {

class GdsLoader {
 public:
  using CreateFn = common::Status (*)(int device_id, std::unique_ptr<GdsLoader>& loader);

  virtual ~GdsLoader() = default;

  virtual common::Status Load(const std::filesystem::path& data_file_path,
                              int64_t data_offset,
                              size_t data_length,
                              Tensor& tensor) const = 0;

  static common::Status Create(int device_id, std::unique_ptr<GdsLoader>& loader);
};

}  // namespace cuda
}  // namespace onnxruntime
