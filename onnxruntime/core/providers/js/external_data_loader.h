// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/external_data_loader.h"

namespace onnxruntime {
namespace js {

enum class ExternalDataLoadType {
  CPU = 0,
  WEBGPU_BUFFER = 1,
};

class ExternalDataLoader : public IExternalDataLoader {
 public:
  ExternalDataLoader() {};
  ~ExternalDataLoader() {};

  bool CanLoad(const OrtMemoryInfo& target_memory_info) const override;

  common::Status LoadTensor(const Env& env,
                            const std::filesystem::path& data_file_path,
                            FileOffsetType data_offset,
                            SafeInt<size_t> data_length,
                            Tensor& tensor) const override;
};

// Entry point for loading external data implementation.
common::Status LoadExternalData(const Env& env,
                                const std::filesystem::path& data_file_path,
                                FileOffsetType data_offset,
                                SafeInt<size_t> data_length,
                                ExternalDataLoadType load_type,
                                void* tensor_data);

}  // namespace js
}  // namespace onnxruntime
