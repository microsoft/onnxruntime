// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/external_data_loader.h"

namespace onnxruntime {
namespace js {

class ExternalDataLoader : public IExternalDataLoader {
 public:
  ExternalDataLoader() {};
  ~ExternalDataLoader() {};

  bool CanLoad(const OrtMemoryInfo& target_memory_info) const override;

  common::Status LoadTensor(const Env& env,
                            const std::filesystem::path& data_file_path,
#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
                            std::string_view tensor_name,
#endif
                            FileOffsetType data_offset,
                            SafeInt<size_t> data_length,
#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
                            const std::shared_ptr<IAllocator>& allocator,
#endif
                            Tensor& tensor) const override;
};

}  // namespace js
}  // namespace onnxruntime
