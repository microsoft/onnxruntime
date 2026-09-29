// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <vector>
#include <filesystem>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/platform/env.h"

struct OrtMemoryInfo;

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class Tensor;
#endif
class Stream;

namespace common {
class Status;
}

// Data transfer interface.
class IExternalDataLoader {
 public:
  virtual ~IExternalDataLoader() = default;

  virtual bool CanLoad(const OrtMemoryInfo& target_memory_info) const = 0;

  // Tensor should be already allocated with the correct memory info and size.
  virtual common::Status LoadTensor([[maybe_unused]] const Env& env,
                                    [[maybe_unused]] const std::filesystem::path& data_file_path,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
};

#if defined(__wasm__)

enum class ExternalDataLoadType {
  CPU = 0,
#if defined(USE_JSEP) || defined(USE_WEBGPU)
  WEBGPU_BUFFER = 1,
#endif
};

// Entry point for loading external data implementation using inline JavaScript.
common::Status LoadWebAssemblyExternalData(const Env& env,
                                           const std::filesystem::path& data_file_path,
                                           FileOffsetType data_offset,
                                           SafeInt<size_t> data_length,
                                           ExternalDataLoadType load_type,
                                           void* tensor_data);

#endif

}  // namespace onnxruntime
