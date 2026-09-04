// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <vector>
#include <filesystem>
#include <string_view>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"
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

  // Returns true when the loader creates the tensor's backing allocation instead of
  // writing into a tensor allocated by the framework.
  virtual bool CreatesTensorForDevice(const OrtDevice& target_device) const;

  // Batch hooks allow loaders to prepare all external tensors before any initializer
  // is exposed to prepacking. The default implementations are no-ops.
  virtual common::Status BeginLoad() const;
  virtual common::Status PrepareTensor(const Env& env,
                                       const std::filesystem::path& data_file_path,
                                       std::string_view tensor_name,
                                       FileOffsetType data_offset,
                                       SafeInt<size_t> data_length) const;
  virtual common::Status FinalizeLoad(const std::function<bool()>& is_cancelled) const;
  virtual void AbortLoad() const noexcept;

  // Tensor should be allocated with the correct memory info and size unless
  // CreatesTensorForDevice() returns true. In that case the loader replaces tensor
  // with one backed by memory owned through allocator.
  virtual common::Status LoadTensor(const Env& env,
                                    const std::filesystem::path& data_file_path,
                                    std::string_view tensor_name,
                                    FileOffsetType data_offset,
                                    SafeInt<size_t> data_length,
                                    const std::shared_ptr<IAllocator>& allocator,
                                    Tensor& tensor) const;
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
