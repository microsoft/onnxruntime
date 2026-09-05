// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <vector>
#include <filesystem>
#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
#include <string_view>
#endif

#include "core/common/common.h"
#include "core/common/safeint.h"
#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"
#endif
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

#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
  virtual bool SupportsDataType(int32_t tensor_data_type) const;

  // Returns true when the loader creates the tensor's backing allocation instead of
  // writing into a tensor allocated by the framework.
  virtual bool CreatesTensorForDevice(const OrtDevice& target_device) const;

  // Optional preload hooks allow a loader to start I/O after model parsing but
  // before execution-provider assignment is finalized.
  virtual bool SupportsPreload() const;
  virtual common::Status BeginPreload() const;
  virtual common::Status PreloadTensor(const Env& env,
                                       const std::filesystem::path& data_file_path,
                                       std::string_view tensor_name,
                                       FileOffsetType data_offset,
                                       SafeInt<size_t> data_length) const;
  virtual common::Status FinalizePreload(const std::function<bool()>& is_cancelled) const;

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
#endif

  // Tensor should be allocated with the correct memory info and size unless
#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
  // CreatesTensorForDevice() returns true. In that case the loader replaces tensor
  // with one backed by memory owned through allocator.
#else
  // the loader writes into the framework-provided allocation.
#endif
  virtual common::Status LoadTensor(const Env& env,
                                    const std::filesystem::path& data_file_path,
#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
                                    std::string_view tensor_name,
#endif
                                    FileOffsetType data_offset,
                                    SafeInt<size_t> data_length,
#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
                                    const std::shared_ptr<IAllocator>& allocator,
#endif
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
