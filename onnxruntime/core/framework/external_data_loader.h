// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/ortdevice.h"
#include "core/platform/env.h"

struct OrtMemoryInfo;

namespace onnxruntime {
class IAllocator;
#ifndef SHARED_PROVIDER
class Tensor;
#endif
class Stream;

namespace common {
class Status;
}

// Data transfer interface.
// Keep default virtual methods inline so shared providers do not require a non-exported RTTI symbol.
class IExternalDataLoader {
 public:
  virtual ~IExternalDataLoader() = default;

  virtual bool CanLoad(const OrtMemoryInfo& target_memory_info) const = 0;

  virtual bool SupportsDataType([[maybe_unused]] int32_t tensor_data_type) const { return true; }

  // Returns true when the loader creates the tensor's backing allocation instead of
  // writing into a tensor allocated by the framework.
  virtual bool CreatesTensorForDevice([[maybe_unused]] const OrtDevice& target_device) const { return false; }

  // Batch hooks allow loaders to prepare all external tensors before any initializer
  // is exposed to prepacking. The default implementations are no-ops.
  virtual common::Status BeginLoad() const { return common::Status::OK(); }
  virtual common::Status PrepareTensor([[maybe_unused]] const Env& env,
                                       [[maybe_unused]] const std::filesystem::path& data_file_path,
                                       [[maybe_unused]] std::string_view tensor_name,
                                       [[maybe_unused]] FileOffsetType data_offset,
                                       [[maybe_unused]] SafeInt<size_t> data_length) const {
    return common::Status::OK();
  }
  virtual common::Status FinalizeLoad([[maybe_unused]] const std::function<bool()>& is_canceled) const {
    return common::Status::OK();
  }
  virtual void AbortLoad() const noexcept {}

  // Tensor should be allocated with the correct memory info and size unless
  // CreatesTensorForDevice() returns true. In that case the loader replaces tensor
  // with one backed by memory owned through allocator.
  virtual common::Status LoadTensor([[maybe_unused]] const Env& env,
                                    [[maybe_unused]] const std::filesystem::path& data_file_path,
                                    [[maybe_unused]] std::string_view tensor_name,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] const std::shared_ptr<IAllocator>& allocator,
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
