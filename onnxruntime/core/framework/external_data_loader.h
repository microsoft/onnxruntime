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

#if defined(ENABLE_D3D12_FILE_LOADING)
  // Returns true when the loader creates the tensor's backing allocation instead of
  // writing into a tensor allocated by the framework.
  virtual bool CreatesTensorForDevice([[maybe_unused]] const OrtDevice& target_device) const { return false; }

  // Begins a batch covering the external initializers for one SessionState
  // initialization. ORT describes all loader-created tensors through PrepareTensor()
  // before calling FinalizeLoad(), allowing an implementation to coalesce file
  // ranges, allocate destination resources together, and perform shared I/O or device
  // work before individual tensors become visible through LoadTensor(). Loaders that
  // can populate each tensor independently do not need to override the batch hooks.
  //
  // This is called once before zero or more PrepareTensor() calls and before any
  // initializer is loaded through LoadTensor(). Implementations should discard stale
  // batch state and initialize any batch-wide resources. Returning an error prevents
  // the batch from continuing and causes AbortLoad() to be called.
  virtual common::Status BeginLoad() const { return common::Status::OK(); }

  // Adds one external tensor to the current batch. The path, offset, and length
  // have already been resolved and validated against the TensorProto by the caller.
  // Implementations may record the range, reserve its destination, or start deferred
  // work. ORT does not call LoadTensor() during this preparation phase. After
  // FinalizeLoad() succeeds, ORT makes a separate initializer-deserialization pass
  // and calls LoadTensor() for each tensor still claimed by this loader. Cancellation
  // or another initialization error may prevent some or all of those calls.
  //
  // PrepareTensor() calls occur after a successful BeginLoad() and before
  // FinalizeLoad(). The same file may be referenced by multiple tensors, and a batch
  // may contain no tensors. Returning an error abandons the batch and causes
  // AbortLoad() to be called. The default implementation performs no preparation.
  virtual common::Status PrepareTensor([[maybe_unused]] const Env& env,
                                       [[maybe_unused]] const std::filesystem::path& data_file_path,
                                       [[maybe_unused]] std::string_view tensor_name,
                                       [[maybe_unused]] FileOffsetType data_offset,
                                       [[maybe_unused]] SafeInt<size_t> data_length) const {
    return common::Status::OK();
  }

  // Completes preparation for the current batch. On success, the loader must
  // either make every tensor for which PrepareTensor() returned OK ready for a
  // subsequent LoadTensor() call, or make CanLoad() return false for those tensors so
  // normal loading can take over. Any asynchronous I/O or device work required to
  // make tensors ready must be completed or synchronized here. The implementation
  // must keep prepared tensors alive until LoadTensor() transfers their ownership or
  // AbortLoad() releases them. This method is called even for an empty batch.
  // Long-running work should check is_canceled and return MODEL_LOAD_CANCELED when
  // cancellation is observed. Returning any error causes AbortLoad() to be called.
  // The default implementation completes immediately.
  virtual common::Status FinalizeLoad([[maybe_unused]] const std::function<bool()>& is_canceled) const {
    return common::Status::OK();
  }

  // Abandons the active batch. Cancel outstanding work and release resources
  // still owned by the batch, without invalidating allocations already transferred by
  // successful LoadTensor() calls. This may be called after BeginLoad(),
  // PrepareTensor(), FinalizeLoad(), or partial tensor consumption, and may be called
  // more than once. Implementations must therefore be noexcept and idempotent.
  virtual void AbortLoad() const noexcept {}

  // A tensor-creating loader replaces tensor with one backed by memory owned through allocator.
  virtual common::Status LoadTensor([[maybe_unused]] const Env& env,
                                    [[maybe_unused]] const std::filesystem::path& data_file_path,
                                    [[maybe_unused]] std::string_view tensor_name,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] const std::shared_ptr<IAllocator>& allocator,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
#endif

  // Tensor should be already allocated with the correct memory info and size.
#if defined(__wasm__)
  virtual common::Status LoadTensor([[maybe_unused]] const Env& env,
                                    [[maybe_unused]] const std::filesystem::path& data_file_path,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
#else
  virtual common::Status LoadTensor([[maybe_unused]] const RandomAccessFile& file,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
#endif
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
