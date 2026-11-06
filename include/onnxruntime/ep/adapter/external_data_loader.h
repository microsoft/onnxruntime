// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#if defined(__wasm__)

#include <memory>
#include <utility>

#include "allocator.h"
#include "tensor_helper.h"
#include "core/framework/external_data_loader.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

class ExternalDataLoader final : public OrtExternalDataLoaderImpl {
 public:
  explicit ExternalDataLoader(std::unique_ptr<IExternalDataLoader> impl)
      : OrtExternalDataLoaderImpl{}, impl_(std::move(impl)) {
    ort_version_supported = ORT_API_VERSION;
    Release = ReleaseImpl;
    CanLoad = CanLoadImpl;
    SupportsDataType = SupportsDataTypeImpl;
    LoadTensor = LoadTensorImpl;
  }

 private:
  static void ORT_API_CALL ReleaseImpl(OrtExternalDataLoaderImpl* this_ptr) noexcept {
    delete static_cast<ExternalDataLoader*>(this_ptr);
  }

  static OrtStatus* ORT_API_CALL CanLoadImpl(const OrtExternalDataLoaderImpl* this_ptr,
                                             const OrtMemoryInfo* memory_info, bool* can_load) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    const auto native_info = CreateMemoryInfoFromApi(memory_info);
    *can_load = static_cast<const ExternalDataLoader*>(this_ptr)->impl_->CanLoad(native_info);
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }

  static OrtStatus* ORT_API_CALL SupportsDataTypeImpl(const OrtExternalDataLoaderImpl* this_ptr,
                                                      ONNXTensorElementDataType element_type,
                                                      bool* supported) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    *supported = static_cast<const ExternalDataLoader*>(this_ptr)->impl_->SupportsDataType(
        static_cast<int32_t>(element_type));
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }

  static OrtStatus* ORT_API_CALL LoadTensorImpl(OrtExternalDataLoaderImpl* this_ptr,
                                                const OrtExternalDataSource* source,
                                                OrtValue* value) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    // Match the Wasm initializer limit before the JavaScript loader narrows offsets to 32 bits.
    constexpr uint64_t kWasmOffsetLimit = uint64_t{1} << 32;
    if (source->file_offset < 0 ||
        static_cast<uint64_t>(source->file_offset) >= kWasmOffsetLimit ||
        source->byte_size >= kWasmOffsetLimit - static_cast<uint64_t>(source->file_offset)) {
      return Ort::Status{"External initializer offset and size are out of bounds or cannot be read in full (>4GB).",
                         ORT_INVALID_ARGUMENT}
          .release();
    }
    auto tensor = CreateTensorFromApiValue(value);
    const auto status = static_cast<ExternalDataLoader*>(this_ptr)->impl_->LoadTensor(
        Env::Default(), std::filesystem::path{source->file_path}, source->file_offset,
        SafeInt<size_t>{source->byte_size}, tensor);
    if (!status.IsOK()) {
      return Ort::Status{status.ErrorMessage().c_str(), static_cast<OrtErrorCode>(status.Code())}.release();
    }
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExternalDataLoader);
  std::unique_ptr<IExternalDataLoader> impl_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime

#endif  // defined(__wasm__)
