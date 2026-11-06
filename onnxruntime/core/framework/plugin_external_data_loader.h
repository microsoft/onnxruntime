// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <utility>

#include "core/framework/error_code_helper.h"
#include "core/framework/external_data_loader.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {
namespace plugin_ep {

struct OrtExternalDataLoaderImplDeleter {
  void operator()(OrtExternalDataLoaderImpl* loader) const noexcept {
    if (loader != nullptr && loader->Release != nullptr) {
      loader->Release(loader);
    }
  }
};

using OrtExternalDataLoaderImplUniquePtr =
    std::unique_ptr<OrtExternalDataLoaderImpl, OrtExternalDataLoaderImplDeleter>;

class ExternalDataLoader final : public IExternalDataLoader {
 public:
  explicit ExternalDataLoader(OrtExternalDataLoaderImplUniquePtr impl) : impl_(std::move(impl)) {
    ORT_ENFORCE(impl_ != nullptr, "OrtExternalDataLoaderImpl must not be null.");
    ORT_ENFORCE(impl_->ort_version_supported >= 32, "External data loader requires ORT API version 32.");
    ORT_ENFORCE(impl_->Release != nullptr, "External data loader must provide Release.");
    ORT_ENFORCE(impl_->CanLoad != nullptr, "External data loader must provide CanLoad.");
    ORT_ENFORCE(impl_->LoadTensor != nullptr, "External data loader must provide LoadTensor.");
  }

  bool CanLoad(const OrtMemoryInfo& memory_info) const override {
    bool can_load = false;
    ORT_THROW_IF_ERROR(ToStatusAndRelease(impl_->CanLoad(impl_.get(), &memory_info, &can_load)));
    return can_load;
  }

  bool SupportsDataType(int32_t element_type) const override {
    if (impl_->SupportsDataType == nullptr) {
      return true;
    }
    bool supported = false;
    ORT_THROW_IF_ERROR(ToStatusAndRelease(impl_->SupportsDataType(
        impl_.get(), static_cast<ONNXTensorElementDataType>(element_type), &supported)));
    return supported;
  }

#if defined(__wasm__)
  Status LoadTensor(const Env&, const std::filesystem::path& path, FileOffsetType offset,
                    SafeInt<size_t> length, Tensor& tensor) const override {
    const OrtExternalDataSource source{ORT_API_VERSION, path.c_str(), offset, length, nullptr, nullptr};
    return Load(source, tensor);
  }
#else
  Status LoadTensor(const RandomAccessFile& file, FileOffsetType offset,
                    SafeInt<size_t> length, Tensor& tensor) const override {
    PathString path;
    ORT_RETURN_IF_ERROR(file.GetCanonicalPath(path));
    const OrtExternalDataSource source{ORT_API_VERSION, path.c_str(), offset, length, &file, Read};
    return Load(source, tensor);
  }
#endif

 private:
  Status Load(const OrtExternalDataSource& source, Tensor& tensor) const {
    ORT_RETURN_IF(source.file_offset < 0 || source.byte_size != tensor.SizeInBytes(),
                  "External data source range does not match the destination tensor.");
    // The wrapper borrows the tensor; releasing it must not release the session's allocation.
    OrtValue value{&tensor, DataTypeImpl::GetType<Tensor>(), [](void*) {}};
    return ToStatusAndRelease(impl_->LoadTensor(impl_.get(), &source, &value));
  }

#if !defined(__wasm__)
  static OrtStatus* ORT_API_CALL Read(const OrtExternalDataSource* source, size_t relative_offset,
                                      void* buffer, size_t length) noexcept {
    API_IMPL_BEGIN
    ORT_API_RETURN_IF(relative_offset > source->byte_size ||
                          length > source->byte_size - relative_offset,
                      ORT_INVALID_ARGUMENT, "External data read exceeds initializer range.");
    ORT_API_RETURN_IF(length != 0 && buffer == nullptr,
                      ORT_INVALID_ARGUMENT, "External data read buffer is null.");
    const auto offset = SafeInt<FileOffsetType>(source->file_offset) + relative_offset;
    const auto& file = *static_cast<const RandomAccessFile*>(source->read_context);
    ORT_API_RETURN_IF_STATUS_NOT_OK(file.Read(offset, gsl::span<char>(static_cast<char*>(buffer), length)));
    return nullptr;
    API_IMPL_END
  }
#endif

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExternalDataLoader);
  OrtExternalDataLoaderImplUniquePtr impl_;
};

inline std::unique_ptr<IExternalDataLoader> CreateExternalDataLoader(const OrtEp& ep) {
  if (ep.ort_version_supported < 32 || ep.CreateExternalDataLoader == nullptr) {
    return {};
  }
  OrtExternalDataLoaderImpl* loader = nullptr;
  auto* status = ep.CreateExternalDataLoader(&ep, &loader);
  OrtExternalDataLoaderImplUniquePtr owned_loader{loader};
  ORT_THROW_IF_ERROR(ToStatusAndRelease(status));
  if (owned_loader == nullptr) {
    return {};
  }
  return std::make_unique<ExternalDataLoader>(std::move(owned_loader));
}

}  // namespace plugin_ep
}  // namespace onnxruntime
