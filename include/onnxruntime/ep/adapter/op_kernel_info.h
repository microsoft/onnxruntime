// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <memory>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/common/status.h"
#include "core/framework/config_options.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"

#include "allocator.h"
#include "node.h"
#include "kernel_def.h"
#include "tensor_helper.h"

namespace onnxruntime {
class DataTransferManager;
class IExecutionProvider;
}  // namespace onnxruntime

namespace onnxruntime {
namespace ep {
namespace adapter {

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::OpKernelInfo`.
/// </summary>
struct OpKernelInfo {
  //
  // A helper struct to cache kernel info data
  //
  // Because `KernelCreatePtrFn` is defined to use `const OpKernelInfo&` as parameter type of the kernel creation function, `OpKernelInfo` has to be copyable.
  // This means we cannot store cached data like `constant_input_tensors_` in `OpKernelInfo` directly to avoid ownership issues.
  //
  // As a workaround, we define this struct `KernelInfoCache` here to represent the cached data. We use a shared pointer to `KernelInfoCache` in `OpKernelInfo`
  // to manage the lifetime of the cached data.
  struct KernelInfoCache {
    explicit KernelInfoCache(const OrtKernelInfo* kernel_info) : kernel_info_(kernel_info) {
      Ort::ConstKernelInfo info{kernel_info};
      ort_ep_ = info.GetEp();
      ORT_ENFORCE(ort_ep_ != nullptr, "Plugin EP adapter requires a non-null OrtEp");
      ep_impl_ = static_cast<const Ep*>(ort_ep_)->EpImpl();

      const size_t input_count = info.GetInputCount();
      constant_input_tensors.resize(input_count);
      for (size_t i = 0; i < input_count; ++i) {
        int is_constant = 0;
        Ort::ConstValue const_input = info.GetTensorConstantInput(gsl::narrow_cast<int>(i), &is_constant);
        if (is_constant && const_input != nullptr && const_input.IsTensor()) {
          constant_input_tensors[i] = CreateTensorFromApiValue(const_cast<OrtValue*>(static_cast<const OrtValue*>(const_input)));
        }
      }
    }
    const OrtKernelInfo* kernel_info_;
    const OrtEp* ort_ep_{};
    const ::onnxruntime::IExecutionProvider* ep_impl_{};
    std::vector<Tensor> constant_input_tensors;

    mutable std::shared_mutex allocator_cache_mutex_;
    mutable InlinedHashMap<OrtMemType, AllocatorPtr> allocator_cache_;

    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelInfoCache);
  };

  explicit OpKernelInfo(const OrtKernelInfo* info) : info_(info), cache_{std::make_shared<KernelInfoCache>(info)} {
  }

  const DataTransferManager& GetDataTransferManager() const noexcept {
    return (static_cast<const Ep*>(cache_->ort_ep_))->GetDataTransferManager();
  }

  AllocatorPtr GetAllocator(OrtMemType mem_type) const {
    {
      std::shared_lock lock(cache_->allocator_cache_mutex_);
      auto it = cache_->allocator_cache_.find(mem_type);
      if (it != cache_->allocator_cache_.end()) {
        return it->second;
      }
    }

    std::unique_lock lock(cache_->allocator_cache_mutex_);
    // Double-check after acquiring exclusive lock
    auto it = cache_->allocator_cache_.find(mem_type);
    if (it != cache_->allocator_cache_.end()) {
      return it->second;
    }

    OrtAllocator* ort_allocator_raw = nullptr;
    Ort::Status status(Ort::GetApi().KernelInfoGetAllocator(cache_->kernel_info_, mem_type, &ort_allocator_raw));

    if (!status.IsOK() || ort_allocator_raw == nullptr) {
      cache_->allocator_cache_.emplace(mem_type, nullptr);
      return nullptr;
    }

    Ort::Allocator ort_allocator{ort_allocator_raw};
    auto allocator = std::make_shared<IAllocatorWrappingOrtAllocator>(std::move(ort_allocator));
    cache_->allocator_cache_.emplace(mem_type, allocator);
    return allocator;
  }

  Node node() const noexcept {
    return Node{cache_->kernel_info_};
  }
  const IExecutionProvider* GetExecutionProvider() const noexcept {
    return cache_->ep_impl_;
  }
  const OrtEp* GetOrtEp() const noexcept {
    return cache_->ort_ep_;
  }

  KernelDef GetKernelDef() const noexcept {
    return KernelDef{cache_->kernel_info_};
  }

  const Ort::ConstKernelInfo GetKernelInfo() const noexcept {
    return Ort::ConstKernelInfo{cache_->kernel_info_};
  }

  ConfigOptions GetConfigOptions() const noexcept {
    ConfigOptions config_options;
    config_options.configurations = info_.GetConfigEntries().GetKeyValuePairs();
    return config_options;
  }

  int GetInputCount() const noexcept {
    return gsl::narrow_cast<int>(info_.GetInputCount());
  }

  const std::vector<Tensor>& GetConstantInputTensors() const noexcept {
    return cache_->constant_input_tensors;
  }

  bool TryGetConstantInput(int input_index, const Tensor** constant_input_value) const {
    if (input_index < 0 || static_cast<size_t>(input_index) >= cache_->constant_input_tensors.size()) {
      return false;
    }
    const Tensor& tensor = cache_->constant_input_tensors[input_index];
    if (tensor.DataRaw() != nullptr) {
      *constant_input_value = &tensor;
      return true;
    }
    return false;
  }

  template <typename T>
  [[nodiscard]] T GetAttrOrDefault(const std::string& name, const T& default_value) const {
    T tmp{};
    return GetAttr<T>(name, &tmp).IsOK() ? tmp : default_value;
  }
  template <typename T>
  void GetAttrOrDefault(const std::string& name, T* value, const T& default_value) const {
    if (!GetAttr<T>(name, value).IsOK())
      *value = default_value;
  }
  template <typename T>
  [[nodiscard]] T GetAttr(const std::string& name) const {
    T value{};
    ORT_THROW_IF_ERROR(GetAttr(name, &value));
    return value;
  }
  template <typename T>
  Status GetAttr(const std::string& name, T* value) const {
    return GetAttrImpl(cache_->kernel_info_, name.c_str(), value);
  }
  template <typename T>
  Status GetAttrs(const std::string& name, std::vector<T>& values) const {
    return GetAttrsImpl(cache_->kernel_info_, name.c_str(), values);
  }

  Status GetAttrs(const std::string& name, TensorShapeVector& out) const {
    std::vector<int64_t> shape;
    Status status = GetAttrs<int64_t>(name, shape);
    if (status.IsOK()) {
      out.reserve(shape.size());
      out.assign(shape.begin(), shape.end());
    }
    return status;
  }

  template <typename T>
  [[nodiscard]] std::vector<T> GetAttrsOrDefault(const std::string& name,
                                                 const std::vector<T>& default_value = {}) const {
    std::vector<T> tmp;
    return GetAttrs<T>(name, tmp).IsOK() ? tmp : default_value;
  }
  [[nodiscard]] TensorShapeVector GetAttrsOrDefault(const std::string& name,
                                                    const TensorShapeVector& default_value = {}) const {
    TensorShapeVector tmp;
    return GetAttrs(name, tmp).IsOK() ? tmp : default_value;
  }

 private:
  // A missing optional attribute is normal control flow for GetAttrOrDefault()/GetAttrsOrDefault(), so these
  // accessors must not depend on catching the exception thrown by the Ort:: C++ wrappers: a plugin EP may be
  // compiled with C++ exception catching disabled. The ORT Web build is one such case -- it builds with
  // `-sDISABLE_EXCEPTION_CATCHING` everywhere except the C API boundary, so a `catch` in this inlined header
  // never matches and the exception escapes all the way out of session creation. Use the non-throwing C API
  // directly instead.
  static Status ToStatus(OrtStatus* ort_status) {
    if (ort_status == nullptr) {
      return Status::OK();
    }
    const Ort::Status status{ort_status};  // takes ownership
    return Status(onnxruntime::common::ONNXRUNTIME, status.GetErrorCode(), status.GetErrorMessage());
  }

  static Status GetAttrImpl(const OrtKernelInfo* info, const char* name, float* out) {
    return ToStatus(Ort::GetApi().KernelInfoGetAttribute_float(info, name, out));
  }

  static Status GetAttrImpl(const OrtKernelInfo* info, const char* name, int64_t* out) {
    return ToStatus(Ort::GetApi().KernelInfoGetAttribute_int64(info, name, out));
  }

  static Status GetAttrImpl(const OrtKernelInfo* info, const char* name, std::string* out) {
    size_t size = 0;
    // Feed nullptr for the data buffer to query the true size of the string attribute.
    ORT_RETURN_IF_ERROR(ToStatus(Ort::GetApi().KernelInfoGetAttribute_string(info, name, nullptr, &size)));

    std::string value;
    value.resize(size);
    ORT_RETURN_IF_ERROR(ToStatus(Ort::GetApi().KernelInfoGetAttribute_string(info, name, value.data(), &size)));
    value.resize(size - 1);  // remove the terminating character '\0'
    *out = std::move(value);
    return Status::OK();
  }

  static Status GetAttrsImpl(const OrtKernelInfo* info, const char* name, std::vector<float>& out) {
    size_t size = 0;
    ORT_RETURN_IF_ERROR(ToStatus(Ort::GetApi().KernelInfoGetAttributeArray_float(info, name, nullptr, &size)));

    std::vector<float> values(size);
    ORT_RETURN_IF_ERROR(ToStatus(Ort::GetApi().KernelInfoGetAttributeArray_float(info, name, values.data(), &size)));
    out.swap(values);
    return Status::OK();
  }

  static Status GetAttrsImpl(const OrtKernelInfo* info, const char* name, std::vector<int64_t>& out) {
    size_t size = 0;
    ORT_RETURN_IF_ERROR(ToStatus(Ort::GetApi().KernelInfoGetAttributeArray_int64(info, name, nullptr, &size)));

    std::vector<int64_t> values(size);
    ORT_RETURN_IF_ERROR(ToStatus(Ort::GetApi().KernelInfoGetAttributeArray_int64(info, name, values.data(), &size)));
    out.swap(values);
    return Status::OK();
  }

  static Status GetAttrsImpl(const OrtKernelInfo* info, const char* name, std::vector<std::string>& out) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t size = 0;
    ORT_RETURN_IF_ERROR(
        ToStatus(Ort::GetApi().KernelInfoGetAttributeArray_string(info, name, allocator, nullptr, &size)));
    if (size == 0) {
      out.clear();
      return Status::OK();
    }

    char** raw_values = nullptr;
    ORT_RETURN_IF_ERROR(
        ToStatus(Ort::GetApi().KernelInfoGetAttributeArray_string(info, name, allocator, &raw_values, &size)));

    // The allocator owns both the array and every string in it, so release them on all exit paths: constructing the
    // std::string copies below can throw. Free through the OrtAllocator function pointer rather than
    // Ort::AllocatorWithDefaultOptions::Free(), which throws on failure and so must not run in this deleter.
    OrtAllocator* raw_allocator = allocator;
    auto free_raw_values = [raw_allocator, size](char** values_to_free) {
      for (size_t i = 0; i < size; ++i) {
        if (values_to_free[i] != nullptr) {
          raw_allocator->Free(raw_allocator, values_to_free[i]);
        }
      }
      raw_allocator->Free(raw_allocator, values_to_free);
    };
    std::unique_ptr<char*, decltype(free_raw_values)> raw_values_guard{raw_values, std::move(free_raw_values)};

    std::vector<std::string> values;
    values.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      values.emplace_back(raw_values[i] != nullptr ? raw_values[i] : "");
    }
    out.swap(values);
    return Status::OK();
  }

  const Ort::ConstKernelInfo info_;
  std::shared_ptr<KernelInfoCache> cache_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
