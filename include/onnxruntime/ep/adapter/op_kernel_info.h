// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <memory>

#include "core/common/narrow.h"
#include "core/common/status.h"
#include "core/framework/config_options.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"

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
      const auto* core_kernel_info = reinterpret_cast<const ::onnxruntime::OpKernelInfo*>(kernel_info);
      execution_provider_ = core_kernel_info->GetExecutionProvider();
      ort_ep_ = execution_provider_ != nullptr ? execution_provider_->GetOrtEp() : nullptr;
      ep_impl_ = ort_ep_ != nullptr ? (static_cast<const Ep*>(ort_ep_))->EpImpl() : execution_provider_;

      Ort::ConstKernelInfo info{kernel_info};
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
    const ::onnxruntime::IExecutionProvider* execution_provider_{};
    const OrtEp* ort_ep_{};
    const ::onnxruntime::IExecutionProvider* ep_impl_{};
    std::vector<Tensor> constant_input_tensors;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelInfoCache);
  };

  explicit OpKernelInfo(const OrtKernelInfo* info) : info_(info), cache_{std::make_shared<KernelInfoCache>(info)} {
  }

  const DataTransferManager& GetDataTransferManager() const noexcept {
    return (static_cast<const Ep*>(cache_->ort_ep_))->GetDataTransferManager();
  }

  // Delegates to the core OpKernelInfo::GetAllocator so the adapter returns
  // exactly the same allocator the framework would provide for each OrtMemType.
  AllocatorPtr GetAllocator(OrtMemType mem_type) const {
    const auto* core_kernel_info = reinterpret_cast<const ::onnxruntime::OpKernelInfo*>(cache_->kernel_info_);
    return core_kernel_info->GetAllocator(mem_type);
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
    T tmp;
    return GetAttr<T>(name, &tmp).IsOK() ? tmp : default_value;
  }
  template <typename T>
  void GetAttrOrDefault(const std::string& name, T* value, const T& default_value) const {
    if (!GetAttr<T>(name, value).IsOK())
      *value = default_value;
  }
  template <typename T>
  [[nodiscard]] T GetAttr(const std::string& name) const {
    T value;
    ORT_THROW_IF_ERROR(GetAttr(name, &value));
    return value;
  }
  template <typename T>
  Status GetAttr(const std::string& name, T* value) const {
    try {
      *value = info_.GetAttribute<T>(name.c_str());
      return Status::OK();
    } catch (const Ort::Exception& ex) {
      return Status(onnxruntime::common::ONNXRUNTIME, ex.GetOrtErrorCode(), ex.what());
    }
  }
  template <typename T>
  Status GetAttrs(const std::string& name, std::vector<T>& values) const {
    try {
      values = info_.GetAttributes<T>(name.c_str());
      return Status::OK();
    } catch (const Ort::Exception& ex) {
      return Status(onnxruntime::common::ONNXRUNTIME, ex.GetOrtErrorCode(), ex.what());
    }
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
  const Ort::ConstKernelInfo info_;
  std::shared_ptr<KernelInfoCache> cache_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
