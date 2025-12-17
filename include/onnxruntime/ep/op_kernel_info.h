// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/pch.h instead."
#endif

#include <memory>

#include "core/common/status.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"

#include "api.h"
#include "node.h"

namespace onnxruntime {
struct DataTransferManager;
struct IExecutionProvider;
}  // namespace onnxruntime

namespace onnxruntime {
namespace ep {
namespace detail {

// A helper struct to cache kernel info data
//
// Because `KernelCreatePtrFn` is defined to use `const OrtKernelInfo&` as parameter type of the kernel creation function, `OpKernelInfo` has to be copyable.
// This means we cannot store cached data like `constant_input_tensors_` in `OpKernelInfo` directly to avoid ownership issues.
//
// As a workaround, we define this struct `KernelInfoCache` here to represent the cached data. An instance of this struct can be created and owned by the `OpKernel`
// during kernel creation, and then passed as pointer to the `OpKernelInfo` for later use.
struct KernelInfoCache {
  explicit KernelInfoCache(const OrtKernelInfo* kernel_info) : node(kernel_info) {
    Ort::ConstKernelInfo info{kernel_info};
    const int input_count = info.GetInputCount();
    constant_input_tensors.resize(input_count);
    for (int i = 0; i < input_count; ++i) {
      int is_constant = 0;
      Ort::ConstValue const_input = info.GetTensorConstantInput(i, &is_constant);
      if (is_constant && const_input != nullptr && const_input.IsTensor()) {
        auto type_and_shape_info = const_input.GetTypeInfo().GetTensorTypeAndShapeInfo();
        auto type = type_and_shape_info.GetElementType();
        auto shape_vec = type_and_shape_info.GetShape();

        auto memory_info = const_input.GetTensorMemoryInfo();
        MLDataType data_type = DataTypeImpl::TensorTypeFromONNXEnum(type);

        constant_input_tensors[i] = std::make_unique<Tensor>(data_type,
                                                             TensorShape{shape_vec},
                                                             const_cast<void*>(const_input.GetTensorRawData()),
                                                             OrtMemoryInfo{
                                                                 memory_info.GetAllocatorName(),
                                                                 memory_info.GetAllocatorType(),
                                                                 OrtDevice{
                                                                     static_cast<OrtDevice::DeviceType>(memory_info.GetDeviceType()),
                                                                     static_cast<OrtDevice::MemoryType>(memory_info.GetMemoryType()),
                                                                     static_cast<OrtDevice::VendorId>(memory_info.GetVendorId()),
                                                                     static_cast<OrtDevice::DeviceId>(memory_info.GetDeviceId()),

                                                                 },
                                                                 memory_info.GetMemoryType()});
      }
    }
  }
  Node node;
  std::vector<std::unique_ptr<Tensor>> constant_input_tensors;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelInfoCache);
};

struct OpKernelInfo {
  explicit OpKernelInfo(const OrtKernelInfo* info) : info_(info), cache_{nullptr} {
  }
  OpKernelInfo(const OrtKernelInfo* info, KernelInfoCache* cache) : info_(info), cache_(cache) {
  }

  const DataTransferManager& GetDataTransferManager() const noexcept {
    return (static_cast<const Ep*>(info_.GetEp()))->GetDataTransferManager();
  }
  const Node& node() const {
    return cache_->node;
  }
  const IExecutionProvider* GetExecutionProvider() const noexcept {
    return (static_cast<const Ep*>(info_.GetEp()))->EpImpl();
  }

  const Ort::ConstKernelInfo GetKernelInfo() const noexcept {
    return info_;
  }

  int GetInputCount() const noexcept {
    return info_.GetInputCount();
  }

  bool TryGetConstantInput(int input_index, const Tensor** constant_input_value) const {
    const Tensor* tensor = cache_->constant_input_tensors[input_index].get();
    if (tensor != nullptr) {
      *constant_input_value = tensor;
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
  KernelInfoCache* cache_;
};

}  // namespace detail
}  // namespace ep
}  // namespace onnxruntime
