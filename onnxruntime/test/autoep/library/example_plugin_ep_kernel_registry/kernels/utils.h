// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../../plugin_ep_utils.h"

/// <summary>
/// Gets an OrtDataType for a tensor type. Throws on error.
/// </summary>
/// <param name="elem_type"></param>
/// <returns></returns>
inline const OrtDataType* GetTensorType(ONNXTensorElementDataType elem_type) {
  const OrtEpApi& ep_api = Ort::GetEpApi();
  const OrtDataType* result = nullptr;

  Ort::ThrowOnError(ep_api.GetTensorDataType(elem_type, &result));
  return result;
}

/// <summary>
/// Copy a tensor using a OrtDataTransferImpl instance. Used by kernel implementations to copy
/// tensors that my reside on different devices.
/// </summary>
/// <param name="data_transfer_impl"></param>
/// <param name="src_tensor"></param>
/// <param name="dst_tensor"></param>
/// <returns></returns>
inline OrtStatus* CopyTensor(OrtDataTransferImpl& data_transfer_impl,
                             Ort::ConstValue src_tensor, Ort::UnownedValue dst_tensor) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  const OrtMemoryDevice* src_device = Ort::GetEpApi().MemoryInfo_GetMemoryDevice(src_tensor.GetTensorMemoryInfo());
  const OrtMemoryDevice* dst_device = Ort::GetEpApi().MemoryInfo_GetMemoryDevice(dst_tensor.GetTensorMemoryInfo());

  RETURN_IF(!data_transfer_impl.CanCopy(&data_transfer_impl, src_device, dst_device), Ort::GetApi(),
            "OrtDataTransferImpl cannot copy src tensor to dst tensor.");

  auto src_type_shape = src_tensor.GetTensorTypeAndShapeInfo();
  auto dst_type_shape = dst_tensor.GetTensorTypeAndShapeInfo();
  bool same_elem_type = src_type_shape.GetElementType() == dst_type_shape.GetElementType();
  bool same_elem_count = src_type_shape.GetElementCount() == dst_type_shape.GetElementCount();
  RETURN_IF(!same_elem_type || !same_elem_count, Ort::GetApi(), "Cannot copy tensors of different types or size.");

  std::array<const OrtValue*, 1> src_tensors = {src_tensor};
  std::array<OrtValue*, 1> dst_tensors = {dst_tensor};

  RETURN_IF_ERROR(data_transfer_impl.CopyTensors(&data_transfer_impl, src_tensors.data(), dst_tensors.data(),
                                                 /*streams*/ nullptr, src_tensors.size()));

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/// <summary>
/// Contains information to create a kernel: kernel definition, creation function + state.
/// </summary>
struct KernelCreateInfo {
  KernelCreateInfo() = default;
  KernelCreateInfo(Ort::KernelDef def, OrtKernelCreateFunc func, void* state)
      : kernel_def{std::move(def)}, kernel_create_func{func}, kernel_create_func_state{state} {}

  Ort::KernelDef kernel_def{nullptr};
  OrtKernelCreateFunc kernel_create_func = nullptr;
  void* kernel_create_func_state = nullptr;
};

using BuildKernelCreateInfoFn = OrtStatus* (*)(const char*, void*, KernelCreateInfo*);

template <typename T>
OrtStatus* BuildKernelCreateInfo(const char* ep_name, void* create_func_state, /*out*/ KernelCreateInfo* result);

template <>
inline OrtStatus* BuildKernelCreateInfo<void>(const char* /*ep_name*/, void* /*create_func_state*/,
                                              /*out*/ KernelCreateInfo* result) {
  result->kernel_def = Ort::KernelDef{nullptr};
  result->kernel_create_func = nullptr;
  result->kernel_create_func_state = nullptr;
  return nullptr;
}

static constexpr const char* kOnnxDomain = "";

// Naming convention for operator kernel classes with a start and end version range.
#define ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, name) \
  example_ep_##name##_##domain##_ver##startver##_##endver

// Naming convention for operator kernel classes for a single version
#define ONNX_OPERATOR_KERNEL_CLASS_NAME(domain, version, name) \
  ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, version, version, name)

// Defines a function of type BuildKernelCreateInfoFn for a kernel implementation with a start and end version range.
#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver, builder, kernel_class)    \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, name);                  \
  template <>                                                                                       \
  OrtStatus*                                                                                        \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, name)>( \
      const char* ep_name,                                                                          \
      void* create_kernel_state,                                                                    \
      KernelCreateInfo* result) {                                                                   \
    try {                                                                                           \
      Ort::KernelDef kernel_def = builder.SetOperatorType(#name)                                    \
                                      .SetDomain(domain)                                            \
                                      .SetSinceVersion(startver, endver)                            \
                                      .SetExecutionProvider(ep_name)                                \
                                      .Build();                                                     \
                                                                                                    \
      auto kernel_create_func = [](void* state, const OrtKernelInfo* info,                          \
                                   OrtKernelImpl** kernel_out) noexcept -> OrtStatus* {             \
        RETURN_IF(kernel_out == nullptr, Ort::GetApi(),                                             \
                  "OrtKernelCreateFunc received a NULL kernel_out argument");                       \
                                                                                                    \
        *kernel_out = nullptr;                                                                      \
        RETURN_IF_ERROR(kernel_class::CreateKernelImpl(info, state, *kernel_out));                  \
        return nullptr;                                                                             \
      };                                                                                            \
                                                                                                    \
      *result = KernelCreateInfo(std::move(kernel_def), kernel_create_func, create_kernel_state);   \
    } catch (const Ort::Exception& ex) {                                                            \
      Ort::Status status(ex);                                                                       \
      return status.release();                                                                      \
    } catch (const std::exception& ex) {                                                            \
      Ort::Status status(ex.what(), ORT_EP_FAIL);                                                   \
      return status.release();                                                                      \
    }                                                                                               \
    return nullptr;                                                                                 \
  }

// Defines a function of type BuildKernelCreateInfoFn for a kernel implementation with a start version.
#define ONNX_OPERATOR_KERNEL_EX(name, domain, version, builder, kernel_class) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, version, version, builder, kernel_class)
