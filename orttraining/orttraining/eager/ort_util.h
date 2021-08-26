// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/session/onnxruntime_cxx_api.h>

#include "ort_backends.h"

namespace torch_ort {
namespace eager {

void CreateMLValue(onnxruntime::AllocatorPtr alloc, 
                   onnxruntime::MLDataType element_type, 
                   const std::vector<int64_t>& dims, 
                   OrtValue* p_mlvalue);

void CreateMLValue(void* data_ptr, onnxruntime::MLDataType element_type, const std::vector<int64_t>& dims, OrtValue* p_mlvalue);

template <typename T>
inline void CopyVectorToTensor(onnxruntime::ORTInvoker& invoker,
                               const std::vector<T>& value,
                               onnxruntime::Tensor& tensor) {
  const auto& execution_provider = invoker.GetCurrentExecutionProvider();

  OrtValue* ort_value;
  int64_t shape = value.size();
  OrtMemoryInfo cpuMemoryInfo;

  Ort::ThrowOnError(Ort::GetApi().CreateTensorWithDataAsOrtValue(
    &cpuMemoryInfo,
    const_cast<void*>(reinterpret_cast<const void*>(value.data())),
    value.size() * sizeof(T),
    &shape,
    1,
    Ort::TypeToTensorType<T>::type,
    &ort_value));

  ORT_THROW_IF_ERROR(execution_provider.GetDataTransfer()->CopyTensor(
    ort_value->Get<onnxruntime::Tensor>(),
    tensor));
}

// vector<bool> is specialized so we need to handle it separately
template <>
inline void CopyVectorToTensor<bool>(onnxruntime::ORTInvoker& /*invoker*/,
                                     const std::vector<bool>& value,
                                     onnxruntime::Tensor& tensor) {
  auto output_span = tensor.MutableDataAsSpan<bool>();
  for (size_t i = 0, end = value.size(); i < end; ++i) {
    output_span[i] = value[i];
  }
}

std::vector<int64_t> GetStrides(const std::vector<int64_t>& shape);

} // namespace eager
} // namespace torch_ort