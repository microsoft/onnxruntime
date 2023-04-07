// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/session/onnxruntime_cxx_api.h>

#include "ort_backends.h"

namespace torch_ort {
namespace eager {

template <typename T>
inline void CopyVectorToTensor(onnxruntime::ORTInvoker& invoker,
                               const T* value_ptr,
                               int64_t size,
                               onnxruntime::Tensor& tensor) {
  const auto& execution_provider = invoker.GetCurrentExecutionProvider();

  OrtValue* ort_value;
  OrtMemoryInfo cpuMemoryInfo;

  Ort::ThrowOnError(Ort::GetApi().CreateTensorWithDataAsOrtValue(
    &cpuMemoryInfo,
    const_cast<void*>(reinterpret_cast<const void*>(value_ptr)),
    size * sizeof(T),
    &size,
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
                                     const bool* value_ptr,
                                     int64_t size,
                                     onnxruntime::Tensor& tensor) {
  auto output_span = tensor.MutableDataAsSpan<bool>();
  for (size_t i = 0, end = size; i < end; ++i) {
    output_span[i] = value_ptr[i];
  }
}

onnxruntime::TensorShapeVector GetStrides(gsl::span<const int64_t> shape);

} // namespace eager
} // namespace torch_ort
